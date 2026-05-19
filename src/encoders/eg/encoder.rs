use crate::core::sealed;
use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::SpaceUsage;
use crate::encoders::eg::quantizer::EgQuantizer;
use crate::encoders::eg::common::find_best_k_rice;
use bytemuck::cast_slice;
use dsi_bitstream::impls::BufBitReader;
use dsi_bitstream::prelude::*;
use std::borrow::Cow;
use rusty_perm::PermApply as _;
use rusty_perm::PermFromSorting as _;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct EgEncoder<Q: EgQuantizer> {
    pub dim: usize,
    pub quantizer: Q,
    pub component_mapping: Option<Box<[u16]>>,
    pub inverse_component_mapping: Option<Box<[u16]>>,
}

impl<Q: EgQuantizer> PartialEq for EgEncoder<Q>
where
    Q: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
            && self.quantizer == other.quantizer
            && self.component_mapping == other.component_mapping
    }
}

impl<Q: EgQuantizer> sealed::Sealed for EgEncoder<Q> {}

impl<Q: EgQuantizer> EgEncoder<Q> {
    pub fn new_with_quantizer(input_dim: usize, quantizer: Q) -> Self {
        Self {
            dim: input_dim,
            quantizer,
            component_mapping: None,
            inverse_component_mapping: None,
        }
    }

    pub fn compute_inverse_mapping(component_mapping: &[u16]) -> Vec<u16> {
        let dim = component_mapping.len();
        let mut inverse = vec![0u16; dim];
        let mut seen = vec![false; dim];

        for (old, &new) in component_mapping.iter().enumerate() {
            let new = new as usize;
            assert!(
                new < dim,
                "component_mapping maps component {} to out-of-bounds index {} (dim={})",
                old,
                new,
                dim
            );
            assert!(
                !seen[new],
                "component_mapping is not a permutation: duplicate mapped index {}",
                new
            );
            seen[new] = true;
            inverse[new] = old as u16;
        }

        inverse
    }

    #[inline]
    pub fn component_mapping(&self) -> Option<&[u16]> {
        self.component_mapping.as_deref()
    }

    #[inline]
    pub fn inverse_component_mapping(&self) -> Option<&[u16]> {
        self.inverse_component_mapping.as_deref()
    }

    pub fn train_components<'a, V>(
        &mut self,
        training_data: impl Iterator<Item = SparseVectorView<'a, u16, V>>,
    ) where
        V: crate::ValueType,
    {
        let components_iter = training_data.map(|v| v.components());
        let permutation =
            crate::utils::permute_components_with_bisection(self.dim, components_iter);
        let component_mapping: Vec<u16> = permutation.iter().map(|i| *i as u16).collect();
        let inverse = Self::compute_inverse_mapping(&component_mapping);
        self.component_mapping = Some(component_mapping.into_boxed_slice());
        self.inverse_component_mapping = Some(inverse.into_boxed_slice());
    }
}

impl<Q> SparseDataEncoder for EgEncoder<Q>
where
    Q: EgQuantizer<InputValue = f32>,
{
    type InputComponentType = u16;
    type InputValueType = Q::InputValue;
    type OutputComponentType = u16;
    type OutputValueType = Q::InputValue; // This is a bit loose but okay

    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let bytes = cast_slice::<u64, u8>(encoded.data());
        let slices = cast_slice::<u64, u32>(encoded.data());
        let mut reader = BufBitReader::<LE, _>::new(MemWordReader::new_inf(slices));

        let n = reader.read_vbyte_le().unwrap() as usize;
        let k = reader.read_bits(4).unwrap() as usize;
        let values_8 = &bytes[bytes.len() - n..];
        let mut decoded_comp = Vec::with_capacity(n);
        let mut values_raw: Vec<u8> = Vec::with_capacity(n);
        let mut last_comp = 0u32;
        for i in 0..n {
            let gap = reader.read_exp_golomb(k).expect("Failed to read EG") as u32;
            last_comp += gap;
            decoded_comp.push(last_comp as u16);
            values_raw.push(values_8[i]);
        }

        if let Some(component_mapping) = self.component_mapping() {
            let inverse: Cow<'_, [u16]> = match self.inverse_component_mapping() {
                Some(inverse) => Cow::Borrowed(inverse),
                None => Cow::Owned(Self::compute_inverse_mapping(component_mapping)),
            };

            for c in decoded_comp.iter_mut() {
                *c = inverse[*c as usize];
            }

            // For per-component quantizers (e.g., ScalarU8Quantizer), we must decode AFTER
            // inverse mapping and permutation to ensure each raw byte is decoded with its
            // original component's scale. Permuting raw bytes along with their components
            // ensures proper alignment.
            let permutation = rusty_perm::PermD::from_sort(decoded_comp.as_slice());
            permutation.apply(values_raw.as_mut_slice()).unwrap();
            permutation.apply(decoded_comp.as_mut_slice()).unwrap();
        }

        let values: Vec<f32> = decoded_comp
            .iter()
            .zip(values_raw.iter())
            .map(|(&c, &v)| self.quantizer.decode_value(c, v))
            .collect();

        SparseVectorOwned::new(decoded_comp, values)
    }
}

impl<Q> PackedSparseVectorEncoder for EgEncoder<Q>
where
    Q: EgQuantizer<InputValue = f32>,
{
    type PackedDataType = u64;
    fn push_encoded<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, u16, Q::InputValue>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<u64>,
    {
        let mut q_values: Vec<u8> = input
            .components()
            .iter()
            .zip(input.values().iter())
            .map(|(&c, &v)| self.quantizer.encode_value(c, v))
            .collect();

        let mut q_components: Vec<u16> = if let Some(mapping) = &self.component_mapping {
            input
                .components()
                .iter()
                .map(|&c| mapping[c as usize])
                .collect()
        } else {
            input.components().to_vec()
        };

        if self.component_mapping.is_some() {
            let permutation = rusty_perm::PermD::from_sort(q_components.as_slice());
            permutation.apply(q_values.as_mut_slice()).unwrap();
            permutation.apply(q_components.as_mut_slice()).unwrap();
        }

        let n = q_values.len();

        let gaps = q_components
            .iter()
            .scan(0u32, |state, &comp| {
                let gap = (comp as u32) - *state;
                *state = comp as u32;
                Some(gap)
            })
            .collect::<Vec<u32>>();

        let best_k = find_best_k_rice(&gaps) as usize;

        let mut bit_writer = <BufBitWriter<LE, _>>::new(MemWordWriterVec::<u8, _>::new(Vec::new()));
        // Header: n (vbyte) + best_k (4 bits)
        bit_writer.write_vbyte_le(n as u64).unwrap();
        bit_writer.write_bits(best_k as u64, 4).unwrap();
        // Gaps compressed with Exp-Golomb with parameter k
        for &gap in &gaps {
            bit_writer.write_exp_golomb(gap as u64, best_k).unwrap();
        }

        let mut payload: Vec<u8> = Vec::new();
        payload.extend_from_slice(bit_writer.into_inner().unwrap().into_inner().as_slice());

        // Padding to ensure q_values are aligned at the end of the payload on an 8-byte boundary,
        // as the entire payload must be compatible with u64 alignment.
        let current_len = payload.len();
        let padding_needed = (8 - (current_len + n) % 8) % 8;
        for _ in 0..padding_needed {
            payload.push(0);
        }

        // Values aligned at the end of the payload: this allows the decoder to read them
        // starting from the end in compute distance, while reading the header and gaps from the start
        payload.extend_from_slice(&q_values);

        let data = payload
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()));

        output.extend(data);
    }
}

impl<Q> VectorEncoder for EgEncoder<Q>
where
    Q: EgQuantizer<InputValue = f32>,
{
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u16, Q::InputValue>;
    type QueryVector<'q> = SparseVectorView<'q, u16, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;
    type Evaluator<'e>
        = EgQueryEvaluator<'e, Q>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        EgQueryEvaluator::new(query, self)
    }
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        EgQueryEvaluator::new(decoded.as_view(), self)
    }
    fn output_dim(&self) -> usize {
        self.dim
    }
    fn input_dim(&self) -> usize {
        self.dim
    }
}

pub struct EgQueryEvaluator<'a, Q> where
    Q: EgQuantizer<InputValue = f32>,{
    pub dense_query: Vec<f32>,
    pub encoder: &'a EgEncoder<Q>,
}

impl<'a, Q> EgQueryEvaluator<'a, Q>
where
    Q: EgQuantizer<InputValue = f32>,
{
    pub fn new(query: SparseVectorView<'_, u16, f32>, encoder: &'a EgEncoder<Q>) -> Self {
        let mut dense_query = vec![0.0f32; encoder.dim];
        for (&c, &v) in query.components().iter().zip(query.values().iter()) {
            let mapped_c = if let Some(mapping) = &encoder.component_mapping {
                mapping[c as usize]
            } else {
                c
            };
            // `query_value` expects the original component index (unmapped),
            // but the dense query is stored at the mapped index.
            dense_query[mapped_c as usize] = encoder.quantizer.query_value(c, v);
        }
        Self {
            dense_query,
            encoder,
        }
    }
}

impl<'a, 'v, Q> QueryEvaluator<PackedVectorView<'v, u64>> for EgQueryEvaluator<'a, Q>
where
    Q: EgQuantizer<InputValue = f32>,
{
    type Distance = DotProduct;
    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> Self::Distance {
        let bytes = cast_slice::<u64, u8>(vector.data());
        let slices = cast_slice::<u64, u32>(vector.data());
        let mut reader = BufBitReader::<LE, _>::new(MemWordReader::new_inf(slices));

        let n = reader.read_vbyte_le().unwrap() as usize;
        let k = reader.read_bits(4).unwrap() as usize;
        let values = &bytes[bytes.len() - n..];
        let query = self.dense_query.as_slice();

        let total_sum = values
            .iter()
            .fold((0f32, 0u32), |(acc, last_res), &value| {
                let gap = reader
                    .read_exp_golomb(k)
                    .expect("Failed to read EG component") as u32;
                let last_res = last_res + gap;
                let comp = unsafe { *query.get_unchecked(last_res as usize) };
                (acc + comp * value as f32, last_res)
            })
            .0;

        DotProduct(total_sum * self.encoder.quantizer.scale())
    }
}

impl<Q: EgQuantizer> SpaceUsage for EgEncoder<Q> {
    fn space_usage_bytes(&self) -> usize {
        let size_of_mapping = match &self.component_mapping {
            Some(component_mapping) => component_mapping.space_usage_bytes(),
            None => std::mem::size_of::<Option<Box<[u16]>>>(),
        };
        let size_of_inverse_mapping = match &self.inverse_component_mapping {
            Some(inverse_component_mapping) => inverse_component_mapping.space_usage_bytes(),
            None => std::mem::size_of::<Option<Box<[u16]>>>(),
        };
        size_of_mapping + size_of_inverse_mapping + self.dim.space_usage_bytes() + self.quantizer.space_usage_bytes()
    }
}
