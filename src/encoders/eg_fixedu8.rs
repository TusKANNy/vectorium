use rusty_perm::PermApply as _;
use rusty_perm::PermFromSorting as _;

use crate::core::sealed;
use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::{FixedU8Q, SpaceUsage};
use bytemuck::cast_slice;
use dsi_bitstream::impls::BufBitReader;
use dsi_bitstream::prelude::*;
use std::borrow::Cow;

const FIXED_U8_SCALE: f32 = 1.0 / ((1u32 << FixedU8Q::FRAC_NBITS) as f32);

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct EgFixedU8Encoder {
    dim: usize,
    component_mapping: Option<Box<[u16]>>,
    #[serde(default)]
    inverse_component_mapping: Option<Box<[u16]>>,
}

impl PartialEq for EgFixedU8Encoder {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim && self.component_mapping == other.component_mapping
    }
}

impl sealed::Sealed for EgFixedU8Encoder {}

impl EgFixedU8Encoder {
    pub fn new(input_dim: usize) -> Self {
        Self {
            dim: input_dim,
            component_mapping: None,
            inverse_component_mapping: None,
        }
    }

    fn compute_inverse_mapping(component_mapping: &[u16]) -> Vec<u16> {
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

    pub fn train<'a, V>(
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

    fn find_best_k_rice(gaps: &[u32]) -> u8 {
        (0..16)
            .map(|k| {
                let total_len: usize = gaps.iter().map(|&g| len_exp_golomb(g as u64, k)).sum();
                (k, total_len as f64 / gaps.len() as f64)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(k, _)| k as u8)
            .unwrap_or(0)
    }
}

impl SparseDataEncoder for EgFixedU8Encoder {
    type InputComponentType = u16;
    type InputValueType = FixedU8Q;
    type OutputComponentType = u16;
    type OutputValueType = FixedU8Q;

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
        let mut values = Vec::with_capacity(n);
        let mut last_comp = 0u32;
        for i in 0..n {
            let gap = reader.read_exp_golomb(k).expect("Failed to read EG") as u32;
            last_comp += gap;
            decoded_comp.push(last_comp as u16);
            values.push(values_8[i] as f32 * FIXED_U8_SCALE);
        }

        if let Some(component_mapping) = self.component_mapping() {
            let inverse: Cow<'_, [u16]> = match self.inverse_component_mapping() {
                Some(inverse) => Cow::Borrowed(inverse),
                None => Cow::Owned(Self::compute_inverse_mapping(component_mapping)),
            };

            for c in decoded_comp.iter_mut() {
                *c = inverse[*c as usize];
            }

            let permutation = rusty_perm::PermD::from_sort(decoded_comp.as_slice());
            permutation.apply(values.as_mut_slice()).unwrap();
            permutation.apply(decoded_comp.as_mut_slice()).unwrap();
        }

        SparseVectorOwned::new(decoded_comp, values)
    }
}

impl PackedSparseVectorEncoder for EgFixedU8Encoder {
    type PackedDataType = u64;
    fn push_encoded<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, u16, FixedU8Q>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<u64>,
    {
        let mut q_values: Vec<u8> = input.values().iter().map(|v| v.to_bits()).collect();
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

        let best_k = Self::find_best_k_rice(&gaps) as usize;

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

        // Layout of the encoded vector in u64 words:
        // [n (vbyte) | best_k (4 bits) | Exp-Golomb encoded gaps | padding | q_values (u8)]
        output.extend(data);
    }
}

impl VectorEncoder for EgFixedU8Encoder {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u16, FixedU8Q>;
    type QueryVector<'q> = SparseVectorView<'q, u16, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;
    type Evaluator<'e>
        = EgFixedU8QueryEvaluator
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        EgFixedU8QueryEvaluator::new(query, self)
    }
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        EgFixedU8QueryEvaluator::new(decoded.as_view(), self)
    }
    fn output_dim(&self) -> usize {
        self.dim
    }
    fn input_dim(&self) -> usize {
        self.dim
    }
}

pub struct EgFixedU8QueryEvaluator{
    dense_query: Vec<f32>,
}

impl EgFixedU8QueryEvaluator {
    pub fn new(query: SparseVectorView<'_, u16, f32>, encoder: &EgFixedU8Encoder) -> Self {
        let mut dense_query = vec![0.0f32; encoder.dim];
        for (&c, &v) in query.components().iter().zip(query.values().iter()) {
            let mapped_c = if let Some(mapping) = &encoder.component_mapping {
                mapping[c as usize]
            } else {
                c
            };
            dense_query[mapped_c as usize] = v;
        }
        Self {
            dense_query,
        }
    }
}

impl<'v> QueryEvaluator<PackedVectorView<'v, u64>> for EgFixedU8QueryEvaluator {
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

        DotProduct(total_sum * FIXED_U8_SCALE)
    }
}

impl SpaceUsage for EgFixedU8Encoder {
    fn space_usage_bytes(&self) -> usize {
        let size_of_mapping = match &self.component_mapping {
            Some(component_mapping) => component_mapping.space_usage_bytes(),
            None => std::mem::size_of::<Option<Box<[u16]>>>(),
        };
        let size_of_inverse_mapping = match &self.inverse_component_mapping {
            Some(inverse_component_mapping) => inverse_component_mapping.space_usage_bytes(),
            None => std::mem::size_of::<Option<Box<[u16]>>>(),
        };
        size_of_mapping + size_of_inverse_mapping + self.dim.space_usage_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FromF32;
    use crate::core::distances::Distance;
    use crate::core::vector::{PackedVectorView, SparseVectorView};
    use num_traits::ToPrimitive;

    fn fixed(val: f32) -> FixedU8Q {
        FixedU8Q::from_f32_saturating(val)
    }

    #[test]
    fn eg_compute_distance_basic() {
        let encoder = EgFixedU8Encoder::new(100);
        let binding = [
            fixed(1.0),
            fixed(3.0),
            fixed(2.0),
            fixed(3.5),
            fixed(1.5),
            fixed(2.0),
            fixed(1.0),
            fixed(2.0),
        ];
        // Components: 0, 4, 8, 24, 36, 48, 53, 90
        let input = SparseVectorView::new(&[0, 4, 8, 24, 36, 48, 53, 90], &binding);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let query = SparseVectorView::new(
            &[2, 4, 6, 8, 24, 70, 90],
            &[0.5, 1.5, 2.5, 1.0, 2.0, 1.0, 2.0],
        );
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        // v = (0, 1.0), (4, 3.0), (8, 2.0), (24, 3.5), (36, 1.5), (48, 2.0), (53, 1.0), (90, 2.0)
        // query = (2, 0.5), (4, 1.5), (6, 2.5), (8, 1.0), (24, 2.0), (70, 1.0), (90, 2.0)
        let expected = 1.5 * 3.0 + 1.0 * 2.0 + 2.0 * 3.5 + 2.0 * 2.0;
        assert!((dist.distance() - expected).abs() < 1e-5);

        let dist_plain = evaluator.compute_distance(PackedVectorView::new(&buffer));
        assert!((dist_plain.distance() - expected).abs() < 1e-5);
    }

    fn verify_eg(gaps: &[u32]) {
        let num_vals = gaps.len();
        let mut components = Vec::new();
        let mut curr = 0u32;
        for &g in gaps {
            curr += g;
            assert!(curr < u16::MAX as u32);
            components.push(curr as u16);
        }

        let encoder = EgFixedU8Encoder::new(u16::MAX as usize + 1);
        let values: Vec<_> = (0..num_vals)
            .map(|i| fixed(1.0 + (i % 7) as f32 / 10.0))
            .collect();
        let input = SparseVectorView::new(&components, &values);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let query_vals: Vec<_> = (0..num_vals).map(|i| 0.1 + (i % 5) as f32 / 10.0).collect();
        let query = SparseVectorView::new(&components, &query_vals);

        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let mut expected = 0.0f32;
        for i in 0..num_vals {
            expected += values[i].to_f32().unwrap() * query_vals[i];
        }

        assert!((dist.distance() - expected).abs() < 1e-3);
    }

    #[test]
    fn test_eg_dist_various() {
        // Just a few selected ones to verify logic
        verify_eg(&[1; 64]);
        verify_eg(&[3; 64]);
        verify_eg(&[7; 64]);
        verify_eg(&[15; 64]);
        verify_eg(&[31; 64]);
        verify_eg(&[127; 8]);
    }

    #[test]
    fn eg_decode_roundtrip() {
        let mut encoder = EgFixedU8Encoder::new(100);
        let binding = [fixed(1.0), fixed(3.0), fixed(2.0), fixed(3.5)];
        let input_components = [0, 4, 8, 24];
        let input0 = SparseVectorView::new(&input_components, &binding);
        let binding1 = [fixed(3.0), fixed(2.0), fixed(3.5)];
        let input_components1 = [4, 8, 24];
        let input1 = SparseVectorView::new(&input_components1, &binding1);

        encoder.train([input0.clone(), input1.clone()].into_iter());

        let mut buffer0 = Vec::new();
        encoder.push_encoded(input0, &mut buffer0);

        let mut buffer1 = Vec::new();
        encoder.push_encoded(input1, &mut buffer1);

        let decoded0 = encoder.decode_vector(PackedVectorView::new(&buffer0));
        let decoded1 = encoder.decode_vector(PackedVectorView::new(&buffer1));

        assert_eq!(decoded0.components(), &input_components);
        let decoded_vals0 = decoded0.values();
        assert_eq!(decoded_vals0.len(), 4);
        assert!((decoded_vals0[0] - 1.0).abs() < 1e-6);
        assert!((decoded_vals0[1] - 3.0).abs() < 1e-6);
        assert!((decoded_vals0[2] - 2.0).abs() < 1e-6);
        assert!((decoded_vals0[3] - 3.5).abs() < 1e-6);

        assert_eq!(decoded1.components(), &[4, 8, 24]);
        let decoded_vals1 = decoded1.values();
        assert_eq!(decoded_vals1.len(), 3);
        assert!((decoded_vals1[0] - 3.0).abs() < 1e-6);
        assert!((decoded_vals1[1] - 2.0).abs() < 1e-6);
        assert!((decoded_vals1[2] - 3.5).abs() < 1e-6);
    }
}
