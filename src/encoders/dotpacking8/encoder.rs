use bytemuck::{cast_slice, from_bytes};
use rusty_perm::{PermApply as _, PermFromSorting as _};
use std::borrow::Cow;
use std::simd::{Mask, Simd, StdFloat, prelude::*};

use crate::SpaceUsage;
use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::encoders::dotpacking8::common::{
    DotPacking8Iter, compute_safe_simd_padding, encode_blocks, simd_prefix_sum,
};
use crate::encoders::dotpacking8::quantizer::DotPacking8Quantizer;

const N: usize = 8;

#[derive(Clone)]
pub struct DotPacking8View<'a> {
    pub n: usize,
    pub selectors: &'a [u8],
    pub payloads: &'a [u8],
    pub values: &'a [u8],
}

impl<'a> DotPacking8View<'a> {
    pub unsafe fn from_unchecked_slice(slice: &'a [u64]) -> Self {
        let bytes = cast_slice::<u64, u8>(slice);
        let n = from_bytes::<u16>(&bytes[0..2]).to_le() as usize;

        let n_blocks_total = (n + N - 1) / N;
        let selectors_size = (n_blocks_total + 1) / 2;
        let selectors_end = 2 + selectors_size;
        let values_start_idx = bytes.len() - n;

        Self {
            n,
            selectors: &bytes[2..selectors_end],
            payloads: &bytes[selectors_end..values_start_idx],
            values: &bytes[values_start_idx..],
        }
    }

    pub fn iter_raw(&self) -> DotPacking8Iter<'a> {
        DotPacking8Iter {
            n: self.n,
            selectors: self.selectors,
            bulk_blocks: self.n / N,
            block_idx: 0,
            payload_ptr: self.payloads.as_ptr(),
            val_ptr: self.values.as_ptr(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DotPacking8Encoder<Q>
where
    Q: DotPacking8Quantizer,
{
    pub dim: usize,
    pub quantizer: Q,
    pub component_mapping: Option<Box<[u16]>>,
    pub inverse_component_mapping: Option<Box<[u16]>>,
}

impl<Q> DotPacking8Encoder<Q>
where
    Q: DotPacking8Quantizer<InputValue = f32>,
{
    pub fn new_with_quantizer(dim: usize, quantizer: Q) -> Self {
        Self {
            dim,
            quantizer,
            component_mapping: None,
            inverse_component_mapping: None,
        }
    }

    #[inline]
    pub fn component_mapping(&self) -> Option<&[u16]> {
        self.component_mapping.as_deref()
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
            crate::utils::permute_components_with_bisection(self.input_dim(), components_iter);
        let component_mapping: Vec<u16> = permutation.iter().map(|i| *i as u16).collect();
        let inverse = Self::compute_inverse_mapping(&component_mapping);
        self.component_mapping = Some(component_mapping.into_boxed_slice());
        self.inverse_component_mapping = Some(inverse.into_boxed_slice());
    }
}

pub struct DotPacking8QueryEvaluator<'a, Q>
where
    Q: DotPacking8Quantizer,
{
    dense_query: Vec<f32>,
    _encoder: &'a DotPacking8Encoder<Q>,
}

impl<'a, Q> DotPacking8QueryEvaluator<'a, Q>
where
    Q: DotPacking8Quantizer<InputValue = f32>,
{
    pub fn new(query: SparseVectorView<'_, u16, f32>, encoder: &'a DotPacking8Encoder<Q>) -> Self {
        let mut dense_query = vec![0.0f32; encoder.dim];
        for (&c, &v) in query.components().iter().zip(query.values().iter()) {
            let mapped_c = if let Some(mapping) = &encoder.component_mapping {
                mapping[c as usize]
            } else {
                c
            };
            dense_query[mapped_c as usize] = encoder.quantizer.query_value(c, v);
        }
        Self {
            dense_query,
            _encoder: encoder,
        }
    }
}

impl<'a, 'v, Q> QueryEvaluator<PackedVectorView<'v, u64>> for DotPacking8QueryEvaluator<'a, Q>
where
    Q: DotPacking8Quantizer<InputValue = f32>,
{
    type Distance = DotProduct;

    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> Self::Distance {
        let view = unsafe { DotPacking8View::from_unchecked_slice(vector.data()) };
        let mut acc = Simd::<f32, N>::splat(0.0);
        let mut query_ptr = self.dense_query.as_slice();

        let mut iter = view.iter_raw();
        for (gaps, values) in &mut iter {
            let components = simd_prefix_sum(gaps);
            let query_values = unsafe {
                Simd::gather_select_unchecked(
                    query_ptr,
                    Mask::splat(true),
                    components.cast(),
                    Simd::splat(0.0),
                )
            };

            acc = query_values.mul_add(values.cast(), acc);
            let last_component = components[N - 1];
            query_ptr = unsafe { query_ptr.split_at_unchecked(last_component as usize).1 };
        }
        let mut total_unscaled = acc.reduce_sum();

        let (gaps, tail_values, remaining) = iter.decode_tail();
        if remaining > 0 {
            let components = simd_prefix_sum(gaps);
            let components_arr = components.to_array();
            for i in 0..remaining {
                total_unscaled += query_ptr[components_arr[i] as usize] * tail_values[i] as f32;
            }
        }
        DotProduct(total_unscaled)
    }
}

impl<Q> VectorEncoder for DotPacking8Encoder<Q>
where
    Q: DotPacking8Quantizer<InputValue = f32>,
{
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u16, Q::InputValue>;
    type QueryVector<'q> = SparseVectorView<'q, u16, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;
    type Evaluator<'e>
        = DotPacking8QueryEvaluator<'e, Q>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        DotPacking8QueryEvaluator::new(query, self)
    }
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        DotPacking8QueryEvaluator::new(decoded.as_view(), self)
    }
    fn output_dim(&self) -> usize {
        self.dim
    }
    fn input_dim(&self) -> usize {
        self.dim
    }
}

impl<Q> SparseDataEncoder for DotPacking8Encoder<Q>
where
    Q: DotPacking8Quantizer<InputValue = f32>,
{
    type InputComponentType = u16;
    type InputValueType = Q::InputValue;
    type OutputComponentType = u16;
    type OutputValueType = u8;

    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let view = unsafe { DotPacking8View::from_unchecked_slice(encoded.data()) };
        let n = view.n;

        let mut decoded_comp = Vec::with_capacity(n);
        let mut values_raw = Vec::with_capacity(n);

        let mut iter = view.iter_raw();
        let mut last_component = 0u32;

        for (gaps, vals) in &mut iter {
            let components = simd_prefix_sum(gaps);
            let absolute_components = components + Simd::splat(last_component);
            let absolute_arr = absolute_components.to_array();
            let vals_arr = vals.to_array();
            for i in 0..N {
                decoded_comp.push(absolute_arr[i] as u16);
                values_raw.push(vals_arr[i]);
            }
            last_component = absolute_arr[N - 1];
        }

        let (gaps, tail_values, remaining) = iter.decode_tail();
        let components = simd_prefix_sum(gaps);
        let absolute_components = components + Simd::splat(last_component);
        let absolute_arr = absolute_components.to_array();
        for i in 0..remaining {
            decoded_comp.push(absolute_arr[i] as u16);
            values_raw.push(tail_values[i]);
        }

        if let Some(component_mapping) = &self.component_mapping {
            let inverse: Cow<'_, [u16]> = match &self.inverse_component_mapping {
                Some(inverse) => Cow::Borrowed(inverse),
                None => Cow::Owned(Self::compute_inverse_mapping(component_mapping)),
            };

            for c in decoded_comp.iter_mut() {
                *c = inverse[*c as usize];
            }

            let mut values: Vec<f32> = decoded_comp
                .iter()
                .zip(values_raw.iter())
                .map(|(&c, &v)| self.quantizer.decode_value(c, v))
                .collect();

            let permutation = rusty_perm::PermD::from_sort(decoded_comp.as_slice());
            permutation.apply(values.as_mut_slice()).unwrap();
            permutation.apply(decoded_comp.as_mut_slice()).unwrap();

            SparseVectorOwned::new(decoded_comp, values)
        } else {
            let values: Vec<f32> = decoded_comp
                .iter()
                .zip(values_raw.iter())
                .map(|(&c, &v)| self.quantizer.decode_value(c, v))
                .collect();
            SparseVectorOwned::new(decoded_comp, values)
        }
    }
}

impl<Q> PackedSparseVectorEncoder for DotPacking8Encoder<Q>
where
    Q: DotPacking8Quantizer<InputValue = f32>,
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
            .zip(input.values())
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

        let n = q_components.len();
        if n == 0 {
            return;
        }

        let mut gaps = Vec::with_capacity(n);
        let mut last = 0u32;
        for &comp in &q_components {
            gaps.push(comp as u32 - last);
            last = comp as u32;
        }

        let (selectors, gap_payloads) = encode_blocks(&gaps);
        let mut payload = Vec::new();
        payload.extend_from_slice(&(n as u16).to_le_bytes());
        payload.extend_from_slice(&selectors);
        payload.extend_from_slice(&gap_payloads);

        let current_len = payload.len();
        let padding_needed =
            compute_safe_simd_padding(n, &selectors, (8 - (current_len + n) % 8) % 8, n);

        for _ in 0..padding_needed {
            payload.push(0);
        }
        payload.extend_from_slice(q_values.as_slice());

        let data = payload
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()));

        output.extend(data);
    }
}

impl<Q> SpaceUsage for DotPacking8Encoder<Q>
where
    Q: DotPacking8Quantizer,
{
    fn space_usage_bytes(&self) -> usize {
        let mapping_size = self
            .component_mapping
            .as_ref()
            .map_or(0, |m| m.len() * std::mem::size_of::<u16>());
        let inverse_mapping_size = self
            .inverse_component_mapping
            .as_ref()
            .map_or(0, |m| m.len() * std::mem::size_of::<u16>());
        mapping_size + inverse_mapping_size + self.quantizer.space_usage_bytes()
    }
}
