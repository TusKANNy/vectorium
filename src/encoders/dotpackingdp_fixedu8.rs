use rusty_perm::PermApply as _;
use rusty_perm::PermFromSorting as _;

use crate::FixedU8Q;
use crate::SpaceUsage;
use crate::core::sealed;
use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::encoders::dotpackingdp::common::simd_prefix_sum;
use crate::encoders::dotpackingdp::common::{
    compute_safe_simd_padding, encode_blocks_dp, DotPackingDpIter,
};
use bytemuck::cast_slice;
use std::simd::StdFloat;
use std::simd::prelude::*;

const FIXED_U8_SCALE: f32 = 1.0 / ((1u32 << FixedU8Q::FRAC_NBITS) as f32);
const N: usize = 8;

pub type DotPackingDp8FixedU8Encoder = DotPackingDpFixedu8Encoder<8>;
pub type DotPackingDp16FixedU8Encoder = DotPackingDpFixedu8Encoder<16>;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct DotPackingDpFixedu8Encoder<const MAX_LEN: usize = 16> {
    dim: usize,
    component_mapping: Option<Box<[u16]>>,
    #[serde(default)]
    inverse_component_mapping: Option<Box<[u16]>>,
}

impl<const MAX_BLOCK_LEN: usize> PartialEq for DotPackingDpFixedu8Encoder<MAX_BLOCK_LEN> {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
            && self.component_mapping == other.component_mapping
            && self.inverse_component_mapping == other.inverse_component_mapping
    }
}

impl<const MAX_BLOCK_LEN: usize> sealed::Sealed for DotPackingDpFixedu8Encoder<MAX_BLOCK_LEN> {}

impl<const MAX_BLOCK_LEN: usize> DotPackingDpFixedu8Encoder<MAX_BLOCK_LEN> {
    pub fn new(input_dim: usize) -> Self {
        Self {
            dim: input_dim,
            component_mapping: None,
            inverse_component_mapping: None,
        }
    }

    #[inline]
    pub fn component_mapping(&self) -> Option<&[u16]> {
        self.component_mapping.as_deref()
    }

    #[inline]
    pub fn inverse_component_mapping(&self) -> Option<&[u16]> {
        self.inverse_component_mapping.as_deref()
    }

    fn compute_inverse_mapping(component_mapping: &[u16]) -> Vec<u16> {
        let dim = component_mapping.len();
        let mut inverse = vec![0u16; dim];
        let mut seen = vec![false; dim];
        for (old, &new) in component_mapping.iter().enumerate() {
            let new = new as usize;
            assert!(new < dim);
            assert!(!seen[new]);
            seen[new] = true;
            inverse[new] = old as u16;
        }
        inverse
    }

    pub fn train<'a, V>(
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

#[derive(Clone)]
struct DotPackingDpFixedu8<'a> {
    n: usize,
    selectors: &'a [u8],
    payloads: &'a [u8],
    values: &'a [u8],
}

impl<'a> DotPackingDpFixedu8<'a> {
    unsafe fn from_unchecked_slice(slice: &'a [u64]) -> Self {
        let bytes = cast_slice::<u64, u8>(slice);
        let n = u16::from_le_bytes(bytes[0..2].try_into().unwrap()) as usize;
        let n_blocks = u16::from_le_bytes(bytes[2..4].try_into().unwrap()) as usize;
        let offset = 4;
        let values_start_idx = bytes.len() - n;
        Self {
            n,
            selectors: &bytes[offset..offset + n_blocks],
            payloads: &bytes[offset + n_blocks..values_start_idx],
            values: &bytes[values_start_idx..],
        }
    }

    pub fn iter_raw<const MAX_BLOCK_LEN: usize>(&self) -> DotPackingDpIter<'a, MAX_BLOCK_LEN> {
        DotPackingDpIter {
            n: self.n,
            processed_count: 0,
            sel_idx: 0,
            chunk_rem: 0,
            b: 0,
            global_bit_pos: 0,
            overflow_count: 0,
            overflow_reg: Simd::splat(0),
            permuted_reg: Simd::splat(0),
            selectors: self.selectors,
            payload_ptr: self.payloads.as_ptr(),
            val_ptr: self.values.as_ptr(),
        }
    }
}

impl<const MAX_BLOCK_LEN: usize> SparseDataEncoder for DotPackingDpFixedu8Encoder<MAX_BLOCK_LEN> {
    type InputComponentType = u16;
    type InputValueType = FixedU8Q;
    type OutputComponentType = u16;
    type OutputValueType = FixedU8Q;
    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let view = unsafe { DotPackingDpFixedu8::from_unchecked_slice(encoded.data()) };
        let mut decoded_comp = Vec::new();
        let mut values = Vec::new();

        let mut iter = view.iter_raw::<MAX_BLOCK_LEN>();
        let mut last_component = 0u32;

        for (gaps, vals) in &mut iter {
            let components = simd_prefix_sum(gaps);
            let absolute_components = components + Simd::splat(last_component);
            let absolute_arr = absolute_components.to_array();
            let vals_arr = vals.to_array();
            for i in 0..N {
                decoded_comp.push(absolute_arr[i] as u16);
                values.push(vals_arr[i] as f32 * FIXED_U8_SCALE);
            }
            last_component = absolute_arr[N - 1];
        }

        let (gaps, tail_values, remaining) = iter.decode_tail();
        if remaining > 0 {
            let components = simd_prefix_sum(gaps);
            let absolute_components = components + Simd::splat(last_component);
            let absolute_arr = absolute_components.to_array();
            for i in 0..remaining {
                decoded_comp.push(absolute_arr[i] as u16);
                values.push(tail_values[i] as f32 * FIXED_U8_SCALE);
            }
        }

        if let Some(component_mapping) = self.component_mapping() {
            let inverse: std::borrow::Cow<'_, [u16]> = match self.inverse_component_mapping() {
                Some(inverse) => std::borrow::Cow::Borrowed(inverse),
                None => std::borrow::Cow::Owned(Self::compute_inverse_mapping(component_mapping)),
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

impl<const MAX_BLOCK_LEN: usize> PackedSparseVectorEncoder
    for DotPackingDpFixedu8Encoder<MAX_BLOCK_LEN>
{
    type PackedDataType = u64;
    fn push_encoded<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, u16, FixedU8Q>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<u64>,
    {
        let mut q_values: Vec<u8> = input.values().iter().map(|v| v.to_bits()).collect();
        let mut q_components: Vec<u16> = if let Some(m) = &self.component_mapping {
            input.components().iter().map(|&c| m[c as usize]).collect()
        } else {
            input.components().to_vec()
        };
        if self.component_mapping.is_some() {
            let perm = rusty_perm::PermD::from_sort(q_components.as_slice());
            perm.apply(q_values.as_mut_slice()).unwrap();
            perm.apply(q_components.as_mut_slice()).unwrap();
        }
        let n = q_components.len();
        let mut gaps = Vec::with_capacity(n);
        let mut last = 0u32;
        for c in q_components {
            gaps.push(c as u32 - last);
            last = c as u32;
        }
        let (selectors, gap_bitstream) = encode_blocks_dp::<MAX_BLOCK_LEN>(&gaps);
        let mut payload = Vec::new();
        payload.extend_from_slice(&(n as u16).to_le_bytes());
        payload.extend_from_slice(&(selectors.len() as u16).to_le_bytes());
        payload.extend_from_slice(&selectors);
        payload.extend_from_slice(&gap_bitstream);
        let current_len = payload.len();
        let mut padding_needed = (8 - (current_len + n) % 8) % 8;
        padding_needed = compute_safe_simd_padding(&selectors, padding_needed, n);
        for _ in 0..padding_needed {
            payload.push(0);
        }
        payload.extend_from_slice(&q_values);
        output.extend(
            payload
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap())),
        );
    }
}

impl<const MAX_BLOCK_LEN: usize> VectorEncoder for DotPackingDpFixedu8Encoder<MAX_BLOCK_LEN> {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u16, FixedU8Q>;
    type QueryVector<'q> = SparseVectorView<'q, u16, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;
    type Evaluator<'e>
        = DotPackingDpFixedu8QueryEvaluator<MAX_BLOCK_LEN>
    where
        Self: 'e;
    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        DotPackingDpFixedu8QueryEvaluator::new(query, self)
    }
    fn vector_evaluator<'e, 'v>(&'e self, v: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = self.decode_vector(v);
        DotPackingDpFixedu8QueryEvaluator::new(decoded.as_view(), self)
    }
    fn output_dim(&self) -> usize {
        self.dim
    }
    fn input_dim(&self) -> usize {
        self.dim
    }
}

pub struct DotPackingDpFixedu8QueryEvaluator<const MAX_BLOCK_LEN: usize = 16> {
    dense_query: Vec<f32>,
}

impl<const MAX_BLOCK_LEN: usize> DotPackingDpFixedu8QueryEvaluator<MAX_BLOCK_LEN> {
    pub fn new(
        query: SparseVectorView<'_, u16, f32>,
        encoder: &DotPackingDpFixedu8Encoder<MAX_BLOCK_LEN>,
    ) -> Self {
        let mut dense_query = vec![0.0f32; encoder.dim];
        for (&c, &v) in query.components().iter().zip(query.values().iter()) {
            let mapped_c = if let Some(m) = &encoder.component_mapping {
                m[c as usize]
            } else {
                c
            };
            dense_query[mapped_c as usize] = v;
        }
        Self { dense_query }
    }

    #[inline]
    unsafe fn simd_compute_distance<'v>(&self, vector: PackedVectorView<'v, u64>) -> DotProduct {
        let view = unsafe { DotPackingDpFixedu8::from_unchecked_slice(vector.data()) };
        let mut acc = Simd::<f32, N>::splat(0.0);
        let mut query_slice = self.dense_query.as_slice();

        let mut iter = view.iter_raw::<MAX_BLOCK_LEN>();
        for (gaps, values) in &mut iter {
            let relative_positions = simd_prefix_sum(gaps);
            let q_vals = unsafe {
                Simd::gather_select_unchecked(
                    query_slice,
                    Mask::splat(true),
                    relative_positions.cast(),
                    Simd::splat(0.0),
                )
            };
            acc = q_vals.mul_add(values.cast(), acc);
            let step = relative_positions[N - 1] as usize;
            query_slice = unsafe { query_slice.split_at_unchecked(step).1 };
        }

        let mut total_unscaled = acc.reduce_sum();

        let (gaps, tail_values, remaining) = iter.decode_tail();
        if remaining > 0 {
            let relative_positions = simd_prefix_sum(gaps);
            let comps = relative_positions.as_array();
            for j in 0..remaining {
                let pos = comps[j] as usize;
                let val = tail_values[j] as f32;
                total_unscaled += query_slice[pos] * val;
            }
        }
        DotProduct(total_unscaled * FIXED_U8_SCALE)
    }
}

impl<'v, const MAX_BLOCK_LEN: usize> QueryEvaluator<PackedVectorView<'v, u64>>
    for DotPackingDpFixedu8QueryEvaluator<MAX_BLOCK_LEN>
{
    type Distance = DotProduct;
    fn compute_distance(&self, v: PackedVectorView<'v, u64>) -> Self::Distance {
        unsafe { self.simd_compute_distance(v) }
    }
}

impl<const MAX_BLOCK_LEN: usize> SpaceUsage for DotPackingDpFixedu8Encoder<MAX_BLOCK_LEN> {
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

    fn verify_dot_product_only<const M: usize>(
        components: &[u16],
        values: &[FixedU8Q],
        dim: usize,
    ) {
        let encoder = DotPackingDpFixedu8Encoder::<M>::new(dim);
        let input = SparseVectorView::new(components, values);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);
        let packed_view = PackedVectorView::new(&buffer);
        let query_values: Vec<f32> = (0..components.len())
            .map(|i| ((i + 1) as f32) * 0.25)
            .collect();
        let query = SparseVectorView::new(components, &query_values);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(packed_view);
        let expected: f32 = values
            .iter()
            .zip(query_values.iter())
            .map(|(v, q)| v.to_f32().unwrap() * q)
            .sum();
        assert!((dist.distance() - expected).abs() < 1e-4);
    }

    #[test]
    fn test_block_dp_n_less_than_8() {
        let components = vec![10, 20, 30, 40, 50];
        let values = vec![fixed(1.0), fixed(2.0), fixed(3.0), fixed(4.0), fixed(5.0)];
        verify_dot_product_only::<16>(&components, &values, 100);
    }

    #[test]
    fn test_block_dp_n_equal_8() {
        let components: Vec<u16> = (0..8).map(|i| (i * 10) as u16).collect();
        let values: Vec<FixedU8Q> = (0..8).map(|i| fixed(i as f32)).collect();
        verify_dot_product_only::<16>(&components, &values, 100);
    }

    #[test]
    fn test_block_dp_n_greater_than_8() {
        let components: Vec<u16> = (0..12).map(|i| (i * 5) as u16).collect();
        let values: Vec<FixedU8Q> = (0..12).map(|i| fixed(i as f32 / 10.0)).collect();
        verify_dot_product_only::<16>(&components, &values, 100);
    }

    #[test]
    fn test_block_dp_max8() {
        let components: Vec<u16> = (0..20).map(|i| (i * 5) as u16).collect();
        let values: Vec<FixedU8Q> = (0..20).map(|i| fixed(i as f32 / 10.0)).collect();
        let encoder = DotPackingDpFixedu8Encoder::<8>::new(1000);
        let input = SparseVectorView::new(&components, &values);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);
        let bytes = cast_slice::<u64, u8>(&buffer);
        let n_blocks = u16::from_le_bytes(bytes[2..4].try_into().unwrap()) as usize;
        let selectors = &bytes[4..4 + n_blocks];
        for &sel in selectors {
            let l = (sel & 0x0F) as usize + 1;
            assert!(l <= 8);
        }
        verify_dot_product_only::<8>(&components, &values, 1000);
    }

    #[test]
    fn test_block_dp_max16() {
        let components: Vec<u16> = (0..20).map(|i| (i * 5) as u16).collect();
        let values: Vec<FixedU8Q> = (0..20).map(|i| fixed(i as f32 / 10.0)).collect();
        let encoder = DotPackingDpFixedu8Encoder::<16>::new(1000);
        let input = SparseVectorView::new(&components, &values);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);
        let bytes = cast_slice::<u64, u8>(&buffer);
        let n_blocks = u16::from_le_bytes(bytes[2..4].try_into().unwrap()) as usize;
        let selectors = &bytes[4..4 + n_blocks];
        for &sel in selectors {
            let l = (sel & 0x0F) as usize + 1;
            assert!(l <= 16);
        }
        verify_dot_product_only::<16>(&components, &values, 1000);
    }

}
