/*
Block8 SIMD sparse decoder/encoder for `FixedU8Q` values.
This implementation decodes 8 gap values in parallel using a 3-step SIMD pipeline:
1) byte-level shuffle to place the relevant 4-byte window per lane,
2) per-lane variable right shift to bit-align each element,
3) bitmasking to keep only `b` significant bits.

The constants table (`BLOCK8_TABLE`) precomputes shuffle indices, shifts, and masks
for bit-widths `b in [1, 16]`, enabling branch-free bulk decoding. Tails are handled
separately in scalar form
*/

use rusty_perm::PermApply as _;
use rusty_perm::PermFromSorting as _;

use crate::core::sealed;
use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::encoders::dotpacking8::common::{
    compute_safe_simd_padding, encode_blocks, DotPacking8Iter, simd_prefix_sum
};
use crate::{FixedU8Q, SpaceUsage};
use bytemuck::cast_slice;
use bytemuck::from_bytes;
use std::borrow::Cow;
use std::simd::StdFloat;
use std::simd::prelude::*;
const FIXED_U8_SCALE: f32 = 1.0 / ((1u32 << FixedU8Q::FRAC_NBITS) as f32);

const N: usize = 8;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct DotPacking8FixedU8Encoder {
    dim: usize,
    component_mapping: Option<Box<[u16]>>,
    #[serde(default)]
    inverse_component_mapping: Option<Box<[u16]>>,
}

impl PartialEq for DotPacking8FixedU8Encoder {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
            && self.component_mapping == other.component_mapping
            && self.inverse_component_mapping == other.inverse_component_mapping
    }
}

impl sealed::Sealed for DotPacking8FixedU8Encoder {}

impl DotPacking8FixedU8Encoder {
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

use std::simd::{Simd};

#[derive(Clone)]
struct DotPacking8Fixedu8<'a> {
    n: usize,
    selectors: &'a [u8],
    payloads: &'a [u8],
    values: &'a [u8],
}

impl<'a> DotPacking8Fixedu8<'a> {
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

impl SparseDataEncoder for DotPacking8FixedU8Encoder {
    type InputComponentType = u16;
    type InputValueType = FixedU8Q;
    type OutputComponentType = u16;
    type OutputValueType = FixedU8Q;

    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let view = unsafe { DotPacking8Fixedu8::from_unchecked_slice(encoded.data()) };
        let n = view.n;

        let mut decoded_comp = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);

        let mut iter = view.iter_raw();
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
        let components = simd_prefix_sum(gaps);
        let absolute_components = components + Simd::splat(last_component);
        let absolute_arr = absolute_components.to_array();
        for i in 0..remaining {
            decoded_comp.push(absolute_arr[i] as u16);
            values.push(tail_values[i] as f32 * FIXED_U8_SCALE);
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

impl PackedSparseVectorEncoder for DotPacking8FixedU8Encoder {
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

        // Header (2 bytes): Total number of elements in the vector.
        payload.extend_from_slice(&(n as u16).to_le_bytes());

        // Selectors: Bit-widths for each bulk gap block.
        // - Each selector byte stores the bit-widths for two blocks (4 bits each).
        // - Storing them separately enables efficient 128-bit loads of gap payloads.
        payload.extend_from_slice(&selectors);

        payload.extend_from_slice(&gap_payloads);

        let current_len = payload.len();
        let padding_needed = (8 - (current_len + n) % 8) % 8;

        let padding_needed = compute_safe_simd_padding(n, &selectors, padding_needed, n);

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

impl VectorEncoder for DotPacking8FixedU8Encoder {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u16, FixedU8Q>;
    type QueryVector<'q> = SparseVectorView<'q, u16, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;
    type Evaluator<'e> = DotPacking8Fixedu8QueryEvaluator;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        DotPacking8Fixedu8QueryEvaluator::new(query, self)
    }
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        DotPacking8Fixedu8QueryEvaluator::new(decoded.as_view(), self)
    }
    fn output_dim(&self) -> usize {
        self.dim
    }
    fn input_dim(&self) -> usize {
        self.dim
    }
}

pub struct DotPacking8Fixedu8QueryEvaluator {
    dense_query: Vec<f32>,
}

impl DotPacking8Fixedu8QueryEvaluator {
    pub fn new(query: SparseVectorView<'_, u16, f32>, encoder: &DotPacking8FixedU8Encoder) -> Self {
        let mut dense_query = vec![0.0f32; encoder.dim];
        for (&c, &v) in query.components().iter().zip(query.values().iter()) {
            let mapped_c = if let Some(mapping) = &encoder.component_mapping {
                mapping[c as usize]
            } else {
                c
            };
            dense_query[mapped_c as usize] = v * FIXED_U8_SCALE; // Scale query values to match the quantized value range
        }
        Self { dense_query }
    }
}

impl<'v> QueryEvaluator<PackedVectorView<'v, u64>> for DotPacking8Fixedu8QueryEvaluator {
    type Distance = DotProduct;
    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> Self::Distance {
        let view = unsafe { DotPacking8Fixedu8::from_unchecked_slice(vector.data()) };
        let mut acc = Simd::<f32, N>::splat(0.0);
        let mut query_ptr = self.dense_query.as_slice();
        // --- BULK PHASE ---
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

        // --- TAIL ---
        let (gaps, tail_values, remaining) = iter.decode_tail();
        if remaining > 0 {
            let components = simd_prefix_sum(gaps);
            let components_arr = components.to_array();
            for i in 0..remaining {
                total_unscaled += query_ptr[components_arr[i] as usize] * tail_values[i] as f32;
            }
        }
        // Final Scale Application
        DotProduct(total_unscaled)
    }
}

impl SpaceUsage for DotPacking8FixedU8Encoder {
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
    fn compute_distance_only_bulk() {
        let encoder = DotPacking8FixedU8Encoder::new(100);
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
        let input = SparseVectorView::new(&[2, 4, 8, 24, 36, 48, 53, 90], &binding);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);
        let query = SparseVectorView::new(
            &[2, 4, 6, 8, 24, 70, 90],
            &[0.5, 1.5, 2.5, 1.0, 2.0, 1.0, 2.0],
        );
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        // expected = 1.0*0.5 + 3.0*1.5 + 2.0*1.0 + 3.5*2.0 + 2.0*2.0
        let expected = 1.0 * 0.5 + 3.0 * 1.5 + 2.0 * 1.0 + 3.5 * 2.0 + 2.0 * 2.0;
        assert!((dist.distance() - expected).abs() < 1e-5);
    }

    #[test]
    fn compute_distance_bulk_and_tail() {
        let encoder = DotPacking8FixedU8Encoder::new(100);
        let binding = [
            fixed(1.0),
            fixed(3.0),
            fixed(2.0),
            fixed(3.5),
            fixed(1.5),
            fixed(2.0),
            fixed(1.0),
            fixed(2.0),
            fixed(3.0),
            fixed(2.5),
        ];
        let input = SparseVectorView::new(&[2, 4, 8, 24, 28, 36, 48, 53, 70, 90], &binding);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);
        let query = SparseVectorView::new(
            &[2, 4, 8, 24, 28, 70, 90],
            &[0.5, 1.5, 2.5, 1.0, 2.0, 1.0, 2.0],
        );
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let expected =
            1.0 * 0.5 + 3.0 * 1.5 + 2.0 * 2.5 + 3.5 * 1.0 + 1.5 * 2.0 + 3.0 * 1.0 + 2.5 * 2.0;
        assert!((dist.distance() - expected).abs() < 1e-5);
    }

    fn verify_gaps(gaps: &[u32]) {
        let num_vals = gaps.len();
        let mut components = Vec::new();
        let mut curr = 0u32;
        for &g in gaps {
            curr += g;
            components.push(curr as u16);
        }

        let encoder = DotPacking8FixedU8Encoder::new(u16::MAX as usize + 1);
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

        assert!(
            (dist.distance() - expected).abs() < 1e-3,
            "Failed: dist={}, expected={}, gaps_len={}",
            dist.distance(),
            expected,
            num_vals
        );
    }

    #[test]
    fn test_plain_multiple_blocks() {
        let gaps = vec![1u32; 32];
        verify_gaps(&gaps);
    }

    #[test]
    fn test_plain_small_gaps() {
        let gaps = vec![1u32; 1];
        verify_gaps(&gaps);
    }

    #[test]
    fn test_plain_with_tail() {
        let gaps = vec![1u32; 11];
        verify_gaps(&gaps);
    }

    #[test]
    fn test_plain_large_gaps() {
        let gaps = vec![1000u32, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000];
        verify_gaps(&gaps);
    }

    #[test]
    fn test_plain_u16_gaps() {
        let gaps = vec![60000, 1, 1, 2, 3, 200, 450, 2000, 3];
        verify_gaps(&gaps);
    }

    #[test]
    fn block8_decode_roundtrip() {
        let mut encoder = DotPacking8FixedU8Encoder::new(100);
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
