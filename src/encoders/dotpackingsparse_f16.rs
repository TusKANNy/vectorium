use bytemuck::cast_slice;
use half::f16;
use std::simd::{Simd, StdFloat, prelude::*};
use rusty_perm::{PermApply as _, PermFromSorting as _};

use crate::{SpaceUsage, Dataset};
use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::encoders::dotpacking8::common::{
    simd_prefix_sum, fast_lane_gather, encode_blocks, compute_safe_simd_padding, DotPacking8f16Iter as DotPackingSparsef16Iter
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{__m256, __m256i, _mm256_setzero_ps};
#[cfg(target_arch = "x86_64")]
use crate::encoders::dotpacking8::common::match_and_fma_8;
use crate::encoders::dotpacking8::sparse_encoder::DOTPACKING8_SPARSE_QUERY_THRESHOLD;

const N: usize = 8;

#[derive(Clone)]
pub struct DotPackingSparsef16View<'a> {
    pub n: usize,
    pub selectors: &'a [u8],
    pub max_components: &'a [u8],
    pub payloads: &'a [u8],
    pub values: &'a [f16],
}

impl<'a> DotPackingSparsef16View<'a> {
    pub unsafe fn from_unchecked_slice(slice: &'a [u64]) -> Self {
        let bytes = cast_slice::<u64, u8>(slice);
        let n = bytemuck::from_bytes::<u16>(&bytes[0..2]).to_le() as usize;

        let n_blocks_total = (n + N - 1) / N;
        let selectors_size = (n_blocks_total + 1) / 2;
        let selectors_end = 2 + selectors_size;
        let max_components_size = n_blocks_total * 2;
        let max_components_end = selectors_end + max_components_size;
        let values_start_idx = bytes.len() - 2 * n; // 2 bytes per f16

        Self {
            n,
            selectors: &bytes[2..selectors_end],
            max_components: &bytes[selectors_end..max_components_end],
            payloads: &bytes[max_components_end..values_start_idx],
            values: cast_slice::<u8, f16>(&bytes[values_start_idx..]),
        }
    }

    #[inline(always)]
    pub fn get_max_component(&self, block_idx: usize) -> u32 {
        let start = block_idx * 2;
        u16::from_le_bytes([self.max_components[start], self.max_components[start + 1]]) as u32
    }

    pub fn iter_raw(&self) -> DotPackingSparsef16Iter<'a> {
        DotPackingSparsef16Iter {
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
pub struct DotPackingSparsef16Encoder {
    pub dim: usize,
    pub component_mapping: Option<Box<[u16]>>,
    pub inverse_component_mapping: Option<Box<[u16]>>,
}

impl DotPackingSparsef16Encoder {
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

    pub fn train<'a>(
        &mut self,
        training_data: &crate::PlainSparseDataset<u16, f32, crate::SquaredEuclideanDistance>,
    ) {
        const SAMPLE_RATE: usize = 20;
        let sample_size = if training_data.len() / SAMPLE_RATE < 50_000 {
            training_data.len()
        } else {
            training_data.len() / SAMPLE_RATE
        };
        self.train_components(training_data.iter().take(sample_size));
    }
}

impl SpaceUsage for DotPackingSparsef16Encoder {
    fn space_usage_bytes(&self) -> usize {
        let mapping_size = self
            .component_mapping
            .as_ref()
            .map_or(0, |m| m.len() * std::mem::size_of::<u16>());
        let inverse_mapping_size = self
            .inverse_component_mapping
            .as_ref()
            .map_or(0, |m| m.len() * std::mem::size_of::<u16>());
        mapping_size + inverse_mapping_size
    }
}

pub struct DotPackingSparsef16QueryEvaluator<'a> {
    dense_query: Option<Vec<f32>>,
    sparse_query: Option<crate::core::vector_encoder::SparseVectorOwned<u16, f32>>,
    _encoder: &'a DotPackingSparsef16Encoder,
}

impl<'a> DotPackingSparsef16QueryEvaluator<'a> {
    pub fn new(query: SparseVectorView<'_, u16, f32>, encoder: &'a DotPackingSparsef16Encoder) -> Self {
        for (c, _) in query.iter() {
            assert!(
                (c as usize) < encoder.dim,
                "Query component {} exceeds dimension {}",
                c,
                encoder.dim
            );
        }

        if query.components().len() >= DOTPACKING8_SPARSE_QUERY_THRESHOLD {
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
                dense_query: Some(dense_query),
                sparse_query: None,
                _encoder: encoder,
            }
        } else {
            let mut mapped_components = Vec::with_capacity(query.components().len());
            let mut mapped_values = Vec::with_capacity(query.components().len());
            for (&c, &v) in query.components().iter().zip(query.values().iter()) {
                let mapped_c = if let Some(mapping) = &encoder.component_mapping {
                    mapping[c as usize]
                } else {
                    c
                };
                mapped_components.push(mapped_c);
                mapped_values.push(v);
            }

            let mut pairs: Vec<(u16, f32)> =
                mapped_components.into_iter().zip(mapped_values).collect();
            pairs.sort_unstable_by_key(|p| p.0);

            let sorted_comps: Vec<u16> = pairs.iter().map(|p| p.0).collect();
            let sorted_vals: Vec<f32> = pairs.iter().map(|p| p.1).collect();

            Self {
                dense_query: None,
                sparse_query: Some(crate::core::vector_encoder::SparseVectorOwned::new(
                    sorted_comps,
                    sorted_vals,
                )),
                _encoder: encoder,
            }
        }
    }
}

impl<'a, 'v> QueryEvaluator<PackedVectorView<'v, u64>> for DotPackingSparsef16QueryEvaluator<'a> {
    type Distance = DotProduct;

    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> Self::Distance {
        if let Some(dense_query) = &self.dense_query {
            let view = unsafe { DotPackingSparsef16View::from_unchecked_slice(vector.data()) };
            let mut acc = Simd::<f32, N>::splat(0.0);
            let mut query_ptr = dense_query.as_slice();

            let mut iter = view.iter_raw();
            for (gaps, values) in &mut iter {
                let components = simd_prefix_sum(gaps);
                let query_values = fast_lane_gather(query_ptr, components);
                
                acc = query_values.mul_add(values, acc);
                let last_component = components[N - 1];
                query_ptr = unsafe { query_ptr.split_at_unchecked(last_component as usize).1 };
            }
            let mut total_unscaled = acc.reduce_sum();

            let (gaps, tail_values, remaining) = iter.decode_tail();
            if remaining > 0 {
                let components = simd_prefix_sum(gaps);
                let components_arr = components.to_array();
                for i in 0..remaining {
                    total_unscaled += query_ptr[components_arr[i] as usize] * tail_values[i].to_f32();
                }
            }
            DotProduct(total_unscaled)
        } else {
            let query = self.sparse_query.as_ref().unwrap().as_view();
            if query.is_empty() {
                return DotProduct::from(0.0f32);
            }

            let query_components = query.components();
            let query_values = query.values();

            let view = unsafe { DotPackingSparsef16View::from_unchecked_slice(vector.data()) };

            #[cfg(not(target_arch = "x86_64"))]
            let mut acc = Simd::<f32, N>::splat(0.0);
            #[cfg(target_arch = "x86_64")]
            let mut acc_m256: __m256 = unsafe { _mm256_setzero_ps() };

            let mut cur_q = 0;
            let mut last_component = 0u32;

            let mut iter = view.iter_raw();
            'block_loop: while iter.block_idx < iter.bulk_blocks {
                let query_c = query_components[cur_q] as u32;

                let mut block_last = view.get_max_component(iter.block_idx);
                while query_c > block_last {
                    last_component = block_last;
                    let sel_byte = unsafe { *iter.selectors.get_unchecked(iter.block_idx / 2) };
                    let b = (((sel_byte >> ((iter.block_idx & 1) << 2)) & 0x0F) as usize) + 1;
                    unsafe {
                        iter.payload_ptr = iter.payload_ptr.add(b);
                        iter.val_ptr = iter.val_ptr.add(N);
                    }
                    iter.block_idx += 1;
                    if iter.block_idx >= iter.bulk_blocks {
                        break 'block_loop;
                    }
                    block_last = view.get_max_component(iter.block_idx);
                }

                let (gaps, values) = iter.next().unwrap();
                let components = simd_prefix_sum(gaps);
                let absolute_components = components + Simd::splat(last_component);

                #[cfg(target_arch = "x86_64")]
                unsafe {
                    let comps_m256i: __m256i = std::mem::transmute(absolute_components);
                    let vbits_m256: __m256 = std::mem::transmute(values);
                    while (query_components[cur_q] as u32) <= block_last {
                        match_and_fma_8(
                            query_components[cur_q] as u32,
                            query_values[cur_q],
                            comps_m256i,
                            vbits_m256,
                            &mut acc_m256,
                        );
                        cur_q += 1;
                        if cur_q >= query_components.len() {
                            break 'block_loop;
                        }
                    }
                }

                #[cfg(not(target_arch = "x86_64"))]
                while (query_components[cur_q] as u32) <= block_last {
                    let qc = query_components[cur_q] as u32;
                    let qv = query_values[cur_q];
                    let match_mask = absolute_components.simd_eq(Simd::splat(qc));
                    let masked_vals = match_mask.select(values, Simd::splat(0.0));
                    acc = Simd::splat(qv).mul_add(masked_vals, acc);
                    cur_q += 1;
                    if cur_q >= query_components.len() {
                        break 'block_loop;
                    }
                }

                last_component = block_last;
            }
            #[cfg(target_arch = "x86_64")]
            let mut total_unscaled = unsafe { crate::core::data_block::hsum256_ps(acc_m256) };
            #[cfg(not(target_arch = "x86_64"))]
            let mut total_unscaled = acc.reduce_sum();

            if cur_q < query_components.len() {
                let (gaps, tail_values, remaining) = iter.decode_tail();
                if remaining > 0 {
                    let components = simd_prefix_sum(gaps);
                    let absolute_components = components + Simd::splat(last_component);
                    let block_comps = absolute_components.to_array();

                    while cur_q < query_components.len() {
                        let qc = query_components[cur_q] as u32;
                        let qv = query_values[cur_q];
                        for i in 0..remaining {
                            if block_comps[i] == qc {
                                total_unscaled += qv * tail_values[i].to_f32();
                            }
                        }
                        cur_q += 1;
                    }
                }
            }

            DotProduct(total_unscaled)
        }
    }
}

impl VectorEncoder for DotPackingSparsef16Encoder {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u16, f32>;
    type QueryVector<'q> = SparseVectorView<'q, u16, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;
    type Evaluator<'e> = DotPackingSparsef16QueryEvaluator<'e>;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        DotPackingSparsef16QueryEvaluator::new(query, self)
    }
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        DotPackingSparsef16QueryEvaluator::new(decoded.as_view(), self)
    }
    fn output_dim(&self) -> usize {
        self.dim
    }
    fn input_dim(&self) -> usize {
        self.dim
    }
}

impl SparseDataEncoder for DotPackingSparsef16Encoder {
    type InputComponentType = u16;
    type InputValueType = f32;
    type OutputComponentType = u16;
    type OutputValueType = f16;

    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let view = unsafe { DotPackingSparsef16View::from_unchecked_slice(encoded.data()) };
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
            values_raw.push(tail_values[i].to_f32());
        }

        if let Some(component_mapping) = &self.component_mapping {
            let inverse: std::borrow::Cow<'_, [u16]> = match &self.inverse_component_mapping {
                Some(inverse) => std::borrow::Cow::Borrowed(inverse),
                None => std::borrow::Cow::Owned(Self::compute_inverse_mapping(component_mapping)),
            };

            for c in decoded_comp.iter_mut() {
                *c = inverse[*c as usize];
            }

            let permutation = rusty_perm::PermD::from_sort(decoded_comp.as_slice());
            permutation.apply(values_raw.as_mut_slice()).unwrap();
            permutation.apply(decoded_comp.as_mut_slice()).unwrap();

            SparseVectorOwned::new(decoded_comp, values_raw)
        } else {
            SparseVectorOwned::new(decoded_comp, values_raw)
        }
    }
}

impl PackedSparseVectorEncoder for DotPackingSparsef16Encoder {
    type PackedDataType = u64;

    fn push_encoded<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, u16, f32>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<u64>,
    {
        let mut q_values: Vec<f16> = input
            .values()
            .iter()
            .map(|&v| f16::from_f32(v))
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
        let mut max_comps = Vec::with_capacity((n + N - 1) / N);
        let mut last = 0u32;

        for chunk in q_components.chunks(N) {
            max_comps.push(*chunk.last().unwrap());
            for &comp in chunk {
                gaps.push(comp as u32 - last);
                last = comp as u32;
            }
        }

        let (selectors, gap_payloads) = encode_blocks(&gaps);
        let mut payload = Vec::new();
        payload.extend_from_slice(&(n as u16).to_le_bytes());
        payload.extend_from_slice(&selectors);
        for &mc in &max_comps {
            payload.extend_from_slice(&mc.to_le_bytes());
        }
        payload.extend_from_slice(&gap_payloads);

        let current_len = payload.len();
        // alignment_padding ensures that the total payload length (metadata + selectors + gap payloads + padding + values)
        // is a multiple of 8 bytes (so it can be chunked into u64) and that the f16 values are aligned to the end of the [u64] slice.
        let alignment_padding = (8 - (current_len + 2 * n) % 8) % 8;
        let padding_needed =
            compute_safe_simd_padding(n, &selectors, alignment_padding, 2 * n);

        for _ in 0..padding_needed {
            payload.push(0);
        }
        payload.extend_from_slice(cast_slice::<f16, u8>(q_values.as_slice()));

        let data = payload
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()));

        output.extend(data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::distances::Distance;
    use crate::core::vector::{PackedVectorView, SparseVectorView};
    use crate::vector_encoder::SparseDataEncoder;
    use crate::{
        DatasetGrowable, PlainSparseDatasetGrowable, PlainSparseQuantizer, QueryEvaluator, VectorEncoder,
    };

    fn calculate_expected_distance(
        vector: &SparseVectorView<u16, f32>,
        query: &SparseVectorView<u16, f32>,
    ) -> f32 {
        let mut expected = 0.0f32;
        let mut vec_iter = vector.iter().peekable();

        for (comp, val) in query.iter() {
            while let Some(&(vec_comp, _)) = vec_iter.peek() {
                if vec_comp < comp {
                    vec_iter.next();
                } else {
                    break;
                }
            }

            if let Some(&(vec_comp, vec_val)) = vec_iter.peek() {
                if vec_comp == comp {
                    expected += f16::from_f32(vec_val).to_f32() * val;
                    vec_iter.next();
                }
            }
        }
        expected
    }

    #[test]
    fn compute_distance_only_bulk() {
        let encoder = DotPackingSparsef16Encoder::new(100);

        let binding = [1.0, 3.0, 2.0, 3.5, 1.5, 2.0, 1.0, 2.0];
        let input = SparseVectorView::new(&[2, 4, 8, 24, 36, 48, 53, 90], &binding);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let query = SparseVectorView::new(
            &[2, 4, 6, 8, 24, 70, 90],
            &[0.5, 1.5, 2.5, 1.0, 2.0, 1.0, 2.0],
        );
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let expected = calculate_expected_distance(&input, &query);
        assert!((dist.distance() - expected).abs() < 1e-5);
    }

    #[test]
    fn compute_distance_bulk_and_tail() {
        let encoder = DotPackingSparsef16Encoder::new(100);

        let binding = [1.0, 3.0, 2.0, 3.5, 1.5, 2.0, 1.0, 2.0, 3.0, 2.5];
        let input = SparseVectorView::new(&[2, 4, 8, 24, 28, 36, 48, 53, 70, 90], &binding);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let query = SparseVectorView::new(
            &[2, 4, 8, 24, 28, 70, 90],
            &[0.5, 1.5, 2.5, 1.0, 2.0, 1.0, 2.0],
        );
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let expected = calculate_expected_distance(&input, &query);
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

        let encoder = DotPackingSparsef16Encoder::new(u16::MAX as usize + 1);

        let values: Vec<_> = (0..num_vals).map(|i| 1.0 + (i % 7) as f32 / 10.0).collect();
        let input = SparseVectorView::new(&components, &values);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let query_vals: Vec<_> = (0..num_vals).map(|i| 0.1 + (i % 5) as f32 / 10.0).collect();
        let query = SparseVectorView::new(&components, &query_vals);

        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let expected = calculate_expected_distance(&input, &query);

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

    fn same_when_quantized(
        before: &SparseVectorView<u16, f32>,
        after: &SparseVectorView<u16, f32>,
    ) {
        assert_eq!(before.components(), after.components());
        for (v1, v2) in before.values().iter().zip(after.values().iter()) {
            let q1 = f16::from_f32(*v1).to_f32();
            let q2 = f16::from_f32(*v2).to_f32();
            assert_eq!(q1, q2);
        }
    }

    #[test]
    fn block8_decode_roundtrip() {
        let mut encoder = DotPackingSparsef16Encoder::new(100);

        let binding = [1.0, 3.0, 2.0, 3.5];
        let input_components = [0, 4, 8, 24];
        let input0 = SparseVectorView::new(&input_components, &binding);
        let binding1 = [3.0, 2.0, 3.5];
        let input_components1 = [4, 8, 24];
        let input1 = SparseVectorView::new(&input_components1, &binding1);

        let quantizer = PlainSparseQuantizer::<u16, f32, crate::SquaredEuclideanDistance>::new(100, 100);
        let mut growable = PlainSparseDatasetGrowable::new(quantizer);

        growable.push(input0.clone());
        growable.push(input1.clone());

        let training_data = growable.into();
        encoder.train(&training_data);

        let mut buffer0 = Vec::new();
        encoder.push_encoded(input0, &mut buffer0);

        let mut buffer1 = Vec::new();
        encoder.push_encoded(input1, &mut buffer1);

        let decoded0 = encoder.decode_vector(PackedVectorView::new(&buffer0));
        let decoded1 = encoder.decode_vector(PackedVectorView::new(&buffer1));

        same_when_quantized(&input0, &decoded0.as_view());
        same_when_quantized(&input1, &decoded1.as_view());
    }
}
