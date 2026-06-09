use bytemuck::{cast_slice, from_bytes};
use rusty_perm::{PermApply as _, PermFromSorting as _};
use std::borrow::Cow;
use std::simd::{Simd, StdFloat, prelude::*};

use crate::SpaceUsage;
use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::encoders::dotpacking8::common::{
    DotPacking8Iter, compute_safe_simd_padding, encode_blocks, fast_lane_gather, simd_prefix_sum,
};
use crate::encoders::dotpacking8::quantizer::DotPacking8Quantizer;

const N: usize = 8;

#[derive(Clone)]
pub struct DotPackingSparseView<'a> {
    pub n: usize,
    pub selectors: &'a [u8],
    pub max_components: &'a [u8],
    pub payloads: &'a [u8],
    pub values: &'a [u8],
}

impl<'a> DotPackingSparseView<'a> {
    pub unsafe fn from_unchecked_slice(slice: &'a [u64]) -> Self {
        let bytes = cast_slice::<u64, u8>(slice);
        let n = from_bytes::<u16>(&bytes[0..2]).to_le() as usize;

        let n_blocks_total = (n + N - 1) / N;
        let selectors_size = (n_blocks_total + 1) / 2;
        let selectors_end = 2 + selectors_size;
        let max_components_size = n_blocks_total * 2;
        let max_components_end = selectors_end + max_components_size;
        let values_start_idx = bytes.len() - n;

        Self {
            n,
            selectors: &bytes[2..selectors_end],
            max_components: &bytes[selectors_end..max_components_end],
            payloads: &bytes[max_components_end..values_start_idx],
            values: &bytes[values_start_idx..],
        }
    }

    #[inline(always)]
    pub fn get_max_component(&self, block_idx: usize) -> u32 {
        let start = block_idx * 2;
        u16::from_le_bytes([self.max_components[start], self.max_components[start + 1]]) as u32
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
pub struct DotPackingSparseEncoder<Q>
where
    Q: DotPacking8Quantizer,
{
    pub dim: usize,
    pub quantizer: Q,
    pub component_mapping: Option<Box<[u16]>>,
    pub inverse_component_mapping: Option<Box<[u16]>>,
}

impl<Q> DotPackingSparseEncoder<Q>
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

pub const DOTPACKING8_SPARSE_QUERY_THRESHOLD: usize = 33;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[inline]
pub(crate) unsafe fn match_and_fma_8(
    query_component: u32,
    query_value: f32,
    comps: core::arch::x86_64::__m256i,
    vbits: core::arch::x86_64::__m256,
    acc: &mut core::arch::x86_64::__m256,
) {
    use core::arch::x86_64::*;
    // 8×u32 compare → 0xFFFFFFFF on match, 0 elsewhere
    let q32: __m256i = _mm256_set1_epi32(query_component as i32);
    let cmp: __m256i = _mm256_cmpeq_epi32(comps, q32);

    // Mask f32 values by the comparison result
    let vbits_masked: __m256 = _mm256_and_ps(vbits, _mm256_castsi256_ps(cmp));

    // Scale and accumulate
    let scale: __m256 = _mm256_set1_ps(query_value);
    *acc = _mm256_fmadd_ps(vbits_masked, scale, *acc);
}


#[derive(Clone)]
pub struct DotPackingSparseQueryEvaluator<'a, Q>
where
    Q: DotPacking8Quantizer,
{
    dense_query: Option<Vec<f32>>,
    sparse_query: Option<crate::core::vector_encoder::SparseVectorOwned<u16, f32>>,
    _encoder: &'a DotPackingSparseEncoder<Q>,
}

impl<'a, Q> DotPackingSparseQueryEvaluator<'a, Q>
where
    Q: DotPacking8Quantizer<InputValue = f32>,
{
    pub fn new(
        query: SparseVectorView<'_, u16, f32>,
        encoder: &'a DotPackingSparseEncoder<Q>,
    ) -> Self {
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
                dense_query[mapped_c as usize] = encoder.quantizer.query_value(c, v);
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
                mapped_values.push(encoder.quantizer.query_value(c, v));
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

impl<'a, 'v, Q> QueryEvaluator<PackedVectorView<'v, u64>> for DotPackingSparseQueryEvaluator<'a, Q>
where
    Q: DotPacking8Quantizer<InputValue = f32>,
{
    type Distance = DotProduct;

    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> Self::Distance {
        if let Some(dense_query) = &self.dense_query {
            let view = unsafe { DotPackingSparseView::from_unchecked_slice(vector.data()) };
            let mut acc = Simd::<f32, N>::splat(0.0);
            let mut query_ptr = dense_query.as_slice();

            let mut iter = view.iter_raw();
            for (gaps, values) in &mut iter {
                let components = simd_prefix_sum(gaps);
                let query_values = fast_lane_gather(query_ptr, components);
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
        } else {
            let query = self.sparse_query.as_ref().unwrap().as_view();
            if query.is_empty() {
                return DotProduct::from(0.0f32);
            }

            let query_components = query.components();
            let query_values = query.values();

            let view = unsafe { DotPackingSparseView::from_unchecked_slice(vector.data()) };
            
            #[cfg(not(target_arch = "x86_64"))]
            let mut acc = Simd::<f32, N>::splat(0.0);
            #[cfg(target_arch = "x86_64")]
            let mut acc_m256: core::arch::x86_64::__m256 = unsafe { core::arch::x86_64::_mm256_setzero_ps() };

            let mut cur_q = 0;
            let mut last_component = 0u32;

            let mut iter = view.iter_raw();
            'block_loop: while iter.block_idx < iter.bulk_blocks {
                let query_c = query_components[cur_q] as u32;

                while iter.block_idx < iter.bulk_blocks && query_c > view.get_max_component(iter.block_idx) {
                    last_component = view.get_max_component(iter.block_idx) as u32;
                    let sel_byte = *unsafe { iter.selectors.get_unchecked(iter.block_idx / 2) };
                    let b = (((sel_byte >> ((iter.block_idx & 1) << 2)) & 0x0F) as usize) + 1;
                    iter.payload_ptr = unsafe { iter.payload_ptr.add(b) };
                    iter.val_ptr = unsafe { iter.val_ptr.add(N) };
                    iter.block_idx += 1;
                }

                if iter.block_idx >= iter.bulk_blocks {
                    break 'block_loop;
                }

                let block_last = view.get_max_component(iter.block_idx);

                let (gaps, values) = iter.next().unwrap();
                let components = simd_prefix_sum(gaps);
                let absolute_components = components + Simd::splat(last_component);
                
                let block_vals = values.cast::<f32>();

                #[cfg(target_arch = "x86_64")]
                unsafe {
                        let comps_m256i: core::arch::x86_64::__m256i =
                            std::mem::transmute(absolute_components);
                        let vbits_m256: core::arch::x86_64::__m256 =
                            std::mem::transmute(block_vals);
                        
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
                        let masked_vals = match_mask.select(block_vals, Simd::splat(0.0));
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
                                total_unscaled += qv * tail_values[i] as f32;
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

impl<Q> VectorEncoder for DotPackingSparseEncoder<Q>
where
    Q: DotPacking8Quantizer<InputValue = f32>,
{
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u16, Q::InputValue>;
    type QueryVector<'q> = SparseVectorView<'q, u16, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;
    type Evaluator<'e>
        = DotPackingSparseQueryEvaluator<'e, Q>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        DotPackingSparseQueryEvaluator::new(query, self)
    }
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        DotPackingSparseQueryEvaluator::new(decoded.as_view(), self)
    }
    fn output_dim(&self) -> usize {
        self.dim
    }
    fn input_dim(&self) -> usize {
        self.dim
    }
}

impl<Q> SparseDataEncoder for DotPackingSparseEncoder<Q>
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
        let view = unsafe { DotPackingSparseView::from_unchecked_slice(encoded.data()) };
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

            // For per-component quantizers (e.g., ScalarU8Quantizer), we must decode AFTER
            // inverse mapping and permutation to ensure each raw byte is decoded with its
            // original component's scale. Permuting raw bytes along with their components
            // ensures proper alignment.
            let permutation = rusty_perm::PermD::from_sort(decoded_comp.as_slice());
            permutation.apply(values_raw.as_mut_slice()).unwrap();
            permutation.apply(decoded_comp.as_mut_slice()).unwrap();

            let values: Vec<f32> = decoded_comp
                .iter()
                .zip(values_raw.iter())
                .map(|(&c, &v)| self.quantizer.decode_value(c, v))
                .collect();

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

impl<Q> PackedSparseVectorEncoder for DotPackingSparseEncoder<Q>
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

impl<Q> SpaceUsage for DotPackingSparseEncoder<Q>
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
