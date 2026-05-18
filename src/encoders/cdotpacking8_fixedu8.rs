use rusty_perm::PermApply as _;
use rusty_perm::PermFromSorting as _;

use crate::core::sealed;
use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::encoders::dotpacking8::common::{
    DotPacking8Iter, compute_safe_simd_padding, encode_blocks, simd_prefix_sum
};
use crate::{FixedU8Q, SpaceUsage};
use bytemuck::cast_slice;
use bytemuck::from_bytes;
use std::simd::StdFloat;
use std::simd::prelude::*;
const FIXED_U8_SCALE: f32 = 1.0 / ((1u32 << FixedU8Q::FRAC_NBITS) as f32);

const N: usize = 8;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct CDotPacking8FixedU8Encoder {
    dim: usize,
    num_clusters: usize,
    all_refs: Vec<u16>,
    max_ref_size: usize,
    component_mapping: Option<Box<[u16]>>,
    #[serde(default)]
    inverse_component_mapping: Option<Box<[u16]>>,
}

impl PartialEq for CDotPacking8FixedU8Encoder {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
            && self.all_refs == other.all_refs
            && self.component_mapping == other.component_mapping
            && self.inverse_component_mapping == other.inverse_component_mapping
    }
}

impl sealed::Sealed for CDotPacking8FixedU8Encoder {}

impl CDotPacking8FixedU8Encoder {
    pub fn new_with_references(
        input_dim: usize,
        reference_lists: Vec<Vec<u16>>,
        max_ref_size: usize,
    ) -> Self {
        let num_clusters = reference_lists.len();
        let mut all_refs = Vec::with_capacity(reference_lists.len() * max_ref_size);
        for mut list in reference_lists {
            list.sort_unstable();
            list.truncate(max_ref_size);
            while list.len() < max_ref_size {
                list.push(u16::MAX);
            }
            all_refs.extend_from_slice(&list);
        }
        Self {
            dim: input_dim,
            all_refs,
            max_ref_size,
            num_clusters,
            component_mapping: None,
            inverse_component_mapping: None,
        }
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
        self.remap_all_refs();
    }

    fn remap_all_refs(&mut self) {
        if let Some(mapping) = &self.component_mapping {
            for ref_id in self.all_refs.iter_mut() {
                if *ref_id != u16::MAX {
                    *ref_id = mapping[*ref_id as usize];
                }
            }
            (0..self.num_clusters).for_each(|i| {
                let s = i * self.max_ref_size;
                let range = s..s + self.max_ref_size;
                self.all_refs[range].sort_unstable();
            });
        }
    }

    fn compute_inverse_mapping(component_mapping: &[u16]) -> Vec<u16> {
        let dim = component_mapping.len();
        let mut inverse = vec![0u16; dim];
        for (old, &new) in component_mapping.iter().enumerate() {
            inverse[new as usize] = old as u16;
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

    #[inline]
    fn get_ref_slice(&self, ref_id: u16) -> &[u16] {
        let start = (ref_id as u32 * self.max_ref_size as u32) as usize;
        unsafe {
            self.all_refs
                .get_unchecked(start..(start + self.max_ref_size))
        }
    }

    pub fn push_vector<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, u16, FixedU8Q>,
        ref_id: u16,
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

        let ref_indices = self.get_ref_slice(ref_id);
        let mut mapped_positions = Vec::new();
        let mut mapped_values = Vec::new();
        let mut residual_indices = Vec::new();
        let mut residual_values = Vec::new();

        for (&comp, &val) in q_components.iter().zip(q_values.iter()) {
            match ref_indices.binary_search(&comp) {
                Ok(pos) => {
                    mapped_positions.push(pos as u32);
                    mapped_values.push(val);
                }
                Err(_) => {
                    residual_indices.push(comp as u32);
                    residual_values.push(val);
                }
            }
        }

        let map_n = mapped_values.len();
        let res_n = residual_values.len();

        // 1. Mapped Gaps
        let mut gaps_m = Vec::with_capacity(map_n);
        let mut last_m = 0u32;
        for &p in &mapped_positions {
            gaps_m.push(p - last_m);
            last_m = p;
        }

        // 2. Residual Gaps
        let mut gaps_r = Vec::with_capacity(res_n);
        let mut last_r = 0u32;
        for &comp in &residual_indices {
            gaps_r.push(comp - last_r);
            last_r = comp;
        }
        let (selectors_m, payloads_m) = encode_blocks(&gaps_m);
        let (selectors_r, payloads_r) = encode_blocks(&gaps_r);

        let mut payload = Vec::new();

        // 1. Header (6 bytes)
        payload.extend_from_slice(&ref_id.to_le_bytes());
        payload.extend_from_slice(&(map_n as u16).to_le_bytes());
        payload.extend_from_slice(&(res_n as u16).to_le_bytes());

        // 2. Selectors
        payload.extend_from_slice(&selectors_m);
        payload.extend_from_slice(&selectors_r);

        // 3. Payloads
        payload.extend_from_slice(&payloads_m);
        payload.extend_from_slice(&payloads_r);

        // 4. Values (padded before to ensure total length is a multiple of 8)
        let n_total = map_n + res_n;
        let mut padding_before = (8 - (payload.len() + n_total) % 8) % 8;

        if res_n > 0 {
            padding_before =
                compute_safe_simd_padding(res_n, &selectors_r, padding_before, n_total);
        } else if map_n > 0 {
            padding_before =
                compute_safe_simd_padding(map_n, &selectors_m, padding_before, n_total);
        }

        payload.resize(payload.len() + padding_before, 0);
        payload.extend_from_slice(&mapped_values);
        payload.extend_from_slice(&residual_values);

        let data = payload
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()));

        output.extend(data);
    }
}

#[derive(Clone)]
struct CDotPacking8Fixedu8<'a> {
    n_m: usize,
    n_r: usize,
    ref_id: u16,
    selectors_m: &'a [u8],
    selectors_r: &'a [u8],
    payloads: &'a [u8],
    values_m: &'a [u8],
    values_r: &'a [u8],
}

impl<'a> CDotPacking8Fixedu8<'a> {
    unsafe fn from_unchecked_slice(slice: &'a [u64]) -> Self {
        let bytes = cast_slice::<u64, u8>(slice);
        let ref_id = from_bytes::<u16>(&bytes[0..2]).to_le();
        let map_n = from_bytes::<u16>(&bytes[2..4]).to_le() as usize;
        let res_n = from_bytes::<u16>(&bytes[4..6]).to_le() as usize;

        let mut offset = 6;
        let n_blocks_m = (map_n + N - 1) / N;
        let n_blocks_r = (res_n + N - 1) / N;
        let selectors_m_size = (n_blocks_m + 1) / 2;
        let selectors_r_size = (n_blocks_r + 1) / 2;

        let selectors_m = &bytes[offset..offset + selectors_m_size];
        offset += selectors_m_size;
        let selectors_r = &bytes[offset..offset + selectors_r_size];
        offset += selectors_r_size;

        let payloads = &bytes[offset..];
        let values_start = bytes.len() - (map_n + res_n);
        let values_m = &bytes[values_start..values_start + map_n];
        let values_r = &bytes[values_start + map_n..values_start + map_n + res_n];

        Self {
            n_m: map_n,
            n_r: res_n,
            ref_id,
            selectors_m,
            selectors_r,
            payloads,
            values_m,
            values_r,
        }
    }

    #[inline(always)]
    fn mapped_iter(&self) -> DotPacking8Iter<'a> {
        DotPacking8Iter {
            bulk_blocks: self.n_m / N,
            block_idx: 0,
            n: self.n_m,
            selectors: self.selectors_m,
            payload_ptr: self.payloads.as_ptr(),
            val_ptr: self.values_m.as_ptr(),
        }
    }

    #[inline(always)]
    fn residual_iter(&self, start_payload_ptr: *const u8) -> DotPacking8Iter<'a> {
        DotPacking8Iter {
            bulk_blocks: self.n_r / N,
            block_idx: 0,
            n: self.n_r,
            selectors: self.selectors_r,
            payload_ptr: start_payload_ptr,
            val_ptr: self.values_r.as_ptr(),
        }
    }
}


impl SparseDataEncoder for CDotPacking8FixedU8Encoder {
    type InputComponentType = u16;
    type InputValueType = FixedU8Q;
    type OutputComponentType = u16;
    type OutputValueType = FixedU8Q;

    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let view = unsafe { CDotPacking8Fixedu8::from_unchecked_slice(encoded.data()) };
        let ref_indices = self.get_ref_slice(view.ref_id);

        let mut mapped_comps = Vec::with_capacity(view.n_m);
        let mut mapped_vals = Vec::with_capacity(view.n_m);
        let mut res_comps = Vec::with_capacity(view.n_r);
        let mut res_vals = Vec::with_capacity(view.n_r);

        // 1. Decode Mapped
        let mut map_iter = view.mapped_iter();
        let mut ref_indices_m = ref_indices;
        for (gaps, vals) in &mut map_iter {
            let pos_in_ref = simd_prefix_sum(gaps);
            let comps = unsafe {
                Simd::gather_select_unchecked(
                    ref_indices_m,
                    Mask::splat(true),
                    pos_in_ref.cast(),
                    Simd::splat(0),
                )
            };
            let vals_arr = vals.to_array();
            let comps_arr = comps.to_array();
            for i in 0..N {
                mapped_comps.push(comps_arr[i] as u16);
                mapped_vals.push(vals_arr[i] as f32 * FIXED_U8_SCALE);
            }
            ref_indices_m = unsafe { ref_indices_m.get_unchecked(pos_in_ref[N - 1] as usize..) };
        }
        let (gaps, tail_values, remaining) = map_iter.decode_tail();
        if remaining > 0 {
            let pos_in_ref = simd_prefix_sum(gaps);
            let rem_gaps_m = pos_in_ref.to_array();
            for i in 0..remaining {
                let abs_comp = *unsafe { ref_indices_m.get_unchecked(rem_gaps_m[i] as usize) };
                mapped_comps.push(abs_comp as u16);
                mapped_vals.push(tail_values[i] as f32 * FIXED_U8_SCALE);
            }
        }

        // 2. Decode Residual
        let mut residual_iter = view.residual_iter(map_iter.payload_ptr);
        let mut last_comp = 0u32;
        for (gaps, vals) in &mut residual_iter {
            let components = simd_prefix_sum(gaps);
            let absolute_components = components + Simd::splat(last_comp);
            let absolute_arr = absolute_components.to_array();
            let vals_arr = vals.to_array();
            for i in 0..N {
                res_comps.push(absolute_arr[i] as u16);
                res_vals.push(vals_arr[i] as f32 * FIXED_U8_SCALE);
            }
            last_comp = absolute_arr[N - 1];
        }
        let (gaps, tail_values, remaining) = residual_iter.decode_tail();
        if remaining > 0 {
            let components = simd_prefix_sum(gaps);
            let absolute_components = components + Simd::splat(last_comp);
            let absolute_arr = absolute_components.to_array();
            for i in 0..remaining {
                res_comps.push(absolute_arr[i] as u16);
                res_vals.push(tail_values[i] as f32 * FIXED_U8_SCALE);
            }
        }

        if let Some(component_mapping) = self.component_mapping() {
            let mut decoded_comp = mapped_comps;
            let mut values = mapped_vals;
            decoded_comp.extend_from_slice(&res_comps);
            values.extend_from_slice(&res_vals);

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

            SparseVectorOwned::new(decoded_comp, values)
        } else {
            let n_total = mapped_comps.len() + res_comps.len();
            let mut decoded_comp = Vec::with_capacity(n_total);
            let mut values = Vec::with_capacity(n_total);

            let (mut i, mut j) = (0, 0);
            while i < mapped_comps.len() && j < res_comps.len() {
                if mapped_comps[i] < res_comps[j] {
                    decoded_comp.push(mapped_comps[i]);
                    values.push(mapped_vals[i]);
                    i += 1;
                } else {
                    decoded_comp.push(res_comps[j]);
                    values.push(res_vals[j]);
                    j += 1;
                }
            }
            decoded_comp.extend_from_slice(&mapped_comps[i..]);
            values.extend_from_slice(&mapped_vals[i..]);
            decoded_comp.extend_from_slice(&res_comps[j..]);
            values.extend_from_slice(&res_vals[j..]);

            SparseVectorOwned::new(decoded_comp, values)
        }
    }
}

impl PackedSparseVectorEncoder for CDotPacking8FixedU8Encoder {
    type PackedDataType = u64;
    fn push_encoded<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, u16, FixedU8Q>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<u64>,
    {
        self.push_vector(input, 0, output);
    }
}

impl VectorEncoder for CDotPacking8FixedU8Encoder {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u16, FixedU8Q>;
    type QueryVector<'q> = SparseVectorView<'q, u16, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;
    type Evaluator<'e>
        = CDotPacking8Fixedu8QueryEvaluator<'e>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        CDotPacking8Fixedu8QueryEvaluator::new(query, self)
    }
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        CDotPacking8Fixedu8QueryEvaluator::new(decoded.as_view(), self)
    }
    fn output_dim(&self) -> usize {
        self.dim
    }
    fn input_dim(&self) -> usize {
        self.dim
    }
}

pub struct CDotPacking8Fixedu8QueryEvaluator<'a> {
    encoder: &'a CDotPacking8FixedU8Encoder,
    dense_query: Vec<f32>,
}

impl<'a, 'v> CDotPacking8Fixedu8QueryEvaluator<'a> {
    pub fn new(
        query: SparseVectorView<'_, u16, f32>,
        encoder: &'a CDotPacking8FixedU8Encoder,
    ) -> Self {
        let mut dense_query = vec![0.0f32; encoder.dim];
        for (&c, &v) in query.components().iter().zip(query.values().iter()) {
            let mapped_c = if let Some(mapping) = &encoder.component_mapping {
                mapping[c as usize]
            } else {
                c
            };
            dense_query[mapped_c as usize] = v * FIXED_U8_SCALE;
        }
        Self {
            encoder,
            dense_query,
        }
    }

    #[inline]
    unsafe fn simd_compute_distance(&self, vector: PackedVectorView<'v, u64>) -> DotProduct {
        let view = unsafe { CDotPacking8Fixedu8::from_unchecked_slice(vector.data()) };
        let ref_indices = self.encoder.get_ref_slice(view.ref_id);

        let mut acc = Simd::<f32, N>::splat(0.0);
        let query = self.dense_query.as_slice();
        let mut total_unscaled = 0.0f32;

        // --- 1. MAPPED PHASE ---
        let mut ref_indices_m = ref_indices;
        let mut map_iter = view.mapped_iter();

        for (gaps, values) in &mut map_iter {
            let pos_in_ref = simd_prefix_sum(gaps);
            let comps = unsafe {
                Simd::gather_select_unchecked(
                    ref_indices_m,
                    Mask::splat(true),
                    pos_in_ref.cast(),
                    Simd::splat(0),
                )
            };
            let q_vals = unsafe {
                Simd::gather_select_unchecked(
                    query,
                    Mask::splat(true),
                    comps.cast(),
                    Simd::splat(0.0),
                )
            };
            acc = q_vals.mul_add(values.cast(), acc);
            ref_indices_m = unsafe { ref_indices_m.get_unchecked(pos_in_ref[N - 1] as usize..) };
        }

        let (gaps, tail_values, remaining) = map_iter.decode_tail();
        if remaining > 0 {
            let components = simd_prefix_sum(gaps);
            let rem_gaps_m = components.to_array();
            for i in 0..remaining {
                let abs_comp = *unsafe { ref_indices_m.get_unchecked(rem_gaps_m[i] as usize) };
                total_unscaled += query[abs_comp as usize] * tail_values[i] as f32;
            }
        }

        // --- 2. RESIDUAL PHASE ---
        let mut query_r = query;
        let mut residual_iter = view.residual_iter(map_iter.payload_ptr);

        for (gaps, values) in &mut residual_iter {
            let comps = simd_prefix_sum(gaps);
            let q_vals = unsafe {
                Simd::gather_select_unchecked(
                    query_r,
                    Mask::splat(true),
                    comps.cast(),
                    Simd::splat(0.0),
                )
            };
            acc = q_vals.mul_add(values.cast(), acc);
            query_r = unsafe { query_r.get_unchecked(comps[N - 1] as usize..) };
        }

        let (gaps, tail_values, remaining) = residual_iter.decode_tail();
        if remaining > 0 {
            let components = simd_prefix_sum(gaps);
            let rem_gaps_r = components.to_array();
            for i in 0..remaining {
                total_unscaled += query_r[rem_gaps_r[i] as usize] * tail_values[i] as f32;
            }
        }

        total_unscaled += acc.reduce_sum();

        DotProduct(total_unscaled)
    }
}

impl<'a, 'v> QueryEvaluator<PackedVectorView<'v, u64>> for CDotPacking8Fixedu8QueryEvaluator<'a> {
    type Distance = DotProduct;
    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> Self::Distance {
        unsafe { self.simd_compute_distance(vector) }
    }
}

impl SpaceUsage for CDotPacking8FixedU8Encoder {
    fn space_usage_bytes(&self) -> usize {
        let size_of_mapping = match &self.component_mapping {
            Some(component_mapping) => component_mapping.space_usage_bytes(),
            None => std::mem::size_of::<Option<Box<[u16]>>>(),
        };
        self.all_refs.len() * 2
            + size_of_mapping
            + self.dim.space_usage_bytes()
            + self.num_clusters.space_usage_bytes()
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
    fn compute_distance_with_only_mapped_bulk() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let encoder = CDotPacking8FixedU8Encoder::new_with_references(100, reference_lists, 512);
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
        let input = SparseVectorView::new(&[0, 4, 8, 24, 36, 48, 53, 90], &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);
        let query = SparseVectorView::new(
            &[2, 4, 6, 8, 24, 70, 90],
            &[0.5, 1.5, 2.5, 1.0, 2.0, 1.0, 2.0],
        );
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
        // v = (0, 1.0), (4, 3.0), (8, 2.0), (24, 3.5), (36, 1.5), (48, 2.0), (53, 1.0), (90, 2.0)
        // query = (2, 0.5), (4, 1.5), (6, 2.5), (8, 1.0), (24, 2.0), (70, 1.0), (90, 2.0)
        let expected = 1.5 * 3.0 + 2.0 * 1.0 + 2.0 * 3.5 + 2.0 * 2.0;
        assert!((dist.distance() - expected).abs() < 1e-5);
    }

    #[test]
    fn compute_distance_with_only_mapped_tail() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let encoder = CDotPacking8FixedU8Encoder::new_with_references(100, reference_lists, 512);
        let binding = [fixed(1.0), fixed(3.0), fixed(2.0), fixed(3.5)];
        let input = SparseVectorView::new(&[0, 4, 8, 24], &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);
        let query = SparseVectorView::new(&[2, 4, 6, 8], &[0.5, 1.5, 2.5, 1.0]);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
        // v = (0, 1.0), (4, 3.0), (8, 2.0), (24, 3.5)
        // query = (2, 0.5), (4, 1.5), (6, 2.5), (8, 1.0), (70, 1.0), (90, 2.0)
        let expected = 1.5 * 3.0 + 2.0 * 1.0;
        assert!((dist.distance() - expected).abs() < 1e-5);
    }

    #[test]
    fn compute_distance_with_mapped_both() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let encoder = CDotPacking8FixedU8Encoder::new_with_references(100, reference_lists, 512);
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
        // v = (0, 1.0), (4, 3.0), (8, 2.0), (24, 3.5), (28, 1.5), (36, 2.0), (48, 1.0), (53, 2.0), (70, 3.0), (90, 2.5)
        let input = SparseVectorView::new(&[0, 4, 8, 24, 28, 36, 48, 53, 70, 90], &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);
        // query = (2, 0.5), (4, 1.5), (8, 2.5), (24, 1.0), (28, 2.0), (70, 1.0), (90, 2.0)
        let query = SparseVectorView::new(
            &[2, 4, 8, 24, 28, 70, 90],
            &[0.5, 1.5, 2.5, 1.0, 2.0, 1.0, 2.0],
        );
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
        // v = (0, 1.0), (4, 3.0), (8, 2.0), (24, 3.5), (28, 1.5), (36, 2.0), (48, 1.0), (53, 2.0), (70, 3.0), (90, 2.5)
        // query = (2, 0.5), (4, 1.5), (8, 2.5), (24, 1.0), (28, 2.0), (70, 1.0), (90, 2.0)
        let expected = 3.0 * 1.5 + 2.0 * 2.5 + 3.5 * 1.0 + 1.5 * 2.0 + 3.0 * 1.0 + 2.5 * 2.0;
        assert!((dist.distance() - expected).abs() < 1e-5);
    }
    
    #[test]
    fn compute_distance_only_res_tail() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8]];
        let encoder = CDotPacking8FixedU8Encoder::new_with_references(25, reference_lists, 512);
        let binding = [fixed(1.0), fixed(3.0)];
        let input = SparseVectorView::new(&[5, 12], &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);
        let query = SparseVectorView::new(&[5, 9, 12], &[0.5f32, 1.5, 2.5]);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
        let expected = 1.0 * 0.5 + 3.0 * 2.5;
        assert!((dist.distance() - expected).abs() < 1e-5);
    }

    #[test]
    fn compute_distance_only_res_bulk() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8]];
        let encoder = CDotPacking8FixedU8Encoder::new_with_references(70, reference_lists, 512);
        let binding = [
            fixed(1.0),
            fixed(2.5),
            fixed(2.0),
            fixed(3.5),
            fixed(1.0),
            fixed(2.0),
            fixed(1.0),
            fixed(2.0),
        ];
        let input = SparseVectorView::new(&[5, 12, 15, 18, 21, 31, 35, 60], &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);
        let query = SparseVectorView::new(&[5, 9, 12, 21, 60], &[0.5f32, 1.5, 2.5, 1.0, 2.0]);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let expected = 1.0 * 0.5 + 2.5 * 2.5 + 1.0 * 1.0 + 2.0 * 2.0;
        assert!((dist.distance() - expected).abs() < 1e-5);
    }

    #[test]
    fn compute_distance_only_res_both() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8]];
        let encoder = CDotPacking8FixedU8Encoder::new_with_references(80, reference_lists, 512);
        let comps = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23];
        let binding = [
            fixed(1.0),
            fixed(2.0),
            fixed(1.5),
            fixed(2.5),
            fixed(3.0),
            fixed(1.0),
            fixed(2.0),
            fixed(1.5),
            fixed(2.5),
            fixed(3.0),
        ];
        let input = SparseVectorView::new(&comps, &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);

        let query_vals = [0.5, 1.5, 0.5, 2.0, 1.0, 1.5, 2.0, 0.5, 1.5, 2.0];
        let query = SparseVectorView::new(&comps, &query_vals);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let mut expected = 0.0f32;
        for i in 0..comps.len() {
            expected += binding[i].to_f32().unwrap() * query_vals[i];
        }
        assert!((dist.distance() - expected).abs() < 1e-5);
    }

    #[test]
    fn compute_distance_mapped_tail_residual_tail() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let encoder = CDotPacking8FixedU8Encoder::new_with_references(100, reference_lists, 512);
        let binding = [
            fixed(1.0),
            fixed(2.5),
            fixed(2.0),
            fixed(3.5),
            fixed(1.0),
            fixed(2.0),
            fixed(1.0),
            fixed(2.0),
        ];
        let input = SparseVectorView::new(&[5, 12, 15, 18, 21, 31, 35, 60], &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);
        let query = SparseVectorView::new(&[5, 9, 12, 21, 60], &[0.5f32, 1.5, 2.5, 1.0, 2.0]);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let expected = 1.0 * 0.5 + 2.5 * 2.5 + 1.0 * 1.0 + 2.0 * 2.0;
        assert!((dist.distance() - expected).abs() < 1e-5);
    }

    #[test]
    fn compute_distance_mapped_bulk_residual_bulk() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let encoder = CDotPacking8FixedU8Encoder::new_with_references(100, reference_lists, 512);

        let comps = [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 24, 36, 48, 53];
        let binding = [
            fixed(1.0),
            fixed(2.0),
            fixed(1.5),
            fixed(2.5),
            fixed(3.0),
            fixed(1.0),
            fixed(2.0),
            fixed(1.5),
            fixed(2.5),
            fixed(3.0),
            fixed(1.0),
            fixed(2.0),
            fixed(1.5),
            fixed(2.5),
            fixed(3.0),
            fixed(1.0),
        ];
        let input = SparseVectorView::new(&comps, &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);

        let query_vals = [
            0.5, 1.5, 0.5, 2.0, 1.0, 1.5, 2.0, 0.5, 1.5, 2.0, 1.0, 0.5, 2.5, 1.0, 1.5, 2.0,
        ];
        let query = SparseVectorView::new(&comps, &query_vals);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let mut expected = 0.0f32;
        for i in 0..comps.len() {
            expected += binding[i].to_f32().unwrap() * query_vals[i];
        }
        assert!((dist.distance() - expected).abs() < 1e-5);
    }

    #[test]
    fn compute_distance_mapped_bulk_tail_residual_bulk_tail() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let encoder = CDotPacking8FixedU8Encoder::new_with_references(120, reference_lists, 512);

        let comps = [
            0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 17, 19, 24, 36, 48, 53, 87, 90,
        ];
        let binding = [
            fixed(1.0),
            fixed(2.0),
            fixed(1.5),
            fixed(2.5),
            fixed(3.0),
            fixed(1.0),
            fixed(2.0),
            fixed(1.5),
            fixed(2.5),
            fixed(3.0),
            fixed(1.0),
            fixed(2.0),
            fixed(1.5),
            fixed(2.5),
            fixed(3.0),
            fixed(1.0),
            fixed(2.0),
            fixed(1.5),
            fixed(2.5),
            fixed(3.0),
        ];
        let input = SparseVectorView::new(&comps, &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);

        let query_vals = [
            0.5, 1.5, 0.5, 2.0, 1.0, 1.5, 2.0, 0.5, 1.5, 2.0, 1.0, 0.5, 2.5, 1.0, 1.5, 2.0,
            1.0, 0.5, 2.5, 1.0,
        ];
        let query = SparseVectorView::new(&comps, &query_vals);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let mut expected = 0.0f32;
        for i in 0..comps.len() {
            expected += binding[i].to_f32().unwrap() * query_vals[i];
        }
        assert!((dist.distance() - expected).abs() < 1e-5);
    }

    #[test]
    fn test_all_bit_widths() {
        for b in 1..=16 {
            let max_val = (1u32 << b) - 1;
            let num_vals = 40;
            let mut gaps = Vec::new();
            let mut refs = Vec::new();
            let mut curr = 0u32;
            for i in 0..num_vals {
                let g = (i as u32 % (max_val + 1)).max(1);
                if curr + g > u16::MAX as u32 {
                    break;
                }
                curr += g;
                gaps.push(g);
                refs.push(curr as u16);
            }
            let n = gaps.len();
            if n == 0 {
                continue;
            }

            let reference_lists = vec![refs.clone()];
            let encoder = CDotPacking8FixedU8Encoder::new_with_references(
                u16::MAX as usize + 1,
                reference_lists,
                512,
            );

            let values: Vec<_> = (0..n).map(|i| fixed(1.0 + (i % 7) as f32 / 10.0)).collect();
            let input = SparseVectorView::new(&refs, &values);

            let mut buffer = Vec::new();
            encoder.push_vector(input, 0, &mut buffer);

            let query_vals: Vec<_> = (0..n).map(|i| 0.1 + (i % 5) as f32 / 10.0).collect();
            let query = SparseVectorView::new(&refs, &query_vals);

            let evaluator = encoder.query_evaluator(query);
            let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

            let mut expected = 0.0f32;
            for i in 0..n {
                expected += values[i].to_f32().unwrap() * query_vals[i];
            }

            assert!(
                (dist.distance() - expected).abs() < 1e-3,
                "Failed at b={}: dist={}, expected={}, diff={}, n={}",
                b,
                dist.distance(),
                expected,
                (dist.distance() - expected).abs(),
                n
            );
        }
    }

    #[test]
    fn test_cdot8_decode_roundtrip() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let encoder = CDotPacking8FixedU8Encoder::new_with_references(100, reference_lists, 512);

        let components = vec![0, 4, 8, 24, 28, 36, 48, 53, 70, 90];
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
        let input = SparseVectorView::new(&components, &binding);

        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &components);
        for (v1, v2) in decoded.values().iter().zip(binding.iter()) {
            assert!((v1 - v2.to_f32().unwrap()).abs() < 1e-5);
        }
    }

    #[test]
    fn test_cdot8_decode_train_roundtrip() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let mut encoder = CDotPacking8FixedU8Encoder::new_with_references(100, reference_lists, 512);
        let binding = [fixed(1.0), fixed(3.0), fixed(2.0), fixed(3.5), fixed(2.5), fixed(1.5)];
        let input_components = [0, 4, 8, 24, 60, 87];
        let input0 = SparseVectorView::new(&input_components, &binding);
        let binding1 = [fixed(3.0), fixed(2.0), fixed(3.5), fixed(2.5), fixed(1.5)];
        let input_components1 = [4, 8, 24, 50, 55];
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
        assert_eq!(decoded_vals0.len(), 6);
        assert!((decoded_vals0[0] - 1.0).abs() < 1e-6);
        assert!((decoded_vals0[1] - 3.0).abs() < 1e-6);
        assert!((decoded_vals0[2] - 2.0).abs() < 1e-6);
        assert!((decoded_vals0[3] - 3.5).abs() < 1e-6);
        assert!((decoded_vals0[4] - 2.5).abs() < 1e-6);
        assert!((decoded_vals0[5] - 1.5).abs() < 1e-6);


        assert_eq!(decoded1.components(), &input_components1);
        let decoded_vals1 = decoded1.values();
        assert_eq!(decoded_vals1.len(), 5);
        assert!((decoded_vals1[0] - 3.0).abs() < 1e-6);
        assert!((decoded_vals1[1] - 2.0).abs() < 1e-6);
        assert!((decoded_vals1[2] - 3.5).abs() < 1e-6);
        assert!((decoded_vals1[3] - 2.5).abs() < 1e-6);
        assert!((decoded_vals1[4] - 1.5).abs() < 1e-6);
    }

}
