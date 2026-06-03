use crate::SpaceUsage;
use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::encoders::dotpacking8::common::fast_lane_gather;
use crate::encoders::dotpackingdp::common::{
    DotPackingDpIter, compute_safe_simd_padding, encode_blocks_dp, simd_prefix_sum,
};
use crate::encoders::dotpackingdp::quantizer::DotPackingDpQuantizer;
use bytemuck::cast_slice;
use rusty_perm::{PermApply, PermFromSorting};
use std::simd::{StdFloat, prelude::*};

const N: usize = 8;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct CDotPackingDpEncoder<Q, const MAX_LEN: usize = 16>
where
    Q: DotPackingDpQuantizer,
{
    pub dim: usize,
    pub quantizer: Q,
    pub num_clusters: usize,
    pub all_refs: Vec<u16>,
    pub max_ref_size: usize,
    pub component_mapping: Option<Box<[u16]>>,
    pub inverse_component_mapping: Option<Box<[u16]>>,
}

impl<Q, const MAX_LEN: usize> PartialEq for CDotPackingDpEncoder<Q, MAX_LEN>
where
    Q: DotPackingDpQuantizer + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
            && self.quantizer == other.quantizer
            && self.all_refs == other.all_refs
            && self.component_mapping == other.component_mapping
            && self.inverse_component_mapping == other.inverse_component_mapping
    }
}

impl<Q, const MAX_LEN: usize> CDotPackingDpEncoder<Q, MAX_LEN>
where
    Q: DotPackingDpQuantizer<InputValue = f32>,
{
    pub fn new(
        input_dim: usize,
        quantizer: Q,
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
            quantizer,
            all_refs,
            max_ref_size,
            num_clusters,
            component_mapping: None,
            inverse_component_mapping: None,
        }
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
        input: SparseVectorView<'a, u16, f32>,
        ref_id: u16,
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

        let mut gaps_m = Vec::with_capacity(map_n);
        let mut last_m = 0u32;
        for &p in &mapped_positions {
            gaps_m.push(p - last_m);
            last_m = p;
        }

        let mut gaps_r = Vec::with_capacity(res_n);
        let mut last_r = 0u32;
        for &comp in &residual_indices {
            gaps_r.push(comp - last_r);
            last_r = comp;
        }

        let (sel_m, bytes_m) = encode_blocks_dp::<MAX_LEN>(&gaps_m);
        let (sel_r, bytes_r) = encode_blocks_dp::<MAX_LEN>(&gaps_r);
        let n_blocks_total = sel_m.len() + sel_r.len();

        let mut payload = Vec::new();
        payload.extend_from_slice(&ref_id.to_le_bytes());
        payload.extend_from_slice(&(map_n as u16).to_le_bytes());
        payload.extend_from_slice(&(res_n as u16).to_le_bytes());
        payload.extend_from_slice(&(n_blocks_total as u16).to_le_bytes());

        payload.extend_from_slice(&sel_m);
        payload.extend_from_slice(&sel_r);
        payload.extend_from_slice(&bytes_m);
        payload.extend_from_slice(&bytes_r);

        let n_total = map_n + res_n;
        let mut padding_before = (8 - (payload.len() + n_total) % 8) % 8;

        if !sel_r.is_empty() {
            padding_before = compute_safe_simd_padding(&sel_r, padding_before, n_total);
        } else if !sel_m.is_empty() {
            padding_before = compute_safe_simd_padding(&sel_m, padding_before, n_total);
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

pub struct CDotPackingDpView<'a, const MAX_LEN: usize> {
    pub map_n: usize,
    pub res_n: usize,
    pub ref_id: u16,
    pub selectors: &'a [u8],
    pub payloads: &'a [u8],
    pub values_m: &'a [u8],
    pub values_r: &'a [u8],
}
impl<'a, const MAX_LEN: usize> CDotPackingDpView<'a, MAX_LEN> {
    pub unsafe fn from_unchecked_slice(slice: &'a [u64]) -> Self {
        let bytes = cast_slice::<u64, u8>(slice);
        let ref_id = u16::from_le_bytes(bytes[0..2].try_into().unwrap());
        let map_n = u16::from_le_bytes(bytes[2..4].try_into().unwrap()) as usize;
        let res_n = u16::from_le_bytes(bytes[4..6].try_into().unwrap()) as usize;
        let n = map_n + res_n;
        let n_blocks = u16::from_le_bytes(bytes[6..8].try_into().unwrap()) as usize;

        let selectors = &bytes[8..8 + n_blocks];
        Self {
            map_n,
            res_n,
            ref_id,
            selectors,
            payloads: &bytes[8 + n_blocks..],
            values_m: &bytes[bytes.len() - n..bytes.len() - res_n],
            values_r: &bytes[bytes.len() - res_n..],
        }
    }

    #[inline(always)]
    pub fn mapped_iter(&self) -> DotPackingDpIter<'a, MAX_LEN> {
        DotPackingDpIter {
            n: self.map_n,
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
            val_ptr: self.values_m.as_ptr(),
        }
    }

    #[inline(always)]
    pub fn residual_iter(
        &self,
        map_iter: &DotPackingDpIter<'a, MAX_LEN>,
    ) -> DotPackingDpIter<'a, MAX_LEN> {
        let mapped_bytes = (map_iter.global_bit_pos + 7) / 8;
        DotPackingDpIter {
            n: self.res_n,
            processed_count: 0,
            sel_idx: map_iter.sel_idx,
            chunk_rem: 0,
            b: 0,
            global_bit_pos: 0,
            overflow_count: 0,
            overflow_reg: Simd::splat(0),
            permuted_reg: Simd::splat(0),
            selectors: self.selectors,
            payload_ptr: unsafe { map_iter.payload_ptr.add(mapped_bytes) },
            val_ptr: self.values_r.as_ptr(),
        }
    }
}

impl<Q, const MAX_LEN: usize> SparseDataEncoder for CDotPackingDpEncoder<Q, MAX_LEN>
where
    Q: DotPackingDpQuantizer<InputValue = f32>,
{
    type InputComponentType = u16;
    type InputValueType = f32;
    type OutputComponentType = u16;
    type OutputValueType = u8;

    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let view = unsafe { CDotPackingDpView::<MAX_LEN>::from_unchecked_slice(encoded.data()) };
        let ref_indices = self.get_ref_slice(view.ref_id);

        let mut mapped_comps = Vec::with_capacity(view.map_n);
        let mut mapped_values_raw = Vec::with_capacity(view.map_n);
        let mut res_comps = Vec::with_capacity(view.res_n);
        let mut res_values_raw = Vec::with_capacity(view.res_n);

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
                mapped_values_raw.push(vals_arr[i]);
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
                mapped_values_raw.push(tail_values[i]);
            }
        }

        // 2. Decode Residual
        let mut residual_iter = view.residual_iter(&map_iter);
        let mut last_comp = 0u32;
        for (gaps, vals) in &mut residual_iter {
            let components = simd_prefix_sum(gaps);
            let absolute_components = components + Simd::splat(last_comp);
            let absolute_arr = absolute_components.to_array();
            let vals_arr = vals.to_array();
            for i in 0..N {
                res_comps.push(absolute_arr[i] as u16);
                res_values_raw.push(vals_arr[i]);
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
                res_values_raw.push(tail_values[i]);
            }
        }

        let mut decoded_comp = mapped_comps;
        let mut values_raw = mapped_values_raw;
        decoded_comp.extend_from_slice(&res_comps);
        values_raw.extend_from_slice(&res_values_raw);

        if let Some(component_mapping) = self.component_mapping() {
            let inverse: std::borrow::Cow<'_, [u16]> = match self.inverse_component_mapping() {
                Some(inverse) => std::borrow::Cow::Borrowed(inverse),
                None => std::borrow::Cow::Owned(Self::compute_inverse_mapping(component_mapping)),
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
            let mut values: Vec<f32> = decoded_comp
                .iter()
                .zip(values_raw.iter())
                .map(|(&c, &v)| self.quantizer.decode_value(c, v))
                .collect();

            let permutation = rusty_perm::PermD::from_sort(decoded_comp.as_slice());
            permutation.apply(values.as_mut_slice()).unwrap();
            permutation.apply(decoded_comp.as_mut_slice()).unwrap();

            SparseVectorOwned::new(decoded_comp, values)
        }
    }
}

impl<Q, const MAX_LEN: usize> PackedSparseVectorEncoder for CDotPackingDpEncoder<Q, MAX_LEN>
where
    Q: DotPackingDpQuantizer<InputValue = f32>,
{
    type PackedDataType = u64;
    fn push_encoded<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, u16, f32>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<u64>,
    {
        self.push_vector(input, 0, output);
    }
}

impl<Q, const MAX_LEN: usize> VectorEncoder for CDotPackingDpEncoder<Q, MAX_LEN>
where
    Q: DotPackingDpQuantizer<InputValue = f32>,
{
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u16, f32>;
    type QueryVector<'q> = SparseVectorView<'q, u16, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;
    type Evaluator<'e>
        = CDotPackingDpQueryEvaluator<'e, Q, MAX_LEN>
    where
        Self: 'e;
    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        CDotPackingDpQueryEvaluator::new(query, self)
    }
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        CDotPackingDpQueryEvaluator::<Q, MAX_LEN>::new(decoded.as_view(), self)
    }
    fn output_dim(&self) -> usize {
        self.dim
    }
    fn input_dim(&self) -> usize {
        self.dim
    }
}

pub struct CDotPackingDpQueryEvaluator<'a, Q, const MAX_LEN: usize = 16>
where
    Q: DotPackingDpQuantizer,
{
    encoder: &'a CDotPackingDpEncoder<Q, MAX_LEN>,
    dense_query: Vec<f32>,
}

impl<'a, Q, const MAX_LEN: usize> CDotPackingDpQueryEvaluator<'a, Q, MAX_LEN>
where
    Q: DotPackingDpQuantizer<InputValue = f32>,
{
    pub fn new(
        query: SparseVectorView<'_, u16, f32>,
        encoder: &'a CDotPackingDpEncoder<Q, MAX_LEN>,
    ) -> Self {
        let mut dense_query = vec![0.0f32; encoder.dim];
        for (&c, &v) in query.components().iter().zip(query.values().iter()) {
            let mapped_c = if let Some(m) = &encoder.component_mapping {
                m[c as usize]
            } else {
                c
            };
            dense_query[mapped_c as usize] = encoder.quantizer.query_value(c, v);
        }
        Self {
            encoder,
            dense_query,
        }
    }

    #[inline]
    unsafe fn simd_compute_distance(&self, vector: PackedVectorView<'_, u64>) -> DotProduct {
        let view = unsafe { CDotPackingDpView::<MAX_LEN>::from_unchecked_slice(vector.data()) };
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
            let q_vals = fast_lane_gather(query, comps.cast());

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
        let mut residual_iter = view.residual_iter(&map_iter);

        for (gaps, values) in &mut residual_iter {
            let comps = simd_prefix_sum(gaps);
            let q_vals = fast_lane_gather(query_r, comps);
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

impl<'a, 'v, Q, const MAX_LEN: usize> QueryEvaluator<PackedVectorView<'v, u64>>
    for CDotPackingDpQueryEvaluator<'a, Q, MAX_LEN>
where
    Q: DotPackingDpQuantizer<InputValue = f32>,
{
    type Distance = DotProduct;
    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> Self::Distance {
        unsafe { self.simd_compute_distance(vector) }
    }
}

impl<Q, const MAX_LEN: usize> SpaceUsage for CDotPackingDpEncoder<Q, MAX_LEN>
where
    Q: DotPackingDpQuantizer,
{
    fn space_usage_bytes(&self) -> usize {
        let size_of_mapping = self
            .component_mapping
            .as_ref()
            .map_or(std::mem::size_of::<Option<Box<[u16]>>>(), |m| {
                m.len() * 2 + 8
            });
        self.all_refs.len() * 2 + size_of_mapping + 16 + self.quantizer.space_usage_bytes()
    }
}
