//! `cdotpacking8` component compression blended with 4-bit per-component **centroid**
//! value compression.
//!
//! This encoder keeps the **component** codec of [`crate::CDotPacking8ScalarU8Encoder`]
//! verbatim (two-tier reference-list scheme: mapped positions vs. residual gaps,
//! variable-bit gap packing, bisection component permutation, dense-query SIMD gather)
//! but stores the per-component **values** as 4-bit nibbles whose codes index a
//! per-component **codebook** of 16 centroids, instead of the linear scalar steps used by
//! [`crate::CDotPacking8Scalar4BitEncoder`].
//!
//! The value model is the one implemented by
//! [`super::packed_centroid_based_quantization_sparse_scalar::PackedCentroidSparseQuantizer`]
//! restricted to `nbits == 4`: `dequant(c, code) = centroids[c·16 + code]`, and scoring a
//! query value `q` against a doc code computes `q · centroids[c·16 + code]`. The query
//! evaluator therefore precomputes a dense LUT `lut[c·16 + v] = q · centroids[c·16 + v]`
//! and the hot kernel reduces to a 2-D gather `lut[comp·16 + code]` + add (no FMA, no
//! widening) — the same kernel as `dot_product_centroid_nbits4`, driven over
//! `cdotpacking8`'s component walk.
//!
//! This module is intentionally self-contained: the entire component skeleton and the
//! nibble byte layout are identical to [`crate::encoders::cdotpacking8_scalar4bit`]; only
//! the value quantizer and the value-combine in the kernel differ. It reuses only the
//! *public* helpers from [`crate::encoders::dotpacking8::common`] and the *public*
//! `PackedCentroidSparseQuantizer::train` + `centroids()` for codebook fitting, so no
//! existing algorithm file is modified.
//!
//! ## Per-vector byte layout (identical to `cdotpacking8_scalar4bit`)
//! ```text
//! [ref_id u16][map_n u16][res_n u16]
//! [selectors_m][selectors_r][gap payloads...][pad]
//! [mapped values: nibble-packed]   (nibble_bytes(map_n) bytes)
//! [residual values: nibble-packed] (nibble_bytes(res_n) bytes)
//! ```
//! Each value stream is packed as **half-split bulk + sequential tail**: every full block
//! of 8 codes becomes 4 bytes with `byte[i] = code[i] | (code[i+4] << 4)`, and the trailing
//! `rem < 8` codes are stored as sequential nibble pairs (`ceil(rem/2)` bytes).

use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::encoders::dotpacking8::common::{
    compute_safe_simd_padding, encode_blocks, simd_prefix_sum,
};
use crate::encoders::dotpacking8::nibble4bit::{
    block_bit_width, gap_iter, nibble_bytes, pack_nibbles, Centroid4BitQuantizer, NibbleReader,
    DEFAULT_KMEANS_ITERS, DEFAULT_LOWER_PCT, DEFAULT_UPPER_PCT, NUM_CENTROIDS,
};
use crate::{Dataset, PlainSparseDataset, SpaceUsage, SquaredEuclideanDistance};
use bytemuck::{cast_slice, from_bytes};
use rusty_perm::{PermApply, PermFromSorting};
use std::simd::prelude::*;

const N: usize = 8;

// ---------------------------------------------------------------------------
// Encoder.
// ---------------------------------------------------------------------------

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct CDotPacking8Centroid4BitEncoder {
    pub dim: usize,
    pub quantizer: Centroid4BitQuantizer,
    pub num_clusters: usize,
    pub all_refs: Vec<u16>,
    pub max_ref_size: usize,
    pub component_mapping: Option<Box<[u16]>>,
    pub inverse_component_mapping: Option<Box<[u16]>>,
}

impl PartialEq for CDotPacking8Centroid4BitEncoder {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
            && self.quantizer == other.quantizer
            && self.all_refs == other.all_refs
            && self.component_mapping == other.component_mapping
            && self.inverse_component_mapping == other.inverse_component_mapping
    }
}

impl CDotPacking8Centroid4BitEncoder {
    pub fn new_with_references(
        input_dim: usize,
        reference_lists: Vec<Vec<u16>>,
        max_ref_size: usize,
    ) -> Self {
        let num_clusters = reference_lists.len();
        let mut all_refs = Vec::with_capacity(num_clusters * max_ref_size);
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
            quantizer: Centroid4BitQuantizer::new(
                input_dim,
                vec![0.0f32; input_dim * NUM_CENTROIDS].into_boxed_slice(),
            ),
            all_refs,
            max_ref_size,
            num_clusters,
            component_mapping: None,
            inverse_component_mapping: None,
        }
    }

    /// Train the component permutation (bisection) and the per-component 4-bit codebook,
    /// using the default centroid k-means knobs ([`DEFAULT_LOWER_PCT`], [`DEFAULT_UPPER_PCT`],
    /// [`DEFAULT_KMEANS_ITERS`]). This is the entry point the clustered-dataset `From` macro
    /// calls; to tune the codebook fit, call [`Self::train_with_params`] directly.
    pub fn train(&mut self, training_data: &PlainSparseDataset<u16, f32, SquaredEuclideanDistance>) {
        self.train_with_params(
            training_data,
            DEFAULT_LOWER_PCT,
            DEFAULT_UPPER_PCT,
            DEFAULT_KMEANS_ITERS,
        );
    }

    /// Training entry point with explicit centroid k-means knobs — the real API for tuning
    /// (benchmarks, programmatic callers). [`Self::train`] delegates here with defaults.
    pub fn train_with_params(
        &mut self,
        training_data: &PlainSparseDataset<u16, f32, SquaredEuclideanDistance>,
        lower_percentile: f32,
        upper_percentile: f32,
        n_iterations: usize,
    ) {
        const SAMPLE_RATE: usize = 20;
        let sample_size = if training_data.len() / SAMPLE_RATE < 50_000 {
            training_data.len()
        } else {
            training_data.len() / SAMPLE_RATE
        };
        self.train_components(training_data.iter().take(sample_size));
        self.quantizer = Centroid4BitQuantizer::train(
            training_data,
            lower_percentile,
            upper_percentile,
            n_iterations,
        );
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

        let (selectors_m, payloads_m) = encode_blocks(&gaps_m);
        let (selectors_r, payloads_r) = encode_blocks(&gaps_r);

        let mut payload = Vec::new();
        payload.extend_from_slice(&ref_id.to_le_bytes());
        payload.extend_from_slice(&(map_n as u16).to_le_bytes());
        payload.extend_from_slice(&(res_n as u16).to_le_bytes());

        payload.extend_from_slice(&selectors_m);
        payload.extend_from_slice(&selectors_r);
        payload.extend_from_slice(&payloads_m);
        payload.extend_from_slice(&payloads_r);

        // Values are nibble-packed, so the trailing-byte count is half that of the `u8`
        // path. Use it for both the alignment padding and the SIMD over-read safety pad.
        let n_total = nibble_bytes(map_n) + nibble_bytes(res_n);
        let mut padding_before = (8 - (payload.len() + n_total) % 8) % 8;

        if res_n > 0 {
            padding_before =
                compute_safe_simd_padding(res_n, &selectors_r, padding_before, n_total);
        } else if map_n > 0 {
            padding_before =
                compute_safe_simd_padding(map_n, &selectors_m, padding_before, n_total);
        }

        payload.resize(payload.len() + padding_before, 0);
        pack_nibbles(&mut payload, &mapped_values);
        pack_nibbles(&mut payload, &residual_values);

        let data = payload
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()));

        output.extend(data);
    }
}

// ---------------------------------------------------------------------------
// View.
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct CDot4BitView<'a> {
    pub n_m: usize,
    pub n_r: usize,
    pub ref_id: u16,
    pub selectors_m: &'a [u8],
    pub selectors_r: &'a [u8],
    pub payloads: &'a [u8],
    pub values_m: &'a [u8],
    pub values_r: &'a [u8],
}

impl<'a> CDot4BitView<'a> {
    pub unsafe fn from_unchecked_slice(slice: &'a [u64]) -> Self {
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

        let m_bytes = nibble_bytes(map_n);
        let r_bytes = nibble_bytes(res_n);
        let values_start = bytes.len() - (m_bytes + r_bytes);
        let values_m = &bytes[values_start..values_start + m_bytes];
        let values_r = &bytes[values_start + m_bytes..values_start + m_bytes + r_bytes];

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
}

// ---------------------------------------------------------------------------
// Trait impls.
// ---------------------------------------------------------------------------

impl SparseDataEncoder for CDotPacking8Centroid4BitEncoder {
    type InputComponentType = u16;
    type InputValueType = f32;
    type OutputComponentType = u16;
    type OutputValueType = u8;

    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let view = unsafe { CDot4BitView::from_unchecked_slice(encoded.data()) };
        let ref_indices = self.get_ref_slice(view.ref_id);

        let mut mapped_comps = Vec::with_capacity(view.n_m);
        let mut mapped_values_raw = Vec::with_capacity(view.n_m);
        let mut res_comps = Vec::with_capacity(view.n_r);
        let mut res_values_raw = Vec::with_capacity(view.n_r);

        // 1. Decode Mapped
        let mut it = gap_iter(view.n_m, view.selectors_m, view.payloads.as_ptr());
        let mut nib_m = NibbleReader::new(view.values_m);
        let mut ref_indices_m = ref_indices;
        for _ in 0..(view.n_m / N) {
            let gaps = it.decode_lane();
            it.block_idx += 1;
            let pos_in_ref = simd_prefix_sum(gaps);
            let comps = unsafe {
                Simd::gather_select_unchecked(
                    ref_indices_m,
                    Mask::splat(true),
                    pos_in_ref.cast(),
                    Simd::splat(0),
                )
            };
            let vals = nib_m.next_block_u8();
            let comps_arr = comps.to_array();
            for i in 0..N {
                mapped_comps.push(comps_arr[i]);
                mapped_values_raw.push(vals[i]);
            }
            ref_indices_m = unsafe { ref_indices_m.get_unchecked(pos_in_ref[N - 1] as usize..) };
        }
        let rem_m = view.n_m % N;
        let residual_payload_ptr = if rem_m > 0 {
            let tail_start = it.payload_ptr;
            let b = block_bit_width(view.selectors_m, view.n_m / N);
            let gaps = it.decode_lane();
            let pos = simd_prefix_sum(gaps).to_array();
            let codes = nib_m.tail_codes(rem_m);
            for i in 0..rem_m {
                let abs_comp = *unsafe { ref_indices_m.get_unchecked(pos[i] as usize) };
                mapped_comps.push(abs_comp);
                mapped_values_raw.push(codes[i]);
            }
            unsafe { tail_start.add((rem_m * b + 7) / 8) }
        } else {
            it.payload_ptr
        };

        // 2. Decode Residual
        let mut rit = gap_iter(view.n_r, view.selectors_r, residual_payload_ptr);
        let mut nib_r = NibbleReader::new(view.values_r);
        let mut last_comp = 0u32;
        for _ in 0..(view.n_r / N) {
            let gaps = rit.decode_lane();
            rit.block_idx += 1;
            let components = simd_prefix_sum(gaps);
            let absolute_components = components + Simd::splat(last_comp);
            let absolute_arr = absolute_components.to_array();
            let vals = nib_r.next_block_u8();
            for i in 0..N {
                res_comps.push(absolute_arr[i] as u16);
                res_values_raw.push(vals[i]);
            }
            last_comp = absolute_arr[N - 1];
        }
        let rem_r = view.n_r % N;
        if rem_r > 0 {
            let gaps = rit.decode_lane();
            let components = simd_prefix_sum(gaps);
            let absolute_components = components + Simd::splat(last_comp);
            let absolute_arr = absolute_components.to_array();
            let codes = nib_r.tail_codes(rem_r);
            for i in 0..rem_r {
                res_comps.push(absolute_arr[i] as u16);
                res_values_raw.push(codes[i]);
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

impl PackedSparseVectorEncoder for CDotPacking8Centroid4BitEncoder {
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

impl VectorEncoder for CDotPacking8Centroid4BitEncoder {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u16, f32>;
    type QueryVector<'q> = SparseVectorView<'q, u16, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;
    type Evaluator<'e>
        = CDotPacking8Centroid4BitQueryEvaluator<'e>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        CDotPacking8Centroid4BitQueryEvaluator::new(query, self)
    }
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        CDotPacking8Centroid4BitQueryEvaluator::new(decoded.as_view(), self)
    }
    fn output_dim(&self) -> usize {
        self.dim
    }
    fn input_dim(&self) -> usize {
        self.dim
    }
}

pub struct CDotPacking8Centroid4BitQueryEvaluator<'a> {
    encoder: &'a CDotPacking8Centroid4BitEncoder,
    /// Dense LUT `dense_lut[mapped_c·16 + v] = q · centroids[orig_c·16 + v]`. Rows for
    /// components absent from the query stay 0, so their gathers contribute 0.
    dense_lut: Vec<f32>,
}

impl<'a> CDotPacking8Centroid4BitQueryEvaluator<'a> {
    pub fn new(
        query: SparseVectorView<'_, u16, f32>,
        encoder: &'a CDotPacking8Centroid4BitEncoder,
    ) -> Self {
        let mut dense_lut = vec![0.0f32; encoder.dim * NUM_CENTROIDS];
        let centroids = encoder.quantizer.centroids();
        for (&c, &qv) in query.components().iter().zip(query.values().iter()) {
            let mapped_c = if let Some(mapping) = &encoder.component_mapping {
                mapping[c as usize]
            } else {
                c
            };
            let qbase = mapped_c as usize * NUM_CENTROIDS;
            let cbase = c as usize * NUM_CENTROIDS;
            for v in 0..NUM_CENTROIDS {
                dense_lut[qbase + v] = qv * centroids[cbase + v];
            }
        }
        Self { encoder, dense_lut }
    }

    #[inline]
    unsafe fn simd_compute_distance<'v>(&self, vector: PackedVectorView<'v, u64>) -> DotProduct {
        let view = unsafe { CDot4BitView::from_unchecked_slice(vector.data()) };
        let ref_indices = self.encoder.get_ref_slice(view.ref_id);

        let mut acc = Simd::<f32, N>::splat(0.0);
        let lut = self.dense_lut.as_slice();
        let stride = Simd::<usize, N>::splat(NUM_CENTROIDS);
        let mut total = 0.0f32;

        // --- 1. MAPPED PHASE ---
        let mut ref_indices_m = ref_indices;
        let mut it = gap_iter(view.n_m, view.selectors_m, view.payloads.as_ptr());
        let mut nib_m = NibbleReader::new(view.values_m);

        for _ in 0..(view.n_m / N) {
            let gaps = it.decode_lane();
            it.block_idx += 1;
            let pos_in_ref = simd_prefix_sum(gaps);
            let comps = unsafe {
                Simd::gather_select_unchecked(
                    ref_indices_m,
                    Mask::splat(true),
                    pos_in_ref.cast(),
                    Simd::splat(0),
                )
            };
            let codes = nib_m.next_block_codes();
            let idx = comps.cast::<usize>() * stride + codes.cast::<usize>();
            let g = unsafe {
                Simd::gather_select_unchecked(lut, Mask::splat(true), idx, Simd::splat(0.0))
            };
            acc += g;
            ref_indices_m = unsafe { ref_indices_m.get_unchecked(pos_in_ref[N - 1] as usize..) };
        }

        let rem_m = view.n_m % N;
        let residual_payload_ptr = if rem_m > 0 {
            let tail_start = it.payload_ptr;
            let b = block_bit_width(view.selectors_m, view.n_m / N);
            let gaps = it.decode_lane();
            let pos = simd_prefix_sum(gaps).to_array();
            let codes = nib_m.tail_codes(rem_m);
            for i in 0..rem_m {
                let abs_comp = *unsafe { ref_indices_m.get_unchecked(pos[i] as usize) };
                total += lut[abs_comp as usize * NUM_CENTROIDS + codes[i] as usize];
            }
            unsafe { tail_start.add((rem_m * b + 7) / 8) }
        } else {
            it.payload_ptr
        };

        // --- 2. RESIDUAL PHASE ---
        let mut rit = gap_iter(view.n_r, view.selectors_r, residual_payload_ptr);
        let mut nib_r = NibbleReader::new(view.values_r);
        let mut last_comp = 0u32;

        for _ in 0..(view.n_r / N) {
            let gaps = rit.decode_lane();
            rit.block_idx += 1;
            let comps = simd_prefix_sum(gaps) + Simd::splat(last_comp);
            last_comp = comps[N - 1];
            let codes = nib_r.next_block_codes();
            let idx = comps.cast::<usize>() * stride + codes.cast::<usize>();
            let g = unsafe {
                Simd::gather_select_unchecked(lut, Mask::splat(true), idx, Simd::splat(0.0))
            };
            acc += g;
        }

        let rem_r = view.n_r % N;
        if rem_r > 0 {
            let gaps = rit.decode_lane();
            let comps = (simd_prefix_sum(gaps) + Simd::splat(last_comp)).to_array();
            let codes = nib_r.tail_codes(rem_r);
            for i in 0..rem_r {
                total += lut[comps[i] as usize * NUM_CENTROIDS + codes[i] as usize];
            }
        }

        total += acc.reduce_sum();

        DotProduct(total)
    }
}

impl<'a, 'v> QueryEvaluator<PackedVectorView<'v, u64>>
    for CDotPacking8Centroid4BitQueryEvaluator<'a>
{
    type Distance = DotProduct;
    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> Self::Distance {
        unsafe { self.simd_compute_distance(vector) }
    }
}

impl SpaceUsage for CDotPacking8Centroid4BitEncoder {
    fn space_usage_bytes(&self) -> usize {
        let size_of_mapping = match &self.component_mapping {
            Some(component_mapping) => component_mapping.space_usage_bytes(),
            None => std::mem::size_of::<Option<Box<[u16]>>>(),
        };
        self.all_refs.len() * 2
            + size_of_mapping
            + self.dim.space_usage_bytes()
            + self.num_clusters.space_usage_bytes()
            + self.quantizer.space_usage_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::distances::Distance;
    use crate::core::vector::{PackedVectorView, SparseVectorView};
    use crate::vector_encoder::SparseDataEncoder;
    use crate::{
        DatasetGrowable, PlainSparseDatasetGrowable, PlainSparseQuantizer, QueryEvaluator,
        VectorEncoder,
    };

    /// Push enough vectors per component (with a spread of values across ~[0.2, 4.0]) that
    /// per-component k-means yields a full, well-separated 16-centroid codebook — so test
    /// inputs map to a *spread* of codes 0..15 (exercising both nibbles incl. high values),
    /// rather than collapsing to a 1- or 2-entry codebook.
    fn build_training_data(
        dim: usize,
        components: &[u16],
    ) -> PlainSparseDataset<u16, f32, SquaredEuclideanDistance> {
        let quantizer = PlainSparseQuantizer::<u16, f32, SquaredEuclideanDistance>::new(dim, dim);
        let mut growable = PlainSparseDatasetGrowable::new(quantizer);
        for k in 0..32usize {
            let vals: Vec<f32> = components
                .iter()
                .enumerate()
                .map(|(idx, _)| 0.2 + ((idx + k) % 20) as f32 * 0.2)
                .collect();
            growable.push(SparseVectorView::new(components, &vals));
        }
        growable.into()
    }

    fn quantized_values(
        encoder: &CDotPacking8Centroid4BitEncoder,
        components: &[u16],
        values: &[f32],
    ) -> Vec<f32> {
        components
            .iter()
            .zip(values.iter())
            .map(|(&c, &v)| {
                let encoded = encoder.quantizer.encode_value(c, v);
                encoder.quantizer.decode_value(c, encoded)
            })
            .collect()
    }

    fn dot_with_query(
        components: &[u16],
        values: &[f32],
        query_components: &[u16],
        query_values: &[f32],
    ) -> f32 {
        let mut sum = 0.0f32;
        let mut i = 0usize;
        let mut j = 0usize;
        while i < components.len() && j < query_components.len() {
            match components[i].cmp(&query_components[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    sum += values[i] * query_values[j];
                    i += 1;
                    j += 1;
                }
            }
        }
        sum
    }

    /// Encode `input`, score against `query`, and assert the SIMD score equals the
    /// brute-force dot over the encoder's own centroid-dequantized values. A bug in nibble
    /// packing/unpacking or the LUT-gather indexing would break this equality.
    fn assert_parity(
        encoder: &CDotPacking8Centroid4BitEncoder,
        comps: &[u16],
        in_vals: &[f32],
        query_comps: &[u16],
        query_vals: &[f32],
    ) {
        let input = SparseVectorView::new(comps, in_vals);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);

        let query = SparseVectorView::new(query_comps, query_vals);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let q_values = quantized_values(encoder, comps, in_vals);
        let expected = dot_with_query(comps, &q_values, query_comps, query_vals);

        let tol = 1e-4 * expected.abs().max(1.0);
        assert!(
            (dist.distance() - expected).abs() <= tol,
            "score {} != expected {} (n={})",
            dist.distance(),
            expected,
            comps.len()
        );
    }

    // Sizes chosen to exercise every bulk/tail boundary, including odd counts that leave
    // a dangling high nibble in the tail byte.
    const NS: [usize; 12] = [1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 23, 40];

    #[test]
    fn parity_mapped_only_various_n() {
        const DIM: usize = 4000;
        let all: Vec<u16> = (0u16..3000).collect();
        // Reference list covers every component, so all input components are "mapped".
        let refs: Vec<u16> = (0u16..3000).collect();
        let training = build_training_data(DIM, &all);
        let mut enc = CDotPacking8Centroid4BitEncoder::new_with_references(DIM, vec![refs], 4096);
        enc.train_with_params(&training, 0.0, 1.0, 10);

        for &n in &NS {
            let comps: Vec<u16> = (0..n as u16).map(|i| i * 3).collect();
            let in_vals: Vec<f32> = (0..n).map(|i| 0.5 + (i % 11) as f32 * 0.3).collect();
            let q_vals: Vec<f32> = (0..n).map(|i| 0.25 + (i % 7) as f32 * 0.2).collect();
            assert_parity(&enc, &comps, &in_vals, &comps, &q_vals);
        }
    }

    #[test]
    fn parity_residual_only_various_n() {
        const DIM: usize = 4000;
        let all: Vec<u16> = (0u16..3000).collect();
        let training = build_training_data(DIM, &all);
        // Empty reference list => every component falls into the residual tier.
        let mut enc = CDotPacking8Centroid4BitEncoder::new_with_references(DIM, vec![vec![]], 4096);
        enc.train_with_params(&training, 0.0, 1.0, 10);

        for &n in &NS {
            let comps: Vec<u16> = (0..n as u16).map(|i| i * 5).collect();
            let in_vals: Vec<f32> = (0..n).map(|i| 0.6 + (i % 9) as f32 * 0.35).collect();
            let q_vals: Vec<f32> = (0..n).map(|i| 0.2 + (i % 5) as f32 * 0.25).collect();
            assert_parity(&enc, &comps, &in_vals, &comps, &q_vals);
        }
    }

    #[test]
    fn parity_mixed_mapped_and_residual() {
        const DIM: usize = 4000;
        let all: Vec<u16> = (0u16..3000).collect();
        let training = build_training_data(DIM, &all);
        // Only even components are in the reference list: inputs split across both tiers.
        let refs: Vec<u16> = (0u16..3000).filter(|c| c % 2 == 0).collect();
        let mut enc = CDotPacking8Centroid4BitEncoder::new_with_references(DIM, vec![refs], 4096);
        enc.train_with_params(&training, 0.0, 1.0, 10);

        for &n in &NS {
            let comps: Vec<u16> = (0..n as u16).map(|i| i * 7).collect();
            let in_vals: Vec<f32> = (0..n).map(|i| 0.5 + (i % 13) as f32 * 0.28).collect();
            let q_vals: Vec<f32> = (0..n).map(|i| 0.3 + (i % 6) as f32 * 0.22).collect();
            assert_parity(&enc, &comps, &in_vals, &comps, &q_vals);
        }
    }

    #[test]
    fn parity_query_partial_overlap() {
        // Query overlaps only part of the document, exercising the gather/scoring path
        // with mismatched component sets (mapped + residual, bulk + tail).
        const DIM: usize = 200;
        let all: Vec<u16> = (0u16..100).collect();
        let training = build_training_data(DIM, &all);
        let refs = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 70, 87, 90]];
        let mut enc = CDotPacking8Centroid4BitEncoder::new_with_references(DIM, refs, 128);
        enc.train_with_params(&training, 0.0, 1.0, 10);

        let comps: Vec<u16> = vec![0, 4, 8, 24, 28, 36, 48, 53, 70, 90];
        let in_vals = [1.0, 3.0, 2.0, 3.5, 1.5, 2.0, 1.0, 2.0, 3.0, 2.5];
        let query_comps: Vec<u16> = vec![2, 4, 8, 24, 28, 70, 90];
        let query_vals = [0.5, 1.5, 2.5, 1.0, 2.0, 1.0, 2.0];
        assert_parity(&enc, &comps, &in_vals, &query_comps, &query_vals);
    }

    fn same_when_quantized(
        before_c: &[u16],
        before_v: &[f32],
        after: &SparseVectorView<u16, f32>,
        encoder: &CDotPacking8Centroid4BitEncoder,
    ) {
        assert_eq!(before_c, after.components());
        for (c, (v1, v2)) in before_c
            .iter()
            .zip(before_v.iter().zip(after.values().iter()))
        {
            let encoded = encoder.quantizer.encode_value(*c, *v1);
            let quantized = encoder.quantizer.decode_value(*c, encoded);
            assert!(
                (quantized - v2).abs() < 1e-5,
                "component {c}: {quantized} != {v2}"
            );
        }
    }

    #[test]
    fn decode_roundtrip_various_n() {
        const DIM: usize = 4000;
        let all: Vec<u16> = (0u16..3000).collect();
        let training = build_training_data(DIM, &all);
        // Mixed tiers so both mapped and residual decode paths are covered.
        let refs: Vec<u16> = (0u16..3000).filter(|c| c % 2 == 0).collect();
        let mut enc = CDotPacking8Centroid4BitEncoder::new_with_references(DIM, vec![refs], 4096);
        enc.train_with_params(&training, 0.0, 1.0, 10);

        for &n in &NS {
            let comps: Vec<u16> = (0..n as u16).map(|i| i * 7).collect();
            let in_vals: Vec<f32> = (0..n).map(|i| 0.5 + (i % 13) as f32 * 0.28).collect();
            let input = SparseVectorView::new(&comps, &in_vals);
            let mut buffer = Vec::new();
            enc.push_encoded(input, &mut buffer);
            let decoded = enc.decode_vector(PackedVectorView::new(&buffer));
            same_when_quantized(&comps, &in_vals, &decoded.as_view(), &enc);
        }
    }

    #[test]
    fn nibble_pack_roundtrip_unit() {
        // Direct check of the packing/unpacking helpers for a full block + odd tail.
        let codes: Vec<u8> = (0..11u8).map(|i| i % 16).collect(); // 8 bulk + 3 tail
        let mut packed = Vec::new();
        pack_nibbles(&mut packed, &codes);
        assert_eq!(packed.len(), nibble_bytes(codes.len())); // 4 + 2 = 6

        let mut reader = NibbleReader::new(&packed);
        let bulk = reader.next_block_u8();
        assert_eq!(&bulk[..], &codes[0..8]);
        let tail = reader.tail_codes(3);
        assert_eq!(&tail[0..3], &codes[8..11]);

        // And the SIMD-order integer expansion matches the same 8 bulk codes.
        let mut reader2 = NibbleReader::new(&packed);
        let bulk_simd = reader2.next_block_codes().to_array();
        assert_eq!(&bulk_simd[..], &codes[0..8]);
    }
}
