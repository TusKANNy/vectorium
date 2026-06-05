//! `cdotpacking8` component compression blended with 4-bit scalar value compression.
//!
//! This encoder keeps the **component** codec of [`crate::CDotPacking8ScalarU8Encoder`]
//! verbatim (two-tier reference-list scheme: mapped positions vs. residual gaps,
//! variable-bit gap packing, bisection component permutation, dense-query SIMD gather)
//! but stores the per-component **values** as 4-bit nibbles instead of full `u8` bytes,
//! roughly halving the value footprint.
//!
//! The value quantizer is the per-component uniform scalar quantizer used by
//! `packed_variable_bit_uniform_quantization_sparse_scalar` restricted to `nbits == 4`:
//! `quants[c] = max_value[c] / 15`, `code = clamp(v / quants[c], 0, 15)`, and scoring
//! computes `Σ code · (v_query · quants[c])`.
//!
//! This module is intentionally self-contained: it reuses only the *public* component-side
//! helpers from [`crate::encoders::dotpacking8::common`] (gap encoding/decoding, prefix
//! sums, the SIMD gather) and owns its entire nibble value path, so the optimized `u8`
//! path is untouched.
//!
//! ## Per-vector byte layout
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
    compute_safe_simd_padding, encode_blocks, fast_lane_gather, simd_prefix_sum, DotPacking8Iter,
};
use crate::encoders::dotpacking8::quantizer::DotPacking8Quantizer;
use crate::utils::train_sparse_scalar_quantizer_with_levels;
use crate::{Dataset, PlainSparseDataset, SpaceUsage, SquaredEuclideanDistance};
use bytemuck::{cast_slice, from_bytes};
use rusty_perm::{PermApply, PermFromSorting};
use std::simd::{prelude::*, StdFloat};

const N: usize = 8;
/// 4-bit codes span 15 quantization levels (`2^4 - 1`).
const NUM_LEVELS: f32 = 15.0;

// ---------------------------------------------------------------------------
// Nibble packing helpers (the only part that differs from the `u8` cencoder).
// ---------------------------------------------------------------------------

/// Number of bytes one value stream of `n` 4-bit codes occupies in the
/// half-split-bulk + sequential-tail layout.
#[inline(always)]
fn nibble_bytes(n: usize) -> usize {
    4 * (n / N) + (n % N + 1) / 2
}

/// Per-block gap bit width for `block_idx`, read directly from the selector stream.
/// Matches the packing in [`encode_blocks`] (4-bit selector per block, two per byte,
/// stored value is `b - 1`).
#[inline(always)]
fn block_bit_width(selectors: &[u8], block_idx: usize) -> usize {
    let sel = selectors[block_idx / 2];
    (((sel >> ((block_idx & 1) << 2)) & 0x0F) as usize) + 1
}

/// Append `codes` (each already in `0..=15`) to `out` in the half-split-bulk +
/// sequential-tail nibble layout. Appends exactly `nibble_bytes(codes.len())` bytes.
fn pack_nibbles(out: &mut Vec<u8>, codes: &[u8]) {
    let n = codes.len();
    let full = n / N;
    for blk in 0..full {
        let base = blk * N;
        for i in 0..4 {
            out.push((codes[base + i] & 0x0F) | ((codes[base + i + 4] & 0x0F) << 4));
        }
    }
    let rem = n - full * N;
    let base = full * N;
    let mut i = 0;
    while i < rem {
        let lo = codes[base + i] & 0x0F;
        let hi = if i + 1 < rem { codes[base + i + 1] & 0x0F } else { 0 };
        out.push(lo | (hi << 4));
        i += 2;
    }
}

/// Sequential cursor over one value stream's nibble-packed codes. Mirrors
/// [`pack_nibbles`]: full blocks are half-split (4 bytes -> 8 codes), the tail is
/// sequential nibble pairs.
struct NibbleReader<'a> {
    bytes: &'a [u8],
    off: usize,
}

impl<'a> NibbleReader<'a> {
    #[inline(always)]
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, off: 0 }
    }

    /// Expand the next full block (8 codes) to `Simd<f32, 8>` in natural lane order
    /// `[v0..v7]` (no cross-lane shuffle): `dup=[b0..b3,b0..b3] >> [0,0,0,0,4,4,4,4] & 0xF`.
    #[inline(always)]
    fn next_block_f32(&mut self) -> Simd<f32, N> {
        let b = unsafe { self.bytes.get_unchecked(self.off..self.off + 4) };
        self.off += 4;
        let dup = Simd::<u8, N>::from_array([b[0], b[1], b[2], b[3], b[0], b[1], b[2], b[3]]);
        let shifts = Simd::<u8, N>::from_array([0, 0, 0, 0, 4, 4, 4, 4]);
        ((dup >> shifts) & Simd::splat(0x0F)).cast::<f32>()
    }

    /// Expand the next full block (8 codes) to raw `u8` codes in natural order.
    #[inline(always)]
    fn next_block_u8(&mut self) -> [u8; N] {
        let b = unsafe { self.bytes.get_unchecked(self.off..self.off + 4) };
        self.off += 4;
        [
            b[0] & 0x0F,
            b[1] & 0x0F,
            b[2] & 0x0F,
            b[3] & 0x0F,
            b[0] >> 4,
            b[1] >> 4,
            b[2] >> 4,
            b[3] >> 4,
        ]
    }

    /// Read the `rem` (`< 8`) tail codes (sequential nibble pairs). Lanes `>= rem` are 0.
    #[inline(always)]
    fn tail_codes(&mut self, rem: usize) -> [u8; N] {
        let mut out = [0u8; N];
        for i in 0..rem {
            let byte = unsafe { *self.bytes.get_unchecked(self.off + i / 2) };
            out[i] = (byte >> ((i & 1) << 2)) & 0x0F;
        }
        self.off += (rem + 1) / 2;
        out
    }
}

/// Build a gap-only [`DotPacking8Iter`] over a component stream. The value cursor is
/// never used (we read nibbles from a [`NibbleReader`] instead), so `val_ptr` is null;
/// only the public `decode_lane()` / field access is exercised, which never touch it.
#[inline(always)]
fn gap_iter<'a>(n: usize, selectors: &'a [u8], payload_ptr: *const u8) -> DotPacking8Iter<'a> {
    DotPacking8Iter {
        bulk_blocks: n / N,
        block_idx: 0,
        n,
        selectors,
        payload_ptr,
        val_ptr: std::ptr::null(),
    }
}

// ---------------------------------------------------------------------------
// 4-bit scalar value quantizer.
// ---------------------------------------------------------------------------

/// Per-component uniform scalar quantizer producing 4-bit codes (`0..=15`).
/// `quants[c]` is the per-component step; `code = clamp(v / quants[c], 0, 15)`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Scalar4BitQuantizer {
    quants: Box<[f32]>,
}

impl SpaceUsage for Scalar4BitQuantizer {
    fn space_usage_bytes(&self) -> usize {
        self.quants.len() * std::mem::size_of::<f32>()
    }
}

impl Scalar4BitQuantizer {
    pub fn new(quants: Box<[f32]>) -> Self {
        Self { quants }
    }

    /// Fit per-component steps over the full `[min, max]` range using 15 levels
    /// (`lower_percentile = 0.0`, `upper_percentile = 1.0`), mirroring the
    /// `ScalarU8Quantizer` fit but for 4-bit codes.
    pub fn train(&mut self, training_data: &PlainSparseDataset<u16, f32, SquaredEuclideanDistance>) {
        self.quants =
            train_sparse_scalar_quantizer_with_levels(training_data, 0.0, 1.0, NUM_LEVELS)
                .into_boxed_slice();
    }
}

impl DotPacking8Quantizer for Scalar4BitQuantizer {
    type InputValue = f32;

    #[inline(always)]
    fn encode_value(&self, component: u16, value: f32) -> u8 {
        let q = self.quants[component as usize];
        if q > 0.0 {
            (value / q).clamp(0.0, NUM_LEVELS) as u8
        } else {
            0u8
        }
    }

    #[inline(always)]
    fn decode_value(&self, component: u16, output: u8) -> f32 {
        output as f32 * self.quants[component as usize]
    }

    #[inline(always)]
    fn query_value(&self, component: u16, input: f32) -> f32 {
        input * self.quants[component as usize]
    }
}

// ---------------------------------------------------------------------------
// Encoder.
// ---------------------------------------------------------------------------

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct CDotPacking8Scalar4BitEncoder {
    pub dim: usize,
    pub quantizer: Scalar4BitQuantizer,
    pub num_clusters: usize,
    pub all_refs: Vec<u16>,
    pub max_ref_size: usize,
    pub component_mapping: Option<Box<[u16]>>,
    pub inverse_component_mapping: Option<Box<[u16]>>,
}

impl PartialEq for CDotPacking8Scalar4BitEncoder {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
            && self.quantizer == other.quantizer
            && self.all_refs == other.all_refs
            && self.component_mapping == other.component_mapping
            && self.inverse_component_mapping == other.inverse_component_mapping
    }
}

impl CDotPacking8Scalar4BitEncoder {
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
            quantizer: Scalar4BitQuantizer::new(vec![0.0f32; input_dim].into_boxed_slice()),
            all_refs,
            max_ref_size,
            num_clusters,
            component_mapping: None,
            inverse_component_mapping: None,
        }
    }

    /// Train the component permutation (bisection) and the 4-bit value quantizer.
    /// Mirrors `CDotPacking8ScalarU8Encoder::train`.
    pub fn train(&mut self, training_data: &PlainSparseDataset<u16, f32, SquaredEuclideanDistance>) {
        const SAMPLE_RATE: usize = 20;
        let sample_size = if training_data.len() / SAMPLE_RATE < 50_000 {
            training_data.len()
        } else {
            training_data.len() / SAMPLE_RATE
        };
        self.train_components(training_data.iter().take(sample_size));
        self.quantizer.train(training_data);
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

impl SparseDataEncoder for CDotPacking8Scalar4BitEncoder {
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

impl PackedSparseVectorEncoder for CDotPacking8Scalar4BitEncoder {
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

impl VectorEncoder for CDotPacking8Scalar4BitEncoder {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u16, f32>;
    type QueryVector<'q> = SparseVectorView<'q, u16, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;
    type Evaluator<'e>
        = CDotPacking8Scalar4BitQueryEvaluator<'e>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        CDotPacking8Scalar4BitQueryEvaluator::new(query, self)
    }
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        CDotPacking8Scalar4BitQueryEvaluator::new(decoded.as_view(), self)
    }
    fn output_dim(&self) -> usize {
        self.dim
    }
    fn input_dim(&self) -> usize {
        self.dim
    }
}

pub struct CDotPacking8Scalar4BitQueryEvaluator<'a> {
    encoder: &'a CDotPacking8Scalar4BitEncoder,
    dense_query: Vec<f32>,
}

impl<'a> CDotPacking8Scalar4BitQueryEvaluator<'a> {
    pub fn new(
        query: SparseVectorView<'_, u16, f32>,
        encoder: &'a CDotPacking8Scalar4BitEncoder,
    ) -> Self {
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
            encoder,
            dense_query,
        }
    }

    #[inline]
    unsafe fn simd_compute_distance<'v>(&self, vector: PackedVectorView<'v, u64>) -> DotProduct {
        let view = unsafe { CDot4BitView::from_unchecked_slice(vector.data()) };
        let ref_indices = self.encoder.get_ref_slice(view.ref_id);

        let mut acc = Simd::<f32, N>::splat(0.0);
        let query = self.dense_query.as_slice();
        let mut total_unscaled = 0.0f32;

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
            let q_vals = fast_lane_gather(query, comps.cast());
            let vals = nib_m.next_block_f32();
            acc = q_vals.mul_add(vals, acc);
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
                total_unscaled += query[abs_comp as usize] * codes[i] as f32;
            }
            unsafe { tail_start.add((rem_m * b + 7) / 8) }
        } else {
            it.payload_ptr
        };

        // --- 2. RESIDUAL PHASE ---
        let mut query_r = query;
        let mut rit = gap_iter(view.n_r, view.selectors_r, residual_payload_ptr);
        let mut nib_r = NibbleReader::new(view.values_r);

        for _ in 0..(view.n_r / N) {
            let gaps = rit.decode_lane();
            rit.block_idx += 1;
            let comps = simd_prefix_sum(gaps);
            let q_vals = fast_lane_gather(query_r, comps);
            let vals = nib_r.next_block_f32();
            acc = q_vals.mul_add(vals, acc);
            query_r = unsafe { query_r.get_unchecked(comps[N - 1] as usize..) };
        }

        let rem_r = view.n_r % N;
        if rem_r > 0 {
            let gaps = rit.decode_lane();
            let pos = simd_prefix_sum(gaps).to_array();
            let codes = nib_r.tail_codes(rem_r);
            for i in 0..rem_r {
                total_unscaled += query_r[pos[i] as usize] * codes[i] as f32;
            }
        }

        total_unscaled += acc.reduce_sum();

        DotProduct(total_unscaled)
    }
}

impl<'a, 'v> QueryEvaluator<PackedVectorView<'v, u64>> for CDotPacking8Scalar4BitQueryEvaluator<'a> {
    type Distance = DotProduct;
    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> Self::Distance {
        unsafe { self.simd_compute_distance(vector) }
    }
}

impl SpaceUsage for CDotPacking8Scalar4BitEncoder {
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

    /// Training data scaled so that test inputs (values ~0.5..4) map to a *spread* of
    /// non-trivial 4-bit codes: per-component max is ~4, so `quants[c] ~= 4/15 ~= 0.27`.
    fn build_training_data(
        dim: usize,
        components: &[u16],
    ) -> PlainSparseDataset<u16, f32, SquaredEuclideanDistance> {
        let quantizer = PlainSparseQuantizer::<u16, f32, SquaredEuclideanDistance>::new(dim, dim);
        let mut growable = PlainSparseDatasetGrowable::new(quantizer);
        let values0: Vec<f32> = components
            .iter()
            .enumerate()
            .map(|(idx, _)| 1.0 + (idx % 7) as f32 * 0.45)
            .collect();
        let values1: Vec<f32> = components
            .iter()
            .enumerate()
            .map(|(idx, _)| 2.0 + (idx % 5) as f32 * 0.4)
            .collect();
        growable.push(SparseVectorView::new(components, &values0));
        if components.len() > 1 {
            growable.push(SparseVectorView::new(components, &values1));
        }
        growable.into()
    }

    fn quantized_values(
        encoder: &CDotPacking8Scalar4BitEncoder,
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
    /// brute-force dot over the encoder's own dequantized values. A bug in nibble
    /// packing/unpacking would corrupt the stored codes and break this equality.
    fn assert_parity(
        encoder: &CDotPacking8Scalar4BitEncoder,
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
        let mut enc =
            CDotPacking8Scalar4BitEncoder::new_with_references(DIM, vec![refs], 4096);
        enc.train(&training);

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
        let mut enc =
            CDotPacking8Scalar4BitEncoder::new_with_references(DIM, vec![vec![]], 4096);
        enc.train(&training);

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
        let mut enc =
            CDotPacking8Scalar4BitEncoder::new_with_references(DIM, vec![refs], 4096);
        enc.train(&training);

        for &n in &NS {
            // Interleave even (mapped) and odd (residual) components.
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
        let mut enc = CDotPacking8Scalar4BitEncoder::new_with_references(DIM, refs, 128);
        enc.train(&training);

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
        encoder: &CDotPacking8Scalar4BitEncoder,
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
        let mut enc =
            CDotPacking8Scalar4BitEncoder::new_with_references(DIM, vec![refs], 4096);
        enc.train(&training);

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
    }
}
