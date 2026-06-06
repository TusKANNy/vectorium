//! Shared building blocks for the 4-bit-value `dotpacking8` / `cdotpacking8` forks.
//!
//! The `*_scalar4bit` and `*_centroid4bit` encoders all keep a `dotpacking8`-family
//! component codec but store values as **4-bit nibbles**. This module owns the parts they
//! share so nothing is duplicated across the four encoder files:
//!
//! - the **nibble value layout** (`nibble_bytes`, `pack_nibbles`, [`NibbleReader`]) — every
//!   full block of 8 codes packs into 4 bytes as `byte[i] = code[i] | (code[i+4] << 4)`,
//!   with a trailing `rem < 8` codes stored as sequential nibble pairs;
//! - two **value quantizers** — [`Scalar4BitQuantizer`] (per-component uniform step, linear
//!   scoring) and [`Centroid4BitQuantizer`] (per-component 16-entry codebook, LUT scoring);
//! - small **gap-stream helpers** ([`gap_iter`], [`block_bit_width`]) that drive
//!   [`DotPacking8Iter`]'s public gap decoding while values are read from a [`NibbleReader`]
//!   rather than its byte value cursor.
//!
//! Each encoder file still owns its own component skeleton (single-stream for `dotpacking8`,
//! two-tier mapped/residual for `cdotpacking8`), view, query evaluator, and kernel.

use crate::encoders::dotpacking8::common::DotPacking8Iter;
use crate::encoders::dotpacking8::quantizer::DotPacking8Quantizer;
use crate::encoders::packed_centroid_based_quantization_sparse_scalar::PackedCentroidSparseQuantizer;
use crate::utils::train_sparse_scalar_quantizer_with_levels;
use crate::{PlainSparseDataset, SpaceUsage, SquaredEuclideanDistance};
use std::simd::prelude::*;

/// Gap block size of the `dotpacking8` codec.
pub(crate) const N: usize = 8;
/// Centroids per component for the codebook quantizer: `2^4 == 16`.
pub(crate) const NUM_CENTROIDS: usize = 16;
/// 4-bit scalar codes span 15 quantization levels (`2^4 - 1`).
const NUM_LEVELS: f32 = 15.0;

/// Default centroid k-means knobs for the parameterless `train(&td)` entry point of the
/// `*_centroid4bit` encoders (the one the dataset `From` macros call). Tuning callers use
/// `train_with_params` instead.
pub(crate) const DEFAULT_LOWER_PCT: f32 = 0.0;
pub(crate) const DEFAULT_UPPER_PCT: f32 = 1.0;
pub(crate) const DEFAULT_KMEANS_ITERS: usize = 10;

// ---------------------------------------------------------------------------
// Nibble value layout.
// ---------------------------------------------------------------------------

/// Number of bytes one value stream of `n` 4-bit codes occupies in the
/// half-split-bulk + sequential-tail layout.
#[inline(always)]
pub(crate) fn nibble_bytes(n: usize) -> usize {
    4 * (n / N) + (n % N + 1) / 2
}

/// Per-block gap bit width for `block_idx`, read directly from the selector stream.
/// Matches the packing in `encode_blocks` (4-bit selector per block, two per byte, stored
/// value is `b - 1`). Used by the two-tier `cdotpacking8` forks to advance the gap payload
/// pointer across the mapped→residual stream boundary.
#[inline(always)]
pub(crate) fn block_bit_width(selectors: &[u8], block_idx: usize) -> usize {
    let sel = selectors[block_idx / 2];
    (((sel >> ((block_idx & 1) << 2)) & 0x0F) as usize) + 1
}

/// Append `codes` (each already in `0..=15`) to `out` in the half-split-bulk +
/// sequential-tail nibble layout. Appends exactly `nibble_bytes(codes.len())` bytes.
pub(crate) fn pack_nibbles(out: &mut Vec<u8>, codes: &[u8]) {
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
pub(crate) struct NibbleReader<'a> {
    bytes: &'a [u8],
    off: usize,
}

impl<'a> NibbleReader<'a> {
    #[inline(always)]
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, off: 0 }
    }

    /// Expand the next full block (8 codes) to `Simd<f32, 8>` in natural lane order
    /// `[v0..v7]` (no cross-lane shuffle): `dup=[b0..b3,b0..b3] >> [0,0,0,0,4,4,4,4] & 0xF`.
    /// Used by the linear scalar kernels (`acc = q.mul_add(codes_f32, acc)`).
    #[inline(always)]
    pub(crate) fn next_block_f32(&mut self) -> Simd<f32, N> {
        let b = unsafe { self.bytes.get_unchecked(self.off..self.off + 4) };
        self.off += 4;
        let dup = Simd::<u8, N>::from_array([b[0], b[1], b[2], b[3], b[0], b[1], b[2], b[3]]);
        let shifts = Simd::<u8, N>::from_array([0, 0, 0, 0, 4, 4, 4, 4]);
        ((dup >> shifts) & Simd::splat(0x0F)).cast::<f32>()
    }

    /// Expand the next full block (8 codes) to `Simd<u8, 8>` in natural lane order, ready to
    /// be widened into LUT gather indices. Used by the centroid kernels.
    #[inline(always)]
    pub(crate) fn next_block_codes(&mut self) -> Simd<u8, N> {
        let b = unsafe { self.bytes.get_unchecked(self.off..self.off + 4) };
        self.off += 4;
        let dup = Simd::<u8, N>::from_array([b[0], b[1], b[2], b[3], b[0], b[1], b[2], b[3]]);
        let shifts = Simd::<u8, N>::from_array([0, 0, 0, 0, 4, 4, 4, 4]);
        (dup >> shifts) & Simd::splat(0x0F)
    }

    /// Expand the next full block (8 codes) to raw `u8` codes in natural order (decode path).
    #[inline(always)]
    pub(crate) fn next_block_u8(&mut self) -> [u8; N] {
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
    pub(crate) fn tail_codes(&mut self, rem: usize) -> [u8; N] {
        let mut out = [0u8; N];
        for i in 0..rem {
            let byte = unsafe { *self.bytes.get_unchecked(self.off + i / 2) };
            out[i] = (byte >> ((i & 1) << 2)) & 0x0F;
        }
        self.off += (rem + 1) / 2;
        out
    }
}

/// Build a gap-only [`DotPacking8Iter`] over a component stream. The value cursor is never
/// used (values are read from a [`NibbleReader`] instead), so `val_ptr` is null; only the
/// public `decode_lane()` / field access is exercised, which never touch it.
#[inline(always)]
pub(crate) fn gap_iter<'a>(
    n: usize,
    selectors: &'a [u8],
    payload_ptr: *const u8,
) -> DotPacking8Iter<'a> {
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
// 4-bit uniform scalar value quantizer (linear scoring).
// ---------------------------------------------------------------------------

/// Per-component uniform scalar quantizer producing 4-bit codes (`0..=15`).
/// `quants[c]` is the per-component step; `code = clamp(v / quants[c], 0, 15)`. Scoring is
/// linear in the code: `dequant(c, code)·q = code · (q·quants[c])`.
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

    /// Fit per-component steps over the full `[min, max]` range using 15 levels, mirroring
    /// the `ScalarU8Quantizer` fit but for 4-bit codes.
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
// 4-bit per-component centroid value quantizer (codebook / LUT scoring).
// ---------------------------------------------------------------------------

/// Per-component centroid quantizer producing 4-bit codes (`0..=15`). `centroids` is a flat
/// `dim·16` table; `centroids[c·16 + v]` is the v-th centroid for component `c`, with each
/// per-component row sorted ascending so [`Self::encode_value`] can binary-search.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Centroid4BitQuantizer {
    dim: usize,
    centroids: Box<[f32]>,
}

impl SpaceUsage for Centroid4BitQuantizer {
    fn space_usage_bytes(&self) -> usize {
        self.centroids.len() * std::mem::size_of::<f32>()
    }
}

impl Centroid4BitQuantizer {
    pub fn new(dim: usize, centroids: Box<[f32]>) -> Self {
        Self { dim, centroids }
    }

    #[inline]
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    /// Fit the per-component codebook by delegating to the existing, tested per-component
    /// k-means in [`PackedCentroidSparseQuantizer::train`] (percentile-clip + Lloyd), then
    /// copying out its `dim·16` codebook. No k-means is reimplemented here.
    pub fn train(
        training_data: &PlainSparseDataset<u16, f32, SquaredEuclideanDistance>,
        lower_percentile: f32,
        upper_percentile: f32,
        n_iterations: usize,
    ) -> Self {
        let inner = PackedCentroidSparseQuantizer::train(
            training_data,
            lower_percentile,
            upper_percentile,
            n_iterations,
        );
        let centroids: Box<[f32]> = inner.centroids().to_vec().into_boxed_slice();
        let dim = centroids.len() / NUM_CENTROIDS;
        Self { dim, centroids }
    }

    /// Nearest-centroid index for `value` in `component` (binary search over the ascending
    /// per-component row). Mirrors `PackedCentroidSparseQuantizer::quantize`.
    #[inline(always)]
    pub(crate) fn encode_value(&self, component: u16, value: f32) -> u8 {
        let base = component as usize * NUM_CENTROIDS;
        let table = &self.centroids[base..base + NUM_CENTROIDS];
        match table.binary_search_by(|c| c.partial_cmp(&value).unwrap()) {
            Ok(idx) => idx as u8,
            Err(idx) => {
                if idx == 0 {
                    0
                } else if idx >= NUM_CENTROIDS {
                    (NUM_CENTROIDS - 1) as u8
                } else {
                    let lo = table[idx - 1];
                    let hi = table[idx];
                    if (value - lo) <= (hi - value) {
                        (idx - 1) as u8
                    } else {
                        idx as u8
                    }
                }
            }
        }
    }

    #[inline(always)]
    pub(crate) fn decode_value(&self, component: u16, output: u8) -> f32 {
        self.centroids[component as usize * NUM_CENTROIDS + output as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nibble_pack_roundtrip_unit() {
        // Full block + odd tail. 11 codes -> 4 (bulk) + 2 (tail) = 6 bytes.
        let codes: Vec<u8> = (0..11u8).map(|i| i % 16).collect();
        let mut packed = Vec::new();
        pack_nibbles(&mut packed, &codes);
        assert_eq!(packed.len(), nibble_bytes(codes.len()));

        let mut reader = NibbleReader::new(&packed);
        assert_eq!(&reader.next_block_u8()[..], &codes[0..8]);
        let tail = reader.tail_codes(3);
        assert_eq!(&tail[0..3], &codes[8..11]);

        // The f32 and integer SIMD expansions agree with the same 8 bulk codes.
        let mut r2 = NibbleReader::new(&packed);
        let f = r2.next_block_f32().to_array();
        let expect_f: Vec<f32> = codes[0..8].iter().map(|&c| c as f32).collect();
        assert_eq!(&f[..], &expect_f[..]);

        let mut r3 = NibbleReader::new(&packed);
        assert_eq!(&r3.next_block_codes().to_array()[..], &codes[0..8]);
    }
}
