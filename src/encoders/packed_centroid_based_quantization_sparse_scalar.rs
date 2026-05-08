//! Bitpacked per-component centroid quantizer for sparse vectors at `nbits == 4`.
//!
//! Specialised sister of
//! [`CentroidSparseQuantizer`](super::centroid_quantization_sparse_scalar::CentroidSparseQuantizer)
//! (which is generic on `nbits` but stores 8-bit codes regardless) for the 4-bit
//! operating point: 16 centroids per component, codes packed as nibbles.
//!
//! ## Layout convention
//!
//! Byte-compatible with [`super::packed_variable_bit_uniform_quantization_sparse_scalar`]
//! at `nbits == 4`: the per-vector blob starts with a `u16` original length, followed by
//! the components, followed by the SIMD-friendly *interleaved* nibble layout (16 codes
//! per 8-byte block; codes 0..7 in the low nibbles, codes 8..15 in the high nibbles),
//! with a trailing pad of ≥ 8 zero bytes so the total length is a multiple of 8.
//!
//! What changes versus the uniform packed encoder is *how the codes are interpreted at
//! scoring time*. Uniform: `dequant(c, code) = code · quants[c]`. Centroid: a per-dim
//! codebook of 16 floats, `dequant(c, code) = centroids[c · 16 + code]`. The query
//! evaluator therefore precomputes a dense LUT
//! `lut[c · 16 + v] = q[c] · centroids[c · 16 + v]` and the inner kernel reduces to a
//! gather-and-sum (no widening, no FMA) over those LUT entries.
//!
//! Dense-LUT-only: dim must satisfy `dim < 2^20` (LUT is `dim · 16 · 4 B`). For larger
//! dims the unpacked [`CentroidSparseQuantizer`] sparse-merge path is the appropriate
//! fallback.

use std::marker::PhantomData;

use bytemuck::{cast_slice, try_cast_slice};
use half::f16;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::clustering::{KMeans, KMeansBuilder};
use crate::core::sealed;
use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::encoders::centroid_quantization_sparse_scalar::{
    KmeansIterStats, greedy_kmeans, print_kmeans_iter_summary,
};
use crate::encoders::dense_scalar::PlainDenseQuantizer;
use crate::{Dataset, PlainDenseDataset, PlainSparseDataset, SpaceUsage, SquaredEuclideanDistance};

/// Hardcoded bit width for this specialised encoder.
const NBITS: u8 = 4;
/// Centroids per component: `2^NBITS == 16`.
const NUM_CENTROIDS: usize = 1 << NBITS;
/// Largest input dim supported by the dense-LUT path: `dim < 2^20`. Beyond that
/// the LUT (`dim · 16 · 4 B`) becomes too large for the gather-friendly layout.
const MAX_DIM: usize = 1 << 20;

/// Bytes used by the packed-codes section for `n` codes in the interleaved
/// 4-bit layout (16 codes per 8-byte block, last partial block zero-padded).
#[inline]
const fn codes_byte_len(n: usize) -> usize {
    n.div_ceil(16) * 8
}

/// Append the codes section in the interleaved nbits=4 layout (mirrors
/// [`super::packed_variable_bit_uniform_quantization_sparse_scalar`]).
fn write_codes_nbits4_interleaved(out: &mut Vec<u8>, values: &[u8]) {
    let n = values.len();
    let full_blocks = n / 16;
    for b in 0..full_blocks {
        let base = b * 16;
        for i in 0..8 {
            let lo = values[base + i] & 0x0F;
            let hi = values[base + i + 8] & 0x0F;
            out.push(lo | (hi << 4));
        }
    }
    let m = n - full_blocks * 16;
    if m > 0 {
        let base = full_blocks * 16;
        for i in 0..8 {
            let lo = if i < m { values[base + i] & 0x0F } else { 0 };
            let hi = if i + 8 < m {
                values[base + i + 8] & 0x0F
            } else {
                0
            };
            out.push(lo | (hi << 4));
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Encoder
// ────────────────────────────────────────────────────────────────────────────

/// Bitpacked centroid-based sparse quantizer at 4 bits / 16 centroids per dim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedCentroidSparseQuantizer {
    dim: usize,
    /// Flat array of centroids: `centroids[c · 16 + v]` is the `v`-th centroid for
    /// dimension `c`. Per-dim slices are sorted ascending so `quantize` can binary-search.
    centroids: Box<[f32]>,
}

impl sealed::Sealed for PackedCentroidSparseQuantizer {}

impl PartialEq for PackedCentroidSparseQuantizer {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim && self.centroids == other.centroids
    }
}

impl PackedCentroidSparseQuantizer {
    /// Build a quantizer from an existing per-dim codebook.
    ///
    /// `centroids` must have length `dim · 16`; per-dim slices must be sorted ascending.
    pub fn from_centroids(dim: usize, centroids: Box<[f32]>) -> Self {
        assert!(
            dim < MAX_DIM,
            "PackedCentroidSparseQuantizer requires dim < 2^20 (got {dim}); \
             use CentroidSparseQuantizer for larger dims."
        );
        assert_eq!(
            centroids.len(),
            dim * NUM_CENTROIDS,
            "centroids length must be dim · 16 = {}",
            dim * NUM_CENTROIDS
        );
        Self { dim, centroids }
    }

    /// Train the quantizer by running per-component k-means (Lloyd) with uniform init,
    /// after percentile-clipping each component's value distribution.
    pub fn train(
        training_data: &PlainSparseDataset<u16, f32, SquaredEuclideanDistance>,
        lower_percentile: f32,
        upper_percentile: f32,
        n_iterations: usize,
    ) -> Self {
        let dim = training_data.output_dim();
        assert!(
            dim < MAX_DIM,
            "PackedCentroidSparseQuantizer requires dim < 2^20 (got {dim})."
        );
        assert!(
            (0.0..1.0).contains(&lower_percentile),
            "lower_percentile must be in [0.0, 1.0), got {lower_percentile}"
        );
        assert!(
            (0.0..=1.0).contains(&upper_percentile) && upper_percentile > lower_percentile,
            "upper_percentile must be in (lower_percentile, 1.0], got {upper_percentile}"
        );

        let mut per_component: Vec<Vec<f32>> = vec![Vec::new(); dim];
        for doc in training_data.iter() {
            for (&c, &v) in doc.components().iter().zip(doc.values()) {
                per_component[c as usize].push(v);
            }
        }

        let mut centroids = vec![0.0f32; dim * NUM_CENTROIDS];

        let per_dim_stats: Vec<Option<(usize, Vec<KmeansIterStats>)>> = per_component
            .par_iter_mut()
            .zip(centroids.par_chunks_mut(NUM_CENTROIDS))
            .map(|(vals, slot)| {
                if vals.is_empty() {
                    return None;
                }
                vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

                let min = if lower_percentile == 0.0 {
                    *vals.first().unwrap()
                } else {
                    let idx = ((vals.len() as f32) * lower_percentile) as usize;
                    vals[idx.min(vals.len() - 1)]
                };
                let max = if upper_percentile >= 1.0 {
                    *vals.last().unwrap()
                } else {
                    let idx = ((vals.len() as f32) * upper_percentile) as usize;
                    vals[idx.min(vals.len() - 1)]
                };

                for v in vals.iter_mut() {
                    *v = v.clamp(min, max);
                }

                let (dim_centroids, dim_stats) =
                    greedy_kmeans(vals, min, max, NUM_CENTROIDS, n_iterations);
                slot.copy_from_slice(&dim_centroids);
                Some((vals.len(), dim_stats))
            })
            .collect();

        print_kmeans_iter_summary(&per_dim_stats, n_iterations);

        Self {
            dim,
            centroids: centroids.into_boxed_slice(),
        }
    }

    #[inline]
    pub fn nbits(&self) -> u8 {
        NBITS
    }

    #[inline]
    pub fn num_centroids(&self) -> usize {
        NUM_CENTROIDS
    }

    #[inline]
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    #[inline]
    pub fn centroids_for_dim(&self, dim: usize) -> &[f32] {
        &self.centroids[dim * NUM_CENTROIDS..(dim + 1) * NUM_CENTROIDS]
    }

    /// Find the nearest centroid index for `value` in dimension `dim`. Mirrors
    /// `CentroidSparseQuantizer::quantize` (binary search + nearest-neighbour pick).
    #[inline]
    fn quantize(&self, dim: usize, value: f32) -> u8 {
        let table = self.centroids_for_dim(dim);
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

    #[inline]
    fn dequantize(&self, dim: usize, code: u8) -> f32 {
        self.centroids[dim * NUM_CENTROIDS + code as usize]
    }
}

impl SparseDataEncoder for PackedCentroidSparseQuantizer {
    type InputComponentType = u16;
    type InputValueType = f32;
    type OutputComponentType = u16;
    type OutputValueType = u8;

    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let view = unsafe { PackedCentroidScalar::from_unchecked_slice(encoded.data()) };
        let mut components: Vec<u16> = Vec::with_capacity(view.n as usize);
        let mut values: Vec<f32> = Vec::with_capacity(view.n as usize);
        for (c, code) in view.iter() {
            components.push(c);
            values.push(self.dequantize(c as usize, code));
        }
        SparseVectorOwned::new(components, values)
    }
}

impl PackedSparseVectorEncoder for PackedCentroidSparseQuantizer {
    type PackedDataType = u64;

    fn push_encoded<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, Self::InputComponentType, Self::InputValueType>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<Self::PackedDataType>,
    {
        let mut q_values: Vec<u8> = input
            .components()
            .iter()
            .zip(input.values())
            .map(|(&c, &v)| self.quantize(c as usize, v))
            .collect();
        let mut q_components: Vec<u16> = input.components().to_vec();

        let mut encoded_u8 = Vec::new();
        PackedCentroidScalar::push_vector(&mut encoded_u8, &mut q_components, &mut q_values);

        debug_assert!(
            encoded_u8.len() % std::mem::size_of::<u64>() == 0,
            "encoded_u8 length {} is not a multiple of 8",
            encoded_u8.len()
        );

        let words = encoded_u8
            .chunks_exact(std::mem::size_of::<u64>())
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()));
        output.extend(words);
    }
}

impl VectorEncoder for PackedCentroidSparseQuantizer {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u16, f32>;
    type QueryVector<'q> = SparseVectorView<'q, u16, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;

    type Evaluator<'e>
        = PackedCentroidSparseQueryEvaluator<'e>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        PackedCentroidSparseQueryEvaluator::new(query, self)
    }

    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        PackedCentroidSparseQueryEvaluator::new(decoded.as_view(), self)
    }

    #[inline]
    fn output_dim(&self) -> usize {
        self.dim
    }

    #[inline]
    fn input_dim(&self) -> usize {
        self.dim
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Query evaluator
// ────────────────────────────────────────────────────────────────────────────

/// Holds the dense LUT `lut[c · 16 + v] = q[c] · centroids[c · 16 + v]`. Only rows
/// for query components are filled; the rest stay at zero, so the kernel's gather
/// over absent doc components correctly contributes 0.
#[derive(Debug, Clone)]
pub struct PackedCentroidSparseQueryEvaluator<'e> {
    dense_lut: Vec<f32>,
    _phantom: PhantomData<&'e ()>,
}

impl<'e> PackedCentroidSparseQueryEvaluator<'e> {
    pub fn new(
        query: SparseVectorView<'_, u16, f32>,
        quantizer: &'e PackedCentroidSparseQuantizer,
    ) -> Self {
        let max_c = query.components().iter().copied().max().unwrap_or(0) as usize;
        assert!(
            max_c < quantizer.input_dim(),
            "Query component {max_c} exceeds quantizer dim {}",
            quantizer.input_dim()
        );
        assert_eq!(
            query.components().len(),
            query.values().len(),
            "Query components/values length mismatch."
        );

        let lut_size = quantizer.dim * NUM_CENTROIDS;
        let mut lut = vec![0.0f32; lut_size];
        for (&c, &qv) in query.components().iter().zip(query.values()) {
            let base = (c as usize) * NUM_CENTROIDS;
            for v in 0..NUM_CENTROIDS {
                lut[base + v] = qv * quantizer.centroids[base + v];
            }
        }

        Self {
            dense_lut: lut,
            _phantom: PhantomData,
        }
    }
}

impl<'e, 'v> QueryEvaluator<PackedVectorView<'v, u64>> for PackedCentroidSparseQueryEvaluator<'e> {
    type Distance = DotProduct;

    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> DotProduct {
        let view = unsafe { PackedCentroidScalar::from_unchecked_slice(vector.data()) };
        DotProduct::from(view.dot_product(&self.dense_lut))
    }
}

impl SpaceUsage for PackedCentroidSparseQuantizer {
    fn space_usage_bytes(&self) -> usize {
        self.dim.space_usage_bytes() + self.centroids.space_usage_bytes()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Per-vector layout helper
// ────────────────────────────────────────────────────────────────────────────

/// Typed view onto a single packed-vector blob. Encapsulates the shared 4-bit
/// interleaved layout — the only difference vs the uniform packed analogue is the
/// scoring kernel used by `dot_product`.
#[derive(Clone)]
struct PackedCentroidScalar<'a> {
    n: u16,
    components: &'a [u16],
    /// `codes_byte_len(n) + 8` bytes (the +8 is the SIMD-safety pad written by
    /// `push_vector`, exposed here so the bulk loop can do unconditional 8-byte loads).
    codes: &'a [u8],
}

impl<'a> PackedCentroidScalar<'a> {
    /// Parse a packed `&[u64]` produced by `push_vector` into typed sub-slices.
    ///
    /// # Safety
    /// Caller guarantees `slice` was produced by `push_vector`.
    unsafe fn from_unchecked_slice(slice: &'a [u64]) -> Self {
        let bytes: &[u8] = unsafe { try_cast_slice::<u64, u8>(slice).unwrap_unchecked() };

        let n = u16::from_le_bytes([bytes[0], bytes[1]]);
        let n_usize = n as usize;

        let comp_start = std::mem::size_of::<u16>();
        let comp_end = comp_start + n_usize * std::mem::size_of::<u16>();
        let codes_start = comp_end;
        let codes_end = codes_start + codes_byte_len(n_usize) + 8;

        let components: &[u16] = cast_slice(unsafe { bytes.get_unchecked(comp_start..comp_end) });
        let codes = unsafe { bytes.get_unchecked(codes_start..codes_end) };

        Self {
            n,
            components,
            codes,
        }
    }

    pub fn push_vector(out: &mut Vec<u8>, components: &mut [u16], values: &mut [u8]) {
        assert_eq!(
            components.len(),
            values.len(),
            "components/values length mismatch"
        );
        assert!(
            components.len() < u16::MAX as usize,
            "vector longer than u16::MAX is not supported (got {})",
            components.len()
        );
        debug_assert!(
            components.windows(2).all(|w| w[0] <= w[1]),
            "components must be sorted ascending"
        );

        let n = components.len() as u16;

        out.extend_from_slice(&n.to_le_bytes());
        for &c in components.iter() {
            out.extend_from_slice(&c.to_le_bytes());
        }
        write_codes_nbits4_interleaved(out, values);

        // Pad so (a) ≥ 8 trailing bytes past codes_end (the kernel does an
        // unconditional u64 load on the last bulk chunk), and (b) total length is a
        // multiple of 8 (required by `PackedDataType = u64`).
        let target_len = (out.len() + 8 + 7) & !7usize;
        out.resize(target_len, 0);
    }

    /// Sequential decode over `(component, code)`. Address arithmetic matches the
    /// uniform packed encoder's nbits=4 path: code `i` lives in block `i / 16` at
    /// byte offset `(i & 7)`, shifted by 0 for low half / 4 for high half.
    pub fn iter(self) -> impl Iterator<Item = (u16, u8)> + 'a {
        let components = self.components;
        let codes = self.codes;
        gen move {
            for (i, &c) in components.iter().enumerate() {
                let block = i / 16;
                let byte_off = block * 8 + (i & 7);
                let shift = ((i & 8) >> 1) as u8;
                let v = (codes[byte_off] >> shift) & 0x0F;
                yield (c, v);
            }
        }
    }

    /// Score against the dense LUT: `Σ lut[c · 16 + code]`.
    #[inline]
    pub fn dot_product(&self, lut: &[f32]) -> f32 {
        kernels::dot_product_centroid_nbits4(self.components, self.codes, lut)
    }

    /// Same as `dot_product`, but the LUT is f16-compressed and stored as adjacent
    /// pairs in `[u32; dim · 8]`: `lut_pairs[c · 8 + p] = (lo as u32) | ((hi as u32) << 16)`
    /// where `lo`/`hi` are the f16 bits at codes `2p` / `2p+1` for component `c`. The
    /// kernel does an aligned 32-bit gather of pairs and extracts the right f16 per lane.
    #[inline]
    pub fn dot_product_f16_lut(&self, lut_pairs: &[u32]) -> f32 {
        kernels::dot_product_centroid_nbits4_f16_lut(self.components, self.codes, lut_pairs)
    }

    /// Score using the clustered codebook: `Σ q_dense[c] · shared_centroid[cluster_of[c] · 16 + code]`.
    /// Three indirect gathers per element instead of one (see kernel docs).
    #[inline]
    pub fn dot_product_clustered(
        &self,
        q_dense: &[f32],
        cluster_of: &[u32],
        shared_centroid: &[f32],
    ) -> f32 {
        kernels::dot_product_centroid_clustered_nbits4(
            self.components,
            self.codes,
            q_dense,
            cluster_of,
            shared_centroid,
        )
    }

}

// ────────────────────────────────────────────────────────────────────────────
// Specialised dot-product kernel
// ────────────────────────────────────────────────────────────────────────────
//
// Safety contract:
// - `components.len() == n` (logical number of codes).
// - `codes.len() >= codes_byte_len(n) + 8` (encoder pad + the +8 slack added in
//   `from_unchecked_slice`).
// - `lut.len() == dim · 16` and `max(components) < dim` (enforced by the evaluator).
mod kernels {
    use std::simd::num::{SimdFloat, SimdUint};
    use std::simd::{Mask, Simd};

    use super::NUM_CENTROIDS;

    /// 4-bit centroid kernel. 16 codes per 8-byte block, two independent 8-wide
    /// SIMD pipelines (so the dependency chain is `chunks` rather than `2 · chunks`).
    ///
    /// Per chunk:
    /// ```text
    ///   load 8 bytes                        → Simd<u8, 8>
    ///   nibble extract (& 0x0F / >> 4)      → 2 × Simd<u8, 8>
    ///   widen to usize                      → 2 × Simd<usize, 8>
    ///   idx = comp · 16 + code              → 2 × Simd<usize, 8>
    ///   gather lut[idx]                     → 2 × Simd<f32, 8>   (vgatherqps)
    ///   acc += gathered                     → 2 × Simd<f32, 8>
    /// ```
    /// vs the uniform packed kernel which additionally widens `u8 → f32` and runs an
    /// FMA against a per-component query value. Here `q[c]` is folded into the LUT,
    /// so the kernel collapses to gather + add.
    #[inline]
    pub(super) fn dot_product_centroid_nbits4(
        components: &[u16],
        codes: &[u8],
        lut: &[f32],
    ) -> f32 {
        let n = components.len();
        let chunks = n / 16;
        let tail_start = chunks * 16;

        let mut acc_lo = Simd::<f32, 8>::splat(0.0);
        let mut acc_hi = Simd::<f32, 8>::splat(0.0);

        let stride = Simd::<usize, 8>::splat(NUM_CENTROIDS);

        for chunk_i in 0..chunks {
            let byte_off = chunk_i * 8;
            let buf_bytes: [u8; 8] = unsafe {
                codes
                    .get_unchecked(byte_off..byte_off + 8)
                    .try_into()
                    .unwrap_unchecked()
            };
            let buf: Simd<u8, 8> = Simd::from_array(buf_bytes);

            // Two-op nibble extraction lands in natural code order:
            // codes_lo = codes 0..7, codes_hi = codes 8..15.
            let codes_lo: Simd<u8, 8> = buf & Simd::splat(0x0F);
            let codes_hi: Simd<u8, 8> = (buf >> Simd::splat(4)) & Simd::splat(0x0F);

            let base = chunk_i * 16;
            let comp_lo: Simd<u16, 8> =
                Simd::from_slice(unsafe { components.get_unchecked(base..base + 8) });
            let comp_hi: Simd<u16, 8> =
                Simd::from_slice(unsafe { components.get_unchecked(base + 8..base + 16) });

            // idx = comp * 16 + code, computed in usize so comp · 16 can't overflow.
            let idx_lo = comp_lo.cast::<usize>() * stride + codes_lo.cast::<usize>();
            let idx_hi = comp_hi.cast::<usize>() * stride + codes_hi.cast::<usize>();

            let g_lo: Simd<f32, 8> = unsafe {
                Simd::gather_select_unchecked(lut, Mask::splat(true), idx_lo, Simd::splat(0.0))
            };
            let g_hi: Simd<f32, 8> = unsafe {
                Simd::gather_select_unchecked(lut, Mask::splat(true), idx_hi, Simd::splat(0.0))
            };

            acc_lo = acc_lo + g_lo;
            acc_hi = acc_hi + g_hi;
        }

        let bulk_sum = (acc_lo + acc_hi).reduce_sum();

        // Tail: < 16 codes. Same address arithmetic as `iter`.
        let mut tail_acc = 0.0f32;
        for i in tail_start..n {
            let block = i / 16;
            let byte_off = block * 8 + (i & 7);
            let shift = ((i & 8) >> 1) as u8;
            let code = unsafe { (codes.get_unchecked(byte_off) >> shift) & 0x0F };
            let c = unsafe { *components.get_unchecked(i) } as usize;
            let v = unsafe { *lut.get_unchecked(c * NUM_CENTROIDS + code as usize) };
            tail_acc = tail_acc.algebraic_add(v);
        }

        bulk_sum.algebraic_add(tail_acc)
    }

    /// Variant of [`dot_product_centroid_nbits4`] that reads from an f16-compressed
    /// LUT laid out as `[u32; dim · 8]` (each u32 packs two adjacent f16 entries:
    /// low halfword = even code, high halfword = odd code). The bulk loop does an
    /// aligned 32-bit pair gather, extracts the right f16 per lane via a
    /// parity-derived shift, and converts 8 f16 → 8 f32 with `_mm256_cvtph_ps`.
    ///
    /// Built with `-C target-cpu=native` on Skylake-X (AVX-512 + f16c). On other
    /// targets the intrinsic call requires the f16c CPU feature.
    #[inline]
    pub(super) fn dot_product_centroid_nbits4_f16_lut(
        components: &[u16],
        codes: &[u8],
        lut_pairs: &[u32],
    ) -> f32 {
        use core::arch::x86_64::{__m128i, __m256, _mm256_cvtph_ps};

        let n = components.len();
        let chunks = n / 16;
        let tail_start = chunks * 16;

        let mut acc_lo = Simd::<f32, 8>::splat(0.0);
        let mut acc_hi = Simd::<f32, 8>::splat(0.0);

        // Each component contributes NUM_CENTROIDS / 2 = 8 pairs in the LUT.
        let stride_pairs = Simd::<usize, 8>::splat(NUM_CENTROIDS / 2);
        let lo_mask = Simd::<u32, 8>::splat(0xFFFF);

        for chunk_i in 0..chunks {
            let byte_off = chunk_i * 8;
            let buf_bytes: [u8; 8] = unsafe {
                codes
                    .get_unchecked(byte_off..byte_off + 8)
                    .try_into()
                    .unwrap_unchecked()
            };
            let buf: Simd<u8, 8> = Simd::from_array(buf_bytes);

            let codes_lo: Simd<u8, 8> = buf & Simd::splat(0x0F);
            let codes_hi: Simd<u8, 8> = (buf >> Simd::splat(4)) & Simd::splat(0x0F);

            let base = chunk_i * 16;
            let comp_lo: Simd<u16, 8> =
                Simd::from_slice(unsafe { components.get_unchecked(base..base + 8) });
            let comp_hi: Simd<u16, 8> =
                Simd::from_slice(unsafe { components.get_unchecked(base + 8..base + 16) });

            // pair_idx = comp * 8 + (code >> 1)
            let codes_lo_us = codes_lo.cast::<usize>();
            let codes_hi_us = codes_hi.cast::<usize>();
            let pair_idx_lo =
                comp_lo.cast::<usize>() * stride_pairs + (codes_lo_us >> Simd::splat(1));
            let pair_idx_hi =
                comp_hi.cast::<usize>() * stride_pairs + (codes_hi_us >> Simd::splat(1));

            // 32-bit aligned gather: 1 µop, fully L2-fitting (917 KiB at dim ≈ 28K).
            let pair_lo: Simd<u32, 8> = unsafe {
                Simd::gather_select_unchecked(
                    lut_pairs,
                    Mask::splat(true),
                    pair_idx_lo,
                    Simd::splat(0),
                )
            };
            let pair_hi: Simd<u32, 8> = unsafe {
                Simd::gather_select_unchecked(
                    lut_pairs,
                    Mask::splat(true),
                    pair_idx_hi,
                    Simd::splat(0),
                )
            };

            // Per-lane extract: shift by (code & 1) * 16 then mask. Result is f16
            // bits in low 16 bits per u32 lane.
            let shift_lo = (codes_lo.cast::<u32>() & Simd::splat(1)) << Simd::splat(4);
            let shift_hi = (codes_hi.cast::<u32>() & Simd::splat(1)) << Simd::splat(4);
            let bits32_lo = (pair_lo >> shift_lo) & lo_mask;
            let bits32_hi = (pair_hi >> shift_hi) & lo_mask;

            // Truncate to Simd<u16, 8> = 128 bits (= __m128i layout).
            let bits16_lo: Simd<u16, 8> = bits32_lo.cast::<u16>();
            let bits16_hi: Simd<u16, 8> = bits32_hi.cast::<u16>();

            // f16 → f32 via vcvtph2ps. Same lane layout in / out.
            let f_lo: Simd<f32, 8> = unsafe {
                let m: __m128i = std::mem::transmute(bits16_lo);
                let f: __m256 = _mm256_cvtph_ps(m);
                std::mem::transmute(f)
            };
            let f_hi: Simd<f32, 8> = unsafe {
                let m: __m128i = std::mem::transmute(bits16_hi);
                let f: __m256 = _mm256_cvtph_ps(m);
                std::mem::transmute(f)
            };

            acc_lo = acc_lo + f_lo;
            acc_hi = acc_hi + f_hi;
        }

        let bulk_sum = (acc_lo + acc_hi).reduce_sum();

        // Tail: < 16 codes. Same address arithmetic, scalar f16 → f32.
        let mut tail_acc = 0.0f32;
        for i in tail_start..n {
            let block = i / 16;
            let byte_off = block * 8 + (i & 7);
            let shift = ((i & 8) >> 1) as u8;
            let code = unsafe { (codes.get_unchecked(byte_off) >> shift) & 0x0F };
            let c = unsafe { *components.get_unchecked(i) } as usize;
            let pair_idx = c * (NUM_CENTROIDS / 2) + (code as usize >> 1);
            let pair = unsafe { *lut_pairs.get_unchecked(pair_idx) };
            let bits = ((pair >> ((code as u32 & 1) * 16)) & 0xFFFF) as u16;
            let v = super::f16::from_bits(bits).to_f32();
            tail_acc = tail_acc.algebraic_add(v);
        }

        bulk_sum.algebraic_add(tail_acc)
    }

    /// 4-bit clustered-centroid kernel. Doc bytes are unchanged (4-bit interleaved
    /// codes); the LUT is replaced by three small per-query / shared tables that
    /// all fit in L2:
    ///   - `q_dense[c]`        : densified query weights (`dim · 4 B`).
    ///   - `cluster_of[c]`     : component → cluster id (`dim · 4 B`, u32 — AVX-512
    ///                           lacks a 16-bit gather, so u32 keeps the gather
    ///                           single-µop at the cost of ~57 KiB extra).
    ///   - `shared_centroid[k · 16 + e]` : codebook (`K · 16 · 4 B` ≈ 64 KiB at K=1024).
    ///
    /// Per chunk: three indirect gathers per side (q[c], cluster_of[c],
    /// shared_centroid[cluster*16+code]) plus an FMA. Two independent 8-wide
    /// pipelines as in the f32-LUT path.
    #[inline]
    pub(super) fn dot_product_centroid_clustered_nbits4(
        components: &[u16],
        codes: &[u8],
        q_dense: &[f32],
        cluster_of: &[u32],
        shared_centroid: &[f32],
    ) -> f32 {
        let n = components.len();
        let chunks = n / 16;
        let tail_start = chunks * 16;

        let mut acc_lo = Simd::<f32, 8>::splat(0.0);
        let mut acc_hi = Simd::<f32, 8>::splat(0.0);

        let stride = Simd::<usize, 8>::splat(NUM_CENTROIDS);

        for chunk_i in 0..chunks {
            let byte_off = chunk_i * 8;
            let buf_bytes: [u8; 8] = unsafe {
                codes
                    .get_unchecked(byte_off..byte_off + 8)
                    .try_into()
                    .unwrap_unchecked()
            };
            let buf: Simd<u8, 8> = Simd::from_array(buf_bytes);

            let codes_lo: Simd<u8, 8> = buf & Simd::splat(0x0F);
            let codes_hi: Simd<u8, 8> = (buf >> Simd::splat(4)) & Simd::splat(0x0F);

            let base = chunk_i * 16;
            let comp_lo: Simd<u16, 8> =
                Simd::from_slice(unsafe { components.get_unchecked(base..base + 8) });
            let comp_hi: Simd<u16, 8> =
                Simd::from_slice(unsafe { components.get_unchecked(base + 8..base + 16) });

            let comp_lo_us = comp_lo.cast::<usize>();
            let comp_hi_us = comp_hi.cast::<usize>();

            // Gather q[c] — vpgatherqps over the densified query.
            let q_lo: Simd<f32, 8> = unsafe {
                Simd::gather_select_unchecked(
                    q_dense,
                    Mask::splat(true),
                    comp_lo_us,
                    Simd::splat(0.0),
                )
            };
            let q_hi: Simd<f32, 8> = unsafe {
                Simd::gather_select_unchecked(
                    q_dense,
                    Mask::splat(true),
                    comp_hi_us,
                    Simd::splat(0.0),
                )
            };

            // Gather cluster_of[c] — vpgatherqd. u32 ids (see fn doc).
            let k_lo: Simd<u32, 8> = unsafe {
                Simd::gather_select_unchecked(
                    cluster_of,
                    Mask::splat(true),
                    comp_lo_us,
                    Simd::splat(0),
                )
            };
            let k_hi: Simd<u32, 8> = unsafe {
                Simd::gather_select_unchecked(
                    cluster_of,
                    Mask::splat(true),
                    comp_hi_us,
                    Simd::splat(0),
                )
            };

            // idx = cluster · 16 + code, computed in usize.
            let idx_lo = k_lo.cast::<usize>() * stride + codes_lo.cast::<usize>();
            let idx_hi = k_hi.cast::<usize>() * stride + codes_hi.cast::<usize>();

            // Gather shared_centroid[idx] — vpgatherqps into the 64 KiB codebook.
            let c_lo: Simd<f32, 8> = unsafe {
                Simd::gather_select_unchecked(
                    shared_centroid,
                    Mask::splat(true),
                    idx_lo,
                    Simd::splat(0.0),
                )
            };
            let c_hi: Simd<f32, 8> = unsafe {
                Simd::gather_select_unchecked(
                    shared_centroid,
                    Mask::splat(true),
                    idx_hi,
                    Simd::splat(0.0),
                )
            };

            // FMA: acc += q · c. Compiler emits vfmadd231ps under target-cpu=native.
            acc_lo = acc_lo + q_lo * c_lo;
            acc_hi = acc_hi + q_hi * c_hi;
        }

        let bulk_sum = (acc_lo + acc_hi).reduce_sum();

        // Tail: < 16 codes. Same address arithmetic as iter().
        let mut tail_acc = 0.0f32;
        for i in tail_start..n {
            let block = i / 16;
            let byte_off = block * 8 + (i & 7);
            let shift = ((i & 8) >> 1) as u8;
            let code = unsafe { (codes.get_unchecked(byte_off) >> shift) & 0x0F };
            let c = unsafe { *components.get_unchecked(i) } as usize;
            let q = unsafe { *q_dense.get_unchecked(c) };
            let k = unsafe { *cluster_of.get_unchecked(c) } as usize;
            let v = unsafe { *shared_centroid.get_unchecked(k * NUM_CENTROIDS + code as usize) };
            tail_acc = tail_acc.algebraic_add(q * v);
        }

        bulk_sum.algebraic_add(tail_acc)
    }

}

// ────────────────────────────────────────────────────────────────────────────
// f16-LUT variant: query evaluator + wrapper quantizer
// ────────────────────────────────────────────────────────────────────────────
//
// Encoding is identical to the f32-LUT path (same packed bytes per doc); only
// the query-side LUT and scoring kernel change. We expose this as a thin wrapper
// quantizer so a `PackedSparseDataset<PackedCentroidSparseQuantizerF16Lut>` can
// be A/B-benchmarked against the f32-LUT path against the same encoded blobs.

/// Holds the f16-compressed LUT as packed adjacent pairs:
/// `dense_lut_pairs[c · 8 + p] = (lo_bits as u32) | ((hi_bits as u32) << 16)`,
/// where `lo`/`hi` are the f16 bits of `q[c] · centroids[c · 16 + 2p]` and
/// `q[c] · centroids[c · 16 + 2p + 1]`. Only rows for query components are filled.
#[derive(Debug, Clone)]
pub struct PackedCentroidSparseQueryEvaluatorF16Lut<'e> {
    dense_lut_pairs: Vec<u32>,
    _phantom: PhantomData<&'e ()>,
}

impl<'e> PackedCentroidSparseQueryEvaluatorF16Lut<'e> {
    pub fn new(
        query: SparseVectorView<'_, u16, f32>,
        quantizer: &'e PackedCentroidSparseQuantizer,
    ) -> Self {
        let max_c = query.components().iter().copied().max().unwrap_or(0) as usize;
        assert!(
            max_c < quantizer.input_dim(),
            "Query component {max_c} exceeds quantizer dim {}",
            quantizer.input_dim()
        );
        assert_eq!(
            query.components().len(),
            query.values().len(),
            "Query components/values length mismatch."
        );

        let pairs_per_dim = NUM_CENTROIDS / 2;
        let lut_size = quantizer.dim * pairs_per_dim;
        let mut lut = vec![0u32; lut_size];
        for (&c, &qv) in query.components().iter().zip(query.values()) {
            let pair_base = (c as usize) * pairs_per_dim;
            let cent_base = (c as usize) * NUM_CENTROIDS;
            for p in 0..pairs_per_dim {
                let lo = f16::from_f32(qv * quantizer.centroids[cent_base + 2 * p]).to_bits()
                    as u32;
                let hi = f16::from_f32(qv * quantizer.centroids[cent_base + 2 * p + 1])
                    .to_bits() as u32;
                lut[pair_base + p] = lo | (hi << 16);
            }
        }

        Self {
            dense_lut_pairs: lut,
            _phantom: PhantomData,
        }
    }
}

impl<'e, 'v> QueryEvaluator<PackedVectorView<'v, u64>>
    for PackedCentroidSparseQueryEvaluatorF16Lut<'e>
{
    type Distance = DotProduct;

    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> DotProduct {
        let view = unsafe { PackedCentroidScalar::from_unchecked_slice(vector.data()) };
        DotProduct::from(view.dot_product_f16_lut(&self.dense_lut_pairs))
    }
}

/// Thin wrapper around [`PackedCentroidSparseQuantizer`] that swaps in the f16-LUT
/// query evaluator at search time. Doc-side encoding is delegated unchanged.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedCentroidSparseQuantizerF16Lut {
    inner: PackedCentroidSparseQuantizer,
}

impl sealed::Sealed for PackedCentroidSparseQuantizerF16Lut {}

impl PartialEq for PackedCentroidSparseQuantizerF16Lut {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl PackedCentroidSparseQuantizerF16Lut {
    pub fn from_inner(inner: PackedCentroidSparseQuantizer) -> Self {
        Self { inner }
    }

    pub fn train(
        training_data: &PlainSparseDataset<u16, f32, SquaredEuclideanDistance>,
        lower_percentile: f32,
        upper_percentile: f32,
        n_iterations: usize,
    ) -> Self {
        Self {
            inner: PackedCentroidSparseQuantizer::train(
                training_data,
                lower_percentile,
                upper_percentile,
                n_iterations,
            ),
        }
    }

    #[inline]
    pub fn inner(&self) -> &PackedCentroidSparseQuantizer {
        &self.inner
    }
}

impl SparseDataEncoder for PackedCentroidSparseQuantizerF16Lut {
    type InputComponentType = u16;
    type InputValueType = f32;
    type OutputComponentType = u16;
    type OutputValueType = u8;

    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        self.inner.decode_vector(encoded)
    }
}

impl PackedSparseVectorEncoder for PackedCentroidSparseQuantizerF16Lut {
    type PackedDataType = u64;

    fn push_encoded<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, Self::InputComponentType, Self::InputValueType>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<Self::PackedDataType>,
    {
        self.inner.push_encoded(input, output);
    }
}

impl VectorEncoder for PackedCentroidSparseQuantizerF16Lut {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u16, f32>;
    type QueryVector<'q> = SparseVectorView<'q, u16, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;

    type Evaluator<'e>
        = PackedCentroidSparseQueryEvaluatorF16Lut<'e>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        PackedCentroidSparseQueryEvaluatorF16Lut::new(query, &self.inner)
    }

    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <PackedCentroidSparseQuantizer as SparseDataEncoder>::decode_vector(
            &self.inner,
            vector,
        );
        PackedCentroidSparseQueryEvaluatorF16Lut::new(decoded.as_view(), &self.inner)
    }

    #[inline]
    fn output_dim(&self) -> usize {
        self.inner.output_dim()
    }

    #[inline]
    fn input_dim(&self) -> usize {
        self.inner.input_dim()
    }
}

impl SpaceUsage for PackedCentroidSparseQuantizerF16Lut {
    fn space_usage_bytes(&self) -> usize {
        self.inner.space_usage_bytes()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Clustered-centroid variant: shared codebook keyed by `cluster_of[c]`
// ────────────────────────────────────────────────────────────────────────────
//
// Training is two-stage: (1) per-component k-means as in
// [`PackedCentroidSparseQuantizer::train`] (re-used here verbatim) producing a
// `dim × 16` matrix of per-component centroid rows; (2) Euclidean k-means over
// those rows (each treated as a 16-dim point) using vectorium's [`KMeans`],
// producing `K` shared centroid rows and a `cluster_of: u32; dim` assignment.
// The doc set is then re-encoded against the shared codebook so each 4-bit code
// names a position in `shared_centroid[cluster_of[c]]`.
//
// At search time the kernel does three indirect gathers per element
// (`q_dense[c]`, `cluster_of[c]`, `shared_centroid[cluster · 16 + code]`); all
// three buffers fit in L2 at typical SPLADE dims. See
// [`kernels::dot_product_centroid_clustered_nbits4`] for the details.

/// Bitpacked clustered-centroid sparse quantizer at 4 bits / 16 codes.
///
/// `cluster_of` is stored as `u32` (not `u16`) on purpose: AVX-512 only has
/// 32/64-bit gathers, so a `u32` array gives a single-µop `vpgatherqd` for the
/// component-to-cluster lookup. The extra ~57 KiB at dim ≈ 30 K is well within
/// L2 (1 MiB) on the target hardware.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedClusteredCentroidSparseQuantizer {
    dim: usize,
    n_clusters: usize,
    /// `cluster_of[c]` is the cluster id assigned to component `c`. u32 for
    /// SIMD-friendly gathers (see struct doc).
    cluster_of: Box<[u32]>,
    /// Shared codebook: `shared_centroid[k · 16 + e]` is the e-th centroid for
    /// cluster `k`. Per-row sorted ascending so `quantize` can binary-search.
    shared_centroid: Box<[f32]>,
}

impl sealed::Sealed for PackedClusteredCentroidSparseQuantizer {}

impl PartialEq for PackedClusteredCentroidSparseQuantizer {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
            && self.n_clusters == other.n_clusters
            && self.cluster_of == other.cluster_of
            && self.shared_centroid == other.shared_centroid
    }
}

impl PackedClusteredCentroidSparseQuantizer {
    /// Two-stage train: per-component k-means (clip + Lloyd) → cluster the rows
    /// of the resulting `dim × 16` matrix into `n_clusters` groups via vectorium
    /// [`KMeans`] (squared-Euclidean). `cluster_iters` / `cluster_redo` are the
    /// outer Lloyd config for that second stage.
    pub fn train(
        training_data: &PlainSparseDataset<u16, f32, SquaredEuclideanDistance>,
        lower_percentile: f32,
        upper_percentile: f32,
        n_iterations: usize,
        n_clusters: usize,
        cluster_iters: usize,
        cluster_redo: usize,
    ) -> Self {
        assert!(n_clusters >= 1, "n_clusters must be ≥ 1");

        // Stage 1: per-component k-means → dim × 16 centroid matrix (rows sorted ascending).
        let inner = PackedCentroidSparseQuantizer::train(
            training_data,
            lower_percentile,
            upper_percentile,
            n_iterations,
        );
        let dim = inner.dim;

        assert!(
            n_clusters <= dim,
            "n_clusters ({n_clusters}) cannot exceed dim ({dim})"
        );

        // Stage 2: cluster the rows. Each row is a 16-d point.
        let row_encoder =
            PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(NUM_CENTROIDS);
        let row_data: Box<[f32]> = inner.centroids.clone();
        let rows_dataset =
            PlainDenseDataset::<f32, SquaredEuclideanDistance>::from_raw(row_data, dim, row_encoder);

        let kmeans = KMeansBuilder::new()
            .n_iter(cluster_iters)
            .n_redo(cluster_redo.max(1))
            .verbose(true)
            .build();
        let cluster_centroids = kmeans.train(&rows_dataset, n_clusters, None);

        let assignments = KMeans::compute_assignments(&rows_dataset, &cluster_centroids, 0);
        let cluster_of: Vec<u32> = assignments.iter().map(|&(_, k)| k as u32).collect();

        // Means of ascending vectors are themselves ascending, but kmeans's
        // empty-cluster split perturbs alternating dims by ±ε which can break
        // monotonicity by tiny amounts. Sort each row to restore the invariant
        // expected by quantize() (binary-search assumes ascending).
        let mut shared = cluster_centroids.values().to_vec();
        for k in 0..n_clusters {
            let row = &mut shared[k * NUM_CENTROIDS..(k + 1) * NUM_CENTROIDS];
            row.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        }

        Self {
            dim,
            n_clusters,
            cluster_of: cluster_of.into_boxed_slice(),
            shared_centroid: shared.into_boxed_slice(),
        }
    }

    #[inline]
    pub fn nbits(&self) -> u8 {
        NBITS
    }

    #[inline]
    pub fn num_centroids(&self) -> usize {
        NUM_CENTROIDS
    }

    #[inline]
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    #[inline]
    pub fn cluster_of(&self) -> &[u32] {
        &self.cluster_of
    }

    #[inline]
    pub fn shared_centroid(&self) -> &[f32] {
        &self.shared_centroid
    }

    #[inline]
    fn centroids_for_cluster(&self, k: usize) -> &[f32] {
        &self.shared_centroid[k * NUM_CENTROIDS..(k + 1) * NUM_CENTROIDS]
    }

    /// Find the nearest centroid index for `value` in the cluster row of `dim`.
    /// Mirrors [`PackedCentroidSparseQuantizer::quantize`].
    #[inline]
    fn quantize(&self, dim: usize, value: f32) -> u8 {
        let cluster = self.cluster_of[dim] as usize;
        let table = self.centroids_for_cluster(cluster);
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

    #[inline]
    fn dequantize(&self, dim: usize, code: u8) -> f32 {
        let cluster = self.cluster_of[dim] as usize;
        self.shared_centroid[cluster * NUM_CENTROIDS + code as usize]
    }
}

impl SparseDataEncoder for PackedClusteredCentroidSparseQuantizer {
    type InputComponentType = u16;
    type InputValueType = f32;
    type OutputComponentType = u16;
    type OutputValueType = u8;

    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let view = unsafe { PackedCentroidScalar::from_unchecked_slice(encoded.data()) };
        let mut components: Vec<u16> = Vec::with_capacity(view.n as usize);
        let mut values: Vec<f32> = Vec::with_capacity(view.n as usize);
        for (c, code) in view.iter() {
            components.push(c);
            values.push(self.dequantize(c as usize, code));
        }
        SparseVectorOwned::new(components, values)
    }
}

impl PackedSparseVectorEncoder for PackedClusteredCentroidSparseQuantizer {
    type PackedDataType = u64;

    fn push_encoded<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, Self::InputComponentType, Self::InputValueType>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<Self::PackedDataType>,
    {
        let mut q_values: Vec<u8> = input
            .components()
            .iter()
            .zip(input.values())
            .map(|(&c, &v)| self.quantize(c as usize, v))
            .collect();
        let mut q_components: Vec<u16> = input.components().to_vec();

        let mut encoded_u8 = Vec::new();
        PackedCentroidScalar::push_vector(&mut encoded_u8, &mut q_components, &mut q_values);

        debug_assert!(
            encoded_u8.len() % std::mem::size_of::<u64>() == 0,
            "encoded_u8 length {} is not a multiple of 8",
            encoded_u8.len()
        );

        let words = encoded_u8
            .chunks_exact(std::mem::size_of::<u64>())
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()));
        output.extend(words);
    }
}

impl VectorEncoder for PackedClusteredCentroidSparseQuantizer {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u16, f32>;
    type QueryVector<'q> = SparseVectorView<'q, u16, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;

    type Evaluator<'e>
        = PackedClusteredCentroidSparseQueryEvaluator<'e>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        PackedClusteredCentroidSparseQueryEvaluator::new(query, self)
    }

    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        PackedClusteredCentroidSparseQueryEvaluator::new(decoded.as_view(), self)
    }

    #[inline]
    fn output_dim(&self) -> usize {
        self.dim
    }

    #[inline]
    fn input_dim(&self) -> usize {
        self.dim
    }
}

impl SpaceUsage for PackedClusteredCentroidSparseQuantizer {
    fn space_usage_bytes(&self) -> usize {
        self.dim.space_usage_bytes()
            + self.n_clusters.space_usage_bytes()
            + self.cluster_of.space_usage_bytes()
            + self.shared_centroid.space_usage_bytes()
    }
}

/// Holds the densified per-query weight vector `q_dense[dim]` (zero outside the
/// query support). `cluster_of` and `shared_centroid` live on the quantizer and
/// are shared across queries; the evaluator just borrows them via `quantizer`.
#[derive(Debug, Clone)]
pub struct PackedClusteredCentroidSparseQueryEvaluator<'e> {
    quantizer: &'e PackedClusteredCentroidSparseQuantizer,
    q_dense: Vec<f32>,
}

impl<'e> PackedClusteredCentroidSparseQueryEvaluator<'e> {
    pub fn new(
        query: SparseVectorView<'_, u16, f32>,
        quantizer: &'e PackedClusteredCentroidSparseQuantizer,
    ) -> Self {
        let max_c = query.components().iter().copied().max().unwrap_or(0) as usize;
        assert!(
            max_c < quantizer.input_dim(),
            "Query component {max_c} exceeds quantizer dim {}",
            quantizer.input_dim()
        );
        assert_eq!(
            query.components().len(),
            query.values().len(),
            "Query components/values length mismatch."
        );

        let mut q_dense = vec![0.0f32; quantizer.dim];
        for (&c, &qv) in query.components().iter().zip(query.values()) {
            q_dense[c as usize] = qv;
        }

        Self { quantizer, q_dense }
    }
}

impl<'e, 'v> QueryEvaluator<PackedVectorView<'v, u64>>
    for PackedClusteredCentroidSparseQueryEvaluator<'e>
{
    type Distance = DotProduct;

    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> DotProduct {
        let view = unsafe { PackedCentroidScalar::from_unchecked_slice(vector.data()) };
        DotProduct::from(view.dot_product_clustered(
            &self.q_dense,
            &self.quantizer.cluster_of,
            &self.quantizer.shared_centroid,
        ))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PlainSparseDatasetGrowable;
    use crate::core::dataset::DatasetGrowable;
    use crate::core::vector::{PackedVectorView, SparseVectorView};
    use crate::core::vector_encoder::PackedSparseVectorEncoder;
    use crate::encoders::sparse_scalar::PlainSparseQuantizer;

    fn build_training_data(
        dim: usize,
        vectors: &[(&[u16], &[f32])],
    ) -> PlainSparseDataset<u16, f32, SquaredEuclideanDistance> {
        let q = PlainSparseQuantizer::<u16, f32, SquaredEuclideanDistance>::new(dim, dim);
        let mut g = PlainSparseDatasetGrowable::new(q);
        for &(c, v) in vectors {
            g.push(SparseVectorView::new(c, v));
        }
        g.into()
    }

    fn encode_to_words(
        encoder: &PackedCentroidSparseQuantizer,
        components: &[u16],
        values: &[f32],
    ) -> Vec<u64> {
        let mut buf: Vec<u64> = Vec::new();
        encoder.push_encoded(SparseVectorView::new(components, values), &mut buf);
        buf
    }

    #[test]
    fn encode_then_decode_roundtrips_components() {
        let td = build_training_data(
            5,
            &[
                (&[0, 1, 2, 3, 4], &[0.0; 5]),
                (&[0, 1, 2, 3, 4], &[10.0; 5]),
            ],
        );
        let q = PackedCentroidSparseQuantizer::train(&td, 0.0, 1.0, 10);
        let words = encode_to_words(&q, &[1u16, 3], &[5.0_f32, 7.0]);
        let view = PackedVectorView::new(&words);
        let dec = q.decode_vector(view);
        assert_eq!(dec.components(), &[1u16, 3]);
    }

    #[test]
    fn dot_product_matches_dequantized_reference() {
        let td = build_training_data(
            4,
            &[(&[0, 1, 2, 3], &[0.0; 4]), (&[0, 1, 2, 3], &[10.0; 4])],
        );
        let q = PackedCentroidSparseQuantizer::train(&td, 0.0, 1.0, 10);
        let words = encode_to_words(&q, &[0u16, 1, 2], &[5.0_f32, 7.0, 9.0]);
        let view = PackedVectorView::new(&words);

        let query = SparseVectorView::new(&[0u16, 2], &[1.0_f32, 3.0]);
        let evaluator = q.query_evaluator(query);
        let got = evaluator.compute_distance(view).0;

        let dec = q.decode_vector(view);
        let mut ref_dot = 0.0f32;
        for (&qc, &qv) in [0u16, 2].iter().zip(&[1.0_f32, 3.0]) {
            for (&dc, &dv) in dec.components().iter().zip(dec.values()) {
                if qc == dc {
                    ref_dot += qv * dv;
                }
            }
        }
        assert!(
            (got - ref_dot).abs() < 1e-3,
            "got {got}, expected {ref_dot}"
        );
    }

    /// Exercises both the bulk SIMD path (chunks of 16) and the scalar tail at
    /// every partial-block boundary, plus a long input.
    #[test]
    fn dot_product_kernel_matches_reference_on_bulk_and_tail() {
        let dim = 256usize;

        let big_components: Vec<u16> = (0..dim as u16).collect();
        let big_values_lo = vec![0.0f32; dim];
        let big_values_hi = vec![10.0f32; dim];
        let td = build_training_data(
            dim,
            &[
                (&big_components, &big_values_lo),
                (&big_components, &big_values_hi),
            ],
        );

        let q = PackedCentroidSparseQuantizer::train(&td, 0.0, 1.0, 10);

        // Cover every partial-block size 1..=17, the second-block boundary, and
        // a longer bulk-dominant input.
        let sizes: Vec<usize> = (1..=17usize).chain([23, 32, 200]).collect();

        for &n in &sizes {
            let comps: Vec<u16> = (0..n as u16).collect();
            let vals: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 + 0.25).collect();

            let words = encode_to_words(&q, &comps, &vals);
            let view = PackedVectorView::new(&words);

            let query_components: Vec<u16> = comps.clone();
            let query_values: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.1).collect();
            let query = SparseVectorView::new(&query_components, &query_values);

            let evaluator = q.query_evaluator(query);
            let got = evaluator.compute_distance(view).0;

            let dec = q.decode_vector(view);
            let mut ref_dot = 0.0f32;
            for (&qc, &qv) in query_components.iter().zip(&query_values) {
                for (&dc, &dv) in dec.components().iter().zip(dec.values()) {
                    if qc == dc {
                        ref_dot += qv * dv;
                    }
                }
            }

            let tol = (ref_dot.abs() * 1e-4).max(1e-3);
            assert!(
                (got - ref_dot).abs() < tol,
                "n={n}: got {got}, expected {ref_dot} (tol {tol})"
            );
        }
    }

    #[test]
    fn output_blob_is_u64_aligned() {
        let td = build_training_data(8, &[(&[0, 1, 2, 3, 4, 5, 6, 7], &[10.0; 8])]);
        let q = PackedCentroidSparseQuantizer::train(&td, 0.0, 1.0, 10);
        let words = encode_to_words(&q, &[0u16, 2, 5], &[5.0_f32, 8.0, 1.0]);
        let dec = q.decode_vector(PackedVectorView::new(&words));
        assert_eq!(dec.components(), &[0u16, 2, 5]);
    }

    /// Mirrors `dot_product_kernel_matches_reference_on_bulk_and_tail` for the
    /// f16-LUT variant: the f16 LUT and the pair-gather kernel must agree with the
    /// dequantized reference within an f16-appropriate tolerance, across every
    /// partial-block size and a longer bulk-dominant input.
    #[test]
    fn f16_lut_kernel_matches_reference_on_bulk_and_tail() {
        let dim = 256usize;

        let big_components: Vec<u16> = (0..dim as u16).collect();
        let big_values_lo = vec![0.0f32; dim];
        let big_values_hi = vec![10.0f32; dim];
        let td = build_training_data(
            dim,
            &[
                (&big_components, &big_values_lo),
                (&big_components, &big_values_hi),
            ],
        );

        let q = PackedCentroidSparseQuantizer::train(&td, 0.0, 1.0, 10);
        let q_f16 = PackedCentroidSparseQuantizerF16Lut::from_inner(q.clone());

        let sizes: Vec<usize> = (1..=17usize).chain([23, 32, 200]).collect();

        for &n in &sizes {
            let comps: Vec<u16> = (0..n as u16).collect();
            let vals: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 + 0.25).collect();

            let words = encode_to_words(&q, &comps, &vals);
            let view = PackedVectorView::new(&words);

            let query_components: Vec<u16> = comps.clone();
            let query_values: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.1).collect();
            let query = SparseVectorView::new(&query_components, &query_values);

            let evaluator = q_f16.query_evaluator(query);
            let got = evaluator.compute_distance(view).0;

            let dec = q.decode_vector(view);
            let mut ref_dot = 0.0f32;
            for (&qc, &qv) in query_components.iter().zip(&query_values) {
                for (&dc, &dv) in dec.components().iter().zip(dec.values()) {
                    if qc == dc {
                        ref_dot += qv * dv;
                    }
                }
            }

            // Each LUT entry has ~10 bits of f16 mantissa precision (~1e-3 relative);
            // sums of n such entries grow proportionally. Generous bound below.
            let tol = (ref_dot.abs() * 5e-3).max(5e-2 * (n as f32).sqrt());
            assert!(
                (got - ref_dot).abs() < tol,
                "n={n}: f16-LUT got {got}, expected {ref_dot} (tol {tol})"
            );
        }
    }

    /// Mirrors the f32-LUT kernel test for the clustered variant: the 3-gather
    /// kernel must agree with a dequantized reference (using the *clustered*
    /// dequantize) across every partial-block size. Tolerance is the same as
    /// the f32-LUT path because the only quantization error here is the doc
    /// codes themselves — the centroid table is f32 and the kernel is exact.
    #[test]
    fn clustered_kernel_matches_reference_on_bulk_and_tail() {
        let dim = 256usize;

        let big_components: Vec<u16> = (0..dim as u16).collect();
        let big_values_lo = vec![0.0f32; dim];
        let big_values_hi = vec![10.0f32; dim];
        let td = build_training_data(
            dim,
            &[
                (&big_components, &big_values_lo),
                (&big_components, &big_values_hi),
            ],
        );

        // n_clusters=8 is small but exercises the same code paths and keeps
        // training fast for a unit test. Per-component training uses 10 Lloyd
        // iters; cluster-stage uses 5.
        let q_clustered =
            PackedClusteredCentroidSparseQuantizer::train(&td, 0.0, 1.0, 10, 8, 5, 1);

        let sizes: Vec<usize> = (1..=17usize).chain([23, 32, 200]).collect();

        for &n in &sizes {
            let comps: Vec<u16> = (0..n as u16).collect();
            let vals: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 + 0.25).collect();

            let mut buf: Vec<u64> = Vec::new();
            q_clustered.push_encoded(SparseVectorView::new(&comps, &vals), &mut buf);
            let view = PackedVectorView::new(&buf);

            let query_components: Vec<u16> = comps.clone();
            let query_values: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.1).collect();
            let query = SparseVectorView::new(&query_components, &query_values);

            let evaluator = q_clustered.query_evaluator(query);
            let got = evaluator.compute_distance(view).0;

            let dec = q_clustered.decode_vector(view);
            let mut ref_dot = 0.0f32;
            for (&qc, &qv) in query_components.iter().zip(&query_values) {
                for (&dc, &dv) in dec.components().iter().zip(dec.values()) {
                    if qc == dc {
                        ref_dot += qv * dv;
                    }
                }
            }

            let tol = (ref_dot.abs() * 1e-4).max(1e-3);
            assert!(
                (got - ref_dot).abs() < tol,
                "n={n}: clustered got {got}, expected {ref_dot} (tol {tol})"
            );
        }
    }
}
