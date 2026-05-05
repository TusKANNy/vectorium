//! Bitpacked per-component uniform quantizer for sparse vectors at `nbits ∈ [1, 7]`.
//!
//! Sister of [`UniformSparseQuantizer`](super::uniform_quantization_sparse_scalar::UniformSparseQuantizer)
//! (the 8-bit / one-byte-per-code path). For `nbits == 8` use that encoder; this one
//! exists to avoid wasting `(8 - nbits)` bits per code.
//!
//! ## Layout convention
//!
//! Mirrors [`super::dotvbyte_scalaru8`]: the encoder is a [`PackedSparseVectorEncoder`]
//! with `PackedDataType = u64`, the per-vector blob starts with a `u16` original length,
//! and the inner data layout lives on a private wrapper struct [`BitPackedScalar`] that
//! owns `from_unchecked_slice`, `push_vector`, `iter`, `iter_raw`, and `dot_product`.
//! That split is what lets us drop in a SIMD-optimised inner kernel later without
//! touching the encoder surface.
//!
//! Per-vector byte layout (then padded to a multiple of 8 bytes for `u64` storage):
//!
//! ```text
//! [u16 LE: n_components]
//! [u16 LE × n_components: components, sorted ascending]
//! [packed codes; layout depends on nbits — see "Codes layout" below]
//! [≥ 8 zero bytes; total length rounded up to a multiple of 8]
//! ```
//!
//! Codes are byte-aligned at the end of each vector, so vector boundaries always sit
//! on a byte (and ultimately u64) boundary. The trailing pad is sized so there are
//! always at least 8 bytes past `codes_end` — that lets the hot decode path load 8
//! bytes unconditionally with no end-of-buffer check on the last chunk.
//! Per-vector pad is in `[8, 15]` bytes.
//!
//! ## Codes layout
//!
//! - `nbits ∈ {1, 2, 3, 5, 6, 7}`: standard LSB-first bitpacking via [`BitPacker`].
//!   Byte count is `ceil(n_components * nbits / 8)`.
//! - `nbits == 4`: SIMD-friendly *interleaved* layout. Codes are grouped in blocks
//!   of 16; each block occupies 8 bytes laid out so codes 0..7 sit in the low
//!   nibbles and codes 8..15 sit in the high nibbles of those bytes:
//!   `byte_i = (code[base + i + 8] << 4) | code[base + i]` for `i ∈ [0..8]`.
//!   The trailing partial block (if any) is zero-padded to a full 8 bytes.
//!   Byte count is `ceil(n_components / 16) * 8`. The interleaving aligns the
//!   block exactly with one 8-byte SIMD load: `buf & 0x0F` yields the natural-
//!   order codes 0..7 and `(buf >> 4) & 0x0F` yields codes 8..15, so the
//!   dot-product kernel needs no shuffle to reach natural code order.
//!   Components are stored in their original (sorted-ascending) order. Existing
//!   on-disk data encoded with the prior 8-code / 4-byte interleaved layout for
//!   `nbits == 4` is not readable by this build.
//!
//! ## Draft scope
//!
//! - DotProduct only, matching `dotvbyte_scalaru8`.
//! - Components hardcoded to `u16`. Generalising to `u32` is mechanical.
//! - `iter_raw` and `dot_product` are scalar in this draft. The clean SIMD hook is
//!   `iter_raw`: each chunk of 8 codes occupies exactly `nbits` bytes in the packed
//!   stream (since `8 * nbits / 8 = nbits` for any `nbits ≤ 7`), which is a much
//!   simpler stride than dotvbyte's variable-length chunks.

use std::marker::PhantomData;

use bytemuck::{cast_slice, try_cast_slice};
use serde::{Deserialize, Serialize};

use crate::core::sealed;
use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::encoders::bitpacker::{packed_byte_len, BitPacker, BitUnpacker};

/// Bytes used by the packed-codes section for `n_components` codes at `nbits`.
///
/// Mirrors [`packed_byte_len`] for `nbits != 4`. For `nbits == 4` the encoder uses
/// the SIMD-friendly interleaved layout (16 codes per 8-byte block, last partial
/// block zero-padded to a full 8 bytes), so the byte count is `ceil(n / 16) * 8`.
#[inline]
const fn codes_byte_len(n: usize, nbits: u8) -> usize {
    if nbits == 4 {
        n.div_ceil(16) * 8
    } else {
        packed_byte_len(n, nbits)
    }
}

/// Append the codes section in the interleaved nbits=4 layout. See the module
/// docstring for the layout description.
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
        // Always write a full 8-byte block; missing nibbles are zero.
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
use crate::utils::train_sparse_scalar_quantizer_with_levels;
use crate::{Dataset, PlainSparseDataset, SpaceUsage, SquaredEuclideanDistance};

// ────────────────────────────────────────────────────────────────────────────
// Encoder
// ────────────────────────────────────────────────────────────────────────────

/// Bitpacked uniform sparse quantizer; `nbits ∈ [1, 7]`. For `nbits == 8` use
/// [`UniformSparseQuantizer`](super::uniform_quantization_sparse_scalar::UniformSparseQuantizer).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedVariableBitUniformSparseQuantizer {
    dim: usize,
    nbits: u8,
    max_val: f32,
    quants: Box<[f32]>,
}

impl sealed::Sealed for PackedVariableBitUniformSparseQuantizer {}

impl PartialEq for PackedVariableBitUniformSparseQuantizer {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim && self.nbits == other.nbits
    }
}

impl PackedVariableBitUniformSparseQuantizer {
    #[inline]
    pub fn new(input_dim: usize, output_dim: usize, nbits: u8) -> Self {
        assert_eq!(
            input_dim, output_dim,
            "PackedVariableBitUniformSparseQuantizer requires input_dim == output_dim"
        );
        assert!(
            (1..=7).contains(&nbits),
            "nbits must be in [1, 7] (use UniformSparseQuantizer for 8). Got {nbits}"
        );
        let max_val = ((1u32 << nbits) - 1) as f32;
        Self {
            dim: input_dim,
            nbits,
            max_val,
            quants: vec![0.0; output_dim].into_boxed_slice(),
        }
    }

    pub fn nbits(&self) -> u8 {
        self.nbits
    }

    pub fn max_val(&self) -> f32 {
        self.max_val
    }

    pub fn quants(&self) -> &[f32] {
        &self.quants
    }

    pub fn train(
        training_data: &PlainSparseDataset<u16, f32, SquaredEuclideanDistance>,
        lower_percentile: f32,
        upper_percentile: f32,
        nbits: u8,
    ) -> Self {
        assert!(
            (1..=7).contains(&nbits),
            "nbits must be in [1, 7] (use UniformSparseQuantizer for 8). Got {nbits}"
        );
        let max_val = ((1u32 << nbits) - 1) as f32;
        let quants = train_sparse_scalar_quantizer_with_levels(
            training_data,
            lower_percentile,
            upper_percentile,
            max_val,
        );
        Self {
            dim: training_data.output_dim(),
            nbits,
            max_val,
            quants: quants.into_boxed_slice(),
        }
    }
}

impl SparseDataEncoder for PackedVariableBitUniformSparseQuantizer {
    type InputComponentType = u16;
    type InputValueType = f32;
    type OutputComponentType = u16;
    type OutputValueType = u8; // logical code width is `nbits`; trait still needs a value type

    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let view = unsafe { BitPackedScalar::from_unchecked_slice(encoded.data(), self.nbits) };
        let mut components: Vec<u16> = Vec::with_capacity(view.n as usize);
        let mut values: Vec<f32> = Vec::with_capacity(view.n as usize);
        for (c, code) in view.iter() {
            components.push(c);
            values.push((code as f32) * self.quants[c as usize]);
        }
        SparseVectorOwned::new(components, values)
    }
}

impl PackedSparseVectorEncoder for PackedVariableBitUniformSparseQuantizer {
    type PackedDataType = u64;

    fn push_encoded<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, Self::InputComponentType, Self::InputValueType>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<Self::PackedDataType>,
    {
        // Quantize values to nbits-wide codes.
        let max_val = self.max_val;
        let mut q_values: Vec<u8> = input
            .components()
            .iter()
            .zip(input.values())
            .map(|(&c, &v)| {
                let q = self.quants[c as usize];
                if q > 0.0 {
                    (v / q).clamp(0.0, max_val) as u8
                } else {
                    0u8
                }
            })
            .collect();
        let mut q_components: Vec<u16> = input.components().to_vec();

        // Build the byte blob, then dump as a u64 stream.
        let mut encoded_u8 = Vec::new();
        BitPackedScalar::push_vector(&mut encoded_u8, &mut q_components, &mut q_values, self.nbits);

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

impl VectorEncoder for PackedVariableBitUniformSparseQuantizer {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u16, f32>;
    type QueryVector<'q> = SparseVectorView<'q, u16, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;

    type Evaluator<'e>
        = PackedVariableBitUniformSparseQueryEvaluator<'e>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        PackedVariableBitUniformSparseQueryEvaluator::new(query, self)
    }

    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        PackedVariableBitUniformSparseQueryEvaluator::new(decoded.as_view(), self)
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

/// Pre-densifies the query and folds per-component quantization scales into it, so the
/// inner dot-product is a plain `Σ code * transformed[c]` (matching `dotvbyte_scalaru8`).
#[derive(Debug, Clone)]
pub struct PackedVariableBitUniformSparseQueryEvaluator<'e> {
    dense_query_transformed: Vec<f32>,
    nbits: u8,
    _phantom: PhantomData<&'e ()>,
}

impl<'e> PackedVariableBitUniformSparseQueryEvaluator<'e> {
    pub fn new(
        query: SparseVectorView<'_, u16, f32>,
        quantizer: &'e PackedVariableBitUniformSparseQuantizer,
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

        let mut transformed = vec![0.0f32; quantizer.dim];
        for (&c, &v) in query.components().iter().zip(query.values()) {
            transformed[c as usize] = v * quantizer.quants[c as usize];
        }

        Self {
            dense_query_transformed: transformed,
            nbits: quantizer.nbits,
            _phantom: PhantomData,
        }
    }
}

impl<'e, 'v> QueryEvaluator<PackedVectorView<'v, u64>>
    for PackedVariableBitUniformSparseQueryEvaluator<'e>
{
    type Distance = DotProduct;

    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> DotProduct {
        let view = unsafe { BitPackedScalar::from_unchecked_slice(vector.data(), self.nbits) };
        DotProduct::from(view.dot_product(&self.dense_query_transformed))
    }
}

impl SpaceUsage for PackedVariableBitUniformSparseQuantizer {
    fn space_usage_bytes(&self) -> usize {
        self.dim.space_usage_bytes()
            + std::mem::size_of::<u8>()
            + std::mem::size_of::<f32>()
            + self.quants.space_usage_bytes()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Per-vector layout helper (the dotvbyte_scalaru8 analogue)
// ────────────────────────────────────────────────────────────────────────────

/// Typed view onto a single packed-vector blob.
///
/// Mirrors `DotVbyteScalarU8`: this struct carries the per-vector layout and exposes
/// the encode (`push_vector`), decode (`iter`), batched-decode (`iter_raw`), and
/// scoring (`dot_product`) entry points. The encoder above just dispatches to it.
#[derive(Clone)]
struct BitPackedScalar<'a> {
    n: u16,
    nbits: u8,
    /// `n` components, sorted ascending.
    /// Stored as `&[u16]` (zero-copy view onto the LE-encoded byte region; valid because
    /// the blob is `u64`-aligned and the components region starts at byte offset 2).
    components: &'a [u16],
    /// `ceil(n * nbits / 8)` bytes of bitpacked codes.
    codes: &'a [u8],
}

impl<'a> BitPackedScalar<'a> {
    /// Parse a packed `&[u64]` produced by `push_vector` into typed sub-slices.
    ///
    /// # Safety
    /// Caller guarantees `slice` was produced by `push_vector` for the same `nbits`.
    unsafe fn from_unchecked_slice(slice: &'a [u64], nbits: u8) -> Self {
        let bytes: &[u8] = unsafe { try_cast_slice::<u64, u8>(slice).unwrap_unchecked() };

        // Header: u16 LE original length.
        let n = u16::from_le_bytes([bytes[0], bytes[1]]);
        let n_usize = n as usize;

        let comp_start = std::mem::size_of::<u16>();
        let comp_end = comp_start + n_usize * std::mem::size_of::<u16>();
        let codes_start = comp_end;
        // Extend the codes view by 8 bytes to expose the SIMD-safety pad that
        // `push_vector` writes after the logical codes region. The bulk decoder loads
        // 8 bytes per chunk; the extra slack is masked off so its content doesn't
        // matter, but the read must be in-slice.
        let codes_end = codes_start + codes_byte_len(n_usize, nbits) + 8;

        // SAFETY (alignment): `slice: &[u64]` is 8-byte aligned, so its underlying
        // `&[u8]` view starts at an address that is also 2-aligned. Skipping the
        // 2-byte header keeps that 2-alignment, so casting the components region to
        // `&[u16]` is sound.
        let components: &[u16] =
            cast_slice(unsafe { bytes.get_unchecked(comp_start..comp_end) });
        let codes = unsafe { bytes.get_unchecked(codes_start..codes_end) };

        Self {
            n,
            nbits,
            components,
            codes,
        }
    }

    /// Append a freshly encoded vector (mutable component / code arrays sorted ascending
    /// by component) to `out` as a sequence of bytes whose total length is a multiple of 8.
    ///
    /// `components` must be sorted ascending; `values` are aligned to it.
    pub fn push_vector(
        out: &mut Vec<u8>,
        components: &mut [u16],
        values: &mut [u8],
        nbits: u8,
    ) {
        assert_eq!(components.len(), values.len(), "components/values length mismatch");
        assert!(
            components.len() < u16::MAX as usize,
            "vector longer than u16::MAX is not supported (got {})",
            components.len()
        );
        debug_assert!(
            components.windows(2).all(|w| w[0] <= w[1]),
            "components must be sorted ascending"
        );
        debug_assert!(
            (1..=7).contains(&nbits),
            "nbits must be in [1,7], got {nbits}"
        );

        let n = components.len() as u16;

        // Header.
        out.extend_from_slice(&n.to_le_bytes());

        // Components.
        for &c in components.iter() {
            out.extend_from_slice(&c.to_le_bytes());
        }

        // Packed codes. For `nbits == 4` we use the interleaved layout (see the
        // module docstring); other `nbits` use the generic bitpacker.
        if nbits == 4 {
            write_codes_nbits4_interleaved(out, values);
        } else {
            let mut packer = BitPacker::new();
            for &v in values.iter() {
                packer.write(v, nbits, out);
            }
            packer.flush(out);
        }

        // Pad so (a) there are ≥ 8 trailing bytes past codes_end (lets the hot decode
        // path do an unconditional u64 load on the last chunk), and (b) total length is
        // a multiple of 8 (required by `PackedDataType = u64` storage).
        let target_len = (out.len() + 8 + 7) & !7usize;
        out.resize(target_len, 0);
    }

    /// Sequential decode iterator over `(component, code)` pairs.
    pub fn iter(self) -> impl Iterator<Item = (u16, u8)> + 'a {
        let nbits = self.nbits;
        let components = self.components;
        let codes = self.codes;
        gen move {
            if nbits == 4 {
                // Interleaved layout: code i lives in block (i / 16) at byte
                // offset (i & 7) within that block; shift is 0 if (i & 8) == 0
                // else 4.
                for (i, &c) in components.iter().enumerate() {
                    let block = i / 16;
                    let byte_off = block * 8 + (i & 7);
                    let shift = ((i & 8) >> 1) as u8;
                    let v = (codes[byte_off] >> shift) & 0x0F;
                    yield (c, v);
                }
            } else {
                let mut unpacker = BitUnpacker::new(codes);
                for &c in components {
                    let v = unpacker.read(nbits);
                    yield (c, v);
                }
            }
        }
    }

    /// Batched decode hook reserved for the SIMD path.
    ///
    /// At `nbits ∈ [1, 7]`, every 8 consecutive codes occupy exactly `nbits` bytes in the
    /// packed stream, so a SIMD lane-of-8 implementation is straightforward (load 16 bytes
    /// of components + `nbits` bytes of codes, expand). The draft below reuses `iter` so
    /// behaviour is correct; swap it out for the vectorised version when wiring in SIMD.
    #[allow(dead_code)]
    pub fn iter_raw(self) -> impl Iterator<Item = (u16, u8)> + 'a {
        // TODO: yield (Simd<u16, 8>, Simd<u8, 8>) once the SIMD kernel is in.
        self.iter()
    }

    /// Score against a dense pre-transformed query: `Σ code · transformed[c]`.
    ///
    /// Dispatches once on `nbits` to a kernel specialised for that bit width. NBITS=1,
    /// 2, and 4 each have a hand-written kernel sized to the largest chunk that fills
    /// a u64 cleanly (64 / 32 / 16 codes). NBITS ∈ {3, 5, 6, 7} share the
    /// `dot_product_general` template (8 codes per chunk, NBITS bytes per chunk);
    /// const-generic monomorphisation gives each its own specialised function in the
    /// binary with shift amounts and mask as immediates. The encoder pads codes by
    /// ≥ 8 trailing bytes so all chunk loads are unconditional.
    pub fn dot_product(&self, query: &[f32]) -> f32 {
        match self.nbits {
            1 => kernels::dot_product_nbits1(self.components, self.codes, query),
            2 => kernels::dot_product_nbits2(self.components, self.codes, query),
            3 => kernels::dot_product_general::<3>(self.components, self.codes, query),
            4 => kernels::dot_product_nbits4(self.components, self.codes, query),
            5 => kernels::dot_product_general::<5>(self.components, self.codes, query),
            6 => kernels::dot_product_general::<6>(self.components, self.codes, query),
            7 => kernels::dot_product_general::<7>(self.components, self.codes, query),
            // BitPackedScalar is only constructed for nbits ∈ [1, 7]
            // (the encoder asserts this at construction).
            _ => unsafe { std::hint::unreachable_unchecked() },
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Specialised dot-product kernels
// ────────────────────────────────────────────────────────────────────────────
//
// One kernel per NBITS. Hand-written for 1 / 2 / 4 (different chunk sizes and
// decode patterns); shared const-generic template for 3 / 5 / 6 / 7 (same shape,
// monomorphised to a separate function per NBITS).
//
// Safety contract for every kernel:
// - `components.len() == n` is the logical number of codes.
// - `codes.len() >= packed_byte_len(n, NBITS) + 8` (enforced by the encoder pad
//   and by `BitPackedScalar::from_unchecked_slice`, which extends `codes` by +8).
// - `query.len() >= max(components) + 1` (enforced by the query evaluator).
mod kernels {
    use std::simd::num::{SimdFloat, SimdUint};
    use std::simd::{Mask, Simd, StdFloat};
    use crate::encoders::dotvbyte_scalaru8::simd_scalaru8_to_f32;

    /// NBITS = 1: 64 codes per 8-byte chunk (one bit per code).
    #[inline]
    pub(super) fn dot_product_nbits1(
        components: &[u16],
        codes: &[u8],
        query: &[f32],
    ) -> f32 {
        let n = components.len();
        let chunks = n / 64;
        let tail_start = chunks * 64;
        let mut acc = 0.0f32;

        for chunk_i in 0..chunks {
            let buf_off = chunk_i * 8;
            // SAFETY: codes is padded by ≥ 8 bytes past the logical end.
            let buf = u64::from_le_bytes(unsafe {
                codes
                    .get_unchecked(buf_off..buf_off + 8)
                    .try_into()
                    .unwrap_unchecked()
            });
            let comp_chunk = unsafe {
                components.get_unchecked(chunk_i * 64..chunk_i * 64 + 64)
            };

            // 64 single-bit extracts.
            let codes_64: [u8; 64] =
                std::array::from_fn(|i| ((buf >> i) & 0x01) as u8);

            for i in 0..64 {
                let c = unsafe { *comp_chunk.get_unchecked(i) };
                let t = unsafe { *query.get_unchecked(c as usize) };
                acc = acc.algebraic_add(t.algebraic_mul(codes_64[i] as f32));
            }
        }

        // Tail: < 64 codes. Each code is bit (i & 7) of byte (i / 8).
        for i in tail_start..n {
            let byte_off = i / 8;
            let shift = (i & 7) as u8;
            let code = unsafe { (codes.get_unchecked(byte_off) >> shift) & 0x01 };
            let c = unsafe { *components.get_unchecked(i) };
            let t = unsafe { *query.get_unchecked(c as usize) };
            acc = acc.algebraic_add(t.algebraic_mul(code as f32));
        }

        acc
    }

    /// NBITS = 2: 32 codes per 8-byte chunk (4 codes per byte).
    #[inline]
    pub(super) fn dot_product_nbits2(
        components: &[u16],
        codes: &[u8],
        query: &[f32],
    ) -> f32 {
        let n = components.len();
        let chunks = n / 32;
        let tail_start = chunks * 32;
        let mut acc = 0.0f32;

        for chunk_i in 0..chunks {
            let buf_off = chunk_i * 8;
            let buf = u64::from_le_bytes(unsafe {
                codes
                    .get_unchecked(buf_off..buf_off + 8)
                    .try_into()
                    .unwrap_unchecked()
            });
            let comp_chunk = unsafe {
                components.get_unchecked(chunk_i * 32..chunk_i * 32 + 32)
            };

            // 32 two-bit extracts; shifts 0, 2, 4, ..., 62.
            let codes_32: [u8; 32] =
                std::array::from_fn(|i| ((buf >> (i * 2)) & 0x03) as u8);

            for i in 0..32 {
                let c = unsafe { *comp_chunk.get_unchecked(i) };
                let t = unsafe { *query.get_unchecked(c as usize) };
                acc = acc.algebraic_add(t.algebraic_mul(codes_32[i] as f32));
            }
        }

        // Tail: < 32 codes. Each code lives at shift (i & 3) * 2 of byte (i / 4).
        for i in tail_start..n {
            let byte_off = i / 4;
            let shift = ((i & 3) * 2) as u8;
            let code = unsafe { (codes.get_unchecked(byte_off) >> shift) & 0x03 };
            let c = unsafe { *components.get_unchecked(i) };
            let t = unsafe { *query.get_unchecked(c as usize) };
            acc = acc.algebraic_add(t.algebraic_mul(code as f32));
        }

        acc
    }

    /// NBITS = 4 (operating point): 16 codes per 8-byte block, using the
    /// *interleaved* layout written by the encoder.
    ///
    /// Each 8-byte block holds codes 0..7 in the low nibbles of bytes 0..7 and
    /// codes 8..15 in the high nibbles of bytes 0..7. The two-op nibble
    /// extraction now lands directly in natural code order: `buf & 0x0F` is
    /// codes 0..7, `(buf >> 4) & 0x0F` is codes 8..15. No shuffle is required.
    /// The 16 codes are split across two independent 8-wide SIMD pipelines
    /// (`acc_lo` covers positions 0..7, `acc_hi` covers 8..15) so the
    /// dependency chain has length `chunks` rather than `2 * chunks`.
    ///
    /// Per chunk the vector path is:
    /// ```text
    ///   load 8 bytes      → Simd<u8, 8>     (vmovq xmm)
    ///   & 0x0F  / >> 4    → 2 × Simd<u8, 8> (vpand / vpsrlw + vpand)
    ///   cast u8 → f32     → Simd<f32, 8>    × 2  (vpmovzxbd + vcvtdq2ps)
    ///   gather query[c]   → Simd<f32, 8>    × 2  (vpgatherqps / vgatherdps)
    ///   mul_add into acc  → Simd<f32, 8>    × 2  (vfmadd231ps)
    /// ```
    /// The tail (< 16 codes) and very small inputs stay on the scalar path.
    #[inline]
    pub(super) fn dot_product_nbits4(
        components: &[u16],
        codes: &[u8],
        query: &[f32],
    ) -> f32 {
        let n = components.len();
        let chunks = n / 16;
        let tail_start = chunks * 16;

        let mut acc_lo = Simd::<f32, 8>::splat(0.0);
        let mut acc_hi = Simd::<f32, 8>::splat(0.0);

        for chunk_i in 0..chunks {
            let byte_off = chunk_i * 8; // 8 bytes = 1 block = 16 codes
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

            // u8 → f32 widening (shared with dotvbyte_scalaru8).
            let f_lo: Simd<f32, 8> = simd_scalaru8_to_f32(codes_lo);
            let f_hi: Simd<f32, 8> = simd_scalaru8_to_f32(codes_hi);

            // Components for this chunk (16 of them, two 8-wide halves).
            let base = chunk_i * 16;
            let comp_lo: Simd<u16, 8> = Simd::from_slice(unsafe {
                components.get_unchecked(base..base + 8)
            });
            let comp_hi: Simd<u16, 8> = Simd::from_slice(unsafe {
                components.get_unchecked(base + 8..base + 16)
            });

            // Gather query[c] for both halves.
            let q_lo: Simd<f32, 8> = unsafe {
                Simd::gather_select_unchecked(
                    query,
                    Mask::splat(true),
                    comp_lo.cast::<usize>(),
                    Simd::splat(0.0),
                )
            };
            let q_hi: Simd<f32, 8> = unsafe {
                Simd::gather_select_unchecked(
                    query,
                    Mask::splat(true),
                    comp_hi.cast::<usize>(),
                    Simd::splat(0.0),
                )
            };

            // SIMD FMA into the two independent accumulators.
            acc_lo = f_lo.mul_add(q_lo, acc_lo);
            acc_hi = f_hi.mul_add(q_hi, acc_hi);
        }

        let bulk_sum = (acc_lo + acc_hi).reduce_sum();

        // Tail: < 16 codes. Walk per code; same address arithmetic as `iter`.
        let mut tail_acc = 0.0f32;
        for i in tail_start..n {
            let block = i / 16;
            let byte_off = block * 8 + (i & 7);
            let shift = ((i & 8) >> 1) as u8;
            let code = unsafe { (codes.get_unchecked(byte_off) >> shift) & 0x0F };
            let c = unsafe { *components.get_unchecked(i) };
            let t = unsafe { *query.get_unchecked(c as usize) };
            tail_acc = tail_acc.algebraic_add(t.algebraic_mul(code as f32));
        }

        bulk_sum.algebraic_add(tail_acc)
    }

    /// NBITS ∈ {3, 5, 6, 7}: 8 codes per chunk, NBITS bytes per chunk. Codes straddle
    /// bytes, so the bulk path uses an unaligned u64 load + 8 variable shifts; the
    /// tail walks per-code reading a 2-byte window.
    #[inline]
    pub(super) fn dot_product_general<const NBITS: u8>(
        components: &[u16],
        codes: &[u8],
        query: &[f32],
    ) -> f32 {
        let n = components.len();
        let stride = NBITS as usize;
        let mask = (1u64 << NBITS) - 1;

        let chunks = n / 8;
        let tail = n % 8;
        let mut acc = 0.0f32;

        for chunk_i in 0..chunks {
            let code_off = chunk_i * stride;
            let buf = u64::from_le_bytes(unsafe {
                codes
                    .get_unchecked(code_off..code_off + 8)
                    .try_into()
                    .unwrap_unchecked()
            });
            let comp_chunk = unsafe {
                components.get_unchecked(chunk_i * 8..chunk_i * 8 + 8)
            };

            let codes_8: [u8; 8] = std::array::from_fn(|i| {
                ((buf >> (i * NBITS as usize)) & mask) as u8
            });

            for i in 0..8 {
                let c = unsafe { *comp_chunk.get_unchecked(i) };
                let t = unsafe { *query.get_unchecked(c as usize) };
                acc = acc.algebraic_add(t.algebraic_mul(codes_8[i] as f32));
            }
        }

        if tail > 0 {
            let base = chunks * 8;
            let mask_u8 = mask as u8;
            for i in 0..tail {
                let bit_off = (base + i) * NBITS as usize;
                let byte_off = bit_off / 8;
                let shift = (bit_off % 8) as u8;
                let two = u16::from_le_bytes(unsafe {
                    [
                        *codes.get_unchecked(byte_off),
                        *codes.get_unchecked(byte_off + 1),
                    ]
                });
                let code = ((two >> shift) as u8) & mask_u8;
                let c = unsafe { *components.get_unchecked(base + i) };
                let t = unsafe { *query.get_unchecked(c as usize) };
                acc = acc.algebraic_add(t.algebraic_mul(code as f32));
            }
        }

        acc
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
        encoder: &PackedVariableBitUniformSparseQuantizer,
        components: &[u16],
        values: &[f32],
    ) -> Vec<u64> {
        let mut buf: Vec<u64> = Vec::new();
        encoder.push_encoded(SparseVectorView::new(components, values), &mut buf);
        buf
    }

    #[test]
    fn encode_then_decode_roundtrips_components() {
        for nbits in 1..=7u8 {
            let td = build_training_data(
                5,
                &[(&[0, 1, 2, 3, 4], &[0.0; 5]), (&[0, 1, 2, 3, 4], &[10.0; 5])],
            );
            let q = PackedVariableBitUniformSparseQuantizer::train(&td, 0.0, 1.0, nbits);
            let words = encode_to_words(&q, &[1u16, 3], &[5.0_f32, 7.0]);
            let view = PackedVectorView::new(&words);
            let dec = q.decode_vector(view);
            assert_eq!(dec.components(), &[1u16, 3], "nbits={nbits}");
        }
    }

    #[test]
    fn dot_product_matches_dequantized_reference() {
        for nbits in 1..=7u8 {
            let td = build_training_data(
                4,
                &[(&[0, 1, 2, 3], &[0.0; 4]), (&[0, 1, 2, 3], &[10.0; 4])],
            );
            let q = PackedVariableBitUniformSparseQuantizer::train(&td, 0.0, 1.0, nbits);
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
                "nbits={nbits}: got {got}, expected {ref_dot}"
            );
        }
    }

    /// Exercises every kernel's bulk + tail path:
    ///   - NBITS=1 chunks 64 codes; NBITS=2 chunks 32; NBITS=4 chunks 16;
    ///     NBITS ∈ {3,5,6,7} chunks 8.
    /// For each NBITS we encode docs of size {chunk-1, chunk, chunk+1, 200} and verify
    /// the kernel dot product matches the dequantized reference.
    #[test]
    fn dot_product_kernels_match_reference_on_bulk_and_tail() {
        let dim = 256usize;

        // Build a training set spanning the whole component range so per-component
        // quants are non-degenerate for every NBITS.
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

        for nbits in 1..=7u8 {
            let chunk = match nbits {
                1 => 64usize,
                2 => 32,
                4 => 16,
                _ => 8,
            };

            // For nbits=4 the encoder writes in 8-code sub-blocks. Cover every
            // partial-sub-block size (m ∈ 1..=7) plus the bulk/tail boundary so
            // both encoder and kernel exercise all relevant paths.
            let sizes: Vec<usize> = if nbits == 4 {
                (1..=17usize).chain([23, 32, 200]).collect()
            } else {
                vec![chunk.saturating_sub(1).max(1), chunk, chunk + 1, 200]
            };

            for &n in &sizes {
                let q = PackedVariableBitUniformSparseQuantizer::train(&td, 0.0, 1.0, nbits);

                // Components: a strictly-ascending subset of [0, dim). Stride 1 keeps
                // every component < dim regardless of n (n ≤ 200 ≤ dim).
                let comps: Vec<u16> = (0..n as u16).collect();
                let vals: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 + 0.25).collect();

                let words = encode_to_words(&q, &comps, &vals);
                let view = PackedVectorView::new(&words);

                // Build a dense query touching every component (so every code
                // contributes to the dot product).
                let query_components: Vec<u16> = comps.clone();
                let query_values: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.1).collect();
                let query = SparseVectorView::new(&query_components, &query_values);

                let evaluator = q.query_evaluator(query);
                let got = evaluator.compute_distance(view).0;

                // Reference: dequantize and dot directly.
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
                    "nbits={nbits}, n={n}: got {got}, expected {ref_dot} (tol {tol})"
                );
            }
        }
    }

    #[test]
    fn output_blob_is_u64_aligned() {
        let td = build_training_data(
            8,
            &[(&[0, 1, 2, 3, 4, 5, 6, 7], &[10.0; 8])],
        );
        for nbits in 1..=7u8 {
            let q = PackedVariableBitUniformSparseQuantizer::train(&td, 0.0, 1.0, nbits);
            // Use a deliberately awkward number of nonzeros to force padding.
            let words = encode_to_words(&q, &[0u16, 2, 5], &[5.0_f32, 8.0, 1.0]);
            // Round-trip via decode_vector to confirm the layout is sound.
            let dec = q.decode_vector(PackedVectorView::new(&words));
            assert_eq!(dec.components(), &[0u16, 2, 5]);
        }
    }
}
