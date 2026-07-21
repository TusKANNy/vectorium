//! Fast Hadamard–Kac random orthogonal rotation (RaBitQ-Library's `FhtKacRotator`).
//!
//! Operates on `f32` vectors: dense-encoder queries are always `f32`
//! ([`DenseVectorEncoder::QueryVector`]), so a pre-quantization transform rotates `f32` data
//! regardless of the encoder's input/output value types.
//!
//! [`DenseVectorEncoder::QueryVector`]: crate::core::vector_encoder::DenseVectorEncoder
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::simd::Simd;
use std::simd::num::SimdFloat;
use std::simd::simd_swizzle;

use crate::SpaceUsage;

/// Number of bits packed into a single `u64` word (the rotator requires `d % 64 == 0`).
const WORD_BITS: usize = 64;

/// A fast random orthogonal transform — RaBitQ-Library's `FhtKacRotator`.
///
/// Replaces the dense `d×d` matrix rotation (`O(d²)` per vector, `O(d³)` to build) with an
/// `O(d log d)` transform that stores only `4·d/8` random bytes. Four rounds of
///
/// 1. **sign flip** — negate each coordinate whose random `flip` bit is set (an orthonormal ±1
///    diagonal);
/// 2. **Fast Hadamard Transform** over the largest power-of-two prefix `trunc_dim = 2^⌊log₂ d⌋`,
///    then rescale by `1/√trunc_dim` (orthonormal on that block, identity elsewhere);
/// 3. **Kac's walk** — the butterfly `(x, y) → (x+y, x−y)` across the two halves (only when
///    `d` is not itself a power of two), which mixes the tail the FHT skipped.
///
/// Each Kac's walk scales the norm by `√2`; the four of them are undone by a final global `×0.25`,
/// so the whole transform is orthonormal (norm- and inner-product-preserving) up to `f32` rounding
/// — all the RaBitQ estimator requires of `P`. The library seeds `flip` from `random_device`; we
/// seed from a caller-provided seed via `StdRng` so the rotation is reproducible.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FhtKacRotator {
    /// Dimensionality (a multiple of 64); the library's `padded_dim`, here equal to `dim`.
    d: usize,
    /// Largest power of two `≤ d`; the FHT block length.
    trunc_dim: usize,
    /// FHT normalization `1/√trunc_dim`.
    fac: f32,
    /// Random sign-flip bits, `4·d/8` bytes — `d/8` per round, bit `i` flips coordinate `i`.
    flip: Box<[u8]>,
}

impl FhtKacRotator {
    /// Draw the random sign-flip bits from `seed` and precompute the FHT block size.
    pub fn new(d: usize, seed: u64) -> Self {
        debug_assert!(d.is_multiple_of(WORD_BITS));
        let trunc_dim = 1usize << d.ilog2();
        let fac = 1.0 / (trunc_dim as f32).sqrt();
        let mut flip = vec![0u8; 4 * d / 8].into_boxed_slice();
        StdRng::seed_from_u64(seed).fill(&mut flip[..]);
        Self {
            d,
            trunc_dim,
            fac,
            flip,
        }
    }

    /// Apply `P·src`, returning the rotated vector (length `d`).
    ///
    /// Mirrors `FhtKacRotator::rotate`: when `d` is a power of two the FHT covers the whole vector
    /// and no Kac's walk is needed; otherwise the FHT alternates between the head (`[0, trunc_dim)`)
    /// and tail (`[d−trunc_dim, d)`) blocks with a Kac's walk after every round.
    pub fn rotate(&self, src: &[f32]) -> Vec<f32> {
        let mut v = src.to_vec();
        self.rotate_inplace(&mut v);
        v
    }

    /// In-place `P·v`, so hot encode paths can rotate a reused scratch buffer instead of
    /// allocating a fresh `Vec` per vector.
    pub fn rotate_inplace(&self, v: &mut [f32]) {
        let (d, td, round) = (self.d, self.trunc_dim, self.d / 8);
        debug_assert_eq!(v.len(), d);
        if td == d {
            for r in 0..4 {
                flip_sign(&self.flip[r * round..], v);
                fht(&mut v[..td]);
                rescale(v, self.fac);
            }
        } else {
            let start = d - td;
            for r in 0..4 {
                flip_sign(&self.flip[r * round..], v);
                // Rounds alternate the FHT block between the head and the tail of the vector.
                let off = if r % 2 == 0 { 0 } else { start };
                fht(&mut v[off..off + td]);
                rescale(&mut v[off..off + td], self.fac);
                kacs_walk(v);
            }
            // Undo the four `√2` norm gains from the Kac's walks.
            rescale(v, 0.25);
        }
    }
}

impl SpaceUsage for FhtKacRotator {
    fn space_usage_bytes(&self) -> usize {
        self.d.space_usage_bytes()
            + self.trunc_dim.space_usage_bytes()
            + std::mem::size_of::<f32>()
            + self.flip.space_usage_bytes()
    }
}

/// SIMD lane width for the `f32` rotation kernels (one AVX-512 `zmm`).
const FW: usize = 16;

/// Negate every `data[i]` whose bit `i` in `flip` is set (an orthonormal ±1 diagonal).
///
/// Each 16-float chunk pulls its 16 flip bits as one `u16` and turns them into a lane mask, so the
/// negation is a masked select (`data.len()` is a multiple of 64, so no scalar tail).
#[inline]
fn flip_sign(flip: &[u8], data: &mut [f32]) {
    let (chunks, tail) = data.as_chunks_mut::<FW>();
    let lane = Simd::<u32, FW>::from_array(std::array::from_fn(|i| i as u32));
    for (c, chunk) in chunks.iter_mut().enumerate() {
        let bits = u32::from(u16::from_le_bytes([flip[2 * c], flip[2 * c + 1]]));
        // Lanes selected by the flip bits get their sign bit set (`x → −x`): move bit `i` of
        // `bits` up to bit 31 of lane `i`, then XOR it into the float's sign bit.
        let sel = ((Simd::splat(bits) >> lane) & Simd::splat(1)) << Simd::splat(31);
        let v = Simd::<f32, FW>::from_array(*chunk).to_bits() ^ sel;
        *chunk = Simd::<f32, FW>::from_bits(v).to_array();
    }
    let base = chunks.len() * FW;
    for (k, x) in tail.iter_mut().enumerate() {
        let i = base + k;
        if (flip[i / 8] >> (i % 8)) & 1 == 1 {
            *x = -*x;
        }
    }
}

/// Scale `data` in place by `fac`.
#[inline]
fn rescale(data: &mut [f32], fac: f32) {
    let (chunks, tail) = data.as_chunks_mut::<FW>();
    let f = Simd::<f32, FW>::splat(fac);
    for c in chunks.iter_mut() {
        *c = (Simd::from_array(*c) * f).to_array();
    }
    for x in tail {
        *x *= fac;
    }
}

/// One Kac's-walk butterfly across the halves: `(x, y) → (x+y, x−y)`. Scales the norm by `√2`.
/// `data.len()` is a multiple of 64, so each half is a whole number of 16-wide chunks.
#[inline]
fn kacs_walk(data: &mut [f32]) {
    let (a, b) = data.split_at_mut(data.len() / 2);
    let (ac, _) = a.as_chunks_mut::<FW>();
    let (bc, _) = b.as_chunks_mut::<FW>();
    for (x, y) in ac.iter_mut().zip(bc.iter_mut()) {
        let (xv, yv) = (Simd::from_array(*x), Simd::from_array(*y));
        *x = (xv + yv).to_array();
        *y = (xv - yv).to_array();
    }
}

/// The four sub-register FHT butterfly passes (`len ∈ {1,2,4,8}`) fused into one in-register
/// Hadamard-16. Each pass at stride `s` combines every lane with its partner `l ^ s`: the low lane
/// (bit `s` clear) gets `x + y`, the high lane gets `x − y`, i.e. `swizzle(v, l^s) + sign_s · v`
/// with `sign_s = −1` exactly on the lanes whose index has bit `s` set. Butterfly passes at
/// distinct strides act on independent index bits, so they commute and may run before the
/// inter-block passes without changing the transform.
#[inline]
fn hadamard16(v: Simd<f32, FW>) -> Simd<f32, FW> {
    let s1 = Simd::from_array([
        1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
    ]);
    let s2 = Simd::from_array([
        1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0,
    ]);
    let s4 = Simd::from_array([
        1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0,
    ]);
    let s8 = Simd::from_array([
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
    ]);
    let v = simd_swizzle!(v, [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]) + s1 * v;
    let v = simd_swizzle!(v, [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13]) + s2 * v;
    let v = simd_swizzle!(v, [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11]) + s4 * v;
    simd_swizzle!(v, [8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7]) + s8 * v
}

/// In-place unnormalized Fast Hadamard Transform over a power-of-two-length slice (`O(n log n)`).
///
/// The sub-register strides (`len ∈ {1,2,4,8}`) run as one in-register [`hadamard16`] per 16-wide
/// chunk; the remaining strides (`len ≥ 16`) are contiguous SIMD butterflies over whole chunks. In
/// the encoder `n` is always a multiple of 64 (the rotator pads to that), so the sub-16 scalar
/// fallback only ever runs for tiny transforms in unit tests.
#[inline]
fn fht(a: &mut [f32]) {
    let n = a.len();
    debug_assert!(n.is_power_of_two());
    if n < FW {
        let mut len = 1;
        while len < n {
            let mut i = 0;
            while i < n {
                let (lo, hi) = a[i..i + 2 * len].split_at_mut(len);
                for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                    (*x, *y) = (*x + *y, *x - *y);
                }
                i += 2 * len;
            }
            len <<= 1;
        }
        return;
    }

    // Strides 1..8, fused per 16-wide chunk (n is a multiple of 16 since it is a power of two ≥ 16).
    let (chunks, _) = a.as_chunks_mut::<FW>();
    for c in chunks.iter_mut() {
        *c = hadamard16(Simd::from_array(*c)).to_array();
    }

    // Strides 16..n/2: contiguous SIMD butterflies (both halves are whole chunks).
    let mut len = FW;
    while len < n {
        let mut i = 0;
        while i < n {
            let (lo, hi) = a[i..i + 2 * len].split_at_mut(len);
            let (loc, _) = lo.as_chunks_mut::<FW>();
            let (hic, _) = hi.as_chunks_mut::<FW>();
            for (x, y) in loc.iter_mut().zip(hic.iter_mut()) {
                let (xv, yv) = (Simd::from_array(*x), Simd::from_array(*y));
                *x = (xv + yv).to_array();
                *y = (xv - yv).to_array();
            }
            i += 2 * len;
        }
        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fht_kac_rotation_is_orthonormal() {
        // `d = 192` is a multiple of 64 but not a power of two, so `trunc_dim = 128` and the
        // Kac's-walk path runs — the case that must still come out orthonormal. Rotating each
        // basis vector yields a column of `P`; orthonormal columns ⇒ `PᵀP = I`.
        let d = 192;
        let rot = FhtKacRotator::new(d, 7);
        let cols: Vec<Vec<f32>> = (0..d)
            .map(|i| {
                let mut e = vec![0.0f32; d];
                e[i] = 1.0;
                rot.rotate(&e)
            })
            .collect();
        for i in 0..d {
            for j in i..d {
                let dot: f32 = cols[i].iter().zip(&cols[j]).map(|(&a, &b)| a * b).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-4,
                    "cols {i},{j}: dot = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn rotation_preserves_norm() {
        // Both the power-of-two (pure FHT) and the Kac's-walk paths must preserve the norm.
        for d in [128usize, 192] {
            let rot = FhtKacRotator::new(d, 3);
            let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.37).sin() * 2.0).collect();
            let y = rot.rotate(&x);
            let nx: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
            let ny: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((nx - ny).abs() < 1e-3, "d={d}: norms differ: {nx} vs {ny}");
        }
    }
}
