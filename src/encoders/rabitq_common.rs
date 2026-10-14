//! Shared core of the two RaBitQ encoders: [`RabitqQuantizer`] (1 bit per component) and
//! [`RabitqExtQuantizer`] (`total_bits` per component).
//!
//! The two encoders differ only in how a document is *stored* — a sign code scored with
//! XOR+popcount, versus `total_bits` bit planes scored with AND+popcount. Everything around that
//! is common, and lives here so a change can't land on one path and miss the other:
//!
//! * the geometry — [`RabitqSpace`], the trained means, the rotated means and the rotation `P`,
//!   with the residual `r = P·(x − mean)` and the centroid terms built from them;
//! * the metric abstraction — [`RabitqSupportedDistance`], which decides how a query enters the
//!   rotated space and how an estimated cosine becomes a score;
//! * the query-side scalar quantizer — [`quantize_query_multibit`] and its exact rescale-factor
//!   search [`best_rescale_factor`], plus the plane transpose [`pack_bit_planes`];
//! * the scan kernels — [`hamming`], [`ip_signed_planes`] and their six-way batch forms, and the
//!   scan-ready metadata word ([`pack_metadata`]).
//!
//! See the [`rabitq`](crate::encoders::rabitq) module docs for the estimator math these implement,
//! and [`rabitq_ext`](crate::encoders::rabitq_ext) for the extended-code variant.
//!
//! [`RabitqQuantizer`]: crate::encoders::rabitq::RabitqQuantizer
//! [`RabitqExtQuantizer`]: crate::encoders::rabitq_ext::RabitqExtQuantizer
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::simd::Simd;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdUint;

use crate::core::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use crate::transformations::fht_kac::FhtKacRotator;
use crate::{Dataset, PlainDenseDataset, ScalarDenseSupportedDistance, SpaceUsage};

/// Number of bits packed into a single `u64` word.
pub(crate) const WORD_BITS: usize = 64;

/// Distance metrics that the RaBitQ quantizers can estimate.
///
/// Documents are always stored the same way — the sign code of the centered, rotated residual
/// `r = P·(x − mean)` plus its `(factor, ‖r‖)` metadata. The metric only decides how the *query*
/// enters that space and how the estimated cosine becomes a score:
///
/// * [`SquaredEuclideanDistance`] — the query is centered by the same mean. A shared shift is
///   L2-preserving (`‖x − q‖² = ‖r − r_q‖²`), so the centroid drops out exactly and
///   `est ‖x − q‖² = ‖r‖² + ‖r_q‖² − 2‖r‖·‖r_q‖·cos`.
/// * [`DotProduct`] — centering is **not** inner-product preserving, so the query is only rotated
///   (`q_r = P·q`) and the centroid term is added back in closed form. With `P` orthogonal,
///   `x = mean + Pᵀr`, hence `⟨x, q⟩ = ⟨mean, q⟩ + ⟨r, q_r⟩ ≈ ⟨mean, q⟩ + ‖r‖·‖q_r‖·cos`.
///
/// These mirror RaBitQ-Library's `METRIC_L2` and `METRIC_IP`; [`query_add`] is its per-query
/// `g_add` term (`‖q − c‖²` and `−⟨q, c⟩` respectively, up to our sign convention).
///
/// [`query_add`]: RabitqSupportedDistance::query_add
pub trait RabitqSupportedDistance: Distance {
    /// Whether the query is centered by the trained mean before being rotated.
    const CENTER_QUERY: bool;

    /// The per-query additive term (RaBitQ-Library's `g_add`), from `⟨mean, q⟩` and the rotated
    /// query norm. `mean_dot_query` is only computed (and meaningful) when [`Self::CENTER_QUERY`]
    /// is `false`.
    fn query_add(mean_dot_query: f32, query_norm: f32) -> f32;

    /// Combine the per-query additive `add`, the per-document `f_add = ‖r‖²`, and the shared
    /// estimate `term = ‖r‖·‖q_r‖·cos` into a score. The scan precomputes `term` divide-free from
    /// the stored `s = ‖r‖/factor`; `f_add` is used only by squared Euclidean.
    fn from_terms(add: f32, f_add: f32, term: f32) -> Self;
}

impl RabitqSupportedDistance for SquaredEuclideanDistance {
    const CENTER_QUERY: bool = true;

    #[inline]
    fn query_add(_mean_dot_query: f32, query_norm: f32) -> f32 {
        query_norm * query_norm
    }

    #[inline]
    fn from_terms(add: f32, f_add: f32, term: f32) -> Self {
        SquaredEuclideanDistance::from(f_add + add - 2.0 * term)
    }
}

impl RabitqSupportedDistance for DotProduct {
    const CENTER_QUERY: bool = false;

    #[inline]
    fn query_add(mean_dot_query: f32, _query_norm: f32) -> f32 {
        mean_dot_query
    }

    #[inline]
    fn from_terms(add: f32, _f_add: f32, term: f32) -> Self {
        DotProduct(add + term)
    }
}

/// SIMD lane width for the `f32` sign-packing kernel (one AVX-512 `zmm`).
pub(crate) const FW: usize = 16;

/// Pack the per-document scan constants into the trailing metadata word: `[f_add:f32 | s:f32]`.
///
/// These are the two quantities the scan needs with **no per-candidate divide**:
/// `f_add = ‖r‖²` (the squared-Euclidean additive) and `s = ‖r‖/factor` (which folds the
/// estimator's `/factor` — the residual norm is recoverable as `√f_add` off the hot path).
#[inline]
pub(crate) fn pack_metadata(f_add: f32, s: f32) -> u64 {
    ((f_add.to_bits() as u64) << 32) | (s.to_bits() as u64)
}

/// Unpack the trailing metadata word into `(f_add, s)` (see [`pack_metadata`]).
#[inline]
pub(crate) fn unpack_metadata(word: u64) -> (f32, f32) {
    (
        f32::from_bits((word >> 32) as u32),
        f32::from_bits(word as u32),
    )
}

/// Sign-pack `residual` into `out` (`residual.len() / 64` words, bit set iff `r_i >= 0`).
///
/// Each 64-component word is built from four [`FW`]-lane sign masks instead of a scalar
/// 64-iteration bit test. The single sign-packing kernel: the document paths
/// ([`RabitqQuantizer::encode_into`]) and the 1-bit query path ([`RabitqQuantizer::pack_signs`])
/// both go through here.
#[inline]
pub(crate) fn pack_signs_into(residual: &[f32], out: &mut [u64]) {
    debug_assert_eq!(out.len(), residual.len() / WORD_BITS);
    for (w, out_word) in out.iter_mut().enumerate() {
        let base = w * WORD_BITS;
        let mut word = 0u64;
        for c in 0..WORD_BITS / FW {
            let off = base + c * FW;
            let mask = Simd::<f32, FW>::from_slice(&residual[off..off + FW])
                .simd_ge(Simd::splat(0.0))
                .to_bitmask();
            word |= mask << (c * FW);
        }
        *out_word = word;
    }
}

/// Lane width for the `u64` popcount kernels (8×`u64` = one AVX-512 `zmm` register).
pub(crate) const LANES: usize = 8;

/// Hamming distance between two packed binary code slices: `Σ popcount(a ^ b)`.
///
/// Vectorized over [`LANES`]-wide `u64` chunks (`vpxorq` + `vpopcntq`) with a scalar tail.
#[inline]
pub(crate) fn hamming(a: &[u64], b: &[u64]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    let (ac, at) = a.as_chunks::<LANES>();
    let (bc, bt) = b.as_chunks::<LANES>();
    let mut acc = Simd::<u64, LANES>::splat(0);
    for (x, y) in ac.iter().zip(bc) {
        acc += (Simd::from_array(*x) ^ Simd::from_array(*y)).count_ones();
    }
    let mut sum = acc.reduce_sum();
    for (&x, &y) in at.iter().zip(bt) {
        sum += (x ^ y).count_ones() as u64;
    }
    sum as u32
}

/// Six-way [`hamming`]: XOR+popcount of the same query against six document codes in one
/// interleaved pass.
///
/// Each query chunk is loaded **once** and reused against all six documents' matching chunks, and
/// the six accumulator chains are independent — the cross-candidate ILP that six back-to-back
/// [`hamming`] calls (separate loops the compiler won't interleave) can't get.
#[inline]
pub(crate) fn hamming_batch6(query: &[u64], codes: [&[u64]; 6]) -> [u32; 6] {
    let nw = query.len();
    debug_assert!(codes.iter().all(|c| c.len() == nw));
    let (q_chunks, q_tail) = query.as_chunks::<LANES>();
    let mut acc = [Simd::<u64, LANES>::splat(0); 6];
    for (c, chunk) in q_chunks.iter().enumerate() {
        let qv = Simd::from_array(*chunk);
        let base = c * LANES;
        for (code, a) in codes.iter().zip(acc.iter_mut()) {
            // SAFETY: `base + LANES <= nw == code.len()` (asserted above), so the read of `LANES`
            // contiguous words is in bounds. Skips the bounds check on this hot load.
            let dv = Simd::<u64, LANES>::from_array(unsafe {
                *(code.as_ptr().add(base) as *const [u64; LANES])
            });
            *a += (dv ^ qv).count_ones();
        }
    }
    if !q_tail.is_empty() {
        let base = q_chunks.len() * LANES;
        // Both sides are zero-padded, so the padding lanes XOR to zero and add nothing.
        let qv = Simd::<u64, LANES>::load_or_default(q_tail);
        for (code, a) in codes.iter().zip(acc.iter_mut()) {
            let dv = Simd::<u64, LANES>::load_or_default(&code[base..]);
            *a += (dv ^ qv).count_ones();
        }
    }
    acc.map(|a| a.reduce_sum() as u32)
}

/// Fused multi-bit inner-product kernel: returns `(ip, ppc)` where
/// `ip = Σ_j 2^j · popcount(doc AND plane_j)` and `ppc = popcount(doc)`.
///
/// A single streaming pass over the document code (mirrors RaBitQ-Library's `warmup_ip_x0_q_512`):
/// each `LANES`-wide document chunk is loaded **once** into a register, its popcount folded into
/// `ppc`, then reused across all `QB` query planes — instead of one full pass per plane, which
/// re-reads the document code `QB + 1` times. `QB` is a const generic so the inner plane loop and
/// its per-plane accumulators fully unroll into registers (the C++ template's `acc_bits[b_query]`).
#[inline]
pub(crate) fn ip_signed_planes<const QB: usize>(code: &[u64], planes: &[u64]) -> (u64, u64) {
    let nw = code.len();
    // Layout invariant (upheld by `pack_bit_planes`, which allocates `QB * num_words`): plane `j`
    // occupies `planes[j*nw .. (j+1)*nw]`. Every plane load below is at `j*nw + base` with
    // `base + LANES <= nw`, so `off + LANES <= QB*nw = planes.len()`. This lets the hot loop use
    // unchecked loads (matching the C++ kernel's raw `_mm512_loadu_si512`) — the bounds checks that
    // LLVM would otherwise emit on the computed offset roughly doubled the per-word scan cost.
    debug_assert_eq!(
        planes.len(),
        QB * nw,
        "plane buffer must be QB * code.len()"
    );
    let (code_chunks, code_tail) = code.as_chunks::<LANES>();

    let mut ppc_acc = Simd::<u64, LANES>::splat(0);
    let mut bit_acc = [Simd::<u64, LANES>::splat(0); QB];
    for (c, chunk) in code_chunks.iter().enumerate() {
        let dv = Simd::from_array(*chunk); // load the document chunk once...
        ppc_acc += dv.count_ones();
        let base = c * LANES;
        for (j, acc) in bit_acc.iter_mut().enumerate() {
            // ...and reuse it against every query plane's matching chunk (plane-major layout).
            let off = j * nw + base;
            // SAFETY: `off + LANES <= QB*nw = planes.len()` (see the invariant above); the read of
            // `LANES` contiguous words is in bounds. Skips the bounds check on this hot load.
            let qv = Simd::<u64, LANES>::from_array(unsafe {
                *(planes.as_ptr().add(off) as *const [u64; LANES])
            });
            *acc += (dv & qv).count_ones();
        }
    }

    // Remainder (`nw % LANES` words): pad the document words and each plane's matching words up
    // to a full register and run the same 8-wide popcount. A scalar loop here auto-vectorizes the
    // `QB`-plane inner loop into a strided `vpgatherqq` (the planes for one word sit `nw` apart),
    // which dominates the scan at low dim — e.g. d=128 → nw=2 is *all* tail. The zero padding
    // contributes nothing to the AND-popcount or to `ppc`, and the plane words for the tail are
    // contiguous within each plane, so the load stays a plain register move.
    let tail_len = code_tail.len();
    if tail_len > 0 {
        let tail_base = code_chunks.len() * LANES;
        let dv = Simd::<u64, LANES>::load_or_default(code_tail);
        ppc_acc += dv.count_ones();
        for (j, acc) in bit_acc.iter_mut().enumerate() {
            let off = j * nw + tail_base;
            // SAFETY: `off + tail_len == j*nw + nw <= QB*nw = planes.len()` (see the invariant
            // above), so this sub-`LANES` slice is in bounds. Skips the bounds check on the tail load.
            let qv = Simd::<u64, LANES>::load_or_default(unsafe {
                planes.get_unchecked(off..off + tail_len)
            });
            *acc += (dv & qv).count_ones();
        }
    }

    // Weight each plane by 2^j **in vector form** and sum into one accumulator, so the whole
    // inner product costs a single horizontal reduction instead of one per plane (this is what
    // `warmup_ip_x0_q_512`'s `_mm512_sll_epi64` + one `_mm512_reduce_add_epi64` does; the per-plane
    // `reduce_sum() << j` form put `QB` serialized reductions on the hot path).
    let mut ip_acc = Simd::<u64, LANES>::splat(0);
    for (j, acc) in bit_acc.iter().enumerate() {
        ip_acc += *acc << Simd::<u64, LANES>::splat(j as u64);
    }
    (ip_acc.reduce_sum(), ppc_acc.reduce_sum())
}

/// Six-way [`ip_signed_planes`]: the fused cross-candidate kernel. Returns the per-document
/// `(ip, ppc)` pairs as `([ip; 6], [ppc; 6])`.
///
/// Each query-plane chunk is broadcast **once** per chunk position and reused against six
/// documents' chunks, whose accumulator chains are independent — the cross-candidate ILP that six
/// back-to-back single-document calls can't get. Register pressure is kept flat across `QB` by
/// folding the `2^j` plane weight into the accumulation (`(dv & qv).count_ones() << j`): one `ip`
/// accumulator per document instead of the single-document kernel's per-plane array, which at six
/// documents (`6·QB` registers) would spill for the larger plane counts.
#[inline]
pub(crate) fn ip_signed_planes_batch6<const QB: usize>(
    codes: [&[u64]; 6],
    planes: &[u64],
) -> ([u64; 6], [u64; 6]) {
    let nw = codes[0].len();
    debug_assert!(codes.iter().all(|c| c.len() == nw));
    // Same layout invariant as `ip_signed_planes`: plane `j` occupies `planes[j*nw .. (j+1)*nw]`,
    // so every plane load at `j*nw + base` with `base + LANES <= nw` is in bounds.
    debug_assert_eq!(
        planes.len(),
        QB * nw,
        "plane buffer must be QB * code.len()"
    );
    let full = nw / LANES;
    let tail_len = nw % LANES;

    let mut ppc_acc = [Simd::<u64, LANES>::splat(0); 6];
    let mut ip_acc = [Simd::<u64, LANES>::splat(0); 6];
    for c in 0..full {
        let base = c * LANES;
        // SAFETY: `base + LANES <= nw == codes[k].len()` (asserted above); each read of `LANES`
        // contiguous words is in bounds. Skips the bounds checks on these hot loads.
        let dv: [Simd<u64, LANES>; 6] = std::array::from_fn(|k| {
            Simd::from_array(unsafe { *(codes[k].as_ptr().add(base) as *const [u64; LANES]) })
        });
        for (v, a) in dv.iter().zip(ppc_acc.iter_mut()) {
            *a += v.count_ones();
        }
        for j in 0..QB {
            let off = j * nw + base;
            // SAFETY: `off + LANES <= QB*nw = planes.len()` (see the invariant above).
            let qv = Simd::<u64, LANES>::from_array(unsafe {
                *(planes.as_ptr().add(off) as *const [u64; LANES])
            });
            let shift = Simd::<u64, LANES>::splat(j as u64);
            for (v, a) in dv.iter().zip(ip_acc.iter_mut()) {
                *a += (v & qv).count_ones() << shift;
            }
        }
    }

    // Remainder (`nw % LANES` words): zero-pad both sides up to a full register, as in the
    // single-document kernel; the padding contributes nothing to the AND-popcount or to `ppc`.
    if tail_len > 0 {
        let base = full * LANES;
        let dv: [Simd<u64, LANES>; 6] =
            std::array::from_fn(|k| Simd::load_or_default(&codes[k][base..]));
        for (v, a) in dv.iter().zip(ppc_acc.iter_mut()) {
            *a += v.count_ones();
        }
        for j in 0..QB {
            let off = j * nw + base;
            // SAFETY: `off + tail_len == j*nw + nw <= QB*nw = planes.len()` (see the invariant
            // above), so this sub-`LANES` slice is in bounds.
            let qv = Simd::<u64, LANES>::load_or_default(unsafe {
                planes.get_unchecked(off..off + tail_len)
            });
            let shift = Simd::<u64, LANES>::splat(j as u64);
            for (v, a) in dv.iter().zip(ip_acc.iter_mut()) {
                *a += (v & qv).count_ones() << shift;
            }
        }
    }

    (
        ip_acc.map(|a| a.reduce_sum()),
        ppc_acc.map(|a| a.reduce_sum()),
    )
}

/// Tight lower bounds for the rescale-factor search window, indexed by `ex_bits` (RaBitQ-Library's
/// `kTightStart`): candidate factors below `t_end · TIGHT_START[ex_bits]` never win.
pub(crate) const TIGHT_START: [f64; 9] = [0.0, 0.15, 0.20, 0.52, 0.59, 0.71, 0.75, 0.77, 0.81];

/// Scratch buffers for [`best_rescale_factor`], caller-owned so the exact per-vector search can be
/// driven allocation-free by a caller that encodes many vectors in a row.
#[derive(Default)]
pub(crate) struct RescaleScratch {
    pub(crate) cur_code: Vec<i64>,
    pub(crate) heap: BinaryHeap<Reverse<(u64, usize)>>,
}

/// Exact search for the rescale factor `t` maximizing the cosine between the ex-bit magnitude
/// code (`+0.5`) and `o_abs` (the normalized `|residual|`).
///
/// Port of RaBitQ-Library's `best_rescale_factor`: sweep, in increasing order, exactly the `t`
/// values at which some component's code increments (a min-heap of `(code_i + 1)/o_abs_i`),
/// maintaining the cosine numerator/denominator incrementally. Zero components are skipped —
/// their candidate `t` is infinite and incrementing them can only lower the objective.
/// Must not be called with `ex_bits = 0` (the sweep assumes `max_code ≥ 1`).
///
/// `scratch` is cleared on entry, so a caller can reuse one instance across vectors.
pub(crate) fn best_rescale_factor(
    o_abs: &[f32],
    ex_bits: u32,
    scratch: &mut RescaleScratch,
) -> f64 {
    const EPS: f64 = 1e-5;
    const N_ENUM: usize = 10;
    let dim = o_abs.len();
    let max_o = o_abs.iter().cloned().fold(0.0f32, f32::max) as f64;
    if max_o <= 0.0 {
        return 0.0;
    }
    let max_code = (1i64 << ex_bits) - 1;
    let t_end = (max_code as usize + N_ENUM) as f64 / max_o;
    let t_start = t_end * TIGHT_START[ex_bits as usize];

    let cur_code = &mut scratch.cur_code;
    cur_code.clear();
    cur_code.resize(dim, 0i64);
    let mut sqr_denominator = dim as f64 * 0.25;
    let mut numerator = 0.0f64;
    for (c, &o) in cur_code.iter_mut().zip(o_abs.iter()) {
        *c = ((t_start * o as f64) + EPS) as i64;
        sqr_denominator += (*c * *c + *c) as f64;
        numerator += (*c as f64 + 0.5) * o as f64;
    }

    // Min-heap of candidate `t` values. All candidates are positive finite floats, so ordering
    // their IEEE-754 bit patterns as integers orders the values — no float-Ord wrapper needed.
    let heap = &mut scratch.heap;
    heap.clear();
    for (i, &o) in o_abs.iter().enumerate() {
        if o > 0.0 {
            let t = (cur_code[i] + 1) as f64 / o as f64;
            heap.push(Reverse((t.to_bits(), i)));
        }
    }

    let mut max_ip = 0.0f64;
    let mut best_t = 0.0f64;
    while let Some(Reverse((t_bits, i))) = heap.pop() {
        let cur_t = f64::from_bits(t_bits);
        cur_code[i] += 1;
        let c = cur_code[i];
        sqr_denominator += (2 * c) as f64;
        numerator += o_abs[i] as f64;

        let cur_ip = numerator / sqr_denominator.sqrt();
        if cur_ip > max_ip {
            max_ip = cur_ip;
            best_t = cur_t;
        }

        if c < max_code {
            let t_next = (c + 1) as f64 / o_abs[i] as f64;
            if t_next < t_end {
                heap.push(Reverse((t_next.to_bits(), i)));
            }
        }
    }
    best_t
}

/// Scalar-quantize a query residual to `query_bits` bits per component.
///
/// Returns `(codes, delta, vl)`: per-component codes in `[0, 2^query_bits)` and the
/// reconstruction scale/offset so that `q̂_i = delta·code_i + vl ≈ residual_i`. Follows
/// RaBitQ-Library's RECONSTRUCTION scheme: 1 sign bit plus `query_bits − 1` magnitude bits from
/// [`best_rescale_factor`], negative components complement-coded, and `delta` the least-squares
/// fit of the residual onto the symmetric grid `u_i = code_i + cb`, `cb = −(2^(query_bits−1) − ½)`.
pub(crate) fn quantize_query_multibit(residual: &[f32], query_bits: u32) -> (Vec<u8>, f32, f32) {
    let ex_bits = query_bits - 1;
    let max_code = (1u32 << ex_bits) - 1;
    let cb = -((1u64 << ex_bits) as f64 - 0.5);

    let norm = residual
        .iter()
        .map(|&r| r as f64 * r as f64)
        .sum::<f64>()
        .sqrt();
    if norm <= f64::EPSILON {
        return (vec![0u8; residual.len()], 0.0, 0.0);
    }

    let o_abs: Vec<f32> = residual
        .iter()
        .map(|&r| (r.abs() as f64 / norm) as f32)
        .collect();
    let t = best_rescale_factor(&o_abs, ex_bits, &mut RescaleScratch::default());

    let codes: Vec<u8> = residual
        .iter()
        .zip(o_abs.iter())
        .map(|(&r, &o)| {
            let ex = (((t * o as f64) + 1e-5) as u32).min(max_code);
            let code = if r >= 0.0 {
                (1u32 << ex_bits) | ex
            } else {
                !ex & max_code
            };
            code as u8
        })
        .collect();

    // Least-squares scale of the residual onto the reconstruction grid.
    let mut dot = 0.0f64;
    let mut sqr = 0.0f64;
    for (&r, &c) in residual.iter().zip(codes.iter()) {
        let u = c as f64 + cb;
        dot += r as f64 * u;
        sqr += u * u;
    }
    let delta = if sqr > 0.0 { (dot / sqr) as f32 } else { 0.0 };
    (codes, delta, delta * cb as f32)
}

/// Transpose per-component query codes into **plane-major** bit planes: `planes[j·num_words + w]`
/// holds bit `j` of components `w·64..(w+1)·64`, so each plane `j` is a contiguous
/// `num_words`-long slice and [`ip_signed_planes`] can AND each plane against a document plane in
/// a single fused streaming pass.
pub(crate) fn pack_bit_planes(codes: &[u8], bits: u32) -> Vec<u64> {
    let bits = bits as usize;
    let num_words = codes.len() / WORD_BITS;
    let mut planes = vec![0u64; bits * num_words];
    for w in 0..num_words {
        for i in 0..WORD_BITS {
            let code = codes[w * WORD_BITS + i];
            for j in 0..bits {
                if (code >> j) & 1 == 1 {
                    planes[j * num_words + w] |= 1u64 << i;
                }
            }
        }
    }
    planes
}

/// The trained geometry both RaBitQ encoders share: the per-component means (the centroid
/// replacement), their rotation, and the random orthogonal transform `P` itself.
///
/// A document is always encoded from the residual `r = P·(x − mean)`; what differs between the two
/// encoders is only how that residual is *coded*. Queries enter the same space, centered or not
/// depending on the metric ([`RabitqSupportedDistance::CENTER_QUERY`]).
///
/// Field order is load-bearing for on-disk compatibility: bincode writes struct fields back to back
/// with no names, so embedding this in a quantizer produces the same byte stream as the four
/// inlined fields it replaced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct RabitqSpace {
    /// Input dimensionality (asserted to be a multiple of 64).
    d: usize,
    /// Per-component means used to center vectors (the centroid replacement); length `d`.
    means: Box<[f32]>,
    /// The rotated means `P·mean`, cached at [`Self::train`]. The inner-product build path adds this
    /// to a reconstructed residual to rebuild the un-centered rotated query without re-running the
    /// transform.
    rotated_means: Box<[f32]>,
    /// Fast random orthogonal transform shared by documents and queries; `None` when the encoder was
    /// trained without rotation (identity `P`).
    rotator: Option<FhtKacRotator>,
}

impl RabitqSpace {
    /// Learn the per-component means over `dataset` and build the rotation.
    ///
    /// Panics unless the dataset dimension is a multiple of 64 (the encoders pack sign bits 64 to a
    /// word and the rotator's kernels are written to the same contract).
    pub(crate) fn train<Ds: ScalarDenseSupportedDistance>(
        dataset: &PlainDenseDataset<f32, Ds>,
        rotate: bool,
        seed: u64,
    ) -> Self {
        Self::train_over_values(
            dataset.values(),
            dataset.input_dim(),
            dataset.len(),
            rotate,
            seed,
        )
    }

    /// [`train`](Self::train) for a source held at half precision; values are upcast as they are
    /// summed, so the means are computed in `f32` exactly as for an `f32` source.
    pub(crate) fn train_narrow<V, Ds>(
        dataset: &crate::PlainDenseDataset<V, Ds>,
        rotate: bool,
        seed: u64,
    ) -> Self
    where
        V: crate::ValueType + crate::Float + crate::FromF32,
        Ds: ScalarDenseSupportedDistance,
    {
        Self::train_over_values(
            dataset.values(),
            dataset.input_dim(),
            dataset.len(),
            rotate,
            seed,
        )
    }

    fn train_over_values<V: crate::ValueType>(
        values: &[V],
        d: usize,
        n: usize,
        rotate: bool,
        seed: u64,
    ) -> Self {
        assert!(
            d.is_multiple_of(WORD_BITS),
            "RaBitQ requires dim % 64 == 0, got {d}"
        );

        // Per-component sum as a parallel reduction over vectors (the sequential pass over all n
        // rows was the dominant cost of construction once the per-vector encode was optimized).
        use rayon::prelude::*;
        let mut means = values
            .par_chunks_exact(d)
            .fold(
                || vec![0.0f32; d],
                |mut acc, x| {
                    for (a, v) in acc.iter_mut().zip(x) {
                        *a += v
                            .to_f32()
                            .expect("source value is not representable as f32");
                    }
                    acc
                },
            )
            .reduce(
                || vec![0.0f32; d],
                |mut a, b| {
                    for (x, &y) in a.iter_mut().zip(&b) {
                        *x += y;
                    }
                    a
                },
            );
        if n > 0 {
            let inv = 1.0 / n as f32;
            for m in means.iter_mut() {
                *m *= inv;
            }
        }

        let rotator = rotate.then(|| FhtKacRotator::new(d, seed));
        let rotated_means = match &rotator {
            Some(rotator) => rotator.rotate(&means),
            None => means.clone(),
        };

        Self {
            d,
            means: means.into_boxed_slice(),
            rotated_means: rotated_means.into_boxed_slice(),
            rotator,
        }
    }

    /// Input dimensionality.
    #[inline]
    pub(crate) fn dim(&self) -> usize {
        self.d
    }

    /// Number of `u64` words holding one bit per component (`d / 64`).
    #[inline]
    pub(crate) fn num_words(&self) -> usize {
        self.d / WORD_BITS
    }

    /// The trained per-component means. Only the tests need them directly: the production paths
    /// go through [`Self::residual`], [`Self::residual_into`] and [`Self::mean_dot`].
    #[cfg(test)]
    #[inline]
    pub(crate) fn means(&self) -> &[f32] {
        &self.means
    }

    /// The rotated means `P·mean`.
    #[inline]
    pub(crate) fn rotated_means(&self) -> &[f32] {
        &self.rotated_means
    }

    /// Apply the rotation: `P·v` (a copy of `values` when rotation is disabled, i.e. `P = I`).
    pub(crate) fn rotate(&self, values: &[f32]) -> Vec<f32> {
        match &self.rotator {
            Some(rotator) => rotator.rotate(values),
            None => values.to_vec(),
        }
    }

    /// Center `values` by the means and apply the rotation: `r = P·(x − mean)`.
    ///
    /// Documents always take this path; queries only do so under a centering metric
    /// (see [`RabitqSupportedDistance::CENTER_QUERY`]).
    pub(crate) fn residual(&self, values: &[f32]) -> Vec<f32> {
        let centered: Vec<f32> = values
            .iter()
            .zip(self.means.iter())
            .map(|(&v, &m)| v - m)
            .collect();
        self.rotate(&centered)
    }

    /// [`Self::residual`] into a caller-owned buffer (length `d`), so the batch encode paths reuse
    /// one scratch per worker and allocate nothing per vector.
    #[inline]
    pub(crate) fn residual_into(&self, values: &[f32], scratch: &mut [f32]) {
        debug_assert_eq!(values.len(), self.d);
        debug_assert_eq!(scratch.len(), self.d);
        for ((s, &v), &m) in scratch.iter_mut().zip(values).zip(self.means.iter()) {
            *s = v - m;
        }
        if let Some(rotator) = &self.rotator {
            rotator.rotate_inplace(scratch);
        }
    }

    /// The exact inverse of [`Self::residual`]: `x = mean + P⁻¹·r`.
    ///
    /// Used to lift a reconstruction out of the rotated, centered residual space and back into the
    /// original data space, which is what residual reranking needs — the second stage has to encode
    /// `x − x̂`, and both terms must live in the same space for the subtraction to mean anything.
    #[allow(clippy::wrong_self_convention)]
    pub(crate) fn from_residual(&self, residual: &[f32]) -> Vec<f32> {
        debug_assert_eq!(residual.len(), self.d);
        let mut v = match &self.rotator {
            Some(rotator) => rotator.rotate_inverse(residual),
            None => residual.to_vec(),
        };
        for (x, &m) in v.iter_mut().zip(self.means.iter()) {
            *x += m;
        }
        v
    }

    /// `⟨mean, q⟩` — the centroid term the inner-product metric adds back.
    pub(crate) fn mean_dot(&self, values: &[f32]) -> f32 {
        self.means
            .iter()
            .zip(values.iter())
            .map(|(&m, &v)| m * v)
            .sum()
    }

    /// `⟨P·mean, q_r⟩` — the same centroid term computed on the *rotated* pair. Equal to
    /// [`Self::mean_dot`] on the un-rotated pair because `P` is orthogonal, and it is the only form
    /// available on the build path, where the un-rotated query was never materialized.
    pub(crate) fn rotated_mean_dot(&self, rotated: &[f32]) -> f32 {
        self.rotated_means
            .iter()
            .zip(rotated.iter())
            .map(|(&m, &v)| m * v)
            .sum()
    }
}

impl SpaceUsage for RabitqSpace {
    fn space_usage_bytes(&self) -> usize {
        self.d.space_usage_bytes()
            + self.means.space_usage_bytes()
            + self.rotated_means.space_usage_bytes()
            + self.rotator.space_usage_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vector::DenseVectorView;
    use crate::{DatasetGrowable, PlainDenseDatasetGrowable, PlainDenseQuantizer};

    /// Pins the on-disk layout claim in [`RabitqSpace`]'s docs: bincode writes struct fields back to
    /// back with no names, so a quantizer embedding this struct serializes to exactly the bytes the
    /// four previously-inlined fields produced. A reordered or added field would break every stored
    /// index, silently and only at load time — this test fails instead.
    #[test]
    fn space_serializes_as_its_four_fields_in_order() {
        let d = 128;
        let encoder = PlainDenseQuantizer::<f32, DotProduct>::new(d);
        let mut growable = PlainDenseDatasetGrowable::new(encoder);
        for k in 0..4 {
            let v: Vec<f32> = (0..d)
                .map(|i| ((i * (k + 1)) as f32 * 0.21).sin())
                .collect();
            growable.push(DenseVectorView::new(&v));
        }
        let space = RabitqSpace::train(&growable.into(), true, 7);

        let cfg = bincode::config::standard();
        let as_struct = bincode::serde::encode_to_vec(&space, cfg).unwrap();
        let as_fields = bincode::serde::encode_to_vec(
            (space.d, &space.means, &space.rotated_means, &space.rotator),
            cfg,
        )
        .unwrap();
        assert_eq!(as_struct, as_fields);
    }
}
