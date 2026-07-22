//! RaBitQ-style binary encoder (no centroids).
//!
//! Implements the centroid-free subset of *RaBitQ: Quantizing High-Dimensional Vectors with a
//! Theoretical Error Bound for ANN* (Gao & Long, SIGMOD 2024). Documents are quantized to
//! **1 bit per component**; the query is either sign-binarized (`query_bits = 1`) or
//! scalar-quantized to `query_bits` bits per component.
//!
//! Documents are preprocessed by subtracting the per-component dataset means (the centroid
//! replacement) and applying a seeded random orthogonal transform `P` (a fast Hadamard–Kac
//! rotation, [`FhtKacRotator`]), giving the residual `r = P·(x − mean)`. The rotation spreads
//! information uniformly across dimensions before the sign is taken. It can be disabled via
//! [`RabitqConfig::rotate`] (then `P = I`), which skips the per-vector transform at encode and
//! query time at the cost of the information-spreading guarantee.
//!
//! Each document stores its sign code (bit set iff `r_i >= 0`) plus two floats:
//!
//! * `factor = ⟨ō,o⟩ = Σ|r_i| / (√d·‖r‖)` — the cosine between the quantized code and the true
//!   residual; dividing the raw sign agreement by it gives the paper's unbiased estimator.
//! * `‖r‖` — the residual norm, which turns the estimated cosine into a distance.
//!
//! ## Metrics
//!
//! Both squared Euclidean and inner product are supported, selected by the type parameter on
//! [`RabitqQuantizer`] (see [`RabitqSupportedDistance`]). Document codes are **identical** under
//! either metric; only the query path and the final combine differ, so the choice is free at
//! index-build time:
//!
//! * [`SquaredEuclideanDistance`] — the query is centered by the same mean. A shared shift is
//!   L2-preserving (`‖(c+r) − (c+r_q)‖² = ‖r − r_q‖²`), so the centroid cancels identically.
//! * [`DotProduct`] — centering is *not* inner-product preserving (a per-document `⟨x, mean⟩`
//!   term survives), so the query is rotated but **not** centered, and the centroid term is added
//!   back in closed form: since `P` is orthogonal, `x = mean + Pᵀr` and
//!   `⟨x, q⟩ = ⟨mean, q⟩ + ⟨r, P·q⟩`, where `⟨mean, q⟩` is a per-query constant.
//!
//! Write `q_r` for the rotated query (`P·(q − mean)` or `P·q` per the above) and `‖r_q‖ = ‖q_r‖`.
//!
//! ## Scoring — 1-bit query
//!
//! With both sides sign-packed to unit bi-valued codes (`±1/√d`), `⟨ō, q̄⟩ = (d − 2·hamming)/d`.
//! The query-side correction `⟨q̄,q⟩` is a per-query constant and cannot change the ranking, so
//! only document-side terms appear:
//!
//! ```text
//! ip_bar = (d − 2·hamming) / d                     # sign agreement, in [-1, 1]
//! cos    = ip_bar / factor                         # unbiased cosine estimate
//! ⟨r,q_r⟩ ≈ ‖r‖·‖r_q‖·cos                          # the shared estimate
//!
//! # squared Euclidean (smaller = nearer)
//! est ‖x − q‖² = ‖r‖² + ‖r_q‖² − 2·‖r‖·‖r_q‖·cos
//! # inner product (larger = nearer)
//! est ⟨x, q⟩   = ⟨mean, q⟩ + ‖r‖·‖r_q‖·cos
//! ```
//!
//! These are RaBitQ-Library's `METRIC_L2` and `METRIC_IP`. Its
//! `est_dist = f_add + g_add + f_rescale·(⟨s,q̂⟩ − correction)` maps onto the L2 line with
//! `f_add = ‖r‖²`, `g_add = ‖r_q‖²`, `f_rescale = −2‖r‖²/Σ|r_i|`; its `g_add = −⟨q,c⟩` maps onto
//! the inner-product line. The library always carries a raw, un-centered query and re-adds the
//! centroid; we do the same for inner product, and center symmetrically for L2 where it cancels.
//!
//! ## Scoring — multi-bit query (`query_bits > 1`)
//!
//! The query residual is scalar-quantized (RaBitQ-Library's RECONSTRUCTION scheme: sign bit +
//! magnitude bits with an exact rescale-factor search, complement-coded negatives) and
//! reconstructs as `q̂_i = delta·code_i + vl ≈ r_i`. With doc code bits `b_i ∈ {0,1}`
//! (signs `s_i = 2·b_i − 1`):
//!
//! ```text
//! ip     = Σ_j 2^j · popcount(doc AND plane_j)                  # ⟨b, code⟩ via bit planes
//! ⟨s,q̂⟩  = 2·(delta·ip + vl·popcount(doc)) − (delta·Σcode + vl·d)
//! ip_bar = ⟨s,q̂⟩ / (√d·‖r_q‖)                                   # ≈ ⟨ō, q/‖q‖⟩
//! ```
//!
//! `ip_bar` then flows through the same factor/norm/metric steps above. Documents are
//! byte-identical either way, so 1-bit vs multi-bit query is a pure query-side choice on the same
//! index.
//!
//! ## Storage
//!
//! Each document is `d/64` sign-code words plus **one metadata word** holding two per-document
//! floats bit-cast side by side, i.e. `d/64 + 1` words. The stored pair is the scan-ready
//! `[f_add:f32 | s:f32]` — `f_add = ‖r‖²` and `s = ‖r‖/factor` — precomputed so the per-candidate
//! scan needs no divide (`factor` and `‖r‖ = √f_add` are recoverable off the hot path).
//!
//! Only dimensions that are a multiple of 64 are supported (no bit-padding).
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::marker::PhantomData;
use std::simd::Simd;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdUint;

use crate::core::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use crate::core::vector::{DenseVectorOwned, DenseVectorView};
use crate::core::vector_encoder::{DenseVectorEncoder, QueryEvaluator, VectorEncoder};
use crate::transformations::fht_kac::FhtKacRotator;
use crate::{Dataset, PlainDenseDataset, ScalarDenseSupportedDistance, SpaceUsage};

/// Number of bits packed into a single `u64` word.
const WORD_BITS: usize = 64;

/// RaBitQ encoder parameters. See the module docs for the estimator math.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RabitqConfig {
    /// Bits per component for the scalar-quantized query (`1` = plain sign query). Values in
    /// `1..=8` are supported. Documents are always 1 bit per component.
    pub query_bits: u32,
    /// Seed for the random orthogonal rotation ([`FhtKacRotator`]).
    pub seed: u64,
    /// Apply the random orthogonal rotation ([`FhtKacRotator`]) to residuals. Disabling skips the
    /// `O(d log d)` per-vector transform at encode and query time; the estimator math is unchanged
    /// (`P = I` is a valid orthogonal transform), but the sign codes lose the
    /// information-spreading guarantee, so recall on real data is expected to drop.
    pub rotate: bool,
}

impl Default for RabitqConfig {
    fn default() -> Self {
        Self {
            query_bits: 1,
            seed: 42,
            rotate: true,
        }
    }
}

/// Distance metrics that the [`RabitqQuantizer`] can estimate.
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
const FW: usize = 16;

/// Pack the per-document scan constants into the trailing metadata word: `[f_add:f32 | s:f32]`.
///
/// These are the two quantities the scan needs with **no per-candidate divide**:
/// `f_add = ‖r‖²` (the squared-Euclidean additive) and `s = ‖r‖/factor` (which folds the
/// estimator's `/factor` — the residual norm is recoverable as `√f_add` off the hot path).
#[inline]
fn pack_metadata(f_add: f32, s: f32) -> u64 {
    ((f_add.to_bits() as u64) << 32) | (s.to_bits() as u64)
}

/// Unpack the trailing metadata word into `(f_add, s)` (see [`pack_metadata`]).
#[inline]
fn unpack_metadata(word: u64) -> (f32, f32) {
    (
        f32::from_bits((word >> 32) as u32),
        f32::from_bits(word as u32),
    )
}

/// Lane width for the `u64` popcount kernels (8×`u64` = one AVX-512 `zmm` register).
const LANES: usize = 8;

/// Hamming distance between two packed binary code slices: `Σ popcount(a ^ b)`.
///
/// Vectorized over [`LANES`]-wide `u64` chunks (`vpxorq` + `vpopcntq`) with a scalar tail.
#[inline]
fn hamming(a: &[u64], b: &[u64]) -> u32 {
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

/// Fused multi-bit inner-product kernel: returns `(ip, ppc)` where
/// `ip = Σ_j 2^j · popcount(doc AND plane_j)` and `ppc = popcount(doc)`.
///
/// A single streaming pass over the document code (mirrors RaBitQ-Library's `warmup_ip_x0_q_512`):
/// each `LANES`-wide document chunk is loaded **once** into a register, its popcount folded into
/// `ppc`, then reused across all `QB` query planes — instead of one full pass per plane, which
/// re-reads the document code `QB + 1` times. `QB` is a const generic so the inner plane loop and
/// its per-plane accumulators fully unroll into registers (the C++ template's `acc_bits[b_query]`).
#[inline]
fn ip_signed_planes<const QB: usize>(code: &[u64], planes: &[u64]) -> (u64, u64) {
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

/// Tight lower bounds for the rescale-factor search window, indexed by `ex_bits` (RaBitQ-Library's
/// `kTightStart`): candidate factors below `t_end · TIGHT_START[ex_bits]` never win.
const TIGHT_START: [f64; 9] = [0.0, 0.15, 0.20, 0.52, 0.59, 0.71, 0.75, 0.77, 0.81];

/// Exact search for the rescale factor `t` maximizing the cosine between the ex-bit magnitude
/// code (`+0.5`) and `o_abs` (the normalized `|residual|`).
///
/// Port of RaBitQ-Library's `best_rescale_factor`: sweep, in increasing order, exactly the `t`
/// values at which some component's code increments (a min-heap of `(code_i + 1)/o_abs_i`),
/// maintaining the cosine numerator/denominator incrementally. Zero components are skipped —
/// their candidate `t` is infinite and incrementing them can only lower the objective.
fn best_rescale_factor(o_abs: &[f32], ex_bits: u32) -> f64 {
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

    let mut cur_code = vec![0i64; dim];
    let mut sqr_denominator = dim as f64 * 0.25;
    let mut numerator = 0.0f64;
    for (c, &o) in cur_code.iter_mut().zip(o_abs.iter()) {
        *c = ((t_start * o as f64) + EPS) as i64;
        sqr_denominator += (*c * *c + *c) as f64;
        numerator += (*c as f64 + 0.5) * o as f64;
    }

    // Min-heap of candidate `t` values. All candidates are positive finite floats, so ordering
    // their IEEE-754 bit patterns as integers orders the values — no float-Ord wrapper needed.
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
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
fn quantize_query_multibit(residual: &[f32], query_bits: u32) -> (Vec<u8>, f32, f32) {
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
    let t: f64 = best_rescale_factor(&o_abs, ex_bits);

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

/// Transpose per-component codes into **plane-major** query bit planes: `planes[j·num_words + w]`
/// holds bit `j` of components `w·64..(w+1)·64`, so each plane `j` is a contiguous
/// `num_words`-long slice, so [`ip_signed_planes`] can AND each plane against the document code in
/// a single fused streaming pass.
fn pack_bit_planes(codes: &[u8], bits: u32) -> Vec<u64> {
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

/// RaBitQ-style binary quantizer. See the module docs.
///
/// The metric `D` selects how the stored residual codes are scored against a query; it defaults
/// to [`DotProduct`] (inner product). Document codes are identical under either metric — only the
/// query path and the final combine differ, so `D` is pure type-level state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RabitqQuantizer<D = DotProduct> {
    /// Input dimensionality (asserted to be a multiple of 64).
    d: usize,
    /// Per-component means used to center vectors (the centroid replacement); length `d`.
    means: Box<[f32]>,
    /// Fast random orthogonal transform shared by documents and queries; `None` when the encoder
    /// was trained with `config.rotate == false` (identity `P`).
    rotator: Option<FhtKacRotator>,
    /// Encoder parameters.
    config: RabitqConfig,
    /// The metric is encoded purely in the type; no runtime state.
    _distance: PhantomData<D>,
}

impl<D: RabitqSupportedDistance> RabitqQuantizer<D> {
    /// The number of bits currently used to quantize the query.
    #[inline]
    pub fn query_bits(&self) -> u32 {
        self.config.query_bits
    }

    /// Set the number of bits used to quantize the *query*.
    ///
    /// Document codes do not depend on this value, so it is safe to change on an already-encoded
    /// dataset: it trades scan cost for estimate accuracy without re-encoding. A single index can
    /// therefore serve every `query_bits` setting.
    ///
    /// Panics unless `query_bits ∈ 1..=8`.
    #[inline]
    pub fn set_query_bits(&mut self, query_bits: u32) {
        assert!(
            (1..=8).contains(&query_bits),
            "RabitqQuantizer requires query_bits in 1..=8, got {query_bits}"
        );
        self.config.query_bits = query_bits;
    }

    /// Learn per-component means over `dataset` and build the rotation.
    ///
    /// Panics unless the dataset dimension is a multiple of 64.
    pub fn train<Ds: ScalarDenseSupportedDistance>(
        dataset: &PlainDenseDataset<f32, Ds>,
        config: RabitqConfig,
    ) -> Self {
        let d = dataset.input_dim();
        assert!(
            d.is_multiple_of(WORD_BITS),
            "RabitqQuantizer requires dim % 64 == 0, got {d}"
        );
        assert!(
            (1..=8).contains(&config.query_bits),
            "RabitqQuantizer requires 1 <= query_bits <= 8, got {}",
            config.query_bits
        );

        // Per-component sum as a parallel reduction over vectors (the sequential pass over all n
        // rows was the dominant cost of construction once the per-vector encode was optimized).
        use rayon::prelude::*;
        let mut means = dataset
            .values()
            .par_chunks_exact(d)
            .fold(
                || vec![0.0f32; d],
                |mut acc, x| {
                    for (a, &v) in acc.iter_mut().zip(x) {
                        *a += v;
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
        let n = dataset.len();
        if n > 0 {
            let inv = 1.0 / n as f32;
            for m in means.iter_mut() {
                *m *= inv;
            }
        }

        let rotator = config.rotate.then(|| FhtKacRotator::new(d, config.seed));

        Self {
            d,
            means: means.into_boxed_slice(),
            rotator,
            config,
            _distance: PhantomData,
        }
    }

    /// Train on `dataset` and encode every vector in parallel.
    ///
    /// The `ConvertFrom` idiom used by the other binary encoders can't carry a config, so this
    /// helper is the entry point for building a RaBitQ dataset.
    pub fn encode_dataset<Ds: ScalarDenseSupportedDistance>(
        dataset: &PlainDenseDataset<f32, Ds>,
        config: RabitqConfig,
    ) -> crate::DenseDataset<Self> {
        let encoder = Self::train(dataset, config);
        encoder.encode_flat_par(dataset.values(), dataset.len())
    }

    /// Encode `n_vecs` flat row-major `f32` vectors with an already-trained encoder, in parallel.
    ///
    /// Split out from [`Self::encode_dataset`] so the construction cost (the per-vector rotation)
    /// can be benchmarked without re-running [`Self::train`]. Encodes straight into the final slab:
    /// each output record is written in place, so there is no `Vec<Vec>` of per-vector allocations
    /// and no reassembly copy (what `from_flat_par` does). A per-worker scratch buffer holds
    /// `r = P·(x − mean)`, reused across every vector that worker sees, so the rotation and
    /// sign-packing allocate nothing on the hot path.
    pub fn encode_flat_par(self, input: &[f32], n_vecs: usize) -> crate::DenseDataset<Self> {
        use rayon::prelude::*;

        let d = self.d;
        let num_words = self.num_words();
        let output_dim = num_words + 1;
        assert_eq!(
            input.len(),
            n_vecs * d,
            "input length must equal n_vecs * d"
        );

        let mut data = vec![0u64; n_vecs * output_dim];
        data.par_chunks_mut(output_dim)
            .zip(input.par_chunks_exact(d))
            .for_each_init(
                || vec![0.0f32; d],
                |scratch, (out, x)| {
                    for ((s, &v), &m) in scratch.iter_mut().zip(x).zip(self.means.iter()) {
                        *s = v - m;
                    }
                    if let Some(rotator) = &self.rotator {
                        rotator.rotate_inplace(scratch);
                    }
                    let (factor, norm) = self.metadata(scratch);
                    // Sign-pack each 64-component word from four 16-lane sign masks (bit set iff the
                    // residual component is ≥ 0), instead of a scalar 64-iteration bit test.
                    for (w, out_word) in out[..num_words].iter_mut().enumerate() {
                        let base = w * WORD_BITS;
                        let mut word = 0u64;
                        for c in 0..WORD_BITS / FW {
                            let off = base + c * FW;
                            let mask = Simd::<f32, FW>::from_slice(&scratch[off..off + FW])
                                .simd_ge(Simd::splat(0.0))
                                .to_bitmask();
                            word |= mask << (c * FW);
                        }
                        *out_word = word;
                    }
                    // Scan-ready constants: f_add = ‖r‖² and s = ‖r‖/factor (see `pack_metadata`).
                    out[num_words] = pack_metadata(norm * norm, norm / factor);
                },
            );

        crate::DenseDataset::<Self>::from_raw(data.into_boxed_slice(), n_vecs, self)
    }

    /// Number of `u64` sign-code words per vector (excluding the metadata word).
    #[inline]
    fn num_words(&self) -> usize {
        self.d / WORD_BITS
    }

    /// Apply the rotation: `P·v` (a copy of `values` when rotation is disabled, i.e. `P = I`).
    fn rotate(&self, values: &[f32]) -> Vec<f32> {
        match &self.rotator {
            Some(rotator) => rotator.rotate(values),
            None => values.to_vec(),
        }
    }

    /// Center `values` by the means and apply the rotation: `r = P·(x − mean)`.
    ///
    /// Documents always take this path; queries only do so under a centering metric
    /// (see [`RabitqSupportedDistance::CENTER_QUERY`]).
    fn residual(&self, values: &[f32]) -> Vec<f32> {
        let centered: Vec<f32> = values
            .iter()
            .zip(self.means.iter())
            .map(|(&v, &m)| v - m)
            .collect();
        self.rotate(&centered)
    }

    /// `⟨mean, q⟩` — the centroid term the inner-product metric adds back.
    fn mean_dot(&self, values: &[f32]) -> f32 {
        self.means
            .iter()
            .zip(values.iter())
            .map(|(&m, &v)| m * v)
            .sum()
    }

    /// Sign-pack a residual into `num_words()` words (bit set iff `r_i >= 0`).
    fn pack_signs(&self, residual: &[f32]) -> Vec<u64> {
        (0..self.num_words())
            .map(|w| {
                let base = w * WORD_BITS;
                let mut word = 0u64;
                for i in 0..WORD_BITS {
                    if residual[base + i] >= 0.0 {
                        word |= 1u64 << i;
                    }
                }
                word
            })
            .collect()
    }

    /// Per-document metadata from a residual: `(factor, norm)` with degenerate-vector guards.
    ///
    /// `factor = ⟨ō,o⟩ = Σ|r_i| / (√d·‖r‖)`; a zero residual gets `factor = 1, norm = 0` so the
    /// estimator never divides by zero or produces NaN.
    fn metadata(&self, residual: &[f32]) -> (f32, f32) {
        let norm: f32 = residual.iter().map(|&r| r * r).sum::<f32>().sqrt();
        if norm <= f32::EPSILON {
            return (1.0, 0.0);
        }
        let abs_sum: f32 = residual.iter().map(|&r| r.abs()).sum();
        let factor = abs_sum / ((self.d as f32).sqrt() * norm);
        (factor.max(1e-6), norm)
    }
}

/// Evaluator holding the quantized query (1-bit sign words or multi-bit planes) and its
/// residual norm.
#[derive(Debug, Clone)]
pub struct RabitqQueryEvaluator<'e, D = DotProduct> {
    _encoder: PhantomData<&'e RabitqQuantizer<D>>,
    /// Sign-packed query residual (owned; the evaluator must not borrow the query). Empty when
    /// `query_bits > 1`.
    query_words: Vec<u64>,
    /// Interleaved query bit planes (`planes[w·bits + j]`, see [`pack_bit_planes`]). Empty when
    /// `query_bits == 1`.
    planes: Vec<u64>,
    /// Multi-bit query reconstruction scale (`q̂_i = delta·code_i + vl`).
    delta: f32,
    /// Multi-bit query reconstruction offset.
    vl: f32,
    /// Per-query constant `delta·Σcode + vl·d`, subtracted to recover the ±1-sign inner product.
    q_const: f32,
    /// Effective query bits for this evaluator (forced to 1 on the [`vector_evaluator`] path,
    /// where the "query" is an already sign-packed document).
    ///
    /// [`vector_evaluator`]: VectorEncoder::vector_evaluator
    query_bits: u32,
    /// Per-query multiplier applied to `s·ip_raw` to form `term = ‖r‖·‖q_r‖·cos`. Folds the whole
    /// query-side normalization so the scan needs no divide: `‖q_r‖/d` for the 1-bit sign path
    /// (`ip_raw = d − 2·hamming`), `1/√d` for the multi-bit path (`ip_raw = ip_signed`, where the
    /// query norm has already cancelled).
    scale: f32,
    /// Per-query additive term (RaBitQ-Library's `g_add`): `‖r_q‖²` for squared Euclidean,
    /// `⟨mean, q⟩` for inner product. See [`RabitqSupportedDistance::query_add`].
    add: f32,
    /// Dimensionality `d` (number of code bits).
    d: usize,
    /// Number of sign-code words per document (its slice `[..num_words]`; word `num_words` is metadata).
    num_words: usize,
}

impl<'e, D> RabitqQueryEvaluator<'e, D> {
    /// The signed document/query inner product `⟨s, q̂⟩` for a multi-bit query, in un-normalized
    /// units (the `√d·‖q_r‖` normalization is applied later via [`Self::scale`]). Only valid when
    /// `query_bits > 1`.
    ///
    /// Each plane is a contiguous AND+popcount reduction over the document code words (plane-major
    /// layout, see [`pack_bit_planes`]) — one `<<j` weighting per plane, none inside the reduction.
    #[inline]
    fn ip_signed(&self, code: &[u64]) -> f32 {
        // Dispatch on the (compile-time-known) plane count so `ip_signed_planes` fully unrolls.
        // `query_bits` is validated to `1..=8` in `train` and this path only runs for `> 1`.
        let planes = &self.planes;
        let (ip, ppc) = match self.query_bits {
            2 => ip_signed_planes::<2>(code, planes),
            3 => ip_signed_planes::<3>(code, planes),
            4 => ip_signed_planes::<4>(code, planes),
            5 => ip_signed_planes::<5>(code, planes),
            6 => ip_signed_planes::<6>(code, planes),
            7 => ip_signed_planes::<7>(code, planes),
            _ => ip_signed_planes::<8>(code, planes),
        };
        // ⟨doc_bits, q̂⟩ = delta·ip + vl·ppc; recover the ±1-sign inner product ⟨s, q̂⟩.
        2.0 * (self.delta * ip as f32 + self.vl * ppc as f32) - self.q_const
    }
}

impl<'e, 'v, D: RabitqSupportedDistance> QueryEvaluator<DenseVectorView<'v, u64>>
    for RabitqQueryEvaluator<'e, D>
{
    type Distance = D;

    #[inline]
    fn compute_distance(&self, vector: DenseVectorView<'v, u64>) -> D {
        let words = vector.values();
        let code = &words[..self.num_words];
        // (f_add = ‖r‖², s = ‖r‖/factor) — both precomputed at encode time so this scan does no
        // divide; see `pack_metadata`.
        let (f_add, s) = unpack_metadata(words[self.num_words]);

        // Raw sign agreement, before any normalization: `d − 2·hamming` (1-bit) or `⟨s, q̂⟩`
        // (multi-bit). `scale` folds the per-query normalization, so `term = ‖r‖·‖q_r‖·cos`.
        let ip_raw = if self.query_bits > 1 {
            self.ip_signed(code)
        } else {
            let h = hamming(&self.query_words, code);
            self.d as f32 - 2.0 * h as f32
        };
        let term = s * ip_raw * self.scale;

        // Metric-specific combine (see the module docs and `RabitqSupportedDistance`): both metrics
        // reduce to ⟨r, q_r⟩ ≈ ‖r‖·‖q_r‖·cos plus a per-query additive term.
        D::from_terms(self.add, f_add, term)
    }
}

impl<D: RabitqSupportedDistance> DenseVectorEncoder for RabitqQuantizer<D> {
    type InputValueType = f32;
    type OutputValueType = u64;

    /// Decode the code bits into `±1` `f32` values; the metadata word is ignored.
    fn decode_vector<'a>(&self, encoded: DenseVectorView<'a, u64>) -> DenseVectorOwned<f32> {
        let mut values = Vec::with_capacity(self.d);
        for &word in &encoded.values()[..self.num_words()] {
            for i in 0..WORD_BITS {
                let set = (word >> i) & 1 == 1;
                values.push(if set { 1.0f32 } else { -1.0f32 });
            }
        }
        DenseVectorOwned::new(values)
    }

    #[inline]
    fn push_encoded<'a, OutputContainer>(
        &self,
        input: DenseVectorView<'a, f32>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<u64>,
    {
        assert_eq!(
            input.len(),
            self.d,
            "Input vector length must equal encoder input dimension."
        );
        let residual = self.residual(input.values());
        let (factor, norm) = self.metadata(&residual);
        let mut words = self.pack_signs(&residual);
        // Store the scan-ready constants: f_add = ‖r‖² and s = ‖r‖/factor (see `pack_metadata`).
        words.push(pack_metadata(norm * norm, norm / factor));
        output.extend(words);
    }
}

impl<D: RabitqSupportedDistance> VectorEncoder for RabitqQuantizer<D> {
    type Distance = D;
    type InputVector<'a> = DenseVectorView<'a, f32>;
    type QueryVector<'q> = DenseVectorView<'q, f32>;
    type EncodedVector<'a> = DenseVectorView<'a, u64>;

    type Evaluator<'e>
        = RabitqQueryEvaluator<'e, D>
    where
        Self: 'e;

    /// Build an evaluator by moving the `f32` query into the rotated space — centered by the mean
    /// or not, per the metric (see [`RabitqSupportedDistance::CENTER_QUERY`]) — then sign-packing
    /// it (`query_bits == 1`) or scalar-quantizing it into bit planes.
    #[inline]
    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        assert_eq!(
            query.len(),
            self.d,
            "Query vector length must equal encoder input dimension."
        );
        let residual = if D::CENTER_QUERY {
            self.residual(query.values())
        } else {
            self.rotate(query.values())
        };
        let query_norm = residual.iter().map(|&r| r * r).sum::<f32>().sqrt();
        // The centroid term is only needed by the non-centering (inner-product) path.
        let mean_dot_query = if D::CENTER_QUERY {
            0.0
        } else {
            self.mean_dot(query.values())
        };
        let add = D::query_add(mean_dot_query, query_norm);

        let query_bits = self.config.query_bits;
        // `scale` folds the whole query-side normalization so the scan is divide-free (see the
        // field docs): the multi-bit path already cancelled ‖q_r‖, the 1-bit path carries it here.
        let scale = if query_bits > 1 {
            1.0 / (self.d as f32).sqrt()
        } else {
            query_norm / self.d as f32
        };
        let (query_words, planes, delta, vl, q_const) = if query_bits > 1 {
            let (codes, delta, vl) = quantize_query_multibit(&residual, query_bits);
            let sum_code: u64 = codes.iter().map(|&c| c as u64).sum();
            let q_const = delta * sum_code as f32 + vl * self.d as f32;
            let planes = pack_bit_planes(&codes, query_bits);
            (Vec::new(), planes, delta, vl, q_const)
        } else {
            (self.pack_signs(&residual), Vec::new(), 0.0, 0.0, 0.0)
        };

        RabitqQueryEvaluator {
            _encoder: PhantomData,
            query_words,
            planes,
            delta,
            vl,
            q_const,
            query_bits,
            scale,
            add,
            d: self.d,
            num_words: self.num_words(),
        }
    }

    /// Treat an already-encoded document as a query (build-path only): reuse its sign-code words and
    /// take the query norm from its stored metadata. Always uses the 1-bit sign path.
    ///
    /// A stored document keeps only its *centered* residual, so the raw vector needed for the
    /// inner-product centroid term `⟨mean, q⟩` is gone. This path therefore scores in residual
    /// space: `⟨r_x, r_y⟩` for [`DotProduct`] (not the true `⟨x, y⟩`) and the exact `‖x − y‖²` for
    /// [`SquaredEuclideanDistance`], which is centering-invariant and so unaffected.
    #[inline]
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let words = vector.values();
        // The stored metadata is (f_add = ‖r‖², s); recover the residual norm as √f_add.
        let (f_add, _) = unpack_metadata(words[self.num_words()]);
        let norm = f_add.sqrt();
        RabitqQueryEvaluator {
            _encoder: PhantomData,
            query_words: words[..self.num_words()].to_vec(),
            planes: Vec::new(),
            delta: 0.0,
            vl: 0.0,
            q_const: 0.0,
            query_bits: 1,
            scale: norm / self.d as f32,
            // Residual-space scoring: no centroid term to add back.
            add: D::query_add(0.0, norm),
            d: self.d,
            num_words: self.num_words(),
        }
    }

    fn input_dim(&self) -> usize {
        self.d
    }

    /// Sign-code words plus the metadata word.
    fn output_dim(&self) -> usize {
        self.num_words() + 1
    }
}

impl<D> SpaceUsage for RabitqQuantizer<D> {
    fn space_usage_bytes(&self) -> usize {
        self.d.space_usage_bytes()
            + self.means.space_usage_bytes()
            + self.rotator.space_usage_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DatasetGrowable, FlatIndex, Index, PlainDenseDatasetGrowable, PlainDenseQuantizer,
    };

    /// RaBitQ scoring squared Euclidean distance.
    type RabitqL2 = RabitqQuantizer<SquaredEuclideanDistance>;
    /// RaBitQ scoring inner product.
    type RabitqIp = RabitqQuantizer<DotProduct>;

    /// Build a `PlainDenseDataset<f32, DotProduct>` from a list of equal-length vectors.
    fn plain_dataset(vectors: &[Vec<f32>]) -> PlainDenseDataset<f32, DotProduct> {
        let d = vectors[0].len();
        let encoder = PlainDenseQuantizer::<f32, DotProduct>::new(d);
        let mut growable = PlainDenseDatasetGrowable::new(encoder);
        for v in vectors {
            growable.push(DenseVectorView::new(v));
        }
        growable.into()
    }

    /// Vectors with distinct popcount patterns (from the zero-one experiments): `±1` over 64 dims
    /// with a varying number of `+1` components, so scores are well separated (no rank ties).
    fn varied_vectors() -> Vec<Vec<f32>> {
        [4usize, 12, 30, 48, 60, 64, 0, 33, 7, 51]
            .iter()
            .map(|&ones| {
                let mut v = vec![-1.0f32; 64];
                v[..ones].fill(1.0);
                v
            })
            .collect()
    }

    #[test]
    fn metadata_word_roundtrips_the_two_scan_constants() {
        let (f_add, s) = (123.456f32, 0.797_884_6_f32);
        let (f2, s2) = unpack_metadata(pack_metadata(f_add, s));
        assert_eq!(f_add, f2);
        assert_eq!(s, s2);
    }

    /// Every query must retrieve itself near the top under metric `D`.
    fn assert_self_is_top_ranked<D: RabitqSupportedDistance>(
        vectors: &[Vec<f32>],
        dataset: &PlainDenseDataset<f32, DotProduct>,
        query_bits: u32,
        rotate: bool,
    ) {
        let config = RabitqConfig {
            query_bits,
            seed: 42,
            rotate,
        };
        let rabitq = RabitqQuantizer::<D>::encode_dataset(dataset, config);
        let index = FlatIndex::from(&rabitq);
        for (i, q) in vectors.iter().enumerate() {
            let top = index.search(DenseVectorView::new(q), vectors.len(), &());
            let self_rank = top
                .iter()
                .position(|r| r.vector as usize == i)
                .expect("self must be retrieved");
            // The all-(+1) and all-(-1) vectors binarize identically to others under rotation
            // with so few points, so allow self to sit just below the top.
            assert!(self_rank < 3, "query {i} ranked itself at {self_rank}");
        }
    }

    #[test]
    fn search_finds_self_as_nearest() {
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        // `varied_vectors` all have norm 8; on equal-norm data the two metrics rank alike, so
        // both must put the query itself at the top.
        assert_self_is_top_ranked::<SquaredEuclideanDistance>(&vectors, &dataset, 1, true);
        assert_self_is_top_ranked::<DotProduct>(&vectors, &dataset, 1, true);
    }

    #[test]
    fn search_finds_self_as_nearest_without_rotation() {
        // With rotation disabled the estimator runs with `P = I` (also orthogonal), so
        // self-retrieval must still hold on both metrics.
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        assert_self_is_top_ranked::<SquaredEuclideanDistance>(&vectors, &dataset, 1, false);
        assert_self_is_top_ranked::<DotProduct>(&vectors, &dataset, 1, false);
    }

    /// Scores stay finite across every query bit width and seed under metric `D`.
    fn assert_scores_finite<D: RabitqSupportedDistance>(
        vectors: &[Vec<f32>],
        dataset: &PlainDenseDataset<f32, DotProduct>,
    ) {
        for query_bits in [1, 2, 4, 8] {
            for seed in [1, 42] {
                let config = RabitqConfig {
                    query_bits,
                    seed,
                    rotate: true,
                };
                let ds = RabitqQuantizer::<D>::encode_dataset(dataset, config);
                let index = FlatIndex::from(&ds);
                for q in vectors {
                    for r in index.search(DenseVectorView::new(q), 3, &()) {
                        assert!(
                            r.distance.distance().is_finite(),
                            "non-finite score with {config:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn scores_are_finite_for_every_query_bit_width() {
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        assert_scores_finite::<SquaredEuclideanDistance>(&vectors, &dataset);
        assert_scores_finite::<DotProduct>(&vectors, &dataset);
    }

    #[test]
    #[should_panic(expected = "dim % 64 == 0")]
    fn train_rejects_non_multiple_of_64() {
        let dataset = plain_dataset(&[vec![0.0f32; 10]]);
        let _ = RabitqIp::train(&dataset, RabitqConfig::default());
    }

    #[test]
    #[should_panic(expected = "query_bits")]
    fn train_rejects_out_of_range_query_bits() {
        let dataset = plain_dataset(&[vec![0.0f32; 64]]);
        let config = RabitqConfig {
            query_bits: 9,
            ..Default::default()
        };
        let _ = RabitqIp::train(&dataset, config);
    }

    #[test]
    fn multibit_query_reconstruction_is_accurate() {
        let d = 64;
        let q: Vec<f32> = (0..d).map(|i| (i as f32 * 0.7).sin() * 2.0 + 0.3).collect();
        let (codes, delta, vl) = quantize_query_multibit(&q, 4);
        let recon: Vec<f32> = codes.iter().map(|&c| delta * c as f32 + vl).collect();

        let dot: f32 = q.iter().zip(&recon).map(|(a, b)| a * b).sum();
        let nq: f32 = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nr: f32 = recon.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cos = dot / (nq * nr);
        assert!(cos > 0.98, "4-bit reconstruction cosine too low: {cos}");

        // The sign bit must round-trip: reconstruction and input never disagree on sign.
        for (&orig, &r) in q.iter().zip(&recon) {
            assert!(orig * r >= 0.0, "sign flipped: {orig} vs {r}");
        }
    }

    #[test]
    fn bitplane_ip_matches_direct_dot() {
        let d = 128;
        let bits = 4u32;
        let q: Vec<f32> = (0..d).map(|i| (i as f32 * 0.37).sin() - 0.1).collect();
        let (codes, delta, vl) = quantize_query_multibit(&q, bits);
        let planes = pack_bit_planes(&codes, bits);
        let sum_code: u64 = codes.iter().map(|&c| c as u64).sum();
        let q_const = delta * sum_code as f32 + vl * d as f32;

        let num_words = d / 64;
        for doc in [
            [0u64, 0u64],
            [u64::MAX, u64::MAX],
            [0x5555_5555_5555_5555, 0xF0F0_F0F0_F0F0_F0F0],
            [u64::MAX, 0],
            [0x0123_4567_89AB_CDEF, 0xDEAD_BEEF_CAFE_F00D],
        ] {
            // Mirror the plane-major kernel: one AND+popcount reduction per plane, weighted by 2^j.
            let ppc: u64 = doc.iter().map(|w| w.count_ones() as u64).sum();
            let mut ip: u64 = 0;
            for j in 0..bits as usize {
                let plane = &planes[j * num_words..(j + 1) * num_words];
                let pj: u64 = doc
                    .iter()
                    .zip(plane)
                    .map(|(&dw, &pw)| (dw & pw).count_ones() as u64)
                    .sum();
                ip += pj << j;
            }
            let kernel = 2.0 * (delta * ip as f32 + vl * ppc as f32) - q_const;

            // Direct evaluation of ⟨sign_doc(±1), q̂⟩ with q̂_i = delta·code_i + vl.
            let direct: f32 = (0..d)
                .map(|i| {
                    let bit = (doc[i / 64] >> (i % 64)) & 1;
                    let sign = if bit == 1 { 1.0f32 } else { -1.0 };
                    sign * (delta * codes[i] as f32 + vl)
                })
                .sum();
            assert!(
                (kernel - direct).abs() < 1e-2,
                "kernel {kernel} vs direct {direct} for doc {doc:?}"
            );
        }
    }

    #[test]
    fn four_bit_query_matches_exact_query_estimator() {
        // With query_bits = 4 the estimated sign agreement must be close to the same estimator
        // computed with the *unquantized* query residual — the only error left is query
        // reconstruction. The evaluator's L2 score is inverted back to `ip_bar` using the
        // document metadata so the comparison happens at the sign-agreement level.
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        let config = RabitqConfig {
            query_bits: 4,
            ..Default::default()
        };
        let encoder = RabitqL2::train(&dataset, config);
        let ds = RabitqL2::encode_dataset(&dataset, config);

        let d = 64usize;

        for q in &vectors {
            let q_res = encoder.residual(q);
            let nq: f32 = q_res.iter().map(|v| v * v).sum::<f32>().sqrt();
            if nq <= f32::EPSILON {
                continue;
            }
            // Expected value with the *reconstructed* query q̂ — must match the evaluator exactly
            // (verifies the plane/q_const plumbing end to end).
            let (codes, delta, vl) = quantize_query_multibit(&q_res, 4);
            let q_hat: Vec<f32> = codes.iter().map(|&c| delta * c as f32 + vl).collect();

            let evaluator = encoder.query_evaluator(DenseVectorView::new(q));
            for (i, doc) in vectors.iter().enumerate() {
                let doc_res = encoder.residual(doc);
                let (factor, norm_o) = encoder.metadata(&doc_res);
                if norm_o <= f32::EPSILON {
                    continue;
                }
                let dot_with = |query: &[f32]| -> f32 {
                    doc_res
                        .iter()
                        .zip(query)
                        .map(|(&r, &qr)| if r >= 0.0 { qr } else { -qr })
                        .sum::<f32>()
                        / ((d as f32).sqrt() * nq)
                };
                let expected = dot_with(&q_hat);
                let exact = dot_with(&q_res);
                // Invert est = ‖r‖² + ‖r_q‖² − 2·‖r‖·‖r_q‖·(ip_bar/factor) back to ip_bar.
                let est = evaluator.compute_distance(ds.get(i as u64)).distance();
                let got = (norm_o * norm_o + nq * nq - est) / (2.0 * norm_o * nq) * factor;
                assert!(
                    (got - expected).abs() < 1e-3,
                    "doc {i}: evaluator {got} vs reconstructed-query value {expected}"
                );
                // Loose sanity bound vs the unquantized query: catches gross scale errors while
                // tolerating genuine 4-bit reconstruction noise on this spiky ±1 data.
                assert!(
                    (got - exact).abs() < 0.25,
                    "doc {i}: 4-bit estimate {got} too far from exact-query {exact}"
                );
            }
        }
    }

    #[test]
    fn ip_centroid_decomposition_is_exact() {
        // The identity the inner-product metric rests on. With P orthogonal, x = mean + Pᵀr, so
        //   ⟨x, q⟩ = ⟨mean, q⟩ + ⟨P(x − mean), P·q⟩
        // holds *exactly* — this is why the query is rotated but not centered. Checked before any
        // quantization enters, so a failure here means the transform plumbing is wrong.
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        let encoder = RabitqIp::train(&dataset, RabitqConfig::default());

        for (qi, q) in vectors.iter().enumerate() {
            let q_rot = encoder.rotate(q);
            let mean_dot = encoder.mean_dot(q);
            for (xi, x) in vectors.iter().enumerate() {
                let r = encoder.residual(x);
                let got = mean_dot + r.iter().zip(&q_rot).map(|(&a, &b)| a * b).sum::<f32>();
                let want: f32 = x.iter().zip(q).map(|(&a, &b)| a * b).sum();
                assert!(
                    (got - want).abs() < 1e-2,
                    "x{xi}·q{qi}: decomposition {got} vs direct {want}"
                );
            }
        }
    }

    #[test]
    fn ip_score_matches_direct_estimator_form() {
        // Parity check for the inner-product path: the evaluator's score must equal
        // ⟨mean,q⟩ + ‖r‖·‖q_r‖·(ip_bar/factor) rebuilt here from the encoder's own residuals and
        // the reconstructed 4-bit query — an independent route through the ±1 sign expansion.
        // Note the query is rotated, *not* centered, so `rotate` is the right transform.
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        let config = RabitqConfig {
            query_bits: 4,
            seed: 42,
            rotate: true,
        };
        let encoder = RabitqIp::train(&dataset, config);
        let ds = RabitqIp::encode_dataset(&dataset, config);
        let d = 64usize;

        for q in &vectors {
            let q_rot = encoder.rotate(q);
            let nq: f32 = q_rot.iter().map(|v| v * v).sum::<f32>().sqrt();
            if nq <= f32::EPSILON {
                continue;
            }
            let mean_dot = encoder.mean_dot(q);
            let (codes, delta, vl) = quantize_query_multibit(&q_rot, 4);
            let q_hat: Vec<f32> = codes.iter().map(|&c| delta * c as f32 + vl).collect();

            let evaluator = encoder.query_evaluator(DenseVectorView::new(q));
            for (i, doc) in vectors.iter().enumerate() {
                let doc_res = encoder.residual(doc);
                let (factor, norm_o) = encoder.metadata(&doc_res);
                if norm_o <= f32::EPSILON {
                    continue;
                }
                // ip_bar = ⟨sign(r), q̂⟩ / (√d·‖q_r‖), the same quantity the kernel computes.
                let ip_bar = doc_res
                    .iter()
                    .zip(&q_hat)
                    .map(|(&r, &qh)| if r >= 0.0 { qh } else { -qh })
                    .sum::<f32>()
                    / ((d as f32).sqrt() * nq);
                let expected = mean_dot + norm_o * nq * (ip_bar / factor);

                let got = evaluator.compute_distance(ds.get(i as u64)).0;
                let tol = 1e-2 * expected.abs().max(1.0);
                assert!(
                    (got - expected).abs() <= tol,
                    "doc {i}: evaluator IP {got} vs direct form {expected} (tol {tol})"
                );
            }
        }
    }

    #[test]
    fn l2_score_matches_library_factor_form() {
        // Parity check: with a 4-bit query, our L2 score must equal the RaBitQ-Library
        // plain-path estimator computed here via its *own* decomposition —
        //   est_dist = f_add + g_add + f_rescale·(ip_x0_qr + k1xsumq)
        // built from 0/1 document bits — which is an independent route to the same number than
        // our ±1 / law-of-cosines expression. Residuals come from `encoder.residual` so both
        // routes see the same centered-and-rotated vectors.
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        let config = RabitqConfig {
            query_bits: 4,
            seed: 42,
            rotate: true,
        };
        let encoder = RabitqL2::train(&dataset, config);
        let ds = RabitqL2::encode_dataset(&dataset, config);

        for q in &vectors {
            let q_res = encoder.residual(q);
            let nq: f32 = q_res.iter().map(|v| v * v).sum::<f32>().sqrt();
            if nq <= f32::EPSILON {
                continue;
            }
            // Reconstructed 4-bit query q̂ ≈ r_q (delta·code + vl per component).
            let (codes, delta, vl) = quantize_query_multibit(&q_res, 4);
            let q_hat: Vec<f32> = codes.iter().map(|&c| delta * c as f32 + vl).collect();
            let sumq_hat: f32 = q_hat.iter().sum();
            let k1xsumq = -0.5 * sumq_hat; // library c_1 = -1/2, on the reconstructed residual sum

            let evaluator = encoder.query_evaluator(DenseVectorView::new(q));
            for (i, doc) in vectors.iter().enumerate() {
                let r = encoder.residual(doc);
                let norm_o: f32 = r.iter().map(|v| v * v).sum::<f32>().sqrt();
                if norm_o <= f32::EPSILON {
                    continue;
                }
                let sum_abs: f32 = r.iter().map(|v| v.abs()).sum(); // = 2·ip_resi_xucb
                // Library L2 doc factors (centered query ⇒ no centroid term in f_add):
                let f_add = norm_o * norm_o;
                let f_rescale = -2.0 * (norm_o * norm_o) / (0.5 * sum_abs);
                let g_add = nq * nq;
                // ip_x0_qr = ⟨bit_doc(0/1), q̂⟩ = Σ over positive-sign dims of q̂_i.
                let ip_x0_qr: f32 = r
                    .iter()
                    .zip(&q_hat)
                    .filter(|&(&ri, _)| ri >= 0.0)
                    .map(|(_, &qh)| qh)
                    .sum();
                let expected = f_add + g_add + f_rescale * (ip_x0_qr + k1xsumq);

                let got = evaluator.compute_distance(ds.get(i as u64)).distance();
                let tol = 1e-2 * expected.abs().max(1.0);
                assert!(
                    (got - expected).abs() <= tol,
                    "doc {i}: evaluator L2 {got} vs library-form {expected} (tol {tol})"
                );
            }
        }
    }

    #[test]
    fn search_finds_self_as_nearest_query_bits_4() {
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        assert_self_is_top_ranked::<SquaredEuclideanDistance>(&vectors, &dataset, 4, true);
        assert_self_is_top_ranked::<DotProduct>(&vectors, &dataset, 4, true);
    }
}
