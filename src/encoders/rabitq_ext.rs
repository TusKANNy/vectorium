//! Extended RaBitQ encoder: multi-bit document codes (no centroids).
//!
//! The extended version of [`rabitq`](crate::encoders::rabitq): documents are quantized to a
//! uniform, configurable `total_bits ∈ 2..=9` bits per component — 1 sign bit plus
//! `ex_bits = total_bits − 1` magnitude bits — following RaBitQ-Library's extended scheme
//! (*Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors*, Gao et al.).
//! `total_bits = 1` (the plain sign code, `ex_bits = 0`) is **not** handled here and is rejected by
//! [`train`](RabitqExtQuantizer::train): it degenerates to and scores identically to
//! [`RabitqQuantizer`](crate::encoders::rabitq::RabitqQuantizer), so use that 1-bit encoder for it.
//! Queries keep the same variable `query_bits ∈ 1..=8` treatment as the 1-bit encoder.
//!
//! This file intentionally duplicates the private helpers of `rabitq.rs` (kernels, rescale-factor
//! search, query quantization) so the two implementations stay independent; they may be merged
//! later. The preprocessing is identical: residual `r = P·(x − mean)` with per-component means and
//! a seeded [`FhtKacRotator`] (disable via [`RabitqExtConfig::rotate`]).
//!
//! ## Document code
//!
//! With `o_abs_i = |r_i|/‖r‖` and `t` from the exact rescale-factor search
//! ([`best_rescale_factor`]), each component gets a magnitude code
//! `e_i = min(⌊t·o_abs_i⌋, 2^ex_bits − 1)` and the total code
//!
//! ```text
//! u_i = e_i | (1 << ex_bits)      if r_i ≥ 0        # sign bit on top
//! u_i = (!e_i) & (2^ex_bits − 1)  if r_i < 0        # complement-coded magnitude
//! ```
//!
//! With `cb = −(2^ex_bits − 0.5)` the shifted code is **antisymmetric**:
//! `u_i + cb = sign(r_i)·(e_i + 0.5)`, so `⟨r, u + cb⟩ = ‖r‖·ipnorm` with
//! `ipnorm = Σ (e_i + 0.5)·o_abs_i > 0` — the quantized direction always points with the residual.
//!
//! ## Scoring
//!
//! The unbiased projection estimator (any scaling of the quantized direction cancels):
//!
//! ```text
//! ⟨r, q_r⟩ ≈ (‖r‖² / ⟨r, u + cb⟩)·⟨u + cb, q̂⟩ = s_ext·⟨u + cb, q̂⟩,   s_ext = ‖r‖/ipnorm
//! ```
//!
//! where `q̂` approximates the rotated query `q_r`: the multi-bit reconstruction
//! `q̂_i = delta·c_i + vl` (`query_bits > 1`), or `(‖q_r‖/√d)·sign(q_r)` (`query_bits = 1`).
//! Expanding `⟨u + cb, q̂⟩` leaves one code/code inner product per candidate plus per-query
//! constants:
//!
//! ```text
//! ⟨u, c⟩  = Σ_a Σ_b 2^(a+b) · popcount(doc_plane_a AND query_plane_b)
//! Σu      = Σ_a 2^a · popcount(doc_plane_a)
//! multi-bit: raw = delta·⟨u, c⟩ + vl·Σu + cb·Σq̂            # = ⟨u + cb, q̂⟩
//! 1-bit:     raw = 2·⟨u, b_q⟩ − Σu + cb·Σsign(q_r)         # = ⟨u + cb, sign(q_r)⟩
//! term = s_ext · scale · raw                                # ≈ ⟨r, q_r⟩
//! ```
//!
//! `term` then feeds the same metric combine as the 1-bit encoder
//! ([`RabitqSupportedDistance::from_terms`]): `f_add + g_add − 2·term` for squared Euclidean,
//! `⟨mean, q⟩ + term` for inner product.
//!
//! ## Storage
//!
//! Each document is `total_bits · d/64` code words in **plane-major** layout (plane `a` holds bit
//! `a` of every `u_i`; plane `ex_bits` is the sign plane) plus one metadata word packing the two
//! scan-ready floats `[f_add = ‖r‖² | s_ext = ‖r‖/ipnorm]`, so the per-candidate scan does no
//! divide. Only dimensions that are a multiple of 64 are supported (no bit-padding).
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::marker::PhantomData;
use std::simd::Simd;
use std::simd::num::SimdUint;

use crate::core::distances::DotProduct;
use crate::core::vector::{DenseVectorOwned, DenseVectorView};
use crate::core::vector_encoder::{DenseVectorEncoder, QueryEvaluator, VectorEncoder};
use crate::encoders::rabitq::RabitqSupportedDistance;
use crate::transformations::fht_kac::FhtKacRotator;
use crate::{Dataset, PlainDenseDataset, ScalarDenseSupportedDistance, SpaceUsage};

/// Number of bits packed into a single `u64` word.
const WORD_BITS: usize = 64;

/// Extended RaBitQ encoder parameters. See the module docs for the estimator math.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RabitqExtConfig {
    /// Bits per component for the **document** code: 1 sign bit + `total_bits − 1` magnitude
    /// bits. Values in `1..=9` are supported; `1` degenerates to the plain sign code.
    pub total_bits: u32,
    /// Bits per component for the scalar-quantized query (`1` = plain sign query). Values in
    /// `1..=8` are supported.
    pub query_bits: u32,
    /// Seed for the random orthogonal rotation ([`FhtKacRotator`]).
    pub seed: u64,
    /// Apply the random orthogonal rotation ([`FhtKacRotator`]) to residuals. Disabling skips the
    /// `O(d log d)` per-vector transform at encode and query time; the estimator math is unchanged
    /// (`P = I` is a valid orthogonal transform), but the codes lose the information-spreading
    /// guarantee, so recall on real data is expected to drop.
    pub rotate: bool,
}

impl Default for RabitqExtConfig {
    fn default() -> Self {
        Self {
            total_bits: 4,
            query_bits: 4,
            seed: 42,
            rotate: true,
        }
    }
}

/// Pack the per-document scan constants into the trailing metadata word: `[f_add:f32 | s_ext:f32]`.
///
/// These are the two quantities the scan needs with **no per-candidate divide**:
/// `f_add = ‖r‖²` (the squared-Euclidean additive) and `s_ext = ‖r‖/ipnorm` (the estimator's
/// rescale; the residual norm is recoverable as `√f_add` off the hot path).
#[inline]
fn pack_metadata(f_add: f32, s_ext: f32) -> u64 {
    ((f_add.to_bits() as u64) << 32) | (s_ext.to_bits() as u64)
}

/// Unpack the trailing metadata word into `(f_add, s_ext)` (see [`pack_metadata`]).
#[inline]
fn unpack_metadata(word: u64) -> (f32, f32) {
    (
        f32::from_bits((word >> 32) as u32),
        f32::from_bits(word as u32),
    )
}

/// Lane width for the `u64` popcount kernels (8×`u64` = one AVX-512 `zmm` register).
const LANES: usize = 8;

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

/// The full document/query code inner product over all document planes: returns
/// `(⟨u, c⟩, Σu)` where `⟨u, c⟩ = Σ_a 2^a · Σ_b 2^b · popcount(doc_plane_a AND query_plane_b)`
/// and `Σu = Σ_a 2^a · popcount(doc_plane_a)`.
///
/// One [`ip_signed_planes`] call per document plane, accumulated with weight `2^a`. Overflow-safe
/// in `u64`: `ip_a < 2^8·d` and `a ≤ 8`, so `⟨u, c⟩ < 2^17·d·9`.
#[inline]
fn doc_planes_ip<const QB: usize>(
    code: &[u64],
    query_planes: &[u64],
    num_words: usize,
) -> (u64, u64) {
    let mut ip = 0u64;
    let mut sum_u = 0u64;
    for (a, doc_plane) in code.chunks_exact(num_words).enumerate() {
        let (ip_a, ppc_a) = ip_signed_planes::<QB>(doc_plane, query_planes);
        ip += ip_a << a;
        sum_u += ppc_a << a;
    }
    (ip, sum_u)
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
/// Must not be called with `ex_bits = 0` (the sweep assumes `max_code ≥ 1`).
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
    let t = best_rescale_factor(&o_abs, ex_bits);

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

/// [`pack_bit_planes`] for the `u16` **document** codes (`total_bits = 9` reaches code 511, which
/// overflows `u8`), writing into a caller-provided zeroed buffer of `bits · num_words` words.
fn pack_bit_planes_u16_into(codes: &[u16], bits: u32, planes: &mut [u64]) {
    let bits = bits as usize;
    let num_words = codes.len() / WORD_BITS;
    debug_assert_eq!(planes.len(), bits * num_words);
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
}

/// Extended RaBitQ quantizer: multi-bit document codes. See the module docs.
///
/// The metric `D` selects how the stored residual codes are scored against a query; it defaults
/// to [`DotProduct`] (inner product). Document codes are identical under either metric — only the
/// query path and the final combine differ, so `D` is pure type-level state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RabitqExtQuantizer<D = DotProduct> {
    /// Input dimensionality (asserted to be a multiple of 64).
    d: usize,
    /// Per-component means used to center vectors (the centroid replacement); length `d`.
    means: Box<[f32]>,
    /// Fast random orthogonal transform shared by documents and queries; `None` when the encoder
    /// was trained with `config.rotate == false` (identity `P`).
    rotator: Option<FhtKacRotator>,
    /// Encoder parameters.
    config: RabitqExtConfig,
    /// The metric is encoded purely in the type; no runtime state.
    _distance: PhantomData<D>,
}

impl<D: RabitqSupportedDistance> RabitqExtQuantizer<D> {
    /// Learn per-component means over `dataset` and build the rotation.
    ///
    /// Panics unless the dataset dimension is a multiple of 64 and the configured bit widths are
    /// in range (`total_bits ∈ 2..=9`, `query_bits ∈ 1..=8`). `total_bits = 1` is deliberately
    /// rejected: the plain 1-bit document code is the domain of
    /// [`RabitqQuantizer`](crate::encoders::rabitq::RabitqQuantizer) — use that encoder instead of
    /// the degenerate `ex_bits = 0` case here.
    pub fn train<Ds: ScalarDenseSupportedDistance>(
        dataset: &PlainDenseDataset<f32, Ds>,
        config: RabitqExtConfig,
    ) -> Self {
        let d = dataset.input_dim();
        assert!(
            d.is_multiple_of(WORD_BITS),
            "RabitqExtQuantizer requires dim % 64 == 0, got {d}"
        );
        assert!(
            (2..=9).contains(&config.total_bits),
            "RabitqExtQuantizer requires 2 <= total_bits <= 9, got {}; \
             for total_bits = 1 use RabitqQuantizer (the 1-bit encoder) instead",
            config.total_bits
        );
        assert!(
            (1..=8).contains(&config.query_bits),
            "RabitqExtQuantizer requires 1 <= query_bits <= 8, got {}",
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
    /// helper is the entry point for building an extended-RaBitQ dataset.
    pub fn encode_dataset<Ds: ScalarDenseSupportedDistance>(
        dataset: &PlainDenseDataset<f32, Ds>,
        config: RabitqExtConfig,
    ) -> crate::DenseDataset<Self> {
        let encoder = Self::train(dataset, config);
        encoder.encode_flat_par(dataset.values(), dataset.len())
    }

    /// Encode `n_vecs` flat row-major `f32` vectors with an already-trained encoder, in parallel.
    ///
    /// Encodes straight into the final slab: each output record is written in place, so there is
    /// no `Vec<Vec>` of per-vector allocations and no reassembly copy. Per-worker scratch buffers
    /// hold the residual `r = P·(x − mean)` and the per-component codes, reused across every
    /// vector that worker sees.
    pub fn encode_flat_par(self, input: &[f32], n_vecs: usize) -> crate::DenseDataset<Self> {
        use rayon::prelude::*;

        let d = self.d;
        let total_words = self.config.total_bits as usize * self.num_words();
        let output_dim = total_words + 1;
        assert_eq!(
            input.len(),
            n_vecs * d,
            "input length must equal n_vecs * d"
        );

        let mut data = vec![0u64; n_vecs * output_dim];
        data.par_chunks_mut(output_dim)
            .zip(input.par_chunks_exact(d))
            .for_each_init(
                || (vec![0.0f32; d], vec![0u16; d]),
                |(scratch, codes), (out, x)| {
                    for ((s, &v), &m) in scratch.iter_mut().zip(x).zip(self.means.iter()) {
                        *s = v - m;
                    }
                    if let Some(rotator) = &self.rotator {
                        rotator.rotate_inplace(scratch);
                    }
                    let (f_add, s_ext) = self.encode_residual_into(scratch, codes);
                    // `out` comes from the zeroed slab, so the plane packer can OR bits in place.
                    pack_bit_planes_u16_into(
                        codes,
                        self.config.total_bits,
                        &mut out[..total_words],
                    );
                    out[total_words] = pack_metadata(f_add, s_ext);
                },
            );

        crate::DenseDataset::<Self>::from_raw(data.into_boxed_slice(), n_vecs, self)
    }

    /// Number of `u64` words **per bit plane** (a document stores `total_bits` planes plus the
    /// metadata word).
    #[inline]
    fn num_words(&self) -> usize {
        self.d / WORD_BITS
    }

    /// Magnitude bits per component (`total_bits − 1`).
    #[inline]
    fn ex_bits(&self) -> u32 {
        self.config.total_bits - 1
    }

    /// The document code's symmetric-grid offset `cb = −(2^ex_bits − 0.5)`, so that
    /// `u_i + cb = sign(r_i)·(e_i + 0.5)`.
    #[inline]
    fn cb(&self) -> f32 {
        -(((1u64 << self.ex_bits()) as f32) - 0.5)
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

    /// Sign-pack a residual into `num_words()` words (bit set iff `r_i >= 0`) — the 1-bit query
    /// path.
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

    /// Quantize a residual to per-component total codes, writing them into `codes` (length `d`).
    ///
    /// Returns the scan constants `(f_add = ‖r‖², s_ext = ‖r‖/ipnorm)`. A zero residual gets
    /// all-zero codes and `(0, 0)`, so the estimator contributes `term = 0` and never divides by
    /// zero or produces NaN. Otherwise `ipnorm = Σ (e_i + 0.5)·o_abs_i ≥ 0.5·Σ o_abs_i > 0`.
    fn encode_residual_into(&self, residual: &[f32], codes: &mut [u16]) -> (f32, f32) {
        let ex_bits = self.ex_bits();
        let norm = residual
            .iter()
            .map(|&r| r as f64 * r as f64)
            .sum::<f64>()
            .sqrt();
        if norm <= f64::EPSILON {
            codes.fill(0);
            return (0.0, 0.0);
        }
        let o_abs: Vec<f32> = residual
            .iter()
            .map(|&r| (r.abs() as f64 / norm) as f32)
            .collect();

        let mut ipnorm = 0.0f64;
        if ex_bits == 0 {
            // Plain sign code: `best_rescale_factor` must not run with max_code = 0.
            for ((c, &r), &o) in codes.iter_mut().zip(residual).zip(&o_abs) {
                *c = (r >= 0.0) as u16;
                ipnorm += 0.5 * o as f64;
            }
        } else {
            let max_code = (1u32 << ex_bits) - 1;
            let t = best_rescale_factor(&o_abs, ex_bits);
            for ((c, &r), &o) in codes.iter_mut().zip(residual).zip(&o_abs) {
                let e = (((t * o as f64) + 1e-5) as u32).min(max_code);
                let code = if r >= 0.0 {
                    (1u32 << ex_bits) | e
                } else {
                    !e & max_code
                };
                ipnorm += (e as f64 + 0.5) * o as f64;
                *c = code as u16;
            }
        }

        ((norm * norm) as f32, (norm / ipnorm) as f32)
    }

    /// [`Self::encode_residual_into`] returning the codes in a fresh `Vec` (test/inspection
    /// convenience; the hot paths reuse scratch buffers instead).
    #[cfg(test)]
    fn encode_residual(&self, residual: &[f32]) -> (Vec<u16>, f32, f32) {
        let mut codes = vec![0u16; residual.len()];
        let (f_add, s_ext) = self.encode_residual_into(residual, &mut codes);
        (codes, f_add, s_ext)
    }
}

/// Evaluator holding the quantized query (1-bit sign words or multi-bit planes) and the per-query
/// constants of the extended estimator.
#[derive(Debug, Clone)]
pub struct RabitqExtQueryEvaluator<'e, D = DotProduct> {
    _encoder: PhantomData<&'e RabitqExtQuantizer<D>>,
    /// Sign-packed query residual (owned; the evaluator must not borrow the query). Empty unless
    /// this is the 1-bit *query* path — the build path always uses `planes`.
    query_words: Vec<u64>,
    /// Plane-major query bit planes (see [`pack_bit_planes`]); on the build path, the document's
    /// own stored planes. Empty on the 1-bit query path.
    planes: Vec<u64>,
    /// Query reconstruction scale (`q̂_i = delta·code_i + vl`). Unused on the 1-bit query path.
    delta: f32,
    /// Query reconstruction offset. Unused on the 1-bit query path.
    vl: f32,
    /// Per-query constant `cb·Σq̂` (multi-bit) or `cb·Σsign(q_r)` (1-bit) — the document-code
    /// offset `cb` paired with the query sum, completing `⟨u + cb, q̂⟩`.
    q_const: f32,
    /// Number of query bit planes to dispatch on: `query_bits` (`1..=8`) for real queries, or
    /// `total_bits` (`2..=9`) on the [`vector_evaluator`] path where the "query" is a stored
    /// document code.
    ///
    /// [`vector_evaluator`]: VectorEncoder::vector_evaluator
    query_bits: u32,
    /// Per-query multiplier applied to `s_ext·raw` to form `term ≈ ⟨r, q_r⟩`: `1.0` when `q̂` is
    /// an absolute-units reconstruction (multi-bit and build paths), `‖q_r‖/√d` on the 1-bit sign
    /// path (where `q̂ = (‖q_r‖/√d)·sign(q_r)`).
    scale: f32,
    /// Per-query additive term (RaBitQ-Library's `g_add`): `‖r_q‖²` for squared Euclidean,
    /// `⟨mean, q⟩` for inner product. See [`RabitqSupportedDistance::query_add`].
    add: f32,
    /// Number of `u64` words per bit plane.
    num_words: usize,
    /// Number of stored document planes `B = total_bits` (the metadata word sits at
    /// `B·num_words`).
    total_bits: u32,
}

impl<'e, D> RabitqExtQueryEvaluator<'e, D> {
    /// `(⟨u, c⟩, Σu)` between a document's stored planes and this evaluator's query planes (or
    /// sign words), dispatched on the compile-time plane count so [`ip_signed_planes`] fully
    /// unrolls. `query_bits` is `1..=8` for real queries and up to `9` on the build path.
    #[inline]
    fn doc_ip(&self, code: &[u64]) -> (u64, u64) {
        let buf = if self.planes.is_empty() {
            &self.query_words
        } else {
            &self.planes
        };
        match self.query_bits {
            1 => doc_planes_ip::<1>(code, buf, self.num_words),
            2 => doc_planes_ip::<2>(code, buf, self.num_words),
            3 => doc_planes_ip::<3>(code, buf, self.num_words),
            4 => doc_planes_ip::<4>(code, buf, self.num_words),
            5 => doc_planes_ip::<5>(code, buf, self.num_words),
            6 => doc_planes_ip::<6>(code, buf, self.num_words),
            7 => doc_planes_ip::<7>(code, buf, self.num_words),
            8 => doc_planes_ip::<8>(code, buf, self.num_words),
            _ => doc_planes_ip::<9>(code, buf, self.num_words),
        }
    }
}

impl<'e, 'v, D: RabitqSupportedDistance> QueryEvaluator<DenseVectorView<'v, u64>>
    for RabitqExtQueryEvaluator<'e, D>
{
    type Distance = D;

    #[inline]
    fn compute_distance(&self, vector: DenseVectorView<'v, u64>) -> D {
        let words = vector.values();
        let total_words = self.total_bits as usize * self.num_words;
        let code = &words[..total_words];
        // (f_add = ‖r‖², s_ext = ‖r‖/ipnorm) — both precomputed at encode time so this scan does
        // no divide; see `pack_metadata`.
        let (f_add, s_ext) = unpack_metadata(words[total_words]);

        let (ip, sum_u) = self.doc_ip(code);
        // raw = ⟨u + cb, q̂⟩: the query-sum·cb part lives in `q_const`; the document part is
        // `delta·⟨u,c⟩ + vl·Σu` (multi-bit q̂) or `⟨u, sign⟩ = 2·⟨u, bits⟩ − Σu` (1-bit signs).
        let raw = if self.planes.is_empty() {
            2.0 * ip as f32 - sum_u as f32 + self.q_const
        } else {
            self.delta * ip as f32 + self.vl * sum_u as f32 + self.q_const
        };
        let term = s_ext * self.scale * raw;

        // Metric-specific combine (see the module docs and `RabitqSupportedDistance`): both metrics
        // reduce to ⟨r, q_r⟩ ≈ s_ext·⟨u + cb, q̂⟩ plus a per-query additive term.
        D::from_terms(self.add, f_add, term)
    }
}

impl<D: RabitqSupportedDistance> DenseVectorEncoder for RabitqExtQuantizer<D> {
    type InputValueType = f32;
    type OutputValueType = u64;

    /// Decode the stored planes into the residual reconstruction `r̂_i = s_ext·(u_i + cb)` (the
    /// extended analogue of the 1-bit encoder's `±1` decode; the values live in the rotated,
    /// centered residual space).
    fn decode_vector<'a>(&self, encoded: DenseVectorView<'a, u64>) -> DenseVectorOwned<f32> {
        let nw = self.num_words();
        let b = self.config.total_bits as usize;
        let words = encoded.values();
        let (_, s_ext) = unpack_metadata(words[b * nw]);
        let cb = self.cb();
        let mut values = Vec::with_capacity(self.d);
        for i in 0..self.d {
            let (w, bit) = (i / WORD_BITS, i % WORD_BITS);
            let mut u = 0u32;
            for (a, plane) in words[..b * nw].chunks_exact(nw).enumerate() {
                u |= (((plane[w] >> bit) & 1) as u32) << a;
            }
            values.push(s_ext * (u as f32 + cb));
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
        let mut codes = vec![0u16; self.d];
        let (f_add, s_ext) = self.encode_residual_into(&residual, &mut codes);
        let total_words = self.config.total_bits as usize * self.num_words();
        let mut words = vec![0u64; total_words + 1];
        pack_bit_planes_u16_into(&codes, self.config.total_bits, &mut words[..total_words]);
        // Store the scan-ready constants: f_add = ‖r‖² and s_ext = ‖r‖/ipnorm.
        words[total_words] = pack_metadata(f_add, s_ext);
        output.extend(words);
    }
}

impl<D: RabitqSupportedDistance> VectorEncoder for RabitqExtQuantizer<D> {
    type Distance = D;
    type InputVector<'a> = DenseVectorView<'a, f32>;
    type QueryVector<'q> = DenseVectorView<'q, f32>;
    type EncodedVector<'a> = DenseVectorView<'a, u64>;

    type Evaluator<'e>
        = RabitqExtQueryEvaluator<'e, D>
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

        // The *document* code's offset — distinct from the query quantizer's internal
        // `−(2^(query_bits−1) − 0.5)`; only the former pairs with the query sums in `q_const`.
        let cb = self.cb();
        let query_bits = self.config.query_bits;
        let (query_words, planes, delta, vl, q_const, scale) = if query_bits > 1 {
            let (codes, delta, vl) = quantize_query_multibit(&residual, query_bits);
            let sum_code: u64 = codes.iter().map(|&c| c as u64).sum();
            // Σq̂ = delta·Σcode + vl·d; q̂ approximates q_r in absolute units, so no rescale.
            let sum_q_hat = delta * sum_code as f32 + vl * self.d as f32;
            let planes = pack_bit_planes(&codes, query_bits);
            (Vec::new(), planes, delta, vl, cb * sum_q_hat, 1.0)
        } else {
            let query_words = self.pack_signs(&residual);
            let ones: u64 = query_words.iter().map(|w| w.count_ones() as u64).sum();
            // Σsign(q_r) = 2·popcount − d; the sign query carries ‖q_r‖/√d in `scale`.
            let sum_signs = 2.0 * ones as f32 - self.d as f32;
            let scale = query_norm / (self.d as f32).sqrt();
            (query_words, Vec::new(), 0.0, 0.0, cb * sum_signs, scale)
        };

        RabitqExtQueryEvaluator {
            _encoder: PhantomData,
            query_words,
            planes,
            delta,
            vl,
            q_const,
            query_bits,
            scale,
            add,
            num_words: self.num_words(),
            total_bits: self.config.total_bits,
        }
    }

    /// Treat an already-encoded document as a query (build-path only): its stored planes *are* a
    /// reconstruction `r̂_i = s_ext·(u_i + cb) = delta·u_i + vl` with `delta = s_ext` and
    /// `vl = s_ext·cb`, so they slot straight into the multi-bit query path at full stored
    /// fidelity (`total_bits` planes, up to 9).
    ///
    /// A stored document keeps only its *centered* residual, so the raw vector needed for the
    /// inner-product centroid term `⟨mean, q⟩` is gone. This path therefore scores in residual
    /// space: `⟨r_x, r_y⟩` for [`DotProduct`] (not the true `⟨x, y⟩`) and the exact `‖x − y‖²` for
    /// [`SquaredEuclideanDistance`], which is centering-invariant and so unaffected.
    ///
    /// [`SquaredEuclideanDistance`]: crate::SquaredEuclideanDistance
    #[inline]
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let words = vector.values();
        let nw = self.num_words();
        let total_words = self.config.total_bits as usize * nw;
        // The stored metadata is (f_add = ‖r‖², s_ext); recover the residual norm as √f_add.
        let (f_add, s_ext) = unpack_metadata(words[total_words]);
        let norm = f_add.sqrt();
        let cb = self.cb();
        let planes = words[..total_words].to_vec();
        // Σu = Σ_a 2^a·popcount(plane_a), for the per-query constant cb·Σq̂ (off the hot path).
        let sum_u: u64 = planes
            .chunks_exact(nw)
            .enumerate()
            .map(|(a, plane)| plane.iter().map(|w| w.count_ones() as u64).sum::<u64>() << a)
            .sum();
        let (delta, vl) = (s_ext, s_ext * cb);
        RabitqExtQueryEvaluator {
            _encoder: PhantomData,
            query_words: Vec::new(),
            planes,
            delta,
            vl,
            q_const: cb * (delta * sum_u as f32 + vl * self.d as f32),
            query_bits: self.config.total_bits,
            scale: 1.0,
            // Residual-space scoring: no centroid term to add back.
            add: D::query_add(0.0, norm),
            num_words: nw,
            total_bits: self.config.total_bits,
        }
    }

    fn input_dim(&self) -> usize {
        self.d
    }

    /// `total_bits` planes of code words plus the metadata word.
    fn output_dim(&self) -> usize {
        self.config.total_bits as usize * self.num_words() + 1
    }
}

impl<D> SpaceUsage for RabitqExtQuantizer<D> {
    fn space_usage_bytes(&self) -> usize {
        self.d.space_usage_bytes()
            + self.means.space_usage_bytes()
            + self.rotator.space_usage_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::distances::{Distance, SquaredEuclideanDistance};
    use crate::{
        DatasetGrowable, FlatIndex, Index, PlainDenseDatasetGrowable, PlainDenseQuantizer,
    };

    /// Extended RaBitQ scoring squared Euclidean distance.
    type RabitqExtL2 = RabitqExtQuantizer<SquaredEuclideanDistance>;
    /// Extended RaBitQ scoring inner product.
    type RabitqExtIp = RabitqExtQuantizer<DotProduct>;

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

    /// A smooth, sign-varying vector of length `d` (magnitudes spread over many quantization
    /// levels, unlike the spiky ±1 fixtures).
    fn smooth_vector(d: usize) -> Vec<f32> {
        (0..d).map(|i| (i as f32 * 0.7).sin() * 2.0 + 0.3).collect()
    }

    fn ext_config(total_bits: u32, query_bits: u32, rotate: bool) -> RabitqExtConfig {
        RabitqExtConfig {
            total_bits,
            query_bits,
            seed: 42,
            rotate,
        }
    }

    #[test]
    fn metadata_word_roundtrips_the_two_scan_constants() {
        let (f_add, s_ext) = (123.456f32, 0.797_884_6_f32);
        let (f2, s2) = unpack_metadata(pack_metadata(f_add, s_ext));
        assert_eq!(f_add, f2);
        assert_eq!(s_ext, s2);
    }

    #[test]
    fn shifted_code_is_antisymmetric_and_ip_with_residual_positive() {
        // The identity the whole estimator rests on: u_i + cb = sign(r_i)·(e_i + 0.5), hence
        // ⟨r, u + cb⟩ = ‖r‖·ipnorm = f_add/s_ext > 0.
        let dataset = plain_dataset(&varied_vectors());
        for total_bits in [2u32, 4, 9] {
            let encoder = RabitqExtIp::train(&dataset, ext_config(total_bits, 4, true));
            let r = smooth_vector(64);
            let (codes, f_add, s_ext) = encoder.encode_residual(&r);
            let cb = encoder.cb();
            let ex_bits = total_bits - 1;

            let mut ip = 0.0f64;
            for (&u, &ri) in codes.iter().zip(&r) {
                let xu_cb = u as f32 + cb;
                // Sign agreement (r_i is never exactly 0 for this fixture).
                assert!(
                    xu_cb * ri > 0.0,
                    "total_bits {total_bits}: u {u} + cb {cb} disagrees with r {ri}"
                );
                // The shifted code is a half-integer magnitude in (0, 2^ex_bits].
                let mag = xu_cb.abs();
                assert!(
                    (mag - (mag.floor() + 0.5)).abs() < 1e-6 && mag <= (1u64 << ex_bits) as f32,
                    "total_bits {total_bits}: |u + cb| = {mag} not a half-integer level"
                );
                ip += (ri * xu_cb) as f64;
            }
            let expected = f_add as f64 / s_ext as f64; // ‖r‖·ipnorm
            assert!(
                expected > 0.0 && (ip - expected).abs() < 1e-2 * expected,
                "total_bits {total_bits}: ⟨r, u+cb⟩ = {ip} vs f_add/s_ext = {expected}"
            );
        }
    }

    /// Every query must retrieve itself near the top under metric `D`.
    fn assert_self_is_top_ranked<D: RabitqSupportedDistance>(
        vectors: &[Vec<f32>],
        dataset: &PlainDenseDataset<f32, DotProduct>,
        config: RabitqExtConfig,
    ) {
        let rabitq = RabitqExtQuantizer::<D>::encode_dataset(dataset, config);
        let index = FlatIndex::from(&rabitq);
        for (i, q) in vectors.iter().enumerate() {
            let top = index.search(DenseVectorView::new(q), vectors.len(), &());
            let self_rank = top
                .iter()
                .position(|r| r.vector as usize == i)
                .expect("self must be retrieved");
            // The all-(+1) and all-(-1) vectors binarize identically to others under rotation
            // with so few points, so allow self to sit just below the top.
            assert!(
                self_rank < 3,
                "query {i} ranked itself at {self_rank} with {config:?}"
            );
        }
    }

    #[test]
    fn search_finds_self_as_nearest_across_bit_widths() {
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        // `varied_vectors` all have norm 8; on equal-norm data the two metrics rank alike, so
        // both must put the query itself at the top, at every bit-width combination.
        for (total_bits, query_bits) in [(2, 4), (4, 1), (4, 4), (9, 8)] {
            let config = ext_config(total_bits, query_bits, true);
            assert_self_is_top_ranked::<SquaredEuclideanDistance>(&vectors, &dataset, config);
            assert_self_is_top_ranked::<DotProduct>(&vectors, &dataset, config);
        }
    }

    #[test]
    fn search_finds_self_as_nearest_without_rotation() {
        // With rotation disabled the estimator runs with `P = I` (also orthogonal), so
        // self-retrieval must still hold on both metrics.
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        let config = ext_config(4, 4, false);
        assert_self_is_top_ranked::<SquaredEuclideanDistance>(&vectors, &dataset, config);
        assert_self_is_top_ranked::<DotProduct>(&vectors, &dataset, config);
    }

    /// Scores stay finite across every bit-width combination and seed under metric `D`.
    fn assert_scores_finite<D: RabitqSupportedDistance>(
        vectors: &[Vec<f32>],
        dataset: &PlainDenseDataset<f32, DotProduct>,
    ) {
        for total_bits in [2, 4, 9] {
            for query_bits in [1, 4, 8] {
                for seed in [1, 42] {
                    let config = RabitqExtConfig {
                        total_bits,
                        query_bits,
                        seed,
                        rotate: true,
                    };
                    let ds = RabitqExtQuantizer::<D>::encode_dataset(dataset, config);
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
    }

    #[test]
    fn scores_are_finite_for_every_bit_width() {
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        assert_scores_finite::<SquaredEuclideanDistance>(&vectors, &dataset);
        assert_scores_finite::<DotProduct>(&vectors, &dataset);
    }

    #[test]
    fn identical_vectors_give_finite_scores() {
        // All residuals are exactly zero (every vector equals the mean): the metadata degenerates
        // to (0, 0) and every estimate must stay finite with term = 0.
        let vectors: Vec<Vec<f32>> = vec![smooth_vector(64); 4];
        let dataset = plain_dataset(&vectors);
        for query_bits in [1, 4] {
            let config = ext_config(4, query_bits, true);
            let ds = RabitqExtL2::encode_dataset(&dataset, config);
            let index = FlatIndex::from(&ds);
            for r in index.search(DenseVectorView::new(&vectors[0]), 4, &()) {
                assert!(r.distance.distance().is_finite(), "non-finite score");
            }
        }
    }

    #[test]
    #[should_panic(expected = "dim % 64 == 0")]
    fn train_rejects_non_multiple_of_64() {
        let dataset = plain_dataset(&[vec![0.0f32; 10]]);
        let _ = RabitqExtIp::train(&dataset, RabitqExtConfig::default());
    }

    #[test]
    #[should_panic(expected = "query_bits")]
    fn train_rejects_out_of_range_query_bits() {
        let dataset = plain_dataset(&[vec![0.0f32; 64]]);
        let config = RabitqExtConfig {
            query_bits: 9,
            ..Default::default()
        };
        let _ = RabitqExtIp::train(&dataset, config);
    }

    #[test]
    #[should_panic(expected = "total_bits")]
    fn train_rejects_out_of_range_total_bits() {
        let dataset = plain_dataset(&[vec![0.0f32; 64]]);
        let config = RabitqExtConfig {
            total_bits: 10,
            ..Default::default()
        };
        let _ = RabitqExtIp::train(&dataset, config);
    }

    #[test]
    #[should_panic(expected = "total_bits")]
    fn train_rejects_zero_total_bits() {
        let dataset = plain_dataset(&[vec![0.0f32; 64]]);
        let config = RabitqExtConfig {
            total_bits: 0,
            ..Default::default()
        };
        let _ = RabitqExtIp::train(&dataset, config);
    }

    #[test]
    fn reconstruction_cosine_improves_with_total_bits() {
        let d = 64;
        let x = smooth_vector(d);
        let vectors = vec![x.clone(), x.iter().map(|&v| -v).collect()];
        let dataset = plain_dataset(&vectors);

        let mut prev_cos = 0.0f32;
        for total_bits in [2u32, 3, 5, 7] {
            let encoder = RabitqExtL2::train(&dataset, ext_config(total_bits, 4, true));
            let residual = encoder.residual(&x);
            let mut words: Vec<u64> = Vec::new();
            encoder.push_encoded(DenseVectorView::new(&x), &mut words);
            let decoded = encoder.decode_vector(DenseVectorView::new(&words));

            let dot: f32 = residual
                .iter()
                .zip(decoded.values())
                .map(|(a, b)| a * b)
                .sum();
            let nr: f32 = residual.iter().map(|v| v * v).sum::<f32>().sqrt();
            let nd: f32 = decoded.values().iter().map(|v| v * v).sum::<f32>().sqrt();
            let cos = dot / (nr * nd);
            assert!(
                cos >= prev_cos - 1e-4,
                "cosine dropped from {prev_cos} to {cos} at total_bits {total_bits}"
            );
            prev_cos = cos;
        }
        assert!(
            prev_cos > 0.99,
            "7-bit reconstruction cosine too low: {prev_cos}"
        );
    }

    #[test]
    fn doc_planes_ip_matches_direct_integer_dot() {
        // The full plane×plane accumulation must equal the direct Σ u_i·c_i and Σ 2^a·popcount
        // exactly (all integers).
        let d = 128;
        let num_words = d / WORD_BITS;
        let dataset = plain_dataset(&varied_vectors());
        let encoder = RabitqExtIp::train(&dataset, ext_config(4, 3, true));

        let r: Vec<f32> = (0..d).map(|i| (i as f32 * 0.37).sin() - 0.1).collect();
        let (doc_codes, _, _) = encoder.encode_residual(&r);
        let mut doc_planes = vec![0u64; 4 * num_words];
        pack_bit_planes_u16_into(&doc_codes, 4, &mut doc_planes);

        let q: Vec<f32> = (0..d).map(|i| (i as f32 * 0.53).cos() + 0.2).collect();
        let (q_codes, _, _) = quantize_query_multibit(&q, 3);
        let q_planes = pack_bit_planes(&q_codes, 3);

        let (ip, sum_u) = doc_planes_ip::<3>(&doc_planes, &q_planes, num_words);
        let direct_ip: u64 = doc_codes
            .iter()
            .zip(&q_codes)
            .map(|(&u, &c)| u as u64 * c as u64)
            .sum();
        let direct_sum: u64 = doc_codes.iter().map(|&u| u as u64).sum();
        assert_eq!(ip, direct_ip);
        assert_eq!(sum_u, direct_sum);
    }

    #[test]
    fn l2_score_matches_library_factor_form() {
        // Parity check: the evaluator's L2 score must equal the extended-RaBitQ decomposition
        //   est_dist = f_add + g_add + f_rescale·⟨u + cb, q̂⟩,   f_rescale = −2·s_ext
        // rebuilt here from scalar residuals, codes, and the reconstructed 4-bit query — an
        // independent route through the shifted-code expansion.
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        let config = ext_config(4, 4, true);
        let encoder = RabitqExtL2::train(&dataset, config);
        let ds = RabitqExtL2::encode_dataset(&dataset, config);
        let cb = encoder.cb();

        for q in &vectors {
            let q_res = encoder.residual(q);
            let nq: f32 = q_res.iter().map(|v| v * v).sum::<f32>().sqrt();
            if nq <= f32::EPSILON {
                continue;
            }
            let (codes, delta, vl) = quantize_query_multibit(&q_res, 4);
            let q_hat: Vec<f32> = codes.iter().map(|&c| delta * c as f32 + vl).collect();

            let evaluator = encoder.query_evaluator(DenseVectorView::new(q));
            for (i, doc) in vectors.iter().enumerate() {
                let r = encoder.residual(doc);
                let (u, f_add, s_ext) = encoder.encode_residual(&r);
                if s_ext == 0.0 {
                    continue;
                }
                let ip_xu_q: f32 = u
                    .iter()
                    .zip(&q_hat)
                    .map(|(&ui, &qh)| (ui as f32 + cb) * qh)
                    .sum();
                let expected = f_add + nq * nq - 2.0 * s_ext * ip_xu_q;

                let got = evaluator.compute_distance(ds.get(i as u64)).distance();
                let tol = 1e-2 * expected.abs().max(1.0);
                assert!(
                    (got - expected).abs() <= tol,
                    "doc {i}: evaluator L2 {got} vs factor form {expected} (tol {tol})"
                );
            }
        }
    }

    #[test]
    fn ip_score_matches_direct_estimator_form() {
        // Parity check for the inner-product path: the evaluator's score must equal
        // ⟨mean, q⟩ + s_ext·⟨u + cb, q̂⟩ rebuilt from scalars. Note the query is rotated, *not*
        // centered, so `rotate` is the right transform.
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        let config = ext_config(4, 4, true);
        let encoder = RabitqExtIp::train(&dataset, config);
        let ds = RabitqExtIp::encode_dataset(&dataset, config);
        let cb = encoder.cb();

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
                let r = encoder.residual(doc);
                let (u, _, s_ext) = encoder.encode_residual(&r);
                if s_ext == 0.0 {
                    continue;
                }
                let ip_xu_q: f32 = u
                    .iter()
                    .zip(&q_hat)
                    .map(|(&ui, &qh)| (ui as f32 + cb) * qh)
                    .sum();
                let expected = mean_dot + s_ext * ip_xu_q;

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
    fn one_bit_query_score_matches_direct_form() {
        // The 1-bit query path approximates q̂ = (‖q_r‖/√d)·sign(q_r); the evaluator must equal
        // f_add + g_add − 2·s_ext·(‖q_r‖/√d)·⟨u + cb, sign(q_r)⟩ rebuilt from scalars.
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        let config = ext_config(4, 1, true);
        let encoder = RabitqExtL2::train(&dataset, config);
        let ds = RabitqExtL2::encode_dataset(&dataset, config);
        let cb = encoder.cb();
        let d = 64usize;

        for q in &vectors {
            let q_res = encoder.residual(q);
            let nq: f32 = q_res.iter().map(|v| v * v).sum::<f32>().sqrt();
            if nq <= f32::EPSILON {
                continue;
            }
            let evaluator = encoder.query_evaluator(DenseVectorView::new(q));
            for (i, doc) in vectors.iter().enumerate() {
                let r = encoder.residual(doc);
                let (u, f_add, s_ext) = encoder.encode_residual(&r);
                if s_ext == 0.0 {
                    continue;
                }
                let ip_xu_sign: f32 = u
                    .iter()
                    .zip(&q_res)
                    .map(|(&ui, &qr)| {
                        let sign = if qr >= 0.0 { 1.0f32 } else { -1.0 };
                        (ui as f32 + cb) * sign
                    })
                    .sum();
                let term = s_ext * (nq / (d as f32).sqrt()) * ip_xu_sign;
                let expected = f_add + nq * nq - 2.0 * term;

                let got = evaluator.compute_distance(ds.get(i as u64)).distance();
                let tol = 1e-2 * expected.abs().max(1.0);
                assert!(
                    (got - expected).abs() <= tol,
                    "doc {i}: evaluator L2 {got} vs 1-bit direct form {expected} (tol {tol})"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "total_bits")]
    fn train_rejects_one_total_bits() {
        // total_bits = 1 (the plain sign code) is the domain of RabitqQuantizer, not this encoder;
        // the constructor must reject it rather than silently run the degenerate ex_bits = 0 path.
        let dataset = plain_dataset(&[vec![0.0f32; 64]]);
        let config = RabitqExtConfig {
            total_bits: 1,
            ..Default::default()
        };
        let _ = RabitqExtIp::train(&dataset, config);
    }
}
