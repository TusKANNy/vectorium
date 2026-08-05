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
//! a seeded [`FhtKacRotator`](crate::FhtKacRotator) (disable via [`RabitqExtConfig::rotate`]).
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
use std::marker::PhantomData;
use std::simd::Simd;
use std::simd::cmp::SimdPartialEq;

use crate::core::distances::DotProduct;
use crate::core::vector::{DenseVectorOwned, DenseVectorView};
use crate::core::vector_encoder::{DenseVectorEncoder, QueryEvaluator, VectorEncoder};
use crate::encoders::rabitq_common::{
    RabitqSpace, RabitqSupportedDistance, RescaleScratch, WORD_BITS, best_rescale_factor,
    ip_signed_planes, ip_signed_planes_batch6, pack_bit_planes, pack_metadata, pack_signs_into,
    quantize_query_multibit, unpack_metadata,
};
use crate::{Dataset, PlainDenseDataset, ScalarDenseSupportedDistance, SpaceUsage};

/// Extended RaBitQ encoder parameters. See the module docs for the estimator math.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RabitqExtConfig {
    /// Bits per component for the **document** code: 1 sign bit + `total_bits − 1` magnitude
    /// bits. Values in `1..=9` are supported; `1` degenerates to the plain sign code.
    pub total_bits: u32,
    /// Bits per component for the scalar-quantized query (`1` = plain sign query). Values in
    /// `1..=8` are supported.
    pub query_bits: u32,
    /// Seed for the random orthogonal rotation ([`FhtKacRotator`](crate::FhtKacRotator)).
    pub seed: u64,
    /// Apply the random orthogonal rotation ([`FhtKacRotator`](crate::FhtKacRotator)) to residuals. Disabling skips the
    /// `O(d log d)` per-vector transform at encode and query time; the estimator math is unchanged
    /// (`P = I` is a valid orthogonal transform), but the codes lose the information-spreading
    /// guarantee, so recall on real data is expected to drop.
    pub rotate: bool,
    /// Use the **constant** rescale factor for document quantization instead of the exact per-vector
    /// search ([`best_rescale_factor`]). When set (the default), the factor is estimated **once** at
    /// [`train`](RabitqExtQuantizer::train) over random unit vectors and every document quantizes in a
    /// single `O(d)` pass — RaBitQ-Library's `faster_config` build path, ~20× faster to encode. Clear
    /// it (`--faster-quant false`) to run the exact per-vector optimal search, which costs a heap sweep
    /// per vector for a marginal recall gain. Query quantization always uses the exact search.
    pub faster_quant: bool,
}

impl Default for RabitqExtConfig {
    fn default() -> Self {
        Self {
            total_bits: 4,
            query_bits: 4,
            seed: 42,
            rotate: true,
            faster_quant: true,
        }
    }
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

/// [`doc_planes_ip`] dispatched on a runtime plane count (`1..=9`), so the kernel's inner loops
/// still unroll at each width.
#[inline]
fn doc_planes_ip_dyn(
    bits: u32,
    code: &[u64],
    query_planes: &[u64],
    num_words: usize,
) -> (u64, u64) {
    match bits {
        1 => doc_planes_ip::<1>(code, query_planes, num_words),
        2 => doc_planes_ip::<2>(code, query_planes, num_words),
        3 => doc_planes_ip::<3>(code, query_planes, num_words),
        4 => doc_planes_ip::<4>(code, query_planes, num_words),
        5 => doc_planes_ip::<5>(code, query_planes, num_words),
        6 => doc_planes_ip::<6>(code, query_planes, num_words),
        7 => doc_planes_ip::<7>(code, query_planes, num_words),
        8 => doc_planes_ip::<8>(code, query_planes, num_words),
        _ => doc_planes_ip::<9>(code, query_planes, num_words),
    }
}

/// Six-way [`doc_planes_ip`]: the full document/query code inner product for six documents at
/// once. Returns `([⟨u, c⟩; 6], [Σu; 6])`.
///
/// One [`ip_signed_planes_batch6`] call per document plane `a`, accumulated with weight `2^a` —
/// the query planes are re-broadcast per document plane exactly as in the single-document path,
/// but shared across all six candidates within each pass.
#[inline]
fn doc_planes_ip_batch6<const QB: usize>(
    codes: [&[u64]; 6],
    query_planes: &[u64],
    num_words: usize,
) -> ([u64; 6], [u64; 6]) {
    let mut ip = [0u64; 6];
    let mut sum_u = [0u64; 6];
    let n_planes = codes[0].len() / num_words;
    for a in 0..n_planes {
        let doc_planes: [&[u64]; 6] =
            std::array::from_fn(|k| &codes[k][a * num_words..(a + 1) * num_words]);
        let (ip_a, ppc_a) = ip_signed_planes_batch6::<QB>(doc_planes, query_planes);
        for k in 0..6 {
            ip[k] += ip_a[k] << a;
            sum_u[k] += ppc_a[k] << a;
        }
    }
    (ip, sum_u)
}

/// Number of random unit vectors averaged to estimate the constant rescale factor (RaBitQ-Library's
/// `kConstNum`).
const CONST_SCALE_SAMPLES: usize = 100;

/// Estimate a **constant** rescale factor for `dim`/`ex_bits` by averaging the exact
/// [`best_rescale_factor`] over [`CONST_SCALE_SAMPLES`] random unit vectors — RaBitQ-Library's
/// `get_const_scaling_factors`. Computed once at [`train`](RabitqExtQuantizer::train); every document
/// then quantizes in a single `O(d)` pass with this `t` instead of a per-vector heap sweep.
///
/// The vectors are standard-Gaussian (isotropic ⇒ the factor is rotation-independent), row-normalized
/// and abs'd — exactly the `o_abs` shape the search consumes. The generator is seeded, so the factor
/// is reproducible; the specific RNG need not match the C++ library since this is an average estimate.
fn get_const_scaling_factors(dim: usize, ex_bits: u32, seed: u64) -> f64 {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    let mut rng = StdRng::seed_from_u64(seed);
    let mut scratch = RescaleScratch::default();
    let mut o_abs = vec![0.0f32; dim];
    let mut sum = 0.0f64;

    for _ in 0..CONST_SCALE_SAMPLES {
        // Box–Muller: draw `dim` standard-normal samples straight into `o_abs`.
        let mut k = 0;
        while k < dim {
            let u1: f64 = rng.gen_range(0.0f64..1.0).max(f64::MIN_POSITIVE);
            let u2: f64 = rng.gen_range(0.0f64..1.0);
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = std::f64::consts::TAU * u2;
            o_abs[k] = (r * theta.cos()) as f32;
            if k + 1 < dim {
                o_abs[k + 1] = (r * theta.sin()) as f32;
            }
            k += 2;
        }
        // Normalize to a unit vector, then take absolute values (the search's `o_abs` domain).
        let norm = (o_abs.iter().map(|&v| v as f64 * v as f64).sum::<f64>()).sqrt();
        let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        for v in o_abs.iter_mut() {
            *v = (*v as f64 * inv).abs() as f32;
        }
        sum += best_rescale_factor(&o_abs, ex_bits, &mut scratch);
    }

    sum / CONST_SCALE_SAMPLES as f64
}

/// [`pack_bit_planes`] for the `u16` **document** codes (`total_bits = 9` reaches code 511, which
/// overflows `u8`), writing into a caller-provided buffer of `bits · num_words` words.
///
/// SIMD bit-transpose: each 64-code word is loaded into a 64-lane vector once, and each plane `j` is
/// extracted as `((v >> j) & 1) != 0` and packed straight to a `u64` via `Mask::to_bitmask` (lane
/// `i` → bit `i`) — replacing the scalar `words × 64 × bits` bit-by-bit loop. Each `(j, w)` word is
/// written exactly once, so a plain assignment is correct regardless of the buffer's prior contents.
fn pack_bit_planes_u16_into(codes: &[u16], bits: u32, planes: &mut [u64]) {
    let bits = bits as usize;
    let num_words = codes.len() / WORD_BITS;
    debug_assert_eq!(planes.len(), bits * num_words);
    let one = Simd::<u16, WORD_BITS>::splat(1);
    let zero = Simd::<u16, WORD_BITS>::splat(0);
    for w in 0..num_words {
        let v = Simd::<u16, WORD_BITS>::from_slice(&codes[w * WORD_BITS..]);
        for j in 0..bits {
            let bit = (v >> Simd::splat(j as u16)) & one;
            planes[j * num_words + w] = bit.simd_ne(zero).to_bitmask();
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
    /// The trained geometry — means, rotated means and the rotation `P` — shared with
    /// [`RabitqQuantizer`](crate::encoders::rabitq::RabitqQuantizer).
    space: RabitqSpace,
    /// Constant rescale factor for the fast document-quantization path
    /// ([`get_const_scaling_factors`]), computed once at [`train`](Self::train). `None` unless
    /// `config.faster_quant` is set; the exact per-vector path ignores it.
    t_const: Option<f64>,
    /// Encoder parameters.
    config: RabitqExtConfig,
    /// The metric is encoded purely in the type; no runtime state.
    _distance: PhantomData<D>,
}

impl<D: RabitqSupportedDistance> RabitqExtQuantizer<D> {
    /// The number of bits currently used to quantize the query.
    #[inline]
    pub fn query_bits(&self) -> u32 {
        self.config.query_bits
    }

    /// The number of bits per component used by the stored *document* codes.
    ///
    /// Fixed at encode time: it determines the code layout, so changing it requires re-encoding.
    #[inline]
    pub fn total_bits(&self) -> u32 {
        self.config.total_bits
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
            "RabitqExtQuantizer requires query_bits in 1..=8, got {query_bits}"
        );
        self.config.query_bits = query_bits;
    }

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
        let space = RabitqSpace::train(dataset, config.rotate, config.seed);

        // Constant rescale factor for the fast build path, estimated once here (negligible vs. the
        // per-vector encode). `ex_bits = total_bits − 1 ≥ 1` always holds (total_bits ∈ 2..=9).
        let t_const = config
            .faster_quant
            .then(|| get_const_scaling_factors(space.dim(), config.total_bits - 1, config.seed));

        Self {
            space,
            t_const,
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
        crate::DenseDataset::<Self>::from_flat_par(encoder, dataset.values(), dataset.len())
    }

    /// Number of `u64` words **per bit plane** (a document stores `total_bits` planes plus the
    /// metadata word).
    #[inline]
    fn num_words(&self) -> usize {
        self.space.num_words()
    }

    /// Input dimensionality.
    #[inline]
    fn d(&self) -> usize {
        self.space.dim()
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
        self.space.rotate(values)
    }

    /// The residual `r = P·(x − mean)` — see [`RabitqSpace::residual`].
    fn residual(&self, values: &[f32]) -> Vec<f32> {
        self.space.residual(values)
    }

    /// `⟨mean, q⟩` — see [`RabitqSpace::mean_dot`].
    fn mean_dot(&self, values: &[f32]) -> f32 {
        self.space.mean_dot(values)
    }

    /// `⟨P·mean, q_r⟩` — see [`RabitqSpace::rotated_mean_dot`].
    fn rotated_mean_dot(&self, rotated: &[f32]) -> f32 {
        self.space.rotated_mean_dot(rotated)
    }

    /// Reconstruct the rotated residual from a stored code: `r̂_i = s_ext·(u_i + cb)`.
    ///
    /// This is the reconstruction the estimator itself uses — the stored planes enter the scan as
    /// `q̂ = delta·u + vl` with `delta = s_ext`, `vl = s_ext·cb` — so the build path stays consistent
    /// with the search path. Backs both [`decode_vector`] and the inner-product
    /// [`vector_evaluator`], which adds `P·mean` to it.
    ///
    /// [`decode_vector`]: DenseVectorEncoder::decode_vector
    /// [`vector_evaluator`]: VectorEncoder::vector_evaluator
    fn reconstruct_residual(&self, words: &[u64]) -> Vec<f32> {
        let nw = self.num_words();
        let b = self.config.total_bits as usize;
        let (_, s_ext) = unpack_metadata(words[b * nw]);
        let cb = self.cb();
        (0..self.d())
            .map(|i| {
                let (w, bit) = (i / WORD_BITS, i % WORD_BITS);
                let mut u = 0u32;
                for (a, plane) in words[..b * nw].chunks_exact(nw).enumerate() {
                    u |= (((plane[w] >> bit) & 1) as u32) << a;
                }
                s_ext * (u as f32 + cb)
            })
            .collect()
    }

    /// Build an evaluator from a query that already lives in the rotated space, given its centroid
    /// term `⟨mean, q⟩` (ignored under a centering metric).
    ///
    /// Shared by both entry points: [`query_evaluator`] rotates a raw `f32` query, while the
    /// inner-product [`vector_evaluator`] path reconstructs a stored residual and adds `P·mean`.
    /// Everything downstream — quantization, `scale`, `add` — is therefore identical on the two
    /// paths.
    ///
    /// [`query_evaluator`]: VectorEncoder::query_evaluator
    /// [`vector_evaluator`]: VectorEncoder::vector_evaluator
    fn rotated_query_evaluator<'e>(
        &'e self,
        q_r: &[f32],
        mean_dot_query: f32,
    ) -> RabitqExtQueryEvaluator<'e, D> {
        let query_norm = q_r.iter().map(|&r| r * r).sum::<f32>().sqrt();
        let add = D::query_add(mean_dot_query, query_norm);

        // The *document* code's offset — distinct from the query quantizer's internal
        // `−(2^(query_bits−1) − 0.5)`; only the former pairs with the query sums in `q_const`.
        let cb = self.cb();
        let query_bits = self.config.query_bits;
        let (query_words, planes, delta, vl, q_const, scale) = if query_bits > 1 {
            let (codes, delta, vl) = quantize_query_multibit(q_r, query_bits);
            let sum_code: u64 = codes.iter().map(|&c| c as u64).sum();
            // Σq̂ = delta·Σcode + vl·d; q̂ approximates q_r in absolute units, so no rescale.
            let sum_q_hat = delta * sum_code as f32 + vl * self.d() as f32;
            let planes = pack_bit_planes(&codes, query_bits);
            (Vec::new(), planes, delta, vl, cb * sum_q_hat, 1.0)
        } else {
            let query_words = self.pack_signs(q_r);
            let ones: u64 = query_words.iter().map(|w| w.count_ones() as u64).sum();
            // Σsign(q_r) = 2·popcount − d; the sign query carries ‖q_r‖/√d in `scale`.
            let sum_signs = 2.0 * ones as f32 - self.d() as f32;
            let scale = query_norm / (self.d() as f32).sqrt();
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

    /// Sign-pack a residual into `num_words()` words (bit set iff `r_i >= 0`) — the 1-bit query
    /// path.
    fn pack_signs(&self, residual: &[f32]) -> Vec<u64> {
        let mut words = vec![0u64; self.num_words()];
        pack_signs_into(residual, &mut words);
        words
    }

    /// Quantize a residual to per-component total codes, writing them into `codes` (length `d`).
    ///
    /// Returns the scan constants `(f_add = ‖r‖², s_ext = ‖r‖/ipnorm)`. A zero residual gets
    /// all-zero codes and `(0, 0)`, so the estimator contributes `term = 0` and never divides by
    /// zero or produces NaN. Otherwise `ipnorm = Σ (e_i + 0.5)·o_abs_i ≥ 0.5·Σ o_abs_i > 0`.
    ///
    /// The rescale factor is the constant [`Self::t_const`] (fast path) when set, else the exact
    /// per-vector [`best_rescale_factor`]. `o_abs` and `rescale` are caller-owned scratch, so a
    /// caller encoding many vectors can reuse them; the fast path folds `1/‖r‖` into the scalar and
    /// never touches `o_abs`.
    fn encode_residual_into(
        &self,
        residual: &[f32],
        codes: &mut [u16],
        o_abs: &mut Vec<f32>,
        rescale: &mut RescaleScratch,
    ) -> (f32, f32) {
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
        let inv_norm = 1.0 / norm;
        let mut ipnorm = 0.0f64;

        if ex_bits == 0 {
            // Plain sign code: `best_rescale_factor` must not run with max_code = 0. (Unreachable on
            // a trained encoder — `train` asserts total_bits ∈ 2..=9 — but kept for safety.)
            for (c, &r) in codes.iter_mut().zip(residual) {
                *c = (r >= 0.0) as u16;
                ipnorm += 0.5 * (r.abs() as f64 * inv_norm);
            }
        } else {
            let max_code = (1u32 << ex_bits) - 1;
            match self.t_const {
                // Fast path: constant factor, single O(d) pass with no per-component o_abs buffer —
                // `scale·|r_i| = t·(|r_i|/‖r‖)` because `scale = t/‖r‖`.
                Some(t_const) => {
                    let scale = t_const * inv_norm;
                    for (c, &r) in codes.iter_mut().zip(residual) {
                        let ar = r.abs() as f64;
                        let e = (((scale * ar) + 1e-5) as u32).min(max_code);
                        let code = if r >= 0.0 {
                            (1u32 << ex_bits) | e
                        } else {
                            !e & max_code
                        };
                        ipnorm += (e as f64 + 0.5) * (ar * inv_norm);
                        *c = code as u16;
                    }
                }
                // Exact path: per-vector search over the normalized abs residual (reused `o_abs`).
                None => {
                    o_abs.clear();
                    o_abs.extend(residual.iter().map(|&r| (r.abs() as f64 * inv_norm) as f32));
                    let t = best_rescale_factor(o_abs, ex_bits, rescale);
                    for ((c, &r), &o) in codes.iter_mut().zip(residual).zip(o_abs.iter()) {
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
            }
        }

        ((norm * norm) as f32, (norm / ipnorm) as f32)
    }

    /// [`Self::encode_residual_into`] returning the codes in a fresh `Vec` (test/inspection
    /// convenience; the hot paths reuse scratch buffers instead).
    #[cfg(test)]
    fn encode_residual(&self, residual: &[f32]) -> (Vec<u16>, f32, f32) {
        let mut codes = vec![0u16; residual.len()];
        let (f_add, s_ext) = self.encode_residual_into(
            residual,
            &mut codes,
            &mut Vec::new(),
            &mut RescaleScratch::default(),
        );
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
    /// `total_bits` (`2..=9`) on the squared-Euclidean [`vector_evaluator`] path, where the "query"
    /// is a stored document code reused verbatim.
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
        doc_planes_ip_dyn(self.query_bits, code, buf, self.num_words)
    }

    /// Six-way [`Self::doc_ip`]: `(⟨u, c⟩, Σu)` for six documents through the fused batch kernel
    /// ([`doc_planes_ip_batch6`]), dispatched on the compile-time plane count.
    #[inline]
    fn doc_ip_batch6(&self, codes: [&[u64]; 6]) -> ([u64; 6], [u64; 6]) {
        let buf = if self.planes.is_empty() {
            &self.query_words
        } else {
            &self.planes
        };
        match self.query_bits {
            1 => doc_planes_ip_batch6::<1>(codes, buf, self.num_words),
            2 => doc_planes_ip_batch6::<2>(codes, buf, self.num_words),
            3 => doc_planes_ip_batch6::<3>(codes, buf, self.num_words),
            4 => doc_planes_ip_batch6::<4>(codes, buf, self.num_words),
            5 => doc_planes_ip_batch6::<5>(codes, buf, self.num_words),
            6 => doc_planes_ip_batch6::<6>(codes, buf, self.num_words),
            7 => doc_planes_ip_batch6::<7>(codes, buf, self.num_words),
            8 => doc_planes_ip_batch6::<8>(codes, buf, self.num_words),
            _ => doc_planes_ip_batch6::<9>(codes, buf, self.num_words),
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

    /// Fused six-candidate scan: the query planes (or sign words) are broadcast once per chunk
    /// and interleaved against all six documents' popcount accumulators in a single pass per
    /// document plane ([`doc_planes_ip_batch6`]) — cross-candidate ILP the default
    /// six-single-calls dispatch can't reach. The scalar combine matches
    /// [`Self::compute_distance`] operation for operation, so batch and single scores are
    /// bit-identical.
    #[inline]
    fn compute_distances_batch6(&self, vectors: [DenseVectorView<'v, u64>; 6]) -> [D; 6] {
        let total_words = self.total_bits as usize * self.num_words;
        let words: [&[u64]; 6] = vectors.map(|v| v.values());
        let codes: [&[u64]; 6] = std::array::from_fn(|k| &words[k][..total_words]);

        let (ips, sum_us) = self.doc_ip_batch6(codes);

        std::array::from_fn(|k| {
            let (f_add, s_ext) = unpack_metadata(words[k][total_words]);
            let raw = if self.planes.is_empty() {
                2.0 * ips[k] as f32 - sum_us[k] as f32 + self.q_const
            } else {
                self.delta * ips[k] as f32 + self.vl * sum_us[k] as f32 + self.q_const
            };
            let term = s_ext * self.scale * raw;
            D::from_terms(self.add, f_add, term)
        })
    }
}

impl<D: RabitqSupportedDistance> DenseVectorEncoder for RabitqExtQuantizer<D> {
    type InputValueType = f32;
    type OutputValueType = u64;

    /// Decode the stored planes into the residual reconstruction `r̂_i = s_ext·(u_i + cb)` (the
    /// extended analogue of the 1-bit encoder's `±1` decode; the values live in the rotated,
    /// centered residual space).
    fn decode_vector<'a>(&self, encoded: DenseVectorView<'a, u64>) -> DenseVectorOwned<f32> {
        DenseVectorOwned::new(self.reconstruct_residual(encoded.values()))
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
            self.d(),
            "Input vector length must equal encoder input dimension."
        );
        let residual = self.residual(input.values());
        let mut codes = vec![0u16; self.d()];
        let (f_add, s_ext) = self.encode_residual_into(
            &residual,
            &mut codes,
            &mut Vec::new(),
            &mut RescaleScratch::default(),
        );
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
            self.d(),
            "Query vector length must equal encoder input dimension."
        );
        let residual = if D::CENTER_QUERY {
            self.residual(query.values())
        } else {
            self.rotate(query.values())
        };
        // The centroid term is only needed by the non-centering (inner-product) path.
        let mean_dot_query = if D::CENTER_QUERY {
            0.0
        } else {
            self.mean_dot(query.values())
        };
        self.rotated_query_evaluator(&residual, mean_dot_query)
    }

    /// Treat an already-encoded document as a query (build-path only).
    ///
    /// A stored document keeps only its *centered* residual `r`, so the two metrics need different
    /// treatment:
    ///
    /// * [`SquaredEuclideanDistance`] — centering is L2-preserving, so residual-space scoring is
    ///   already the exact `‖x − y‖²`. The stored planes *are* the reconstruction
    ///   `r̂_i = s_ext·(u_i + cb) = delta·u_i + vl` with `delta = s_ext`, `vl = s_ext·cb`, so they
    ///   slot straight into the multi-bit query path at full stored fidelity (`total_bits` planes,
    ///   up to 9) with no re-quantization.
    /// * [`DotProduct`] — centering is **not** inner-product preserving. Scoring in residual space
    ///   would return `⟨r_x, r_y⟩`, which drops `⟨P·mean, r_y⟩`; that term varies per candidate, so
    ///   the ranking would be wrong rather than merely shifted. The residual is therefore
    ///   reconstructed (`reconstruct_residual`) and `P·mean` added back, rebuilding the
    ///   un-centered rotated query `P·x̂ = r̂ + P·mean` and reducing this path to
    ///   [`Self::query_evaluator`] on the reconstruction — without ever applying `Pᵀ` and `P` and
    ///   letting them cancel. All four terms of
    ///   `⟨x̂, ŷ⟩ = ‖mean‖² + ⟨P·mean, r̂⟩ + ⟨P·mean, r_y⟩ + ⟨r̂, r_y⟩` then survive: the first two as
    ///   the per-query `add`, the last two out of a single scan pass, since the kernel estimates
    ///   `⟨r_y, r̂ + P·mean⟩` and the dot product is linear in the query. The cost is that the
    ///   stored planes can no longer be reused verbatim — the summed query is re-quantized at
    ///   `query_bits`, below `total_bits`.
    ///
    /// [`SquaredEuclideanDistance`]: crate::SquaredEuclideanDistance
    #[inline]
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let words = vector.values();
        if !D::CENTER_QUERY {
            let mut q_r = self.reconstruct_residual(words);
            for (v, &m) in q_r.iter_mut().zip(self.space.rotated_means().iter()) {
                *v += m;
            }
            // ⟨P·mean, P·x̂⟩ = ⟨mean, x̂⟩ — the centroid term, exactly as on the query path.
            let mean_dot_query = self.rotated_mean_dot(&q_r);
            return self.rotated_query_evaluator(&q_r, mean_dot_query);
        }
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
            q_const: cb * (delta * sum_u as f32 + vl * self.d() as f32),
            query_bits: self.config.total_bits,
            scale: 1.0,
            add: D::query_add(0.0, norm),
            num_words: nw,
            total_bits: self.config.total_bits,
        }
    }

    fn input_dim(&self) -> usize {
        self.d()
    }

    /// `total_bits` planes of code words plus the metadata word.
    fn output_dim(&self) -> usize {
        self.config.total_bits as usize * self.num_words() + 1
    }

    /// Score two stored documents directly, without building an evaluator.
    ///
    /// Only the centering (squared-Euclidean) metric gets the shortcut: `v1` plays the query through
    /// its own stored planes exactly as [`Self::vector_evaluator`] would, but those planes are read
    /// in place instead of being copied into an owned evaluator — no allocation, and the arithmetic
    /// matches operation for operation (its `scale` is `1.0` here, so `term = s_ext·raw`), so both
    /// routes give bit-identical scores.
    ///
    /// Under [`DotProduct`] there is nothing to shortcut: the effective query is `r̂ + P·mean` (see
    /// [`Self::vector_evaluator`]), which has to be materialized and quantized whatever the entry
    /// point, so this delegates. `D::CENTER_QUERY` is a constant, so the branch folds at compile
    /// time.
    #[inline]
    fn compute_distance_between(
        &self,
        v1: Self::EncodedVector<'_>,
        v2: Self::EncodedVector<'_>,
    ) -> Self::Distance {
        if !D::CENTER_QUERY {
            return self.vector_evaluator(v1).compute_distance(v2);
        }
        let nw = self.num_words();
        let total_bits = self.config.total_bits;
        let total_words = total_bits as usize * nw;
        let (q_words, d_words) = (v1.values(), v2.values());

        // v1 as query: its planes *are* the reconstruction q̂_i = delta·u_i + vl with
        // delta = s_ext, vl = s_ext·cb, and its residual norm is √f_add.
        let q_planes = &q_words[..total_words];
        let (q_f_add, q_s_ext) = unpack_metadata(q_words[total_words]);
        let norm = q_f_add.sqrt();
        let cb = self.cb();
        let (delta, vl) = (q_s_ext, q_s_ext * cb);
        // Σu = Σ_a 2^a·popcount(plane_a), for the per-query constant cb·Σq̂.
        let sum_q_u: u64 = q_planes
            .chunks_exact(nw)
            .enumerate()
            .map(|(a, plane)| plane.iter().map(|w| w.count_ones() as u64).sum::<u64>() << a)
            .sum();
        let q_const = cb * (delta * sum_q_u as f32 + vl * self.d() as f32);

        let (f_add, s_ext) = unpack_metadata(d_words[total_words]);
        let (ip, sum_u) = doc_planes_ip_dyn(total_bits, &d_words[..total_words], q_planes, nw);
        let raw = delta * ip as f32 + vl * sum_u as f32 + q_const;

        D::from_terms(D::query_add(0.0, norm), f_add, s_ext * raw)
    }
}

impl<D> SpaceUsage for RabitqExtQuantizer<D> {
    fn space_usage_bytes(&self) -> usize {
        self.space.space_usage_bytes() + self.t_const.space_usage_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::distances::{Distance, SquaredEuclideanDistance};
    use crate::{
        DatasetGrowable, FlatIndex, Index, IndexSerializer, PlainDenseDatasetGrowable,
        PlainDenseQuantizer,
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
            ..Default::default()
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
                        ..Default::default()
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
    fn faster_quant_matches_optimal_reconstruction() {
        // The constant-factor fast path (`faster_quant = true`) must reconstruct essentially as well
        // as the exact per-vector search: it uses an averaged rescale factor, so codes differ
        // slightly, but the reconstruction cosine must stay within a small tolerance. Assert on
        // reconstruction quality (the meaningful quantity), not bit-equality (which is expected to
        // differ by design).
        let d = 128;
        let x = smooth_vector(d);
        let vectors = vec![x.clone(), x.iter().map(|&v| -v * 0.5 + 0.2).collect()];
        let dataset = plain_dataset(&vectors);

        for total_bits in [2u32, 4, 5, 7, 9] {
            let cos = |faster_quant: bool| -> f32 {
                let cfg = RabitqExtConfig {
                    total_bits,
                    query_bits: 4,
                    seed: 42,
                    rotate: true,
                    faster_quant,
                };
                let encoder = RabitqExtL2::train(&dataset, cfg);
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
                dot / (nr * nd)
            };
            let (fast, optimal) = (cos(true), cos(false));
            assert!(
                fast >= optimal - 0.02,
                "fast-path cosine {fast} trails optimal {optimal} by too much at total_bits {total_bits}"
            );
        }
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

    /// The fused batch kernel must be bit-identical to six single `compute_distance` calls,
    /// across the 1-bit and multi-bit query paths and every document plane count, under metric `D`.
    fn assert_batch6_matches_singles<D: RabitqSupportedDistance + std::fmt::Debug>(
        vectors: &[Vec<f32>],
        dataset: &PlainDenseDataset<f32, DotProduct>,
    ) {
        for (total_bits, query_bits) in [(2, 1), (2, 3), (4, 1), (4, 4), (9, 8)] {
            let config = ext_config(total_bits, query_bits, true);
            let ds = RabitqExtQuantizer::<D>::encode_dataset(dataset, config);
            for q in vectors {
                let evaluator = ds.encoder().query_evaluator(DenseVectorView::new(q));
                let views = std::array::from_fn(|k| ds.get(k as u64));
                let batch = evaluator.compute_distances_batch6(views);
                let singles = std::array::from_fn(|k| evaluator.compute_distance(ds.get(k as u64)));
                assert_eq!(batch, singles, "{config:?}");
            }
            // The build path (`vector_evaluator`) dispatches on `total_bits` planes (up to 9):
            // its batch scores must also match its singles.
            let evaluator = ds.encoder().vector_evaluator(ds.get(0));
            let views = std::array::from_fn(|k| ds.get(k as u64));
            let batch = evaluator.compute_distances_batch6(views);
            let singles = std::array::from_fn(|k| evaluator.compute_distance(ds.get(k as u64)));
            assert_eq!(batch, singles, "build path, {config:?}");
        }
    }

    #[test]
    fn compute_distances_batch6_matches_six_singles() {
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        assert_batch6_matches_singles::<SquaredEuclideanDistance>(&vectors, &dataset);
        assert_batch6_matches_singles::<DotProduct>(&vectors, &dataset);
    }

    #[test]
    fn dataset_serialization_round_trip() {
        // The trained geometry lives in a nested `RabitqSpace`, so exercise the serde path end to
        // end: a reloaded index must equal the original *and* score identically.
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        let ds = RabitqExtIp::encode_dataset(&dataset, RabitqExtConfig::default());

        let mut path = std::env::temp_dir();
        path.push(format!("vectorium_rabitq_ext_{}.bin", std::process::id()));
        let path = path.to_str().unwrap().to_string();

        ds.save_index(&path).unwrap();
        let loaded = crate::DenseDataset::<RabitqExtIp>::load_index(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(ds, loaded);
        for q in &vectors {
            let before = ds.encoder().query_evaluator(DenseVectorView::new(q));
            let after = loaded.encoder().query_evaluator(DenseVectorView::new(q));
            for i in 0..ds.len() as u64 {
                assert_eq!(
                    before.compute_distance(ds.get(i)),
                    after.compute_distance(loaded.get(i)),
                    "vector {i}"
                );
            }
        }
    }

    #[test]
    fn compute_distances_batch6_matches_six_singles_wide() {
        // d = 640 → 10 words per plane: exercises the 8-wide unchecked-load chunk loop *and* the
        // 2-word tail (the d = 64 fixture is all tail).
        let vectors: Vec<Vec<f32>> = (0..8)
            .map(|k| {
                (0..640)
                    .map(|i| ((i * (k + 3)) as f32 * 0.13).sin())
                    .collect()
            })
            .collect();
        let dataset = plain_dataset(&vectors);
        assert_batch6_matches_singles::<SquaredEuclideanDistance>(&vectors, &dataset);
        assert_batch6_matches_singles::<DotProduct>(&vectors, &dataset);
    }

    /// The allocation-free two-document score must be bit-identical to the `vector_evaluator`
    /// route it replaces, across the stored plane counts, under metric `D`.
    fn assert_distance_between_matches_evaluator<D: RabitqSupportedDistance + std::fmt::Debug>(
        dataset: &PlainDenseDataset<f32, DotProduct>,
    ) {
        for total_bits in [2, 4, 9] {
            let config = ext_config(total_bits, 4, true);
            let ds = RabitqExtQuantizer::<D>::encode_dataset(dataset, config);
            let encoder = ds.encoder();
            for i in 0..ds.len() as u64 {
                for j in 0..ds.len() as u64 {
                    let direct = encoder.compute_distance_between(ds.get(i), ds.get(j));
                    let via = encoder
                        .vector_evaluator(ds.get(i))
                        .compute_distance(ds.get(j));
                    assert_eq!(direct, via, "pair ({i}, {j}), {config:?}");
                }
            }
        }
    }

    #[test]
    fn compute_distance_between_matches_vector_evaluator() {
        let dataset = plain_dataset(&varied_vectors());
        assert_distance_between_matches_evaluator::<SquaredEuclideanDistance>(&dataset);
        assert_distance_between_matches_evaluator::<DotProduct>(&dataset);
    }

    #[test]
    fn ip_build_path_equals_query_path_on_the_reconstruction() {
        // What the inner-product build path claims to be: `query_evaluator` applied to the stored
        // vector's reconstruction x̂ = Pᵀr̂ + mean. With rotation disabled (P = I) that x̂ can be
        // formed directly here, and both routes must build the same effective query — bit for bit,
        // since `vector_evaluator` skips only the Pᵀ/P pair that cancels.
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        for (total_bits, query_bits) in [(2, 1), (4, 4), (9, 8)] {
            let config = ext_config(total_bits, query_bits, false);
            let ds = RabitqExtIp::encode_dataset(&dataset, config);
            let encoder = ds.encoder();
            for i in 0..ds.len() as u64 {
                let mut recon = encoder.reconstruct_residual(ds.get(i).values());
                for (v, &m) in recon.iter_mut().zip(encoder.space.means().iter()) {
                    *v += m;
                }
                let build = encoder.vector_evaluator(ds.get(i));
                let query = encoder.query_evaluator(DenseVectorView::new(&recon));
                for j in 0..ds.len() as u64 {
                    assert_eq!(
                        build.compute_distance(ds.get(j)),
                        query.compute_distance(ds.get(j)),
                        "pair ({i}, {j}), {config:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn ip_build_path_estimates_the_true_dot_product() {
        // The build path must estimate ⟨x̂, ŷ⟩, not the residual-space ⟨r̂_x, r̂_y⟩ it used to
        // return: the latter drops ⟨P·mean, r_y⟩, which varies per candidate and so reorders
        // results. Run at high fidelity on both sides so anything left is quantization noise
        // rather than a missing term.
        let vectors = varied_vectors();
        let dataset = plain_dataset(&vectors);
        let config = ext_config(9, 8, true);
        let ds = RabitqExtIp::encode_dataset(&dataset, config);
        let encoder = ds.encoder();

        // Reconstructions in rotated space: P·x̂ = r̂_x + P·mean (P orthogonal, so no inverse
        // rotation is needed and every dot product below can be taken here).
        let rotated: Vec<Vec<f32>> = (0..ds.len() as u64)
            .map(|i| {
                let mut v = encoder.reconstruct_residual(ds.get(i).values());
                for (c, &m) in v.iter_mut().zip(encoder.space.rotated_means().iter()) {
                    *c += m;
                }
                v
            })
            .collect();
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(&x, &y)| x * y).sum::<f32>();

        let mut worst_est = 0.0f32;
        let mut worst_residual_only = 0.0f32;
        for i in 0..ds.len() as u64 {
            let evaluator = encoder.vector_evaluator(ds.get(i));
            let r_x = encoder.reconstruct_residual(ds.get(i).values());
            for j in 0..ds.len() as u64 {
                let truth = dot(&rotated[i as usize], &rotated[j as usize]);
                let residual_only = dot(&r_x, &encoder.reconstruct_residual(ds.get(j).values()));
                let est = evaluator.compute_distance(ds.get(j)).0;

                // Errors are measured against ‖x̂‖·‖ŷ‖, the scale of the quantity being estimated;
                // dividing by |truth| would blow up on the pairs that happen to be near-orthogonal.
                let denom = (dot(&rotated[i as usize], &rotated[i as usize])
                    * dot(&rotated[j as usize], &rotated[j as usize]))
                .sqrt();
                worst_est = worst_est.max((est - truth).abs() / denom);
                worst_residual_only =
                    worst_residual_only.max((residual_only - truth).abs() / denom);
            }
        }
        assert!(
            worst_est < 0.02,
            "build-path IP off the true dot product by {worst_est} of ‖x̂‖·‖ŷ‖"
        );
        // Confirms the bound above has teeth: dropping the centroid terms misses by far more.
        assert!(
            worst_residual_only > 0.2,
            "residual-space scoring only off by {worst_residual_only} — fixture centroid too small \
             for this test to discriminate"
        );
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
