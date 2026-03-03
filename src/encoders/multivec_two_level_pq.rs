//! Two-level Product Quantizer for multivector (late-interaction) data.
//!
//! Encoding is two-stage (IVF + residual PQ):
//! 1. **Coarse quantization**: assign each token to the nearest of `ncoarse` centroids
//!    (k-means, L2); store the centroid id as a `u32` little-endian (4 bytes).
//! 2. **Residual PQ**: encode `token − coarse_centroid` with a standard product
//!    quantizer (`M` subspaces, 256 centroids each); store `M` bytes.
//!
//! **Storage layout** (blocked for cache efficiency):
//! ```text
//! [coarse_ids: u32 LE × n_tokens (4×n bytes)] [pq_codes: u8 × n_tokens×M (n×M bytes)]
//! ```
//! All coarse centroid IDs are stored first (contiguous), followed by all PQ codes.
//! This blocked layout provides much better cache locality than per-token interleaving.
//!
//! # Scoring
//!
//! Queries must have exactly [`Q_TOKEN`] tokens (a compile-time constant, default 32).
//! The evaluator precomputes two structures once per query:
//!
//! - **Transposed query layout** `[D × Q_TOKEN]`:
//!   `query_flat_t[d * Q_TOKEN + q]` = value at dimension `d` of query token `q`.
//!   Contiguous Q-slices enable SIMD-friendly inner loops.
//!
//! - **PQ table** `[M × KSUB × Q_TOKEN]`:
//!   `table[m * KSUB * Q_TOKEN + k * Q_TOKEN + q]` = `dot(query_q_subspace_m, pq_centroid_mk)`.
//!   Built via a SAXPY loop over centroid dimensions, broadcasting each scalar to all Q slots.
//!
//! For scoring a document, two phases:
//! 1. **Centroid GEMM**: extract coarse centroid for each doc token into `centroid_cols[doc_n × D]`,
//!    then compute `centroid_scores[doc_n × Q_TOKEN]` via a `D × doc_n × Q_TOKEN` SAXPY loop.
//!    Each centroid dimension is read once and broadcast to all Q_TOKEN accumulators.
//! 2. **MaxSim**: for each doc token, start `acc[Q_TOKEN]` from its centroid scores, add PQ
//!    residual contributions (SAXPY from table), update per-query-token maxima.
//!
//! Both `acc` and `max_scores` are stack-allocated `[f32; Q_TOKEN]` arrays.
//! All inner loops run over the compile-time constant `Q_TOKEN`, enabling full unrolling
//! and mapping to 4 AVX2 `ymm` registers (32 × f32 = 128 bytes).

use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

use matrixmultiply;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Per-phase timing accumulators (in nanoseconds).
// Enabled only when the `phase_timing` cfg flag is set at compile time so
// there is zero overhead in production builds.
// ---------------------------------------------------------------------------
pub static PHASE_CENTROID_EXTRACT_NS: AtomicU64 = AtomicU64::new(0);
pub static PHASE_CENTROID_GEMM_NS: AtomicU64 = AtomicU64::new(0);
pub static PHASE_PQ_MAXSIM_NS: AtomicU64 = AtomicU64::new(0);
pub static PHASE_CALL_COUNT: AtomicU64 = AtomicU64::new(0);
/// Time to build the PQ distance table (once per query evaluator, not per document).
pub static PHASE_TABLE_BUILD_NS: AtomicU64 = AtomicU64::new(0);
pub static PHASE_EVAL_INIT_COUNT: AtomicU64 = AtomicU64::new(0);

/// Reset all phase timing accumulators.
pub fn reset_phase_timings() {
    PHASE_CENTROID_EXTRACT_NS.store(0, Ordering::Relaxed);
    PHASE_CENTROID_GEMM_NS.store(0, Ordering::Relaxed);
    PHASE_PQ_MAXSIM_NS.store(0, Ordering::Relaxed);
    PHASE_CALL_COUNT.store(0, Ordering::Relaxed);
    PHASE_TABLE_BUILD_NS.store(0, Ordering::Relaxed);
    PHASE_EVAL_INIT_COUNT.store(0, Ordering::Relaxed);
}

/// Print a summary of accumulated per-phase timings.
pub fn print_phase_timings() {
    let calls = PHASE_CALL_COUNT.load(Ordering::Relaxed);
    let eval_inits = PHASE_EVAL_INIT_COUNT.load(Ordering::Relaxed);
    let extract_ns = PHASE_CENTROID_EXTRACT_NS.load(Ordering::Relaxed);
    let gemm_ns = PHASE_CENTROID_GEMM_NS.load(Ordering::Relaxed);
    let maxsim_ns = PHASE_PQ_MAXSIM_NS.load(Ordering::Relaxed);
    let table_ns = PHASE_TABLE_BUILD_NS.load(Ordering::Relaxed);
    let total_ns = extract_ns + gemm_ns + maxsim_ns;
    println!("--- evaluator init ({eval_inits} calls) ---");
    if eval_inits > 0 {
        println!(
            "  PQ table build   : {:>10.2} µs/init",
            table_ns as f64 / eval_inits as f64 / 1_000.0,
        );
    }
    if calls == 0 {
        println!("No per-document phase data collected.");
        return;
    }
    println!("--- compute_distance phase breakdown ({calls} calls) ---");
    println!(
        "  Centroid extract : {:>10.2} µs/call  ({:.1}%)",
        extract_ns as f64 / calls as f64 / 1_000.0,
        100.0 * extract_ns as f64 / total_ns as f64,
    );
    println!(
        "  Centroid GEMM    : {:>10.2} µs/call  ({:.1}%)",
        gemm_ns as f64 / calls as f64 / 1_000.0,
        100.0 * gemm_ns as f64 / total_ns as f64,
    );
    println!(
        "  PQ MaxSim        : {:>10.2} µs/call  ({:.1}%)",
        maxsim_ns as f64 / calls as f64 / 1_000.0,
        100.0 * maxsim_ns as f64 / total_ns as f64,
    );
    println!(
        "  Total (3 phases) : {:>10.2} µs/call",
        total_ns as f64 / calls as f64 / 1_000.0,
    );
}

use crate::clustering::KMeansBuilder;
use crate::core::vector::{DenseMultiVectorOwned, DenseMultiVectorView, DenseVectorView};
use crate::core::vector_encoder::{MultiVecEncoder, QueryEvaluator, VectorEncoder};
use crate::distances::{DotProduct, SquaredEuclideanDistance};
use crate::{
    Dataset, DatasetGrowable, Float, PlainDenseDataset, PlainDenseDatasetGrowable,
    PlainDenseQuantizer, SpaceUsage, ValueType, VectorId,
};

/// Number of PQ centroids per subspace (always 256 = 2^8, one u8 code per subspace).
const KSUB: usize = 256;
/// Bytes used to store the coarse centroid id per encoded token (u32 little-endian).
const COARSE_ID_BYTES: usize = 4;
/// Number of query tokens processed per scoring call.
///
/// Fixed at compile time so all inner loops over query tokens are fully unrolled by the
/// compiler and map to 4 AVX2 `ymm` registers (32 × f32 = 128 bytes).
/// Queries must have exactly this many tokens.
pub const Q_TOKEN: usize = 32;

/// Two-level Product Quantizer for multivector (late-interaction) data.
///
/// Combines coarse IVF-style quantization with product quantization on residuals.
/// Each encoded token occupies `M + 4` bytes: 4 bytes for the coarse centroid id
/// followed by `M` bytes of PQ codes for the residual.
///
/// # Type Parameters
/// - `M`: number of PQ subspaces; must divide `token_dim` and be divisible by 4.
/// - `In`: input value type (e.g. `f32`).
///
/// # Training
///
/// Call [`MultiVecTwoLevelProductQuantizer::train`] with a flat dataset of all token
/// vectors and the desired number of coarse centroids.
///
/// # Search (MaxSim via ADC)
///
/// [`MultiVecTwoLevelPQQueryEvaluator`] precomputes a PQ distance table `[M][KSUB][Q]`
/// once per query. For each document token the coarse centroid contribution is computed
/// on the fly via a dot product, then the PQ residual contributions are accumulated
/// via SAXPY from the precomputed table.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MultiVecTwoLevelProductQuantizer<const M: usize, In> {
    /// Dimensionality of each individual token vector (= M * dsub).
    token_dim: usize,
    /// Sub-space dimensionality (= token_dim / M).
    dsub: usize,
    /// Coarse centroids: `ncoarse` vectors of dimension `token_dim`.
    coarse_centroids: PlainDenseDataset<f32, SquaredEuclideanDistance>,
    /// Per-subspace PQ centroids: M datasets, each with KSUB vectors of dimension `dsub`.
    pq_centroids: Box<[PlainDenseDataset<f32, SquaredEuclideanDistance>]>,
    _phantom: PhantomData<In>,
}

impl<const M: usize, In> MultiVecTwoLevelProductQuantizer<M, In> {
    /// Construct from pre-trained components.
    ///
    /// Panics if `M % 4 != 0`, `token_dim % M != 0`, or any centroid dataset has the
    /// wrong number of vectors or dimension.
    pub fn from_pretrained(
        token_dim: usize,
        coarse_centroids: PlainDenseDataset<f32, SquaredEuclideanDistance>,
        pq_centroids: Vec<PlainDenseDataset<f32, SquaredEuclideanDistance>>,
    ) -> Self {
        assert_eq!(M % 4, 0, "M ({}) must be divisible by 4", M);
        assert_eq!(
            token_dim % M,
            0,
            "token_dim ({}) must be divisible by M ({})",
            token_dim,
            M
        );
        let dsub = token_dim / M;
        assert_eq!(
            coarse_centroids.output_dim(),
            token_dim,
            "coarse_centroids dim ({}) must equal token_dim ({})",
            coarse_centroids.output_dim(),
            token_dim
        );
        assert_eq!(
            pq_centroids.len(),
            M,
            "Expected {} PQ centroid datasets, got {}",
            M,
            pq_centroids.len()
        );
        for (i, ds) in pq_centroids.iter().enumerate() {
            assert_eq!(
                ds.len(),
                KSUB,
                "Subspace {} has {} centroids, expected {}",
                i,
                ds.len(),
                KSUB
            );
            assert_eq!(
                ds.output_dim(),
                dsub,
                "Subspace {} has dim {}, expected {}",
                i,
                ds.output_dim(),
                dsub
            );
        }
        Self {
            token_dim,
            dsub,
            coarse_centroids,
            pq_centroids: pq_centroids.into_boxed_slice(),
            _phantom: PhantomData,
        }
    }

    /// Train a two-level product quantizer on a flat dataset of individual token vectors.
    ///
    /// Training proceeds in three steps:
    /// 1. K-means on all token vectors → `ncoarse` coarse centroids (L2).
    /// 2. Compute residuals in parallel: `residual[i] = token[i] − nearest_coarse_centroid[i]`.
    /// 3. Per-subspace k-means on residuals (parallel over `M` subspaces) → PQ centroids.
    ///
    /// Panics if `M % 4 != 0` or `token_dim % M != 0`.
    pub fn train(
        token_vectors: &PlainDenseDataset<f32, SquaredEuclideanDistance>,
        ncoarse: usize,
    ) -> Self {
        let token_dim = token_vectors.output_dim();
        let n = token_vectors.len();

        assert_eq!(M % 4, 0, "M ({}) must be divisible by 4", M);
        assert_eq!(
            token_dim % M,
            0,
            "token_dim ({}) must be divisible by M ({})",
            token_dim,
            M
        );
        let dsub = token_dim / M;

        println!(
            "Training MultiVecTwoLevelProductQuantizer: {} tokens × {} dims, \
             ncoarse={}, M={}, dsub={}, KSUB={}",
            n, token_dim, ncoarse, M, dsub, KSUB
        );

        // Step 1: Train coarse centroids via k-means.
        println!("  Step 1: training {} coarse centroids...", ncoarse);
        let coarse_centroids: PlainDenseDataset<f32, SquaredEuclideanDistance> =
            KMeansBuilder::new()
                .build()
                .train(token_vectors, ncoarse, None);

        // Step 2: Compute residuals in parallel.
        println!("  Step 2: computing residuals ({} vectors)...", n);
        let mut residuals_flat = vec![0f32; n * token_dim];
        {
            let coarse_ref = &coarse_centroids;
            residuals_flat
                .par_chunks_mut(token_dim)
                .enumerate()
                .for_each(|(i, res)| {
                    let token = token_vectors.get(i as VectorId);
                    let nearest = coarse_ref
                        .search_nearest(token)
                        .map(|s| s.vector)
                        .unwrap_or(0);
                    let centroid = coarse_ref.get(nearest);
                    for (r, (&v, &c)) in res
                        .iter_mut()
                        .zip(token.values().iter().zip(centroid.values()))
                    {
                        *r = v - c;
                    }
                });
        }

        // Wrap residuals in a PlainDenseDataset for sub-space extraction.
        let residuals_ds = PlainDenseDataset::<f32, SquaredEuclideanDistance>::from_raw(
            residuals_flat.into_boxed_slice(),
            n,
            PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(token_dim),
        );

        // Step 3: Per-subspace k-means on residuals, parallel over M subspaces.
        println!(
            "  Step 3: training PQ ({} subspaces × {} centroids each)...",
            M, KSUB
        );
        let pq_centroids: Vec<PlainDenseDataset<f32, SquaredEuclideanDistance>> = (0..M)
            .into_par_iter()
            .map(|m| {
                let q = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(dsub);
                let mut sub_ds =
                    PlainDenseDatasetGrowable::<f32, SquaredEuclideanDistance>::with_capacity(q, n);
                for vec in residuals_ds.iter() {
                    sub_ds.push(DenseVectorView::new(
                        &vec.values()[m * dsub..(m + 1) * dsub],
                    ));
                }
                KMeansBuilder::new().build().train(&sub_ds, KSUB, None)
            })
            .collect();

        println!("  Training complete.");

        Self {
            token_dim,
            dsub,
            coarse_centroids,
            pq_centroids: pq_centroids.into_boxed_slice(),
            _phantom: PhantomData,
        }
    }

    /// Number of coarse centroids.
    #[inline]
    pub fn ncoarse(&self) -> usize {
        self.coarse_centroids.len()
    }

    /// Dimensionality of each individual token vector.
    #[inline]
    pub fn token_dim(&self) -> usize {
        self.token_dim
    }

    /// Sub-space dimensionality (`token_dim / M`).
    #[inline]
    pub fn dsub(&self) -> usize {
        self.dsub
    }

    /// Number of PQ centroids per subspace (always 256).
    #[inline]
    pub fn ksub(&self) -> usize {
        KSUB
    }

    /// Decode an encoded multivector back to f32 token vectors.
    ///
    /// For each encoded token the reconstructed vector is
    /// `coarse_centroid[coarse_id] + Σ_m pq_centroid[m][code_m]`.
    ///
    /// Handles blocked layout: first 4*n bytes are coarse IDs, then n*M bytes are PQ codes.
    fn decode_multivec(&self, encoded: DenseMultiVectorView<'_, u8>) -> DenseMultiVectorOwned<f32> {
        let n_tokens = encoded.num_vecs();
        let all_bytes = encoded.values();
        let codes_offset = 4 * n_tokens;

        let mut values = Vec::with_capacity(n_tokens * self.token_dim);

        for t in 0..n_tokens {
            // Read coarse_id from blocked layout
            let byte_offset = t * 4;
            let coarse_id = u32::from_le_bytes([
                all_bytes[byte_offset],
                all_bytes[byte_offset + 1],
                all_bytes[byte_offset + 2],
                all_bytes[byte_offset + 3],
            ]) as VectorId;

            let centroid = self.coarse_centroids.get(coarse_id);
            let mut decoded: Vec<f32> = centroid.values().to_vec();

            // Read PQ codes from blocked layout
            for m in 0..M {
                let code = all_bytes[codes_offset + t * M + m] as VectorId;
                let pq_centroid = self.pq_centroids[m].get(code);
                let sub_start = m * self.dsub;
                for (j, &v) in pq_centroid.values().iter().enumerate() {
                    decoded[sub_start + j] += v;
                }
            }
            values.extend_from_slice(&decoded);
        }
        DenseMultiVectorOwned::new(values, self.token_dim)
    }
}

/// Query evaluator for [`MultiVecTwoLevelProductQuantizer`].
///
/// Requires exactly [`Q_TOKEN`] query tokens. Precomputes two structures:
///
/// - **Transposed query** `query_flat_t[d * Q_TOKEN + q]`: contiguous Q-slices per dimension,
///   enabling SIMD-friendly SAXPY for both the PQ table and the centroid GEMM.
/// - **PQ table** `[M × KSUB × Q_TOKEN]`: built via SAXPY over PQ centroid dimensions.
///
/// Scoring phases:
/// 1. Extract coarse centroid columns `[doc_n × D]` in one pass.
/// 2. GEMM (D-outer, doc_n-middle, Q_TOKEN-inner SAXPY) → `centroid_scores[doc_n × Q_TOKEN]`.
/// 3. Per doc token: `acc[Q_TOKEN]` = centroid row + PQ SAXPY; update `max_scores[Q_TOKEN]`.
///
/// `acc` and `max_scores` are stack arrays — no heap allocation in the hot path.
pub struct MultiVecTwoLevelPQQueryEvaluator<'e, const M: usize, In> {
    /// Reference to the encoder for coarse centroid lookups at scoring time.
    encoder: &'e MultiVecTwoLevelProductQuantizer<M, In>,
    /// Transposed query layout `[D × Q_TOKEN]`: `query_flat_t[d * Q_TOKEN + q] = query[q][d]`.
    /// Contiguous Q_TOKEN-wide rows per dimension enable SIMD-friendly broadcast.
    query_flat_t: Vec<f32>,
    /// PQ residual distance table `[M × KSUB × Q_TOKEN]`:
    /// `table[m * KSUB * Q_TOKEN + k * Q_TOKEN + q]` = `dot(query_q_subspace_m, pq_centroid_mk)`.
    distance_table: Vec<f32>,
}

impl<'e, const M: usize, In> MultiVecTwoLevelPQQueryEvaluator<'e, M, In>
where
    In: ValueType,
{
    fn new(
        encoder: &'e MultiVecTwoLevelProductQuantizer<M, In>,
        query: DenseMultiVectorView<'_, f32>,
    ) -> Self {
        assert_eq!(
            query.num_vecs(),
            Q_TOKEN,
            "Query must have exactly Q_TOKEN={} tokens, got {}",
            Q_TOKEN,
            query.num_vecs()
        );
        let token_dim = encoder.token_dim;
        let dsub = encoder.dsub;
        let query_vals = query.values();

        // Build transposed query layout [D × Q_TOKEN].
        // query_flat_t[d * Q_TOKEN + q] = query[q][d]
        // Contiguous Q_TOKEN-wide rows allow the inner `for q in 0..Q_TOKEN` loops
        // to be a simple SIMD load + fused-multiply-add broadcast.
        let mut query_flat_t = vec![0f32; token_dim * Q_TOKEN];
        for qi in 0..Q_TOKEN {
            for d in 0..token_dim {
                query_flat_t[d * Q_TOKEN + qi] = query_vals[qi * token_dim + d];
            }
        }

        // Build PQ table [M × KSUB × Q_TOKEN] via SAXPY over centroid dimensions.
        let t_table = std::time::Instant::now();
        let mut distance_table = vec![0f32; M * KSUB * Q_TOKEN];
        unsafe {
            let qt_ptr = query_flat_t.as_ptr();
            let dt_ptr = distance_table.as_mut_ptr();

            for m in 0..M {
                let sub_offset = m * dsub;
                for k in 0..KSUB {
                    let centroid = encoder.pq_centroids[m].get(k as VectorId);
                    let entry_base = m * KSUB * Q_TOKEN + k * Q_TOKEN;
                    let centroid_vals = centroid.values();

                    for (d_sub, &c) in centroid_vals.iter().enumerate() {
                        let d = sub_offset + d_sub;
                        let qt_base = qt_ptr.add(d * Q_TOKEN);
                        let out_base = dt_ptr.add(entry_base);

                        for q in 0..Q_TOKEN {
                            let qt_val = *qt_base.add(q);
                            let current = *out_base.add(q);
                            *out_base.add(q) = current.algebraic_add(qt_val.algebraic_mul(c));
                        }
                    }
                }
            }
        }
        PHASE_TABLE_BUILD_NS.fetch_add(t_table.elapsed().as_nanos() as u64, Ordering::Relaxed);
        PHASE_EVAL_INIT_COUNT.fetch_add(1, Ordering::Relaxed);

        Self {
            encoder,
            query_flat_t,
            distance_table,
        }
    }
}

impl<'e, 'v, const M: usize, In> QueryEvaluator<DenseMultiVectorView<'v, u8>>
    for MultiVecTwoLevelPQQueryEvaluator<'e, M, In>
where
    In: ValueType,
{
    type Distance = DotProduct;

    /// Compute MaxSim between the stored query and a two-level PQ-encoded document multivector.
    ///
    /// **Phase 1 — Centroid GEMM**: extract `centroid_cols[doc_n × D]` (one centroid copy per
    /// doc token), then compute `centroid_scores[doc_n × Q_TOKEN]` via the optimized
    /// `two_level_pq_centroid_gemm` function in distances.rs using unsafe pointer arithmetic
    /// and algebraic FMA operations.
    ///
    /// **Phase 2 — MaxSim**: calls the optimized `two_level_pq_maxsim` function from distances.rs
    /// which uses unsafe pointer arithmetic and algebraic operations for maximum performance.
    ///
    /// Returns the sum of per-query-token maxima as a [`DotProduct`].
    fn compute_distance(&self, vector: DenseMultiVectorView<'v, u8>) -> DotProduct {
        let doc_n = vector.num_vecs();
        let token_dim = self.encoder.token_dim;

        // Phase 1a: extract coarse centroid for each doc token into centroid_cols [doc_n × D].
        let t0 = std::time::Instant::now();
        let mut centroid_cols = vec![0f32; doc_n * token_dim];
        let doc_bytes = vector.values();

        // SAFETY: Blocked layout guarantees first 4*doc_n bytes are coarse IDs
        unsafe {
            let doc_ptr = doc_bytes.as_ptr();
            let centroids_data = self.encoder.coarse_centroids.values();
            let centroids_ptr = centroids_data.as_ptr();
            let dest_ptr = centroid_cols.as_mut_ptr();

            for t in 0..doc_n {
                let byte_offset = t * 4;
                let c0 = *doc_ptr.add(byte_offset) as u32;
                let c1 = *doc_ptr.add(byte_offset + 1) as u32;
                let c2 = *doc_ptr.add(byte_offset + 2) as u32;
                let c3 = *doc_ptr.add(byte_offset + 3) as u32;
                let coarse_id = (c0 | (c1 << 8) | (c2 << 16) | (c3 << 24)) as usize;

                let src = centroids_ptr.add(coarse_id * token_dim);
                let dst = dest_ptr.add(t * token_dim);
                std::ptr::copy_nonoverlapping(src, dst, token_dim);
            }
        }
        PHASE_CENTROID_EXTRACT_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // Phase 1b: GEMM centroid contributions → centroid_scores [doc_n × Q_TOKEN].
        // matrixmultiply::sgemm: pure-Rust cache-blocking micro-kernel that correctly
        // holds accumulator registers across the full k-loop regardless of surrounding
        // LTO register pressure (which caused LLVM to horizontally reduce our manual
        // [f32; Q] acc array per (t,q) pair instead of keeping it in 4 YMM regs).
        let t1 = std::time::Instant::now();
        let mut centroid_scores = vec![0f32; doc_n * Q_TOKEN];
        unsafe {
            matrixmultiply::sgemm(
                doc_n,     // m: one row per doc token
                token_dim, // k: contract over token dimensions
                Q_TOKEN,   // n: one column per query token
                1.0,       // alpha
                centroid_cols.as_ptr(),
                token_dim as isize, // rsa: row stride of A (centroid_cols is [doc_n × token_dim])
                1,                  // csa: col stride of A
                self.query_flat_t.as_ptr(),
                Q_TOKEN as isize, // rsb: row stride of B (query_flat_t is [token_dim × Q_TOKEN])
                1,                // csb: col stride of B
                0.0,              // beta
                centroid_scores.as_mut_ptr(),
                Q_TOKEN as isize, // rsc: row stride of C
                1,                // csc: col stride of C
            );
        }
        PHASE_CENTROID_GEMM_NS.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // Phase 2: MaxSim with PQ residuals.
        let t2 = std::time::Instant::now();
        let doc_bytes = vector.values();
        let codes_offset = 4 * doc_n;
        let score = unsafe {
            crate::distances::two_level_pq_maxsim_blocked::<M, Q_TOKEN>(
                &centroid_scores,
                &self.distance_table,
                &doc_bytes[codes_offset..],
                doc_n,
            )
        };
        PHASE_PQ_MAXSIM_NS.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);
        PHASE_CALL_COUNT.fetch_add(1, Ordering::Relaxed);

        DotProduct::from(score)
    }
}

impl<const M: usize, In> VectorEncoder for MultiVecTwoLevelProductQuantizer<M, In>
where
    In: ValueType + Float,
{
    type Distance = DotProduct;
    type InputVector<'a> = DenseMultiVectorView<'a, In>;
    type QueryVector<'q> = DenseMultiVectorView<'q, f32>;
    type EncodedVector<'a> = DenseMultiVectorView<'a, u8>;

    type Evaluator<'e>
        = MultiVecTwoLevelPQQueryEvaluator<'e, M, In>
    where
        Self: 'e;

    #[inline]
    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        assert_eq!(
            query.dim(),
            self.token_dim,
            "Query dim ({}) must match token_dim ({})",
            query.dim(),
            self.token_dim
        );
        MultiVecTwoLevelPQQueryEvaluator::new(self, query)
    }

    #[inline]
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = self.decode_multivec(vector);
        MultiVecTwoLevelPQQueryEvaluator::new(self, decoded.as_view())
    }

    #[inline]
    fn input_dim(&self) -> usize {
        self.token_dim
    }

    #[inline]
    fn output_dim(&self) -> usize {
        COARSE_ID_BYTES + M
    }
}

impl<const M: usize, In> MultiVecEncoder for MultiVecTwoLevelProductQuantizer<M, In>
where
    In: ValueType + Float,
{
    type InputValueType = In;
    type OutputValueType = u8;

    /// Encode each token of the input multivector.
    ///
    /// Uses a **blocked layout** for better cache performance:
    /// 1. Process all tokens: find nearest coarse centroid (L2), compute residuals, PQ encode
    /// 2. Write all coarse IDs as a contiguous block (4 bytes × n_tokens)
    /// 3. Write all PQ codes as a contiguous block (M bytes × n_tokens)
    ///
    /// This layout dramatically improves scoring performance vs per-token interleaving.
    #[inline]
    fn push_encoded<'a, OutputContainer>(
        &self,
        input: DenseMultiVectorView<'a, In>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<u8>,
    {
        let n_tokens = input.num_vecs();
        let mut token_f32 = vec![0f32; self.token_dim];
        let mut residual = vec![0f32; self.token_dim];

        // Temporary buffers for blocked output
        let mut coarse_ids = Vec::with_capacity(n_tokens);
        let mut pq_codes = Vec::with_capacity(n_tokens * M);

        for token in input.iter_vectors() {
            // Convert to f32.
            for (dst, &src) in token_f32.iter_mut().zip(token.values()) {
                // SAFETY: Float::to_f32() is infallible for all Float-bounded types.
                *dst = unsafe { src.to_f32().unwrap_unchecked() };
            }

            // Find nearest coarse centroid (L2 via search_nearest).
            let token_view = DenseVectorView::new(&token_f32);
            let coarse_id = self
                .coarse_centroids
                .search_nearest(token_view)
                .map(|s| s.vector as u32)
                .unwrap_or(0);
            coarse_ids.push(coarse_id);

            // Compute residual = token − coarse_centroid.
            let centroid = self.coarse_centroids.get(coarse_id as VectorId);
            for (r, (&v, &c)) in residual
                .iter_mut()
                .zip(token_f32.iter().zip(centroid.values()))
            {
                *r = v - c;
            }

            // PQ encode each subspace of the residual.
            for m in 0..M {
                let sub = DenseVectorView::new(&residual[m * self.dsub..(m + 1) * self.dsub]);
                let code = self.pq_centroids[m]
                    .search_nearest(sub)
                    .map(|s| s.vector as u8)
                    .unwrap_or(0);
                pq_codes.push(code);
            }
        }

        // Write blocked layout: all coarse IDs first, then all PQ codes
        for &id in &coarse_ids {
            output.extend(id.to_le_bytes());
        }
        output.extend(pq_codes);
    }

    /// Decode an encoded multivector via coarse centroid + PQ residual lookup.
    fn decode_vector<'a>(
        &self,
        encoded: DenseMultiVectorView<'a, Self::OutputValueType>,
    ) -> DenseMultiVectorOwned<f32> {
        self.decode_multivec(encoded)
    }
}

impl<const M: usize, In> SpaceUsage for MultiVecTwoLevelProductQuantizer<M, In>
where
    In: ValueType,
{
    fn space_usage_bytes(&self) -> usize {
        2 * std::mem::size_of::<usize>() // token_dim, dsub
            + self.coarse_centroids.space_usage_bytes()
            + self
                .pq_centroids
                .iter()
                .map(|ds| ds.space_usage_bytes())
                .sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::distances::Distance;
    use crate::core::vector::{DenseMultiVectorView, DenseVectorView};
    use crate::{
        PlainDenseDataset, PlainDenseDatasetGrowable, PlainDenseQuantizer, SquaredEuclideanDistance,
    };

    fn make_training_set(
        token_dim: usize,
        n_vecs: usize,
    ) -> PlainDenseDataset<f32, SquaredEuclideanDistance> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);
        let quantizer = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(token_dim);
        let mut ds = PlainDenseDatasetGrowable::<f32, SquaredEuclideanDistance>::with_capacity(
            quantizer, n_vecs,
        );
        for _ in 0..n_vecs {
            let values: Vec<f32> = (0..token_dim)
                .map(|_| rng.gen_range(-1.0_f32..1.0))
                .collect();
            ds.push(DenseVectorView::new(&values));
        }
        ds.into()
    }

    #[test]
    fn train_dimensions_are_correct() {
        const M: usize = 4;
        let token_dim = 8;
        let ncoarse = 4;
        let training = make_training_set(token_dim, 512);
        let encoder = MultiVecTwoLevelProductQuantizer::<M, f32>::train(&training, ncoarse);

        assert_eq!(encoder.token_dim(), token_dim);
        assert_eq!(encoder.dsub(), token_dim / M);
        assert_eq!(encoder.ksub(), KSUB);
        assert_eq!(encoder.ncoarse(), ncoarse);
        assert_eq!(encoder.input_dim(), token_dim);
        assert_eq!(encoder.output_dim(), COARSE_ID_BYTES + M);
    }

    #[test]
    fn encode_produces_correct_byte_count() {
        const M: usize = 4;
        let token_dim = 8;
        let training = make_training_set(token_dim, 512);
        let encoder = MultiVecTwoLevelProductQuantizer::<M, f32>::train(&training, 4);

        let doc_vals: Vec<f32> = vec![0.5f32; 3 * token_dim]; // 3 tokens
        let doc = DenseMultiVectorView::new(&doc_vals, token_dim);
        let mut encoded = Vec::new();
        encoder.push_encoded(doc, &mut encoded);

        // 3 tokens × (COARSE_ID_BYTES + M) bytes each
        assert_eq!(encoded.len(), 3 * (COARSE_ID_BYTES + M));
    }

    #[test]
    fn query_evaluator_produces_finite_score() {
        const M: usize = 4;
        let token_dim = 8;
        let training = make_training_set(token_dim, 512);
        let encoder = MultiVecTwoLevelProductQuantizer::<M, f32>::train(&training, 4);

        let query_vals: Vec<f32> = vec![0.5f32; Q_TOKEN * token_dim]; // Q_TOKEN query tokens
        let query = DenseMultiVectorView::new(&query_vals, token_dim);

        let doc_vals: Vec<f32> = vec![0.25f32; 3 * token_dim]; // 3 doc tokens
        let doc = DenseMultiVectorView::new(&doc_vals, token_dim);
        let mut encoded_doc = Vec::new();
        encoder.push_encoded(doc, &mut encoded_doc);

        let evaluator = encoder.query_evaluator(query);
        let encoded_view = DenseMultiVectorView::new(&encoded_doc, COARSE_ID_BYTES + M);
        let dist = evaluator.compute_distance(encoded_view);

        assert!(dist.distance().is_finite());
    }

    #[test]
    fn decode_restores_correct_shape() {
        const M: usize = 4;
        let token_dim = 8;
        let training = make_training_set(token_dim, 512);
        let encoder = MultiVecTwoLevelProductQuantizer::<M, f32>::train(&training, 4);

        let doc_vals: Vec<f32> = vec![0.1f32; 2 * token_dim]; // 2 tokens
        let doc = DenseMultiVectorView::new(&doc_vals, token_dim);
        let mut encoded = Vec::new();
        encoder.push_encoded(doc, &mut encoded);

        let encoded_view = DenseMultiVectorView::new(&encoded, COARSE_ID_BYTES + M);
        let decoded = encoder.decode_multivec(encoded_view);

        assert_eq!(decoded.dim(), token_dim);
        assert_eq!(decoded.num_vecs(), 2);
    }

    #[test]
    #[should_panic(expected = "Query dim")]
    fn panics_on_mismatched_query_dim() {
        const M: usize = 4;
        let training = make_training_set(8, 512);
        let encoder = MultiVecTwoLevelProductQuantizer::<M, f32>::train(&training, 4);
        // dim = 2 doesn't match token_dim = 8
        let query = DenseMultiVectorView::new(&[1.0f32, 0.0], 2);
        encoder.query_evaluator(query);
    }
}
