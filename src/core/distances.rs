//! Distance types and utilities.
//!
//! Provides the `Distance` trait (a small wrapper around an `f32` value)
//! and concrete distance types that implement comparison.
//!
//! NaN is NOT permitted for distance values; passing one is undefined behaviour.
//! This is enforced via `debug_assert!` in `From<f32>` so the check is active in
//! debug/test builds but compiled away in release builds.
//!
//! `Ord` is implemented via `partial_cmp(...).unwrap_or(Equal)` — NaN-free inputs
//! are assumed, so the `unwrap_or` branch is unreachable in practice.
//!
//! `DotProduct` implements reversed ordering: larger values are considered better.

use crate::core::vector::{DenseMultiVectorView, DenseVectorView, SparseVectorView};
use crate::utils::is_strictly_sorted;
use crate::{ComponentType, ValueType};

use std::hint::assert_unchecked;

/// A simple trait representing a distance-like value (stored as `f32`).
pub trait Distance: Ord + Copy + Send + Sync + 'static {
    /// Return the numeric distance value.
    fn distance(&self) -> f32;

    /// Returns true if self satisfies a relaxed threshold comparison.
    ///
    /// The relaxation parameter `lambda` (typically 0.0 to 0.5) controls how much
    /// wider the acceptance window becomes:
    /// - For minimize metrics (Euclidean, squared distances): `self ≤ threshold × (1 + λ)`
    /// - For maximize metrics (DotProduct, similarities): `self ≥ threshold - |threshold| × λ`
    ///
    /// When `lambda = 0.0`, this is equivalent to the standard `self <= threshold` (for minimize)
    /// or `self >= threshold` (for maximize) comparison.
    ///
    /// # Arguments
    /// * `threshold` - The base threshold distance to compare against
    /// * `lambda` - Relaxation factor (must be finite and ≥ 0.0). Common values: 0.0 to 0.5
    ///
    /// # Returns
    /// `true` if self is within the relaxed bound, `false` otherwise
    fn is_within_relaxation(&self, threshold: &Self, lambda: f32) -> bool;
}

/// Squared Euclidean distance wrapper around an `f32`.
#[derive(Copy, Clone, Debug, Default)]
pub struct SquaredEuclideanDistance(f32);

impl Distance for SquaredEuclideanDistance {
    fn distance(&self) -> f32 {
        self.0
    }

    #[inline]
    fn is_within_relaxation(&self, threshold: &Self, lambda: f32) -> bool {
        debug_assert!(lambda.is_finite(), "lambda must be finite");
        debug_assert!(lambda >= 0.0, "lambda must be non-negative");
        // Minimize metric: accept if candidate ≤ threshold × (1 + λ)
        self.0 <= threshold.0 * (1.0 + lambda)
    }
}

impl SquaredEuclideanDistance {
    /// Returns the actual Euclidean distance by taking the square root.
    #[inline]
    pub fn sqrt(&self) -> f32 {
        self.0.sqrt()
    }
}

impl From<f32> for SquaredEuclideanDistance {
    fn from(v: f32) -> Self {
        debug_assert!(
            !v.is_nan(),
            "NaN is not allowed for SquaredEuclideanDistance"
        );
        SquaredEuclideanDistance(v)
    }
}

impl PartialEq for SquaredEuclideanDistance {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl PartialOrd for SquaredEuclideanDistance {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for SquaredEuclideanDistance {}

impl Ord for SquaredEuclideanDistance {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Dot product score wrapper. Higher dot-product is better.
#[derive(Copy, Clone, Debug, Default)]
pub struct DotProduct(pub f32);

impl Distance for DotProduct {
    fn distance(&self) -> f32 {
        self.0
    }

    #[inline]
    fn is_within_relaxation(&self, threshold: &Self, lambda: f32) -> bool {
        debug_assert!(lambda.is_finite(), "lambda must be finite");
        debug_assert!(lambda >= 0.0, "lambda must be non-negative");
        // Maximize metric: accept if candidate ≥ threshold - |threshold| × λ
        self.0 >= threshold.0 - threshold.0.abs() * lambda
    }
}

impl From<f32> for DotProduct {
    fn from(v: f32) -> Self {
        debug_assert!(!v.is_nan(), "NaN is not allowed for DotProduct");
        DotProduct(v)
    }
}

impl PartialEq for DotProduct {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl PartialOrd for DotProduct {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for DotProduct {}

impl Ord for DotProduct {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .0
            .partial_cmp(&self.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Computes the dot product between a dense query and a sparse vector.
#[cfg_attr(test, inline(never))]
#[cfg_attr(not(test), inline)]
#[must_use]
pub fn dot_product_dense_sparse<C, Q, V>(
    query: DenseVectorView<'_, Q>,
    vector: SparseVectorView<'_, C, V>,
) -> DotProduct
where
    C: ComponentType,
    Q: ValueType,
    V: ValueType,
{
    let query_len = query.len();
    for &c in vector.components() {
        assert!(
            c.as_() < query_len,
            "sparse vector component {} is out of bounds for query of length {}",
            c.as_(),
            query_len
        );
    }
    // SAFETY: We just validated that all components are within bounds.
    unsafe { dot_product_dense_sparse_unchecked(query, vector) }
}

/// Computes the dot product between a dense query and a sparse vector (unchecked).
///
/// # Safety
/// Caller must ensure that all components in `vector` are valid indices in `query`.
#[cfg_attr(test, inline(never))]
#[cfg_attr(not(test), inline)]
#[must_use]
pub unsafe fn dot_product_dense_sparse_unchecked<C, Q, V>(
    query: DenseVectorView<'_, Q>,
    vector: SparseVectorView<'_, C, V>,
) -> DotProduct
where
    C: ComponentType,
    Q: ValueType,
    V: ValueType,
{
    let query_slice = query.values();
    vector
        .iter()
        .map(|(c, v)| {
            unsafe { *query_slice.get_unchecked(c.as_()) }
                .to_f32()
                .unwrap()
                .algebraic_mul(v.to_f32().unwrap())
        })
        .fold(0.0, |acc: f32, x| acc.algebraic_add(x))
        .into()
}

/// Computes the dot product between a dense query vector and 6 sparse vectors without bounds checking.
///
/// # Safety
///
/// The caller must ensure that all component indices in the sparse vectors are valid indices
/// into the query vector's values array. Using invalid indices will result in undefined behavior.
pub unsafe fn dot_product_dense_sparse_batch6_unchecked<C, Q, V>(
    query: DenseVectorView<'_, Q>,
    vectors: [SparseVectorView<'_, C, V>; 6],
) -> [DotProduct; 6]
where
    C: ComponentType,
    Q: ValueType,
    V: ValueType,
{
    let q = query.values();

    let mut accs = [0.0f32; 6];
    let mut lens = [0usize; 6];
    let comps = [
        vectors[0].components(),
        vectors[1].components(),
        vectors[2].components(),
        vectors[3].components(),
        vectors[4].components(),
        vectors[5].components(),
    ];
    let vals = [
        vectors[0].values(),
        vectors[1].values(),
        vectors[2].values(),
        vectors[3].values(),
        vectors[4].values(),
        vectors[5].values(),
    ];

    for i in 0..6 {
        lens[i] = comps[i].len();
    }

    let min_len = lens.iter().copied().min().unwrap_or(0);

    let mut i = 0;
    while i < min_len {
        for j in 0..6 {
            let q_val = unsafe { *q.get_unchecked((*comps[j].get_unchecked(i)).as_()) }
                .to_f32()
                .unwrap();
            let v_val = unsafe { (*vals[j].get_unchecked(i)).to_f32().unwrap() };
            accs[j] = accs[j].algebraic_add(q_val.algebraic_mul(v_val));
        }
        i += 1;
    }

    // Handle remaining elements for each vector
    for j in 0..6 {
        let mut ij = min_len;
        while ij < lens[j] {
            let q_val = unsafe { *q.get_unchecked((*comps[j].get_unchecked(ij)).as_()) }
                .to_f32()
                .unwrap();
            let v_val = unsafe { (*vals[j].get_unchecked(ij)).to_f32().unwrap() };
            accs[j] = accs[j].algebraic_add(q_val.algebraic_mul(v_val));
            ij += 1;
        }
    }

    [
        DotProduct::from(accs[0]),
        DotProduct::from(accs[1]),
        DotProduct::from(accs[2]),
        DotProduct::from(accs[3]),
        DotProduct::from(accs[4]),
        DotProduct::from(accs[5]),
    ]
}

/// Computes the dot product between two sparse vectors using merge style.
#[cfg_attr(test, inline(never))]
#[cfg_attr(not(test), inline)]
#[must_use]
pub fn dot_product_sparse_with_merge<C, Q, V>(
    query: SparseVectorView<'_, C, Q>,
    vector: SparseVectorView<'_, C, V>,
) -> DotProduct
where
    C: ComponentType,
    Q: ValueType,
    V: ValueType,
{
    assert!(
        is_strictly_sorted(query.components()),
        "query components must be sorted in strictly ascending order"
    );
    assert!(
        is_strictly_sorted(vector.components()),
        "vector components must be sorted in strictly ascending order"
    );
    unsafe { dot_product_sparse_with_merge_unchecked(query, vector) }
}

/// Computes the dot product between two sparse vectors using merge style (unchecked).
/// # Safety
/// * Both `query` and `vector` must be sorted in strictly ascending component order.
/// * Every component index encountered must be valid for the provided slices to avoid out-of-bounds access.
/// * The caller must not drop the slices while this function runs.
#[cfg_attr(test, inline(never))]
#[cfg_attr(not(test), inline)]
#[must_use]
pub unsafe fn dot_product_sparse_with_merge_unchecked<C, Q, V>(
    query: SparseVectorView<'_, C, Q>,
    vector: SparseVectorView<'_, C, V>,
) -> DotProduct
where
    C: ComponentType,
    Q: ValueType,
    V: ValueType,
{
    debug_assert!(is_strictly_sorted(query.components()));
    debug_assert!(is_strictly_sorted(vector.components()));

    let mut result = 0.0;
    let mut v_iter = vector.iter();
    let mut current = v_iter.next();
    let b = current.is_some();

    for (q_id, q_v) in query.iter() {
        if b {
            unsafe { assert_unchecked(current.is_some()) }
        }
        while let Some((v_id, _)) = current
            && v_id < q_id
        {
            current = v_iter.next();
        }
        if !b {
            unsafe { assert_unchecked(current.is_none()) }
        }
        match current {
            Some((v_id, v_v)) if v_id == q_id => {
                result += v_v.to_f32().unwrap() * q_v.to_f32().unwrap();
            }
            None => {
                break;
            }
            _ => {}
        }
    }
    result.into()
}

/// Computes the dot product between two dense vectors.
#[cfg_attr(test, inline(never))]
#[cfg_attr(not(test), inline)]
#[must_use]
pub fn dot_product_dense<Q, V>(
    query: DenseVectorView<'_, Q>,
    vector: DenseVectorView<'_, V>,
) -> DotProduct
where
    Q: ValueType,
    V: ValueType,
{
    assert_eq!(
        query.len(),
        vector.len(),
        "query and vector must have the same length"
    );
    unsafe { dot_product_dense_unchecked(query, vector) }
}

/// Computes the dot product between two dense vectors (unchecked).
/// # Safety
/// The caller must ensure that `query.len() == vector.len()` to keep the zipped iteration within bounds.
#[cfg_attr(test, inline(never))]
#[cfg_attr(not(test), inline)]
#[must_use]
pub unsafe fn dot_product_dense_unchecked<Q, V>(
    query: DenseVectorView<'_, Q>,
    vector: DenseVectorView<'_, V>,
) -> DotProduct
where
    Q: ValueType,
    V: ValueType,
{
    unsafe { assert_unchecked(query.len() == vector.len()) };
    query
        .iter()
        .zip(vector.iter())
        .map(|(q, v)| q.to_f32().unwrap().algebraic_mul(v.to_f32().unwrap()))
        .fold(0.0, |acc: f32, x| acc.algebraic_add(x))
        .into()
}

/// Computes the dot product between a dense query and six dense vectors in a single pass.
///
/// Uses 6 independent partial-sum accumulators so the CPU can execute multiple multiply-add chains
/// simultaneously (better ILP) and reuse the query data before eviction (better cache behaviour).
///
/// # Safety
/// Caller must ensure `query.len() == vi.len()` for all six vectors.
#[inline]
#[must_use]
pub unsafe fn dot_product_dense_batch6_unchecked<Q, V>(
    query: DenseVectorView<'_, Q>,
    [v0, v1, v2, v3, v4, v5]: [DenseVectorView<'_, V>; 6],
) -> [DotProduct; 6]
where
    Q: ValueType,
    V: ValueType,
{
    let d = query.len();
    let q = query.values();
    let a0 = v0.values();
    let a1 = v1.values();
    let a2 = v2.values();
    let a3 = v3.values();
    let a4 = v4.values();
    let a5 = v5.values();

    let mut r0 = 0.0f32;
    let mut r1 = 0.0f32;
    let mut r2 = 0.0f32;
    let mut r3 = 0.0f32;
    let mut r4 = 0.0f32;
    let mut r5 = 0.0f32;

    for i in 0..d {
        unsafe {
            let qi = q.get_unchecked(i).to_f32().unwrap();
            r0 = r0.algebraic_add(qi.algebraic_mul(a0.get_unchecked(i).to_f32().unwrap()));
            r1 = r1.algebraic_add(qi.algebraic_mul(a1.get_unchecked(i).to_f32().unwrap()));
            r2 = r2.algebraic_add(qi.algebraic_mul(a2.get_unchecked(i).to_f32().unwrap()));
            r3 = r3.algebraic_add(qi.algebraic_mul(a3.get_unchecked(i).to_f32().unwrap()));
            r4 = r4.algebraic_add(qi.algebraic_mul(a4.get_unchecked(i).to_f32().unwrap()));
            r5 = r5.algebraic_add(qi.algebraic_mul(a5.get_unchecked(i).to_f32().unwrap()));
        }
    }
    [
        r0.into(),
        r1.into(),
        r2.into(),
        r3.into(),
        r4.into(),
        r5.into(),
    ]
}

/// Computes the squared Euclidean distance between two dense vectors.
#[cfg_attr(test, inline(never))]
#[cfg_attr(not(test), inline)]
#[must_use]
pub fn squared_euclidean_distance_dense<Q, V>(
    query: DenseVectorView<'_, Q>,
    vector: DenseVectorView<'_, V>,
) -> SquaredEuclideanDistance
where
    Q: ValueType,
    V: ValueType,
{
    assert_eq!(
        query.len(),
        vector.len(),
        "query and vector must have the same length"
    );
    unsafe { squared_euclidean_distance_dense_unchecked(query, vector) }
}

/// Computes the squared Euclidean distance between two dense vectors (unchecked).
/// # Safety
/// The caller must ensure `query.len() == vector.len()` so every subtraction uses valid indices.
#[cfg_attr(test, inline(never))]
#[cfg_attr(not(test), inline)]
#[must_use]
pub unsafe fn squared_euclidean_distance_dense_unchecked<Q, V>(
    query: DenseVectorView<'_, Q>,
    vector: DenseVectorView<'_, V>,
) -> SquaredEuclideanDistance
where
    Q: ValueType,
    V: ValueType,
{
    unsafe { assert_unchecked(query.len() == vector.len()) };

    let dist = query
        .values()
        .iter()
        .zip(vector.values().iter())
        .map(|(q, c)| {
            let q = q.to_f32().unwrap();
            let c = c.to_f32().unwrap();
            let diff = q.algebraic_sub(c);
            diff.algebraic_mul(diff)
        })
        .fold(0.0f32, |acc, x| acc.algebraic_add(x));

    dist.into()
}

/// Computes the squared Euclidean distance between one query and six dense vectors in a single pass.
///
/// Uses 6 independent partial-sum accumulators so the CPU can execute multiple multiply-add chains
/// simultaneously (better ILP) and reuse the query data before eviction (better cache behaviour).
///
/// # Safety
/// Caller must ensure `query.len() == vi.len()` for all six vectors.
#[inline]
#[must_use]
pub unsafe fn squared_euclidean_distance_dense_batch6_unchecked<Q, V>(
    query: DenseVectorView<'_, Q>,
    [v0, v1, v2, v3, v4, v5]: [DenseVectorView<'_, V>; 6],
) -> [SquaredEuclideanDistance; 6]
where
    Q: ValueType,
    V: ValueType,
{
    let d = query.len();
    let q = query.values();
    let a0 = v0.values();
    let a1 = v1.values();
    let a2 = v2.values();
    let a3 = v3.values();
    let a4 = v4.values();
    let a5 = v5.values();

    let mut r0 = 0.0f32;
    let mut r1 = 0.0f32;
    let mut r2 = 0.0f32;
    let mut r3 = 0.0f32;
    let mut r4 = 0.0f32;
    let mut r5 = 0.0f32;

    for i in 0..d {
        unsafe {
            let qi = q.get_unchecked(i).to_f32().unwrap();
            let d0 = qi.algebraic_sub(a0.get_unchecked(i).to_f32().unwrap());
            let d1 = qi.algebraic_sub(a1.get_unchecked(i).to_f32().unwrap());
            let d2 = qi.algebraic_sub(a2.get_unchecked(i).to_f32().unwrap());
            let d3 = qi.algebraic_sub(a3.get_unchecked(i).to_f32().unwrap());
            let d4 = qi.algebraic_sub(a4.get_unchecked(i).to_f32().unwrap());
            let d5 = qi.algebraic_sub(a5.get_unchecked(i).to_f32().unwrap());
            r0 = r0.algebraic_add(d0.algebraic_mul(d0));
            r1 = r1.algebraic_add(d1.algebraic_mul(d1));
            r2 = r2.algebraic_add(d2.algebraic_mul(d2));
            r3 = r3.algebraic_add(d3.algebraic_mul(d3));
            r4 = r4.algebraic_add(d4.algebraic_mul(d4));
            r5 = r5.algebraic_add(d5.algebraic_mul(d5));
        }
    }
    [
        r0.into(),
        r1.into(),
        r2.into(),
        r3.into(),
        r4.into(),
        r5.into(),
    ]
}

fn sparse_squared_norm<C, V>(vector: SparseVectorView<'_, C, V>) -> f32
where
    C: ComponentType,
    V: ValueType,
{
    vector
        .values()
        .iter()
        .map(|v| {
            let v = v.to_f32().unwrap();
            v.algebraic_mul(v)
        })
        .fold(0.0f32, |acc, x| acc.algebraic_add(x))
}

/// Computes the squared Euclidean distance between a dense query and a sparse vector.
///
/// `dot_query` must be `dot(query, query)` and is provided by the evaluator so the distance
/// computation can reuse the precomputed value.
#[cfg_attr(test, inline(never))]
#[cfg_attr(not(test), inline)]
#[must_use]
pub fn squared_euclidean_distance_sparse_with_dense_query<C, Q, V>(
    dot_query: f32,
    query: DenseVectorView<'_, Q>,
    vector: SparseVectorView<'_, C, V>,
) -> SquaredEuclideanDistance
where
    C: ComponentType,
    Q: ValueType,
    V: ValueType,
{
    let dot_query_b = dot_product_dense_sparse(query, vector).distance();
    let dot_b_b = sparse_squared_norm(vector);
    let dist = dot_query
        .algebraic_add(dot_b_b)
        .algebraic_sub(2.0f32.algebraic_mul(dot_query_b));

    SquaredEuclideanDistance::from(dist)
}

pub fn squared_euclidean_distance_sparse_with_dense_query_batch6<C, Q, V>(
    dot_query: f32,
    query: DenseVectorView<'_, Q>,
    vectors: [SparseVectorView<'_, C, V>; 6],
) -> [SquaredEuclideanDistance; 6]
where
    C: ComponentType,
    Q: ValueType,
    V: ValueType,
{
    let query_len = query.len();
    for vec in &vectors {
        for &c in vec.components() {
            assert!(
                c.as_() < query_len,
                "sparse vector component {} is out of bounds for query of length {}",
                c.as_(),
                query_len
            );
        }
    }

    let dots = unsafe { dot_product_dense_sparse_batch6_unchecked(query, vectors) };

    let dbs = [
        sparse_squared_norm(vectors[0]),
        sparse_squared_norm(vectors[1]),
        sparse_squared_norm(vectors[2]),
        sparse_squared_norm(vectors[3]),
        sparse_squared_norm(vectors[4]),
        sparse_squared_norm(vectors[5]),
    ];

    [
        SquaredEuclideanDistance::from(
            dot_query
                .algebraic_add(dbs[0])
                .algebraic_sub(2.0f32.algebraic_mul(dots[0].distance())),
        ),
        SquaredEuclideanDistance::from(
            dot_query
                .algebraic_add(dbs[1])
                .algebraic_sub(2.0f32.algebraic_mul(dots[1].distance())),
        ),
        SquaredEuclideanDistance::from(
            dot_query
                .algebraic_add(dbs[2])
                .algebraic_sub(2.0f32.algebraic_mul(dots[2].distance())),
        ),
        SquaredEuclideanDistance::from(
            dot_query
                .algebraic_add(dbs[3])
                .algebraic_sub(2.0f32.algebraic_mul(dots[3].distance())),
        ),
        SquaredEuclideanDistance::from(
            dot_query
                .algebraic_add(dbs[4])
                .algebraic_sub(2.0f32.algebraic_mul(dots[4].distance())),
        ),
        SquaredEuclideanDistance::from(
            dot_query
                .algebraic_add(dbs[5])
                .algebraic_sub(2.0f32.algebraic_mul(dots[5].distance())),
        ),
    ]
}

/// Computes the squared Euclidean distance between two sparse vectors.
///
/// `dot_query` must be `dot(query, query)` and the caller is expected to ensure both views follow
/// the same ordering invariants required by the sparse dot-product helpers.
#[cfg_attr(test, inline(never))]
#[cfg_attr(not(test), inline)]
#[must_use]
pub fn squared_euclidean_distance_sparse_with_merge<C, Q, V>(
    dot_query: f32,
    query: SparseVectorView<'_, C, Q>,
    vector: SparseVectorView<'_, C, V>,
) -> SquaredEuclideanDistance
where
    C: ComponentType,
    Q: ValueType,
    V: ValueType,
{
    let dot_query_b = dot_product_sparse_with_merge(query, vector).distance();
    let dot_b_b = sparse_squared_norm(vector);
    let dist = dot_query
        .algebraic_add(dot_b_b)
        .algebraic_sub(2.0f32.algebraic_mul(dot_query_b));

    SquaredEuclideanDistance::from(dist)
}

/// Compute the MaxSim score between a query multivector and a document multivector.
///
/// For each query token, finds the maximum dot product with any document token, then sums
/// across all query tokens — the standard late-interaction (ColBERT-style) score.
///
/// The outer loop iterates over document tokens: each is converted from `Out` to `f32`
/// exactly once (into `d_buf`), then the decoded f32 slice is reused for all query-tokens.
///
/// # Arguments
/// * `query` — query multivector with `f32` token values
/// * `doc` — document multivector with `Out`-typed token values
/// * `d_buf` — caller-supplied scratch buffer of length `query.dim()` for the decoded doc token
/// * `max_scores` — caller-supplied accumulator of length `query.num_vecs()`, one slot per query token;
///   **must be pre-initialised to `f32::NEG_INFINITY`** before calling
///
/// # Panics
/// Panics if `d_buf.len() != query.dim()`, `max_scores.len() != query.num_vecs()`, or
/// `query.dim() != doc.dim()`.
#[cfg_attr(test, inline(never))]
#[cfg_attr(not(test), inline)]
#[must_use]
pub fn maxsim<Out>(
    query: DenseMultiVectorView<'_, f32>,
    doc: DenseMultiVectorView<'_, Out>,
    d_buf: &mut [f32],
    max_scores: &mut [f32],
) -> f32
where
    Out: ValueType,
{
    assert_eq!(
        d_buf.len(),
        query.dim(),
        "d_buf length must equal query.dim()"
    );
    assert_eq!(
        max_scores.len(),
        query.num_vecs(),
        "max_scores length must equal query.num_vecs()"
    );
    assert_eq!(
        query.dim(),
        doc.dim(),
        "query and doc must have the same token dimension"
    );

    for d_token in doc.iter_vectors() {
        // Materialize this doc token to f32 once.
        for (dst, &src) in d_buf.iter_mut().zip(d_token.values()) {
            *dst = src.to_f32().unwrap();
        }
        let d_f32 = DenseVectorView::new(d_buf);

        // Update per-query-token maxima using the decoded f32 slice.
        for (score, q_token) in max_scores.iter_mut().zip(query.iter_vectors()) {
            let dot = unsafe { dot_product_dense_unchecked(d_f32, q_token) }.distance();
            *score = score.max(dot);
        }
    }

    max_scores.iter().sum()
}

/// Compute the two-level PQ MaxSim score with blocked layout.
///
/// Layout requirements for `pq_codes_block`:
/// - All PQ codes in one contiguous block: `pq_codes_block[t * M + m]` = code for token t, subspace m
/// - No stride, no interleaved data.
///
/// # Safety
/// - `centroid_scores.len() >= doc_n * Q`
/// - `distance_table.len() >= M * 256 * Q`
/// - `pq_codes_block.len() >= doc_n * M`
#[cfg_attr(not(test), inline)]
pub unsafe fn two_level_pq_maxsim_blocked<const M: usize, const Q: usize>(
    centroid_scores: &[f32],
    distance_table: &[f32],
    pq_codes_block: &[u8],
    doc_n: usize,
) -> f32 {
    const KSUB: usize = 256;

    let mut acc = [0f32; Q];
    let mut max_scores = [f32::NEG_INFINITY; Q];

    unsafe {
        let cs_ptr = centroid_scores.as_ptr();
        let dt_ptr = distance_table.as_ptr();
        let codes_ptr = pq_codes_block.as_ptr();

        for t in 0..doc_n {
            // Reset acc and load centroid scores in one pass.
            let cs_t = cs_ptr.add(t * Q);
            for q in 0..Q {
                *acc.get_unchecked_mut(q) = *cs_t.add(q);
            }

            // PQ residual SAXPY: M passes, each Q-wide.
            let codes_t = codes_ptr.add(t * M);
            for m in 0..M {
                let code = *codes_t.add(m) as usize;
                let tbl = dt_ptr.add(m * KSUB * Q + code * Q);
                for q in 0..Q {
                    *acc.get_unchecked_mut(q) += *tbl.add(q);
                }
            }

            // Update per-query-token maxima.
            for q in 0..Q {
                let v = *acc.get_unchecked(q);
                let cur = max_scores.get_unchecked_mut(q);
                if v > *cur {
                    *cur = v;
                }
            }
        }
    }

    // Sum maxima.
    max_scores.iter().fold(0f32, |s, &x| s + x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vector::{DenseVectorView, SparseVectorView};
    use std::panic::{AssertUnwindSafe, catch_unwind};

    #[test]
    fn dot_product_dense_sparse_basic() {
        let query_data = &[1.0f32, 2.0, 3.0];
        let query = DenseVectorView::new(query_data);
        let comps: &[usize] = &[0usize, 2usize];
        let vals: &[f32] = &[1.0f32, 1.0f32];
        let v = SparseVectorView::new(comps, vals);

        let result = dot_product_dense_sparse(query, v);
        assert_eq!(result, DotProduct::from(4.0f32));
    }

    #[test]
    fn squared_euclidean_ordering() {
        let a = SquaredEuclideanDistance::from(1.0);
        let b = SquaredEuclideanDistance::from(2.0);
        assert!(a < b);
    }

    #[test]
    fn dotproduct_reverse_ordering() {
        let a = DotProduct::from(1.0);
        let b = DotProduct::from(2.0);
        assert!(b < a);
    }

    #[test]
    fn dot_product_dense_unchecked_matches_checked() {
        let query = DenseVectorView::new(&[1.0f32, 2.0, 3.0]);
        let vector = DenseVectorView::new(&[2.0f32, 1.5, 0.5]);
        let checked = dot_product_dense(query, vector);
        let unchecked = unsafe { dot_product_dense_unchecked(query, vector) };
        assert_eq!(checked, unchecked);
    }

    #[test]
    fn dot_product_dense_unchecked_direct() {
        let query = DenseVectorView::new(&[2.0f32, 1.0]);
        let vector = DenseVectorView::new(&[1.0f32, 3.0]);
        let result = unsafe { dot_product_dense_unchecked(query, vector) };
        assert_eq!(result, DotProduct::from(5.0));
    }

    #[test]
    fn dot_product_sparse_with_merge_computes_shared_components() {
        let query = SparseVectorView::new(&[0usize, 2], &[1.0f32, 2.0]);
        let vector = SparseVectorView::new(&[0usize, 2], &[3.0f32, 4.0]);
        let result = dot_product_sparse_with_merge(query, vector);
        assert_eq!(result, DotProduct::from(11.0));
    }

    #[test]
    fn squared_euclidean_dense_unchecked_matches_manual_sum() {
        let query = DenseVectorView::new(&[0.0f32, 2.0]);
        let vector = DenseVectorView::new(&[3.0f32, 1.0]);
        let expected = SquaredEuclideanDistance::from(10.0);
        let computed = squared_euclidean_distance_dense(query, vector);
        let unchecked = unsafe { squared_euclidean_distance_dense_unchecked(query, vector) };
        assert_eq!(computed, expected);
        assert_eq!(unchecked, expected);
    }

    #[test]
    fn squared_euclidean_distance_dense_unchecked_direct() {
        let query = DenseVectorView::new(&[1.0f32, 2.0]);
        let vector = DenseVectorView::new(&[2.0f32, 3.0]);
        let result = unsafe { squared_euclidean_distance_dense_unchecked(query, vector) };
        assert_eq!(result, SquaredEuclideanDistance::from(2.0));
    }

    #[test]
    fn dot_product_dense_sparse_unchecked_sum() {
        let query = DenseVectorView::new(&[1.0f32, 2.0, 3.0]);
        let vector = SparseVectorView::new(&[0usize, 2], &[2.0f32, 4.0]);
        let result = unsafe { dot_product_dense_sparse_unchecked(query, vector) };
        assert_eq!(result, DotProduct::from(14.0));
    }

    #[test]
    fn dot_product_sparse_with_merge_unchecked_matches_manual() {
        let query = SparseVectorView::new(&[0usize, 1, 3], &[1.0f32, 2.0, 4.0]);
        let vector = SparseVectorView::new(&[0usize, 3], &[3.0f32, 1.0]);
        let result = unsafe { dot_product_sparse_with_merge_unchecked(query, vector) };
        assert_eq!(result, DotProduct::from(7.0));
    }

    #[test]
    #[should_panic(expected = "sparse vector component 4 is out of bounds")]
    fn dot_product_dense_sparse_out_of_bounds_panics() {
        let query = DenseVectorView::new(&[1.0f32, 2.0]);
        let vector = SparseVectorView::new(&[0usize, 4], &[1.0f32, 1.0]);
        let _ = dot_product_dense_sparse(query, vector);
    }

    #[test]
    fn dot_product_dense_sparse_bounds_assert_coverage() {
        let query = DenseVectorView::new(&[1.0f32, 2.0]);
        let vector = SparseVectorView::new(&[0usize, 4], &[1.0f32, 1.0]);
        let panic = catch_unwind(AssertUnwindSafe(|| dot_product_dense_sparse(query, vector)));
        assert!(panic.is_err());
    }

    #[test]
    fn dot_product_dense_sparse_unchecked_inner_loop() {
        let query = DenseVectorView::new(&[1.0f32, 2.0, 3.0]);
        let vector = SparseVectorView::new(&[0usize, 2], &[2.0f32, 4.0]);
        let result = unsafe { dot_product_dense_sparse_unchecked(query, vector) };
        assert_eq!(result, DotProduct::from(14.0));
    }

    #[test]
    #[should_panic(expected = "query and vector must have the same length")]
    fn dot_product_dense_mismatch_panics() {
        let query = DenseVectorView::new(&[1.0f32, 2.0, 3.0]);
        let vector = DenseVectorView::new(&[1.0f32, 2.0]);
        let _ = dot_product_dense(query, vector);
    }

    #[test]
    fn dot_product_dense_length_assert_coverage() {
        let query = DenseVectorView::new(&[1.0f32, 2.0, 3.0]);
        let vector = DenseVectorView::new(&[1.0f32, 2.0]);
        let panic = catch_unwind(AssertUnwindSafe(|| dot_product_dense(query, vector)));
        assert!(panic.is_err());
    }

    #[test]
    #[should_panic(expected = "query and vector must have the same length")]
    fn squared_euclidean_distance_dense_mismatch_panics() {
        let query = DenseVectorView::new(&[0.0f32, 1.0]);
        let vector = DenseVectorView::new(&[1.0f32, 2.0, 3.0]);
        let _ = squared_euclidean_distance_dense(query, vector);
    }

    #[test]
    fn squared_euclidean_dense_length_assert_coverage() {
        let query = DenseVectorView::new(&[0.0f32, 1.0]);
        let vector = DenseVectorView::new(&[1.0f32, 2.0, 3.0]);
        let panic = catch_unwind(AssertUnwindSafe(|| {
            squared_euclidean_distance_dense(query, vector)
        }));
        assert!(panic.is_err());
    }

    #[test]
    #[should_panic(expected = "query components must be sorted in strictly ascending order")]
    fn dot_product_sparse_with_merge_unsorted_query_panics() {
        let query = SparseVectorView::new(&[0usize, 2, 1], &[1.0f32, 2.0, 3.0]);
        let vector = SparseVectorView::new(&[0usize, 2], &[1.0f32, 1.0]);
        let _ = dot_product_sparse_with_merge(query, vector);
    }

    #[test]
    fn dot_product_sparse_with_merge_vector_unsorted_panics() {
        let query = SparseVectorView::new(&[0usize, 1, 3], &[1.0f32, 2.0, 3.0]);
        let vector = SparseVectorView::new(&[0usize, 2, 1], &[1.0f32, 1.0, 2.0]);
        let panic = catch_unwind(AssertUnwindSafe(|| {
            dot_product_sparse_with_merge(query, vector)
        }));
        assert!(panic.is_err());
    }

    #[test]
    fn dot_product_sparse_with_merge_unchecked_handles_empty_vector() {
        let query = SparseVectorView::new(&[0usize, 2], &[1.0f32, 2.0]);
        let vector: SparseVectorView<'_, usize, f32> = SparseVectorView::new(&[], &[]);
        let result = unsafe { dot_product_sparse_with_merge_unchecked(query, vector) };
        assert_eq!(result, DotProduct::from(0.0));
    }

    #[test]
    fn dot_product_sparse_with_merge_unchecked_skips_non_matching_component() {
        let query = SparseVectorView::new(&[0usize, 1], &[1.0f32, 2.0]);
        let vector = SparseVectorView::new(&[2usize], &[3.0f32]);
        let result = unsafe { dot_product_sparse_with_merge_unchecked(query, vector) };
        assert_eq!(result, DotProduct::from(0.0));
    }

    #[test]
    fn distance_trait_methods_return_inner_value() {
        let distance = DotProduct::from(2.5);
        assert_eq!(distance.distance(), 2.5);
        let euclid = SquaredEuclideanDistance::from(16.0);
        assert_eq!(euclid.distance(), 16.0);
        assert_eq!(euclid.sqrt(), 4.0);
    }

    #[test]
    fn euclidean_no_relaxation() {
        let candidate = SquaredEuclideanDistance::from(100.0);
        let threshold = SquaredEuclideanDistance::from(100.0);
        assert!(candidate.is_within_relaxation(&threshold, 0.0));

        let candidate2 = SquaredEuclideanDistance::from(101.0);
        assert!(!candidate2.is_within_relaxation(&threshold, 0.0));
    }

    #[test]
    fn euclidean_with_relaxation() {
        let threshold = SquaredEuclideanDistance::from(100.0);

        let candidate1 = SquaredEuclideanDistance::from(105.0);
        assert!(candidate1.is_within_relaxation(&threshold, 0.1));

        let candidate2 = SquaredEuclideanDistance::from(110.0);
        assert!(candidate2.is_within_relaxation(&threshold, 0.1));

        let candidate3 = SquaredEuclideanDistance::from(111.0);
        assert!(!candidate3.is_within_relaxation(&threshold, 0.1));
    }

    #[test]
    fn euclidean_boundary_cases() {
        let threshold = SquaredEuclideanDistance::from(100.0);

        let candidate = SquaredEuclideanDistance::from(120.0);
        assert!(candidate.is_within_relaxation(&threshold, 0.2));

        let candidate2 = SquaredEuclideanDistance::from(120.001);
        assert!(!candidate2.is_within_relaxation(&threshold, 0.2));
    }

    #[test]
    fn dotproduct_no_relaxation() {
        let candidate = DotProduct::from(0.9);
        let threshold = DotProduct::from(0.9);
        assert!(candidate.is_within_relaxation(&threshold, 0.0));

        let candidate2 = DotProduct::from(0.89);
        assert!(!candidate2.is_within_relaxation(&threshold, 0.0));
    }

    #[test]
    fn dotproduct_with_relaxation() {
        let threshold = DotProduct::from(0.9);

        let candidate1 = DotProduct::from(0.85);
        assert!(candidate1.is_within_relaxation(&threshold, 0.1));

        let candidate2 = DotProduct::from(0.81);
        assert!(candidate2.is_within_relaxation(&threshold, 0.1));

        let candidate3 = DotProduct::from(0.80);
        assert!(!candidate3.is_within_relaxation(&threshold, 0.1));
    }

    #[test]
    fn dotproduct_boundary_cases() {
        let threshold = DotProduct::from(1.0);

        let candidate = DotProduct::from(0.8);
        assert!(candidate.is_within_relaxation(&threshold, 0.2));

        let candidate2 = DotProduct::from(0.799);
        assert!(!candidate2.is_within_relaxation(&threshold, 0.2));
    }

    #[test]
    fn dotproduct_negative_threshold_relaxation() {
        let threshold = DotProduct::from(-0.5);
        let candidate = DotProduct::from(-0.55);
        assert!(candidate.is_within_relaxation(&threshold, 0.2));

        let candidate2 = DotProduct::from(-0.65);
        assert!(!candidate2.is_within_relaxation(&threshold, 0.2));
    }

    #[test]
    fn zero_distance_euclidean() {
        let candidate = SquaredEuclideanDistance::from(0.0);
        let threshold = SquaredEuclideanDistance::from(100.0);
        assert!(candidate.is_within_relaxation(&threshold, 0.0));
        assert!(candidate.is_within_relaxation(&threshold, 0.5));
    }

    #[test]
    fn large_lambda() {
        let threshold = SquaredEuclideanDistance::from(100.0);
        let candidate = SquaredEuclideanDistance::from(150.0);
        assert!(candidate.is_within_relaxation(&threshold, 1.0));

        let candidate2 = SquaredEuclideanDistance::from(200.0);
        assert!(candidate2.is_within_relaxation(&threshold, 1.0));
    }

    #[test]
    #[should_panic(expected = "lambda must be non-negative")]
    #[cfg(debug_assertions)]
    fn negative_lambda_euclidean() {
        let candidate = SquaredEuclideanDistance::from(100.0);
        let threshold = SquaredEuclideanDistance::from(100.0);
        candidate.is_within_relaxation(&threshold, -0.1);
    }

    #[test]
    #[should_panic(expected = "lambda must be non-negative")]
    #[cfg(debug_assertions)]
    fn negative_lambda_dotproduct() {
        let candidate = DotProduct::from(0.9);
        let threshold = DotProduct::from(0.9);
        candidate.is_within_relaxation(&threshold, -0.1);
    }

    #[test]
    #[should_panic(expected = "lambda must be finite")]
    #[cfg(debug_assertions)]
    fn non_finite_lambda_euclidean() {
        let candidate = SquaredEuclideanDistance::from(100.0);
        let threshold = SquaredEuclideanDistance::from(100.0);
        candidate.is_within_relaxation(&threshold, f32::INFINITY);
    }

    #[test]
    #[should_panic(expected = "lambda must be finite")]
    #[cfg(debug_assertions)]
    fn non_finite_lambda_dotproduct() {
        let candidate = DotProduct::from(0.9);
        let threshold = DotProduct::from(0.9);
        candidate.is_within_relaxation(&threshold, f32::NAN);
    }

    #[test]
    #[should_panic(expected = "NaN is not allowed for DotProduct")]
    fn dot_product_from_nan_panics() {
        let _ = DotProduct::from(f32::NAN);
    }

    #[test]
    #[should_panic(expected = "NaN is not allowed for SquaredEuclideanDistance")]
    fn squared_euclidean_from_nan_panics() {
        let _ = SquaredEuclideanDistance::from(f32::NAN);
    }

    #[test]
    fn squared_euclidean_batch6_matches_six_singles() {
        let q = &[1.0f32, 2.0, 3.0, 4.0];
        let a0 = &[0.0f32, 0.0, 0.0, 0.0];
        let a1 = &[1.0f32, 1.0, 1.0, 1.0];
        let a2 = &[2.0f32, 2.0, 2.0, 2.0];
        let a3 = &[3.0f32, 4.0, 5.0, 6.0];
        let a4 = &[4.0f32, 3.0, 2.0, 1.0];
        let a5 = &[0.5f32, 1.5, 2.5, 3.5];
        let query = DenseVectorView::new(q);
        let vs = [
            DenseVectorView::new(a0),
            DenseVectorView::new(a1),
            DenseVectorView::new(a2),
            DenseVectorView::new(a3),
            DenseVectorView::new(a4),
            DenseVectorView::new(a5),
        ];
        let batch = unsafe { squared_euclidean_distance_dense_batch6_unchecked(query, vs) };
        let singles = vs.map(|v| unsafe { squared_euclidean_distance_dense_unchecked(query, v) });
        assert_eq!(batch, singles);
    }

    #[test]
    fn dot_product_batch6_matches_six_singles() {
        let q = &[1.0f32, 0.5, 2.0];
        let a0 = &[3.0f32, 1.0, 0.0];
        let a1 = &[0.0f32, 0.0, 0.0];
        let a2 = &[-1.0f32, 2.0, 1.0];
        let a3 = &[1.0f32, 1.0, 1.0];
        let a4 = &[2.0f32, 0.0, 1.0];
        let a5 = &[0.5f32, 1.5, 0.5];
        let query = DenseVectorView::new(q);
        let vs = [
            DenseVectorView::new(a0),
            DenseVectorView::new(a1),
            DenseVectorView::new(a2),
            DenseVectorView::new(a3),
            DenseVectorView::new(a4),
            DenseVectorView::new(a5),
        ];
        let batch = unsafe { dot_product_dense_batch6_unchecked(query, vs) };
        let singles = vs.map(|v| unsafe { dot_product_dense_unchecked(query, v) });
        assert_eq!(batch, singles);
    }

    #[test]
    fn squared_euclidean_batch6_dim1_edge_case() {
        let query = DenseVectorView::new(&[5.0f32]);
        let vs = [
            DenseVectorView::new(&[5.0f32]),
            DenseVectorView::new(&[4.0f32]),
            DenseVectorView::new(&[3.0f32]),
            DenseVectorView::new(&[0.0f32]),
            DenseVectorView::new(&[6.0f32]),
            DenseVectorView::new(&[2.0f32]),
        ];
        let batch = unsafe { squared_euclidean_distance_dense_batch6_unchecked(query, vs) };
        assert_eq!(batch[0], SquaredEuclideanDistance::from(0.0));
        assert_eq!(batch[1], SquaredEuclideanDistance::from(1.0));
        assert_eq!(batch[2], SquaredEuclideanDistance::from(4.0));
        assert_eq!(batch[3], SquaredEuclideanDistance::from(25.0));
        assert_eq!(batch[4], SquaredEuclideanDistance::from(1.0));
        assert_eq!(batch[5], SquaredEuclideanDistance::from(9.0));
    }

    #[test]
    fn squared_euclidean_batch6_all_zero_query_and_vectors() {
        let zero_data = &[0.0f32, 0.0, 0.0];
        let query = DenseVectorView::new(zero_data);
        let zero = DenseVectorView::new(zero_data);
        let batch = unsafe {
            squared_euclidean_distance_dense_batch6_unchecked(
                query,
                [zero, zero, zero, zero, zero, zero],
            )
        };
        for d in batch {
            assert_eq!(d, SquaredEuclideanDistance::from(0.0));
        }
    }

    #[test]
    fn squared_euclidean_sparse_with_dense_query_matches_dense_result() {
        let query_values = &[1.0f32, 2.5, 0.0];
        let query = DenseVectorView::new(query_values);
        let components = &[0usize, 2usize];
        let values = &[2.0f32, -1.0];
        let sparse_vector = SparseVectorView::new(components, values);
        let dot_query = dot_product_dense(query, query).distance();

        let result =
            squared_euclidean_distance_sparse_with_dense_query(dot_query, query, sparse_vector);

        let expected_dense = DenseVectorView::new(&[2.0f32, 0.0, -1.0]);
        let expected = squared_euclidean_distance_dense(query, expected_dense);

        assert_eq!(result, expected);
    }

    #[test]
    fn squared_euclidean_sparse_with_merge_matches_dense_result() {
        let query_sparse = SparseVectorView::new(&[0usize, 2], &[3.0f32, 1.0]);
        let dataset_sparse = SparseVectorView::new(&[0usize, 1, 2], &[2.0f32, 4.0, -1.0]);
        let dot_query = dot_product_sparse_with_merge(query_sparse, query_sparse).distance();

        let result =
            squared_euclidean_distance_sparse_with_merge(dot_query, query_sparse, dataset_sparse);

        let dense_query = DenseVectorView::new(&[3.0f32, 0.0, 1.0]);
        let dense_dataset = DenseVectorView::new(&[2.0f32, 4.0, -1.0]);
        let expected = squared_euclidean_distance_dense(dense_query, dense_dataset);

        assert_eq!(result, expected);
    }
}
