//! Distance types and utilities.
//!
//! Provides the `Distance` trait (a small wrapper around an `f32` value)
//! and concrete distance types that implement comparison.
//!
//! NaN (Not a Number) is NOT permitted for distance values. These types
//! implement `Ord` using a total ordering and will panic if a NaN is
//! encountered. `PartialEq` and `PartialOrd` are provided for convenience,
//! but the crate's contract forbids NaN.
//!
//! `DotProduct` implements reversed ordering: larger values are considered
//! better.

use crate::core::vector1d::{DenseVector1DView, SparseVector1DView};
use crate::utils::is_strictly_sorted;
use crate::{ComponentType, ValueType};

use std::hint::assert_unchecked;

/// A simple trait representing a distance-like value (stored as `f32`).
pub trait Distance: Ord + Copy + Send + Sync + 'static {
    /// Return the numeric distance value.
    fn distance(&self) -> f32;
}

/// Squared Euclidean distance wrapper around an `f32`.
#[derive(Copy, Clone, Debug, Default)]
pub struct SquaredEuclideanDistance(f32);

impl Distance for SquaredEuclideanDistance {
    fn distance(&self) -> f32 {
        self.0
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
        assert!(
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
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Dot product score wrapper. Higher dot-product is better.
#[derive(Copy, Clone, Debug, Default)]
pub struct DotProduct(pub f32);

impl Distance for DotProduct {
    fn distance(&self) -> f32 {
        self.0
    }
}

impl From<f32> for DotProduct {
    fn from(v: f32) -> Self {
        assert!(!v.is_nan(), "NaN is not allowed for DotProduct");
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
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.0.total_cmp(&self.0)
    }
}

/// Computes the dot product between a dense query and a sparse vector.
#[inline]
#[must_use]
pub fn dot_product_dense_sparse<C, Q, V>(
    query: DenseVector1DView<'_, Q>,
    vector: SparseVector1DView<'_, C, V>,
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
#[inline]
#[must_use]
pub unsafe fn dot_product_dense_sparse_unchecked<C, Q, V>(
    query: DenseVector1DView<'_, Q>,
    vector: SparseVector1DView<'_, C, V>,
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

/// Computes the dot product between two sparse vectors using merge style.
#[inline]
#[must_use]
pub fn dot_product_sparse_with_merge<C, Q, V>(
    query: SparseVector1DView<'_, C, Q>,
    vector: SparseVector1DView<'_, C, V>,
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
#[inline]
#[must_use]
pub unsafe fn dot_product_sparse_with_merge_unchecked<C, Q, V>(
    query: SparseVector1DView<'_, C, Q>,
    vector: SparseVector1DView<'_, C, V>,
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
#[inline]
#[must_use]
pub fn dot_product_dense<Q, V>(
    query: DenseVector1DView<'_, Q>,
    vector: DenseVector1DView<'_, V>,
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
#[inline]
#[must_use]
pub unsafe fn dot_product_dense_unchecked<Q, V>(
    query: DenseVector1DView<'_, Q>,
    vector: DenseVector1DView<'_, V>,
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

/// Computes the squared Euclidean distance between two dense vectors.
#[inline]
#[must_use]
pub fn squared_euclidean_distance_dense<Q, V>(
    query: DenseVector1DView<'_, Q>,
    vector: DenseVector1DView<'_, V>,
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
#[inline]
#[must_use]
pub unsafe fn squared_euclidean_distance_dense_unchecked<Q, V>(
    query: DenseVector1DView<'_, Q>,
    vector: DenseVector1DView<'_, V>,
) -> SquaredEuclideanDistance
where
    Q: ValueType,
    V: ValueType,
{
    unsafe { assert_unchecked(query.len() == vector.len()) };

    let sum_sq = query
        .iter()
        .zip(vector.iter())
        .map(|(q, v)| {
            let q = q.to_f32().unwrap();
            let v = v.to_f32().unwrap();
            let diff = q.algebraic_sub(v);
            diff.algebraic_mul(diff)
        })
        .fold(0.0f32, |acc, x| acc.algebraic_add(x));

    sum_sq.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vector1d::{DenseVector1DView, SparseVector1DView};

    #[test]
    fn dot_product_dense_sparse_basic() {
        let query_data = &[1.0f32, 2.0, 3.0];
        let query = DenseVector1DView::new(query_data);
        let comps: &[usize] = &[0usize, 2usize];
        let vals: &[f32] = &[1.0f32, 1.0f32];
        let v = SparseVector1DView::new(comps, vals);

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
}
