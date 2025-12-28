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

use crate::DenseVector1D;
use crate::SparseVector1D;
use crate::Vector1D;
use crate::{ComponentType, ValueType};

use std::hint::assert_unchecked;

/// A simple trait representing a distance-like value (stored as `f32`).
pub trait Distance: Ord + Copy + Send + Sync {
    /// Return the numeric distance value.
    fn distance(&self) -> f32;
}

/// Errors returned when attempting to construct a distance value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceError {
    /// Value was NaN which is not allowed for distances.
    NaNValue,
}

impl std::fmt::Display for DistanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistanceError::NaNValue => write!(f, "NaN is not allowed for distance types"),
        }
    }
}

impl std::error::Error for DistanceError {}

/// Squared Euclidean distance wrapper around an `f32`.
///
/// For performance, this type stores the *squared* Euclidean distance
/// (without taking the square root). This avoids unnecessary `sqrt`
/// computations when only comparisons are required.
///
/// If you need the actual Euclidean distance, call `.sqrt()` on the result.
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

// Note: NaN is NOT permitted as a distance value.
// These implementations assume that `f32` values are finite numeric
// distances. If a NaN is encountered, `Ord::cmp` will panic to make the
// contract explicit.
impl Eq for SquaredEuclideanDistance {}

impl Ord for SquaredEuclideanDistance {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Use total ordering for floats via `total_cmp`.
        self.0.total_cmp(&other.0)
    }
}

/// Dot product score wrapper. Higher dot-product is better: ordering is
/// reversed so that greater values compare as "smaller" distance.
///
/// This means comparisons are reversed: a larger dot-product is considered
/// better. (i.e. compare in reverse order — greater is better)
#[derive(Copy, Clone, Debug, Default)]
pub struct DotProduct(f32);

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

// Note: NaN is NOT permitted as a dot-product score.
// `Ord::cmp` will panic if a NaN is provided. Ordering is reversed so that
// larger dot-product values are treated as "better" (see doc comment above).
impl Eq for DotProduct {}

impl Ord for DotProduct {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse total ordering: larger dot product is considered smaller
        // in ordering (i.e. better).
        other.0.total_cmp(&self.0)
    }
}

/// Computes the dot product between a dense query and a sparse vector.
/// Before using this function, the query must be made dense. In some cases,
/// especially when queries have many non-zero components, this is faster
/// than computing the dot product with a "merge" style.
///
/// # Arguments
///
/// * `query` - The dense query vector.
/// * `vector` - The sparse vector.
///
/// # Returns
///
/// The dot product between the query and the vector.
///
/// # Examples
///
/// ```rust
/// use vectorium::distances::dot_product_dense_sparse;
/// use vectorium::{DenseVector1D, SparseVector1D};
///
/// // Create a dense query and a sparse vector and compute the dot product.
/// let query = DenseVector1D::new(&[1.0f32, 2.0, 3.0]);
/// let comps = &[0usize, 2usize];
/// let vals = &[1.0f32, 1.0f32];
/// let v = SparseVector1D::new(comps, vals);
///
/// let result = unsafe { dot_product_dense_sparse(&query, &v) };
/// assert_eq!(result, vectorium::distances::DotProduct::from(4.0f32));
/// ```
///
/// # Safety
/// - All components `c` in `vector.components_as_slice()` must satisfy `c.as_() < query.len()`.
#[inline]
#[must_use]
pub unsafe fn dot_product_dense_sparse<C, Q, V>(
    query: &DenseVector1D<Q, impl AsRef<[Q]>>,
    vector: &SparseVector1D<C, V, impl AsRef<[C]>, impl AsRef<[V]>>,
) -> DotProduct
where
    C: ComponentType,
    Q: ValueType,
    V: ValueType,
{
    let query = query.values_as_slice();
    vector
        .components_as_slice()
        .iter()
        .zip(vector.values_as_slice())
        .map(|(&c, &v): (&C, &V)| {
            // SAFETY: il chiamante garantisce che `c.as_() < query.len()`.
            unsafe { *query.get_unchecked(c.as_()) }
                .to_f32()
                .unwrap()
                .algebraic_mul(v.to_f32().unwrap())
        })
        .fold(0.0, |acc: f32, x| acc.algebraic_add(x))
        .into()
}

/// Computes the dot product between a query and a vector using merge style.
/// This function should be used when the query has just a few components.
/// Both the query's and vector's terms must be sorted by id.
///
/// # Arguments
///
/// * `query` - Sparse query vector.
/// * `vector` - Sparse vector.
///
/// # Returns
///
/// The dot product between the query and the vector.
///
/// # Examples
///
/// ```
/// use vectorium::distances::dot_product_sparse_with_merge;
///
/// let query_components = [1_u32, 2, 7];
/// let query_values = [1.0, 1.0, 1.0];
/// let v_components = [0_u32, 1, 2, 3, 4];
/// let v_values = [0.1, 1.0, 1.0, 1.0, 0.5];
///
/// let query = vectorium::SparseVector1D::new(&query_components, &query_values);
/// let vector = vectorium::SparseVector1D::new(&v_components, &v_values);
/// let result = unsafe { dot_product_sparse_with_merge(&query, &vector) };
/// assert_eq!(result, vectorium::distances::DotProduct::from(2.0));
/// ```
///
/// # Safety
/// - `query.components_as_slice().len() == query.values_as_slice().len()`.
/// - `vector.components_as_slice().len() == vector.values_as_slice().len()`.
/// - Entrambe le liste di componenti devono essere ordinate (crescente).
#[inline]
#[must_use]
pub unsafe fn dot_product_sparse_with_merge<C, Q, V>(
    query: &SparseVector1D<C, Q, impl AsRef<[C]>, impl AsRef<[Q]>>,
    vector: &SparseVector1D<C, V, impl AsRef<[C]>, impl AsRef<[V]>>,
) -> DotProduct
where
    C: ComponentType,
    Q: ValueType,
    V: ValueType,
{
    let v_components = vector.components_as_slice();
    let v_values = vector.values_as_slice();
    let query_components = query.components_as_slice();
    let query_values = query.values_as_slice();

    unsafe {
        assert_unchecked(
            v_components.len() == v_values.len() && query_components.len() == query_values.len(),
        )
    }
    let mut result = 0.0;
    let mut v_iter = v_components.iter().zip(v_values);
    let mut current = v_iter.next();
    let b = current.is_some();
    for (&q_id, &q_v) in query_components.iter().zip(query_values) {
        // This assert actually improves performance: https://github.com/rust-lang/rust/issues/134667
        if b {
            unsafe { assert_unchecked(current.is_some()) }
        }
        while let Some((&v_id, _)) = current
            && v_id < q_id
        {
            current = v_iter.next();
        }
        if !b {
            unsafe { assert_unchecked(current.is_none()) }
        }
        match current {
            Some((&v_id, v_v)) if v_id == q_id => {
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
///
/// The function accepts `DenseVector1D`.
///
/// ```rust
/// use vectorium::distances::dot_product_dense;
/// use vectorium::DotProduct;
/// use vectorium::DenseVector1D;
///
/// let a = DenseVector1D::new(&[1.0f32, 2.0]);
/// let b = DenseVector1D::new(&[3.0f32, 4.0]);
/// let result = unsafe { dot_product_dense(a, b) };
/// assert_eq!(result, DotProduct::from(11.0f32));
/// ```
///
/// # Safety
/// - `query.len() == vector.len()`.
#[inline]
#[must_use]
pub unsafe fn dot_product_dense<Q, V>(
    query: DenseVector1D<Q, impl AsRef<[Q]>>,
    vector: DenseVector1D<V, impl AsRef<[V]>>,
) -> DotProduct
where
    Q: ValueType,
    V: ValueType,
{
    let query = query.values_as_slice();
    let vector = vector.values_as_slice();
    unsafe { assert_unchecked(query.len() == vector.len()) };
    query
        .iter()
        .zip(vector.iter())
        .map(|(&q, &v): (&Q, &V)| q.to_f32().unwrap().algebraic_mul(v.to_f32().unwrap()))
        .fold(0.0, |acc: f32, x| acc.algebraic_add(x))
        .into()
}

/// Computes the squared Euclidean distance between two dense vectors.
///
/// # Safety
/// - `query.len() == vector.len()`.
#[inline]
#[must_use]
pub unsafe fn squared_euclidean_distance_dense<Q, V>(
    query: DenseVector1D<Q, impl AsRef<[Q]>>,
    vector: DenseVector1D<V, impl AsRef<[V]>>,
) -> SquaredEuclideanDistance
where
    Q: ValueType,
    V: ValueType,
{
    let query = query.values_as_slice();
    let vector = vector.values_as_slice();
    unsafe { assert_unchecked(query.len() == vector.len()) };

    let sum_sq = query
        .iter()
        .zip(vector.iter())
        .map(|(&q, &v): (&Q, &V)| {
            let q = q.to_f32().unwrap();
            let v = v.to_f32().unwrap();

            let diff = q.algebraic_sub(v);

            // diff^2
            diff.algebraic_mul(diff)
        })
        .fold(0.0f32, |acc, x| acc.algebraic_add(x));

    sum_sq.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_product_dense_sparse_basic() {
        // query: [1, 2, 3]
        let query = DenseVector1D::new(&[1.0f32, 2.0, 3.0]);
        // sparse vector with components at 0 and 2
        let comps: &[usize] = &[0usize, 2usize];
        let vals: &[f32] = &[1.0f32, 1.0f32];
        let v = SparseVector1D::new(comps, vals);

        let result = unsafe { dot_product_dense_sparse(&query, &v) };
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
        // b has larger dot-product and should be considered "better" — i.e.
        // in reversed ordering b < a.
        assert!(b < a);
    }

    #[test]
    fn dot_product_dense_slice_slice() {
        let q = DenseVector1D::new(&[1.0f32, 2.0, 3.0]);
        let v = DenseVector1D::new(&[4.0f32, 5.0, 6.0]);
        let res = unsafe { dot_product_dense(q, v) };
        // 1*4 + 2*5 + 3*6 = 32
        assert_eq!(res, DotProduct::from(32.0f32));
    }

    #[test]
    fn dot_product_dense_vec_vec() {
        let q = DenseVector1D::new(vec![1.0f32, 2.0, 3.0]);
        let v = DenseVector1D::new(vec![4.0f32, 5.0, 6.0]);
        let res = unsafe { dot_product_dense(q, v) };
        assert_eq!(res, DotProduct::from(32.0f32));
    }

    #[test]
    fn dot_product_dense_vec_slice() {
        let q = DenseVector1D::new(vec![1.0f32, 2.0, 3.0]);
        let v = DenseVector1D::new(&[4.0f32, 5.0, 6.0]);
        let res = unsafe { dot_product_dense(q, v) };
        assert_eq!(res, DotProduct::from(32.0f32));
    }
}
