use crate::quantizers::{Quantizer, QueryEvaluator};
use crate::{Distance, Vector1D};

use itertools::Itertools;

pub mod dense_dataset;
pub mod dense_dataset_scalar;
pub mod sparse_dataset;
pub mod sparse_dataset_scalar;

pub type VectorId = u64;
pub type VectorKey = u64;

#[derive(Debug, PartialOrd, Eq, Ord, PartialEq, Copy, Clone)]
pub struct ResultGeneric<D: Distance, T> {
    pub distance: D,
    pub vector: T,
}

pub type Result<D> = ResultGeneric<D, VectorId>;
pub type ResultWithKey<D> = ResultGeneric<D, VectorKey>;

/// A `Dataset` stores a collection of dense or sparse embedding vectors.
///
/// At the moment we have two implementations:
/// - `DenseDataset` in which we store fixed-length dense vectors, i.e., vectors for which there is no need to store compoennt indices.
/// - `SparseDataset` in which we store variable-length sparse vectors, i.e., vectors for which we need to store component indices.
///
/// A quantizer is associated with the dataset, defining how input vectors
/// are encoded and how queries are evaluated against the encoded vectors.
/// Important: iterators iterate over all **encoded** representation of the
/// vectors in the dataset.
///
/// Each vector has a logical `VectorId` (a `u64`), which is (and MUST be) its index in the
/// dataset and is the stable way to refer to that vector across components.
///
/// Sparse datasets store variable‑length vectors in a packed array and keep
/// metadata (offsets, lengths) in a separate structure. Accessing a vector by
/// `VectorId` therefore involves an extra level of indirection:
///
///     VectorId -> (offset, length) -> vector data
///
/// This extra lookup can slow down accesses and reduce the effectiveness
/// of software prefetching.
///
/// To avoid this, datasets can also expose a `VectorKey` (an opaque `u64`)
/// for each vector. A `VectorKey` uniquely identifies a vector *within a
/// particular dataset implementation* and is chosen so that the dataset can
/// locate and prefetch the vector without going through the offset table.
/// For dense datasets, the `VectorKey` is typically equal to the `VectorId`.
/// For sparse datasets, it may encode the vector’s offset and length in the
/// underlying storage, so not all `u64` values are valid keys.
///
/// The dataset provides conversion routines between `VectorId` and
/// `VectorKey`. Index structures using the dataset can then choose:
///
/// - to store `VectorId`s, keeping a purely logical identifier but paying
///   the extra indirection on each access; or
/// - to store `VectorKey`s, allowing direct, more efficient access and
///   prefetching at the cost of tying the index to this specific dataset
///   representation.
///
/// Conversion between `VectorId` and `VectorKey` is provided by the dataset.
/// However, conversion from `VectorKey` back to `VectorId` may be
/// expensive (e.g., requiring a binary search), so it should be used sparingly.

pub trait Dataset<Q>
where
    Q: Quantizer,
{
    fn quantizer(&self) -> &Q;

    fn shape(&self) -> (usize, usize);

    // Dimensionality of the vectors, i.e., largest possible component index + 1.
    fn dim(&self) -> usize;

    fn len(&self) -> usize;

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn nnz(&self) -> usize;

    fn key_from_id(&self, id: VectorId) -> VectorKey;

    fn id_from_key(&self, key: VectorKey) -> VectorId;

    fn get(
        &self,
        key: VectorKey,
    ) -> impl Vector1D<ComponentType = Q::OutputComponentType, ValueType = Q::OutputValueType>;

    fn prefetch(&self, key: VectorKey);

    fn iter(
        &self,
    ) -> impl Iterator<
        Item = impl Vector1D<ComponentType = Q::OutputComponentType, ValueType = Q::OutputValueType>,
    >;

    /// Performs a brute-force search to find the K-nearest neighbors (KNN) of the queried vector.
    ///
    /// This method scans the entire dataset to find the K-nearest neighbors of the queried vector.
    /// It computes the *dot product* between the queried vector and each vector in the dataset and returns
    /// the indices of the K-nearest neighbors along with their distances.
    ///
    #[inline]
    fn search(
        &self,
        query: impl Vector1D<ComponentType = Q::QueryComponentType, ValueType = Q::QueryValueType>,
        k: usize,
    ) -> Vec<Result<<Q as Quantizer>::Distance>> {
        let evaluator = self.quantizer().get_query_evaluator(query);

        if k == 0 {
            return Vec::new();
        }

        self.iter()
            .enumerate()
            .map(|(i, vector)| Result {
                distance: evaluator.compute_distance(vector),
                vector: i as u64,
            })
            .k_smallest(k)
            .collect()
    }
}

pub trait GrowableDataset<Q>: Dataset<Q>
where
    Q: Quantizer,
{
    fn new(quantizer: Q, d: usize) -> Self;
    fn push(
        &mut self,
        vec: impl Vector1D<ComponentType = Q::InputComponentType, ValueType = Q::InputValueType>,
    );
}
