use crate::{Distance, SpaceUsage, Vector1D};
use crate::{QueryEvaluator, QueryVectorFor, VectorEncoder};

use itertools::Itertools;

pub type VectorId = u64;

#[derive(Debug, PartialOrd, Eq, Ord, PartialEq, Copy, Clone)]
pub struct ScoredItemGeneric<D: Distance, T> {
    pub distance: D,
    pub vector: T,
}

// We are not using ScoredItemGeneric here because Range does not implement Ord.
// So, we have to implement PartialOrd and Ord manually.
#[derive(Debug, PartialEq, Clone)]
pub struct ScoredRange<D: Distance> {
    pub distance: D,
    pub range: std::ops::Range<usize>,
}

impl<D: Distance> Eq for ScoredRange<D> {}

impl<D: Distance> PartialOrd for ScoredRange<D> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<D: Distance> Ord for ScoredRange<D> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Note: for our use, equal starts should imply equal ends, but we compare
        // by end as a defensive tie-breaker.
        self.distance
            .cmp(&other.distance)
            .then_with(|| self.range.start.cmp(&other.range.start))
            .then_with(|| self.range.end.cmp(&other.range.end))
    }
}

pub type ScoredVector<D> = ScoredItemGeneric<D, VectorId>;
pub type ScoredVectorDotProduct = ScoredVector<crate::core::distances::DotProduct>;
pub type ScoredRangeDotProduct = ScoredRange<crate::core::distances::DotProduct>;
pub type ScoredVectorSquaredEuclidean =
    ScoredVector<crate::core::distances::SquaredEuclideanDistance>;
pub type ScoredRangeSquaredEuclidean =
    ScoredRange<crate::core::distances::SquaredEuclideanDistance>;

/// Deprecated alias for backwards compatibility.
#[deprecated(since = "0.2.0", note = "Use ScoredVectorSquaredEuclidean instead")]
pub type ScoredVectorEuclidean = ScoredVectorSquaredEuclidean;
/// Deprecated alias for backwards compatibility.
#[deprecated(since = "0.2.0", note = "Use ScoredRangeSquaredEuclidean instead")]
pub type ScoredRangeEuclidean = ScoredRangeSquaredEuclidean;

/// A `Dataset` stores a collection of dense or sparse embedding vectors.
///
/// At the moment we have two implementations:
/// - `DenseDataset` in which we store fixed-length dense vectors, i.e., vectors for which there is no need to store compoennt indices.
/// - `SparseDataset` in which we store variable-length sparse vectors, i.e., vectors for which we need to store component indices.
/// - `PackedDataset` in which we store variable-length packed encodings (not `Vector1D`) concatenated in a single array and delimited by `offsets`.
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
/// ```text
/// VectorId -> (offset, length) -> vector data
/// ```
///
/// This extra lookup can slow down accesses and reduce the effectiveness
/// of software prefetching.
///
/// To avoid this, datasets can also expose the underlying range of their
/// packed storage for each vector. A `Range<usize>` identifies the slice
/// containing the encoded vector in the dataset’s storage. For dense datasets,
/// this is a fixed-size range derived from the vector index; for sparse and
/// packed datasets it comes from the offsets table. Converting a range back
/// to a `VectorId` may still be expensive (e.g., a binary search on offsets).
///
/// We recommend an indexing data structure (e.g., Seismic, IVF, HNSW) to
/// store internally (a compact version) of ranges instead of VectorIds.
///
/// # Example
/// ```
/// use vectorium::{
///     Dataset, DenseVector1D, DotProduct, GrowableDataset, PlainDenseDatasetGrowable,
///     PlainDenseQuantizer, Vector1D, VectorEncoder,
/// };
///
/// let quantizer = PlainDenseQuantizer::<f32, DotProduct>::new(3);
/// let mut dataset = PlainDenseDatasetGrowable::new(quantizer);
/// dataset.push(DenseVector1D::new(vec![1.0, 0.0, 2.0]));
///
/// let v = dataset.get(0);
/// assert_eq!(v.values_as_slice(), &[1.0, 0.0, 2.0]);
/// ```
pub trait Dataset<Q>: SpaceUsage
where
    Q: VectorEncoder,
{
    fn quantizer(&self) -> &Q;

    /// Get a query evaluator for the given query vector.
    #[inline]
    fn get_query_evaluator<'a, QueryVector>(&'a self, query: &'a QueryVector) -> Q::Evaluator<'a>
    where
        QueryVector: QueryVectorFor<Q> + ?Sized,
    {
        self.quantizer().get_query_evaluator(query)
    }

    fn shape(&self) -> (usize, usize) {
        (self.len(), self.output_dim())
    }

    /// Dimensionality of the input vectors (original vector space).
    #[inline]
    fn input_dim(&self) -> usize {
        self.quantizer().input_dim()
    }

    /// Dimensionality of the encoded vectors (stored representation).
    /// For example, this is equal to `m` for PQ quantizers, or the number of values per vector for scalar quantizers.
    #[inline]
    fn output_dim(&self) -> usize {
        self.quantizer().output_dim()
    }

    fn len(&self) -> usize;

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn nnz(&self) -> usize;

    fn range_from_id(&self, id: VectorId) -> std::ops::Range<usize>;

    fn id_from_range(&self, range: std::ops::Range<usize>) -> VectorId;

    /// Get the representation of the vector with the given id.
    #[inline]
    fn get<'a>(&'a self, id: VectorId) -> Q::EncodedVector<'a> {
        let range = self.range_from_id(id);
        self.get_by_range(range)
    }

    /// Get the representation of the vector with the given range.
    fn get_by_range<'a>(&'a self, range: std::ops::Range<usize>) -> Q::EncodedVector<'a>;

    fn prefetch(&self, range: std::ops::Range<usize>);

    /// Returns an iterator over all encoded vectors in the dataset.
    fn iter<'a>(&'a self) -> impl Iterator<Item = Q::EncodedVector<'a>>;

    /// Performs a brute-force search to find the K-nearest neighbors (KNN) of the queried vector.
    ///
    /// This method scans the entire dataset to find the K-nearest neighbors of the queried vector.
    /// It computes the *dot product* between the queried vector and each vector in the dataset and returns
    /// the indices of the K-nearest neighbors along with their distances.
    ///
    #[inline]
    fn search(
        &self,
        query: impl QueryVectorFor<Q>,
        k: usize,
    ) -> Vec<ScoredVector<<Q as VectorEncoder>::Distance>> {
        if k == 0 {
            return Vec::new();
        }

        let evaluator = self.quantizer().get_query_evaluator(&query);

        self.iter()
            .enumerate()
            .map(|(i, vector)| ScoredVector {
                distance: evaluator.compute_distance(vector),
                vector: i as u64,
            })
            .k_smallest(k)
            .collect()
    }
}

pub trait GrowableDataset<Q>: Dataset<Q>
where
    Q: VectorEncoder,
{
    /// Growable dataset API.
    ///
    /// # Example
    /// ```
    /// use vectorium::{
    ///     Dataset, DenseVector1D, DotProduct, GrowableDataset, PlainDenseDatasetGrowable,
    ///     PlainDenseQuantizer, VectorEncoder,
    /// };
    ///
    /// let quantizer = PlainDenseQuantizer::<f32, DotProduct>::new(2);
    /// let mut dataset = PlainDenseDatasetGrowable::new(quantizer);
    /// dataset.push(DenseVector1D::new(vec![1.0, 2.0]));
    /// dataset.push(DenseVector1D::new(vec![3.0, 4.0]));
    /// assert_eq!(dataset.len(), 2);
    /// ```
    /// Create a new growable dataset with the given quantizer and dimensionality.
    fn new(quantizer: Q) -> Self;

    /// Push a new vector into the dataset.
    /// The vector must have the appropriate component and value types as defined by the quantizer.
    /// The encoding defined by the quantizer is applied to the input vector before storing it in the dataset.
    fn push(
        &mut self,
        vec: impl Vector1D<ComponentType = Q::InputComponentType, ValueType = Q::InputValueType>,
    );
}
