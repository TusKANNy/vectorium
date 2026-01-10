use crate::ValueType;
use crate::core::sealed;
use crate::core::vector_encoder::{QueryEvaluator, VectorEncoder};
use itertools::Itertools;

pub type VectorId = u64;

#[derive(Debug, PartialOrd, Eq, Ord, PartialEq, Copy, Clone)]
pub struct ScoredItemGeneric<D, T> {
    pub distance: D,
    pub vector: T,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ScoredRange<D> {
    pub distance: D,
    pub range: std::ops::Range<usize>,
}

impl<D: Eq> Eq for ScoredRange<D> {}

impl<D: Ord> PartialOrd for ScoredRange<D> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<D: Ord> Ord for ScoredRange<D> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .cmp(&other.distance)
            .then_with(|| self.range.start.cmp(&other.range.start))
            .then_with(|| self.range.end.cmp(&other.range.end))
    }
}

pub type ScoredVector<D> = ScoredItemGeneric<D, VectorId>;

pub trait ConvertFrom<T: Dataset>: Dataset {
    fn convert_from(value: T) -> Self;
}

pub trait ConvertInto<T: Dataset>: Dataset {
    fn convert_into(self) -> T;
}

impl<T, U> ConvertInto<U> for T
where
    T: Dataset,
    U: ConvertFrom<T>,
{
    fn convert_into(self) -> U {
        U::convert_from(self)
    }
}

// Note: Removed concrete type aliases for DotProduct/etc to avoid coupling with specific distance implementations here,
// or I can keep them if I import the concrete distances. I'll omit them for brevity unless required.

pub trait Dataset: sealed::Sealed {
    type Encoder: VectorEncoder;

    fn encoder(&self) -> &Self::Encoder;

    fn input_dim(&self) -> usize {
        self.encoder().input_dim()
    }

    fn output_dim(&self) -> usize {
        self.encoder().output_dim()
    }

    fn len(&self) -> usize;

    fn nnz(&self) -> usize;

    fn range_from_id(&self, id: VectorId) -> std::ops::Range<usize>;

    fn id_from_range(&self, range: std::ops::Range<usize>) -> VectorId;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the representation of the vector with the given id.
    fn get(&self, index: VectorId) -> <Self::Encoder as VectorEncoder>::EncodedVector<'_>;

    /// Get a vector directy specifying the range in the underlying storage.
    fn get_with_range(
        &self,
        range: std::ops::Range<usize>,
    ) -> <Self::Encoder as VectorEncoder>::EncodedVector<'_>;

    fn iter(&self) -> impl Iterator<Item = <Self::Encoder as VectorEncoder>::EncodedVector<'_>>;

    fn prefetch_with_range(&self, range: std::ops::Range<usize>);

    fn search<'d, 'q, V: ValueType>(
        &'d self,
        query: <Self::Encoder as VectorEncoder>::QueryVector<'q, V>,
        k: usize,
    ) -> Vec<ScoredVector<<Self::Encoder as VectorEncoder>::Distance>> {
        if k == 0 {
            return Vec::new();
        }

        let evaluator = self.encoder().query_evaluator(query);

        self.iter()
            .enumerate()
            .map(|(i, vector)| ScoredVector {
                distance: evaluator.compute_distance(vector),
                vector: i as VectorId,
            })
            .k_smallest(k)
            .collect()
    }
}

impl<T> sealed::Sealed for &T where T: Dataset {}

impl<T> Dataset for &T
where
    T: Dataset,
{
    type Encoder = T::Encoder;

    #[inline]
    fn encoder(&self) -> &Self::Encoder {
        (*self).encoder()
    }

    #[inline]
    fn len(&self) -> usize {
        (*self).len()
    }

    #[inline]
    fn nnz(&self) -> usize {
        (*self).nnz()
    }

    #[inline]
    fn range_from_id(&self, id: VectorId) -> std::ops::Range<usize> {
        (*self).range_from_id(id)
    }

    #[inline]
    fn id_from_range(&self, range: std::ops::Range<usize>) -> VectorId {
        (*self).id_from_range(range)
    }

    #[inline]
    fn get(&self, index: VectorId) -> <Self::Encoder as VectorEncoder>::EncodedVector<'_> {
        (*self).get(index)
    }

    #[inline]
    fn get_with_range(
        &self,
        range: std::ops::Range<usize>,
    ) -> <Self::Encoder as VectorEncoder>::EncodedVector<'_> {
        (*self).get_with_range(range)
    }

    #[inline]
    fn prefetch_with_range(&self, range: std::ops::Range<usize>) {
        (*self).prefetch_with_range(range)
    }

    #[inline]
    fn iter(&self) -> impl Iterator<Item = <Self::Encoder as VectorEncoder>::EncodedVector<'_>> {
        (*self).iter()
    }
}

/// Marker trait representing any dataset that stores dense vectors.
///
/// In the current design, the memory layout and structural constraints of a dataset are
/// rigidly determined by its associated `VectorEncoder` (e.g., `DenseVectorEncoder`).
/// It is impossible to have a "Dense Dataset" with a "Sparse Encoder" due to type system constraints.
///
/// However, external libraries or generic functions often need to express bounds like
/// "this function accepts any dense dataset" without listing every specific encoder trait or implementation.
///
/// This marker trait serves as a semantic category generic over the implementation details.
pub trait DenseData: Dataset {}

/// Marker trait representing any dataset that stores sparse vectors.
///
/// This covers both "plain" sparse datasets (using `SparseVectorEncoder`) and
/// "packed" sparse datasets (using `PackedSparseVectorEncoder`).
///
/// While `SparseDataset` and `PackedSparseDataset` are distinct types with distinct memory layouts
/// enforced by their specific encoders, they both logically represent sparse data.
/// This trait allows downstream code to treat them uniformly when the exact storage details
/// (e.g., whether it's component-value pairs or bit-packed blocks) are abstracted away.
pub trait SparseData: Dataset {}

pub trait GrowableDataset: Dataset {
    /// Create a new growable dataset with the given encoder.
    fn new(encoder: Self::Encoder) -> Self;

    /// Create a new growable dataset with the given encoder and capacity.
    fn with_capacity(encoder: Self::Encoder, capacity: usize) -> Self;

    /// Push a new vector into the dataset.
    fn push<'a>(&mut self, vec: <Self::Encoder as VectorEncoder>::InputVector<'a>);
}
