use crate::core::sealed;
use crate::core::vector_encoder::{
    DenseVectorEncoder, QueryEvaluator, SparseDataEncoder, VectorEncoder,
};
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

    fn search<'d, 'q>(
        &'d self,
        query: <Self::Encoder as VectorEncoder>::QueryVector<'q>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DenseDataset;
    use crate::core::vector::DenseVectorView;
    use crate::datasets::dense_dataset::DenseDatasetGrowable;
    use crate::distances::DotProduct;
    use crate::encoders::dense_scalar::ScalarDenseQuantizer;

    #[test]
    fn dataset_search_returns_expected_order() {
        type Encoder = ScalarDenseQuantizer<f32, f32, DotProduct>;

        let encoder = Encoder::new(2);
        let mut growable = DenseDatasetGrowable::new(encoder);
        growable.push(DenseVectorView::new(&[1.0f32, 0.5]));
        growable.push(DenseVectorView::new(&[0.0f32, 1.0]));
        growable.push(DenseVectorView::new(&[2.0f32, 1.0]));

        let dataset: DenseDataset<Encoder> = growable.into();
        let query = DenseVectorView::new(&[1.5f32, 1.0]);
        let results = dataset.search(query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].vector, 2);
        assert_eq!(results[0].distance, DotProduct::from(4.0));
        let second = &results[1];
        assert!(matches!(second.vector, 0));
        assert_eq!(second.distance, DotProduct::from(2.0));
    }

    #[test]
    fn scored_range_order_obeys_distance_then_start() {
        let r1 = ScoredRange {
            distance: DotProduct::from(1.0),
            range: 0..2,
        };
        let r2 = ScoredRange {
            distance: DotProduct::from(1.0),
            range: 1..3,
        };
        assert!(r1 < r2);
    }
}

/// Marker trait representing any dataset whose encoder exposes the dense-vector contract.
///
/// Dense layout, query expectations, and decoding helpers are defined by `DenseVectorEncoder`, so this marker
/// only makes sense when the encoder implements that trait. Requiring the encoder bound keeps downstream consumers
/// honest about what they can rely on.
pub trait DenseData: Dataset
where
    Self::Encoder: DenseVectorEncoder,
{
}

/// Marker trait representing datasets whose encoder satisfies the shared sparse-input API.
///
/// Both plain `SparseVectorEncoder`s and packed encoders implement `SparseDataEncoder`, which exposes the common
/// component/value types and decoding helpers. Bounding this marker by `SparseDataEncoder` ensures consumers have
/// access to the shared behavior without needing to know whether the underlying layout is packed or unpacked.
pub trait SparseData: Dataset<Encoder: SparseDataEncoder> {}

pub trait GrowableDataset: Dataset {
    /// Create a new growable dataset with the given encoder.
    fn new(encoder: Self::Encoder) -> Self;

    /// Create a new growable dataset with the given encoder and capacity.
    fn with_capacity(encoder: Self::Encoder, capacity: usize) -> Self;

    /// Push a new vector into the dataset.
    fn push<'a>(&mut self, vec: <Self::Encoder as VectorEncoder>::InputVector<'a>);
}
