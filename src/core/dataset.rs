//! Core dataset abstractions used by the search/indexing primitives.
//! Includes type-safe traits to iterate, look up ranges, and integrate with encoders.
use crate::core::sealed;
use crate::core::vector_encoder::{
    DenseVectorEncoder, QueryEvaluator, SparseDataEncoder, VectorEncoder,
};
use itertools::Itertools;

/// Unique identifier assigned to each vector stored inside a dataset.
pub type VectorId = u64;

/// Holds a vector identifier together with the distance reported by a search evaluator.
#[derive(Debug, PartialOrd, Eq, Ord, PartialEq, Copy, Clone)]
pub struct ScoredItemGeneric<D, T> {
    pub distance: D,
    pub vector: T,
}

/// Helper type that maintains the ordering guarantees expected by callers of range-based APIs.
/// The tuple order (distance, range start, range end) is enforced so callers can merge ranges deterministically.
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

/// Specialized alias where the vector handle is a `VectorId`.
pub type ScoredVector<D> = ScoredItemGeneric<D, VectorId>;

/// Conversion helper that enforces datasets must be constructible from other datasets.
pub trait ConvertFrom<T: Dataset>: Dataset {
    /// Build this dataset type from `value`.
    fn convert_from(value: T) -> Self;
}

/// Mirror helper that forwards into `ConvertFrom` implementations.
pub trait ConvertInto<T: Dataset>: Dataset {
    /// Consume this dataset and produce another dataset wired by `ConvertFrom`.
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

/// Core abstraction for any collection of vectors paired with a `VectorEncoder`.
/// Implementations must expose the encoder but may choose arbitrary storage strategies.
/// Each vector is identified by a `VectorId` assigned at insertion time. The id is just the index of the vector in the dataset.
/// Each vector can be retrieved either by id or by specifying the underlying storage range directly.
/// The latter provides a faster access time on sparse datasets because we don't need to translate the id into a range first, by looking up an offsets table. This saves one memory access per vector retrieval and provides the oppurtunity for prefetching the vector data.
/// No advantage is expected for dense datasets, where the range can be trivially computed from the id and the vector dimensionality.
pub trait Dataset: sealed::Sealed {
    type Encoder: VectorEncoder;

    /// Shared encoder instance used to encode, query, and decode vectors.
    fn encoder(&self) -> &Self::Encoder;

    /// Dimensionality of the input vectors accepted by the encoder.
    fn input_dim(&self) -> usize {
        self.encoder().input_dim()
    }

    /// Dimensionality of the output vectors produced by the encoder.
    fn output_dim(&self) -> usize {
        self.encoder().output_dim()
    }

    /// Number of vectors stored in the dataset.
    fn len(&self) -> usize;

    /// Number of non-zero components in the underlying representation.
    fn nnz(&self) -> usize;

    /// Translate a persisted `VectorId` into a storage range.
    fn range_from_id(&self, id: VectorId) -> std::ops::Range<usize>;

    /// Translate a storage range back to the opaque `VectorId`.
    fn id_from_range(&self, range: std::ops::Range<usize>) -> VectorId;

    /// Check whether the dataset contains any vectors.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the representation of the vector with the given id.
    fn get(&self, index: VectorId) -> <Self::Encoder as VectorEncoder>::EncodedVector<'_>;

    /// Get a vector directly specifying the range in the underlying storage.
    fn get_with_range(
        &self,
        range: std::ops::Range<usize>,
    ) -> <Self::Encoder as VectorEncoder>::EncodedVector<'_>;

    /// Iterate through raw encoded vectors for cases where the caller is responsible for decoding.
    fn iter(&self) -> impl Iterator<Item = <Self::Encoder as VectorEncoder>::EncodedVector<'_>>;

    /// Touch the provided range to hint that it will be accessed soon.
    fn prefetch_with_range(&self, range: std::ops::Range<usize>);

    /// Exhaustive search that scores every vector via the configured query evaluator.
    /// Complexity is `θ(n log k)` because of the heap built by `k_smallest`.
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

/// Allows borrowing any dataset, so callers can pass references through the public APIs
/// without having to duplicate impls or rely on trait objects.
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

/// Marker trait representing any dataset whose encoder exposes the dense-vector contract.
///
/// Dense layout, query expectations, and decoding helpers are defined by `DenseVectorEncoder`,
/// so this marker only makes sense when the encoder implements that trait.
/// Requiring the encoder bound keeps downstream consumers honest about what they can rely on.
pub trait DenseData: Dataset<Encoder: DenseVectorEncoder> {}

/// Marker trait for datasets backed by shared sparse-data helpers.
/// Both plain `SparseVectorEncoder`s and packed encoders expose the common component/value types
/// via `SparseDataEncoder`. Consumers can rely on those helpers without needing to distinguish the layout.
pub trait SparseData: Dataset<Encoder: SparseDataEncoder> {}

pub trait DatasetGrowable: Dataset {
    /// Create a new growable dataset owning the provided encoder.
    fn new(encoder: Self::Encoder) -> Self;

    /// Create a new growable dataset with the given encoder and reserved certain capacity for vectors.
    fn with_capacity(encoder: Self::Encoder, capacity: usize) -> Self;

    /// Append another vector, encoded through the dataset encoder, into storage.
    fn push<'a>(&mut self, vec: <Self::Encoder as VectorEncoder>::InputVector<'a>);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vector::DenseVectorView;
    use crate::datasets::dense_dataset::DenseDatasetGrowable;
    use crate::distances::DotProduct;
    use crate::distances::SquaredEuclideanDistance;
    use crate::{DenseDataset, PlainDenseQuantizer};

    #[test]
    fn dataset_search_returns_expected_order() {
        type Encoder = PlainDenseQuantizer<f32, DotProduct>;

        let encoder = Encoder::new(2);
        let mut growable = DenseDatasetGrowable::new(encoder);
        growable.push(DenseVectorView::new(&[1.0f32, 0.5]));
        growable.push(DenseVectorView::new(&[0.0f32, 1.0]));
        growable.push(DenseVectorView::new(&[2.0f32, 1.0]));

        let query = DenseVectorView::new(&[1.5f32, 1.0]);
        let results = growable.search(query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].vector, 2);
        assert_eq!(results[0].distance, DotProduct::from(4.0));
        let second = &results[1];
        assert!(matches!(second.vector, 0));
        assert_eq!(second.distance, DotProduct::from(2.0));

        growable.push(DenseVectorView::new(&[5.0f32, 5.0]));
        growable.push(DenseVectorView::new(&[6.0f32, 6.0]));

        let results = growable.search(query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].vector, 4);
        assert_eq!(results[1].vector, 3);
    }

    #[test]
    fn scored_range_order_obeys_distance_then_start() {
        let r1 = ScoredRange {
            distance: DotProduct::from(2.0),
            range: 1..5,
        };
        let r2 = ScoredRange {
            distance: DotProduct::from(1.0),
            range: 0..2,
        };
        assert!(r1 < r2);

        let r1 = ScoredRange {
            distance: DotProduct::from(1.0),
            range: 0..2,
        };
        let r2 = ScoredRange {
            distance: DotProduct::from(1.0),
            range: 1..3,
        };
        assert!(r1 < r2);
        let r1 = ScoredRange {
            distance: SquaredEuclideanDistance::from(1.0),
            range: 1..5,
        };
        let r2 = ScoredRange {
            distance: SquaredEuclideanDistance::from(2.0),
            range: 0..2,
        };
        assert!(r1 < r2);

        let r1 = ScoredRange {
            distance: SquaredEuclideanDistance::from(1.0),
            range: 0..2,
        };
        let r2 = ScoredRange {
            distance: SquaredEuclideanDistance::from(1.0),
            range: 1..3,
        };
        assert!(r1 < r2);
    }

    #[test]
    fn dataset_trait_helpers_report_dimensions_and_empty() {
        type Encoder = PlainDenseQuantizer<f32, DotProduct>;

        let encoder = Encoder::new(2);
        let mut growable = DenseDatasetGrowable::new(encoder);
        growable.push(DenseVectorView::new(&[1.0f32, 0.0]));

        assert_eq!(growable.input_dim(), 2);
        assert_eq!(growable.output_dim(), 2);
        assert!(!growable.is_empty());
        assert!(growable.nnz() > 0);
    }

    #[test]
    fn dataset_search_with_zero_k_returns_empty() {
        type Encoder = PlainDenseQuantizer<f32, SquaredEuclideanDistance>;

        let encoder = Encoder::new(1);
        let mut growable = DenseDatasetGrowable::new(encoder);
        growable.push(DenseVectorView::new(&[0.0f32]));
        let dataset: DenseDataset<Encoder> = growable.into();
        let query = DenseVectorView::new(&[0.0f32]);

        assert!(dataset.search(query, 0).is_empty());
    }

    #[test]
    fn dataset_reference_forwarding_implements_all_methods() {
        type Encoder = PlainDenseQuantizer<f32, DotProduct>;

        let encoder = Encoder::new(2);
        let mut growable = DenseDatasetGrowable::new(encoder);
        growable.push(DenseVectorView::new(&[1.0f32, 0.5]));
        growable.push(DenseVectorView::new(&[0.0f32, 1.0]));
        growable.push(DenseVectorView::new(&[2.0f32, 1.0]));

        let dataset: DenseDataset<Encoder> = growable.into();

        fn exercise(dataset: &DenseDataset<Encoder>) {
            assert_eq!(<&DenseDataset<Encoder> as Dataset>::len(&dataset), 3);
            assert_eq!(<&DenseDataset<Encoder> as Dataset>::nnz(&dataset), 6);
            assert!(!<&DenseDataset<Encoder> as Dataset>::is_empty(&dataset));
            assert_eq!(<&DenseDataset<Encoder> as Dataset>::input_dim(&dataset), 2);
            assert_eq!(<&DenseDataset<Encoder> as Dataset>::output_dim(&dataset), 2);

            let first_range = <&DenseDataset<Encoder> as Dataset>::range_from_id(&dataset, 0);
            let second_range = <&DenseDataset<Encoder> as Dataset>::range_from_id(&dataset, 1);
            let third_range = <&DenseDataset<Encoder> as Dataset>::range_from_id(&dataset, 2);
            assert_eq!(first_range, 0..2);
            assert_eq!(second_range, 2..4);
            assert_eq!(third_range, 4..6);

            assert_eq!(
                <&DenseDataset<Encoder> as Dataset>::id_from_range(&dataset, first_range.clone()),
                0
            );
            assert_eq!(
                <&DenseDataset<Encoder> as Dataset>::id_from_range(&dataset, second_range.clone()),
                1
            );
            assert_eq!(
                <&DenseDataset<Encoder> as Dataset>::id_from_range(&dataset, third_range.clone()),
                2
            );

            <&DenseDataset<Encoder> as Dataset>::prefetch_with_range(&dataset, first_range.clone());
            <&DenseDataset<Encoder> as Dataset>::prefetch_with_range(
                &dataset,
                second_range.clone(),
            );
            <&DenseDataset<Encoder> as Dataset>::prefetch_with_range(&dataset, third_range.clone());

            let first_vector = <&DenseDataset<Encoder> as Dataset>::get(&dataset, 0);
            assert_eq!(first_vector, DenseVectorView::new(&[1.0f32, 0.5]));

            let second_vector =
                <&DenseDataset<Encoder> as Dataset>::get_with_range(&dataset, second_range.clone());
            assert_eq!(second_vector, DenseVectorView::new(&[0.0f32, 1.0]));

            let iterated = <&DenseDataset<Encoder> as Dataset>::iter(&dataset)
                .map(|vec| vec.values().to_vec())
                .collect::<Vec<_>>();

            assert_eq!(
                iterated,
                vec![vec![1.0f32, 0.5], vec![0.0f32, 1.0], vec![2.0f32, 1.0]]
            );
        }

        exercise(&dataset);
    }
}
