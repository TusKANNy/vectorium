//! Multivector dataset with offset-based variable-length storage.
//!
//! Documents are stored flat in a single buffer; an offsets array records where
//! each document starts and ends, analogous to a CSR sparse-matrix row pointer array.

use serde::{Deserialize, Serialize};

use crate::SpaceUsage;
use crate::core::sealed;
use crate::core::vector::DenseVectorView;
use crate::core::vector_encoder::MultiVecEncoder;
use crate::utils::prefetch_read_slice;
use crate::{Dataset, DatasetGrowable, VectorId};

use rayon::prelude::*;

/// Growable multivector dataset backed by `Vec` flat buffers.
pub type MultiVectorDatasetGrowable<E> =
    MultiVectorDatasetGeneric<E, Vec<usize>, Vec<<E as MultiVecEncoder>::OutputValueType>>;

/// Immutable multivector dataset backed by boxed-slice flat buffers.
pub type MultiVectorDataset<E> =
    MultiVectorDatasetGeneric<E, Box<[usize]>, Box<[<E as MultiVecEncoder>::OutputValueType]>>;

/// Shared implementation for growable and frozen multivector datasets.
///
/// Documents are stored as flat buffers of `token_dim * n_tokens` values concatenated
/// in insertion order. Variable per-document token counts are tracked via `offsets`:
/// document `i` occupies `data[offsets[i]..offsets[i+1]]`.
/// The sentinel `offsets[0] == 0` is always present; `offsets.len() == n_docs + 1`.
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct MultiVectorDatasetGeneric<E, Offsets, Data>
where
    E: MultiVecEncoder,
    Offsets: AsRef<[usize]>,
    Data: AsRef<[E::OutputValueType]>,
{
    data: Data,
    offsets: Offsets,
    encoder: E,
}

impl<E, Offsets, Data> sealed::Sealed for MultiVectorDatasetGeneric<E, Offsets, Data>
where
    E: MultiVecEncoder,
    Offsets: AsRef<[usize]>,
    Data: AsRef<[E::OutputValueType]>,
{
}

impl<E, Offsets, Data> MultiVectorDatasetGeneric<E, Offsets, Data>
where
    E: MultiVecEncoder,
    Offsets: AsRef<[usize]>,
    Data: AsRef<[E::OutputValueType]>,
{
    /// Build a dataset from pre-encoded raw buffers.
    ///
    /// `offsets` must satisfy `offsets[0] == 0` and `*offsets.last() == data.len()`.
    pub fn from_raw(data: Data, offsets: Offsets, encoder: E) -> Self {
        let offsets_slice = offsets.as_ref();
        assert!(
            !offsets_slice.is_empty(),
            "offsets must contain at least the sentinel 0"
        );
        assert_eq!(offsets_slice[0], 0, "offsets[0] must be 0");
        assert_eq!(
            *offsets_slice.last().unwrap(),
            data.as_ref().len(),
            "offsets.last() must equal data.len()"
        );
        Self {
            data,
            offsets,
            encoder,
        }
    }

    /// Access the flat encoded buffer.
    pub fn values(&self) -> &[E::OutputValueType] {
        self.data.as_ref()
    }

    /// Access the raw offsets array.
    pub fn offsets(&self) -> &[usize] {
        self.offsets.as_ref()
    }

    /// Parallel iterator over documents as `EncodedVector` views.
    pub fn par_iter(&self) -> impl ParallelIterator<Item = E::EncodedVector<'_>> + '_
    where
        for<'a> E::EncodedVector<'a>: Send,
    {
        let data = self.data.as_ref();
        self.offsets.as_ref().par_windows(2).map(move |window| {
            let &[start, end] = window else {
                unsafe { std::hint::unreachable_unchecked() }
            };
            DenseVectorView::new(&data[start..end])
        })
    }
}

impl<E, Offsets, Data> Dataset for MultiVectorDatasetGeneric<E, Offsets, Data>
where
    E: MultiVecEncoder,
    Offsets: AsRef<[usize]>,
    Data: AsRef<[E::OutputValueType]>,
{
    type Encoder = E;

    fn encoder(&self) -> &E {
        &self.encoder
    }

    fn len(&self) -> usize {
        self.offsets.as_ref().len().saturating_sub(1)
    }

    fn nnz(&self) -> usize {
        self.data.as_ref().len()
    }

    fn range_from_id(&self, id: VectorId) -> std::ops::Range<usize> {
        let offsets = self.offsets.as_ref();
        let idx = id as usize;
        assert!(idx + 1 < offsets.len(), "Index out of bounds.");
        offsets[idx]..offsets[idx + 1]
    }

    fn id_from_range(&self, range: std::ops::Range<usize>) -> VectorId {
        let offsets = self.offsets.as_ref();
        let idx = offsets.binary_search(&range.start).unwrap();
        assert_eq!(
            offsets[idx + 1],
            range.end,
            "Range does not match vector boundaries."
        );
        idx as VectorId
    }

    fn get(&self, index: VectorId) -> E::EncodedVector<'_> {
        let range = self.range_from_id(index);
        DenseVectorView::new(&self.data.as_ref()[range])
    }

    fn get_with_range(&self, range: std::ops::Range<usize>) -> E::EncodedVector<'_> {
        DenseVectorView::new(&self.data.as_ref()[range])
    }

    fn prefetch_with_range(&self, range: std::ops::Range<usize>) {
        prefetch_read_slice(&self.data.as_ref()[range]);
    }

    fn iter(&self) -> impl Iterator<Item = E::EncodedVector<'_>> {
        let data = self.data.as_ref();
        self.offsets
            .as_ref()
            .windows(2)
            .map(move |w| DenseVectorView::new(&data[w[0]..w[1]]))
    }
}

impl<E> DatasetGrowable for MultiVectorDatasetGrowable<E>
where
    E: MultiVecEncoder,
{
    fn new(encoder: E) -> Self {
        Self {
            data: Vec::new(),
            offsets: vec![0],
            encoder,
        }
    }

    fn with_capacity(encoder: E, capacity: usize) -> Self {
        let mut offsets = Vec::with_capacity(capacity + 1);
        offsets.push(0);
        Self {
            data: Vec::new(),
            offsets,
            encoder,
        }
    }

    fn push<'a>(&mut self, vec: E::InputVector<'a>) {
        assert!(
            vec.len() % self.encoder.input_dim() == 0,
            "Input length must be a multiple of token_dim ({}), got {}",
            self.encoder.input_dim(),
            vec.len(),
        );
        self.encoder.push_encoded(vec, &mut self.data);
        self.offsets.push(self.data.len());
    }
}

impl<E> MultiVectorDatasetGrowable<E>
where
    E: MultiVecEncoder,
{
    /// Build a new empty growable multivector dataset.
    pub fn new(encoder: E) -> Self {
        crate::DatasetGrowable::new(encoder)
    }

    /// Build a growable dataset with preallocated capacity for `doc_capacity` documents
    /// and `data_capacity` encoded scalar values.
    pub fn with_data_capacity(encoder: E, doc_capacity: usize, data_capacity: usize) -> Self {
        let mut offsets = Vec::with_capacity(doc_capacity + 1);
        offsets.push(0);
        Self {
            data: Vec::with_capacity(data_capacity),
            offsets,
            encoder,
        }
    }
}

impl<E> From<MultiVectorDatasetGrowable<E>> for MultiVectorDataset<E>
where
    E: MultiVecEncoder,
{
    fn from(dataset: MultiVectorDatasetGrowable<E>) -> Self {
        Self {
            data: dataset.data.into_boxed_slice(),
            offsets: dataset.offsets.into_boxed_slice(),
            encoder: dataset.encoder,
        }
    }
}

impl<E, Offsets, Data> SpaceUsage for MultiVectorDatasetGeneric<E, Offsets, Data>
where
    E: MultiVecEncoder + SpaceUsage,
    Offsets: AsRef<[usize]> + SpaceUsage,
    Data: AsRef<[E::OutputValueType]> + SpaceUsage,
{
    fn space_usage_bytes(&self) -> usize {
        self.encoder.space_usage_bytes()
            + self.data.space_usage_bytes()
            + self.offsets.space_usage_bytes()
    }
}

/// Marker trait for datasets whose encoder implements the multivector contract.
pub trait MultiVecData: Dataset<Encoder: MultiVecEncoder> {}

impl<E, Offsets, Data> MultiVecData for MultiVectorDatasetGeneric<E, Offsets, Data>
where
    E: MultiVecEncoder,
    Offsets: AsRef<[usize]>,
    Data: AsRef<[E::OutputValueType]>,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vector::DenseVectorView;
    use crate::encoders::multivec_scalar::PlainMultiVecQuantizer;

    #[test]
    fn multivec_dataset_push_and_get() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(2);
        let mut dataset = MultiVectorDatasetGrowable::new(encoder);

        dataset.push(DenseVectorView::new(&[1.0f32, 2.0, 3.0, 4.0])); // 2 tokens
        dataset.push(DenseVectorView::new(&[5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0])); // 3 tokens

        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.nnz(), 10);
        assert_eq!(dataset.get(0).values(), &[1.0f32, 2.0, 3.0, 4.0]);
        assert_eq!(dataset.get(1).values(), &[5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn multivec_dataset_range_roundtrip() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(2);
        let mut dataset = MultiVectorDatasetGrowable::new(encoder);

        dataset.push(DenseVectorView::new(&[1.0f32, 2.0, 3.0, 4.0]));
        dataset.push(DenseVectorView::new(&[5.0f32, 6.0, 7.0, 8.0]));

        assert_eq!(dataset.range_from_id(0), 0..4);
        assert_eq!(dataset.range_from_id(1), 4..8);
        assert_eq!(dataset.id_from_range(0..4), 0);
        assert_eq!(dataset.id_from_range(4..8), 1);
    }

    #[test]
    fn multivec_dataset_iter_matches_get() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(2);
        let mut dataset = MultiVectorDatasetGrowable::new(encoder);

        dataset.push(DenseVectorView::new(&[1.0f32, 2.0, 3.0, 4.0]));
        dataset.push(DenseVectorView::new(&[5.0f32, 6.0]));

        let collected: Vec<Vec<f32>> = dataset.iter().map(|v| v.values().to_vec()).collect();
        assert_eq!(
            collected,
            vec![vec![1.0f32, 2.0, 3.0, 4.0], vec![5.0f32, 6.0]]
        );
    }

    #[test]
    fn multivec_dataset_search_works() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(2);
        let mut dataset = MultiVectorDatasetGrowable::new(encoder);

        // doc0: [[1,0],[0,1]], doc1: [[2,0],[0,2]]
        dataset.push(DenseVectorView::new(&[1.0f32, 0.0, 0.0, 1.0]));
        dataset.push(DenseVectorView::new(&[2.0f32, 0.0, 0.0, 2.0]));

        // query: [1,0] — MaxSim with doc0=1, doc1=2, so doc1 wins
        let query = DenseVectorView::new(&[1.0f32, 0.0]);
        let results = dataset.search(query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].vector, 1);
        assert_eq!(results[1].vector, 0);
    }

    #[test]
    fn multivec_dataset_frozen_roundtrip() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(2);
        let mut growable = MultiVectorDatasetGrowable::new(encoder);

        growable.push(DenseVectorView::new(&[1.0f32, 2.0]));
        growable.push(DenseVectorView::new(&[3.0f32, 4.0, 5.0, 6.0]));

        let frozen: MultiVectorDataset<_> = growable.into();
        assert_eq!(frozen.len(), 2);
        assert_eq!(frozen.nnz(), 6);
        assert_eq!(frozen.get(1).values(), &[3.0f32, 4.0, 5.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "Input length must be a multiple of token_dim")]
    fn multivec_dataset_panics_on_misaligned_push() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(3);
        let mut dataset = MultiVectorDatasetGrowable::new(encoder);
        dataset.push(DenseVectorView::new(&[1.0f32, 2.0])); // 2 % 3 != 0
    }

    #[test]
    fn multivec_dataset_frozen_boxes_offsets() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(2);
        let mut growable = MultiVectorDatasetGrowable::new(encoder);
        growable.push(DenseVectorView::new(&[1.0f32, 0.0]));
        growable.push(DenseVectorView::new(&[0.0f32, 1.0]));

        let frozen: MultiVectorDataset<_> = growable.into();
        // offsets is now Box<[usize]>: [0, 2, 4]
        assert_eq!(frozen.offsets(), &[0usize, 2, 4]);
    }
}
