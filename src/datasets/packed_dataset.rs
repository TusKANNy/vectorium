//! Packed sparse dataset helpers keep the encodings in a single concatenated buffer and expose growable/immutable APIs.
//! The module reuses `PackedSparseVectorEncoder` implementations so conversions and queries stay efficient.
use serde::{Deserialize, Serialize};

use crate::PackedSparseVectorEncoder;
use crate::SpaceUsage;
use crate::core::sealed;
use crate::dataset::ConvertFrom;
use crate::utils::prefetch_read_slice;
use crate::{Dataset, GrowableDataset, PackedVectorView, SparseData, VectorId};

use rayon::prelude::*;

/// A growable packed dataset.
///
/// # Examples
///
/// ```
/// use vectorium::{
///     DotVByteFixedU8Encoder, FixedU8Q, FromF32, PackedSparseDatasetGrowable, GrowableDataset,
///     Dataset, SparseVectorView,
/// };
///
/// let encoder = DotVByteFixedU8Encoder::new(4, 4);
/// let mut dataset = PackedSparseDatasetGrowable::new(encoder);
/// dataset.push(SparseVectorView::new(
///     &[0_u16, 2],
///     &[
///         FixedU8Q::from_f32_saturating(1.0),
///         FixedU8Q::from_f32_saturating(2.0),
///     ],
/// ));
/// assert_eq!(dataset.len(), 1);
/// ```
pub type PackedSparseDatasetGrowable<E> = PackedSparseDatasetGeneric<
    E,
    Vec<usize>,
    Vec<<E as PackedSparseVectorEncoder>::PackedDataType>,
>;

impl<E> PackedSparseDatasetGrowable<E>
where
    E: PackedSparseVectorEncoder,
{
    #[inline]
    pub fn new(encoder: E) -> Self {
        crate::GrowableDataset::new(encoder)
    }

    #[inline]
    pub fn with_capacity(encoder: E, capacity: usize) -> Self {
        crate::GrowableDataset::with_capacity(encoder, capacity)
    }
}

/// An immutable packed dataset.
///
/// # Examples
///
/// ```
/// use vectorium::{
///     Dataset, DotProduct, DotVByteFixedU8Encoder, GrowableDataset, PackedSparseDataset,
///     PlainSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView,
/// };
///
/// let quantizer = PlainSparseQuantizer::<u16, f32, DotProduct>::new(3, 3);
/// let mut sparse = PlainSparseDatasetGrowable::new(quantizer);
/// sparse.push(SparseVectorView::new(&[0_u16], &[1.0_f32]));
/// let frozen: PlainSparseDataset<u16, f32, DotProduct> = sparse.into();
/// let packed: PackedSparseDataset<DotVByteFixedU8Encoder> = frozen.into();
/// assert_eq!(packed.len(), 1);
/// ```
pub type PackedSparseDataset<E> = PackedSparseDatasetGeneric<
    E,
    Box<[usize]>,
    Box<[<E as PackedSparseVectorEncoder>::PackedDataType]>,
>;

/// Dataset storing variable-length packed encodings in a single concatenated `data` array.
///
/// Vector boundaries are stored in `offsets`, exactly like in `SparseDataset`:
/// - `offsets.len() == len() + 1`
/// - `offsets[0] == 0`
/// - vector `i` lives in `data[offsets[i]..offsets[i+1]]`.
///
/// Packed dataset storing variable-length encodings with offsets.
///
/// # Example
/// ```
/// use vectorium::{
///     Dataset, DotProduct, DotVByteFixedU8Encoder, GrowableDataset, PackedSparseDataset,
///     PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView, VectorEncoder,
/// };
///
/// let quantizer = PlainSparseQuantizer::<u16, f32, DotProduct>::new(5, 5);
/// let mut sparse = PlainSparseDatasetGrowable::new(quantizer);
/// sparse.push(SparseVectorView::new(&[1_u16, 3], &[1.0, 2.0]));
///
/// let frozen: vectorium::PlainSparseDataset<u16, f32, DotProduct> = sparse.into();
/// let packed: PackedSparseDataset<DotVByteFixedU8Encoder> = frozen.into();
/// let range = packed.range_from_id(0);
/// let v = packed.get_with_range(range);
/// assert!(!v.data().is_empty());
/// ```
#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct PackedSparseDatasetGeneric<E, Offsets, Data>
where
    E: PackedSparseVectorEncoder,
    Offsets: AsRef<[usize]>,
    Data: AsRef<[E::PackedDataType]>,
{
    offsets: Offsets,
    data: Data,
    encoder: E,
    nnz: usize,
}

impl<E, Offsets, Data> sealed::Sealed for PackedSparseDatasetGeneric<E, Offsets, Data>
where
    E: PackedSparseVectorEncoder,
    Offsets: AsRef<[usize]>,
    Data: AsRef<[E::PackedDataType]>,
{
}

impl<E, Offsets, Data> PackedSparseDatasetGeneric<E, Offsets, Data>
where
    E: PackedSparseVectorEncoder,
    Offsets: AsRef<[usize]>,
    Data: AsRef<[E::PackedDataType]>,
{
    #[inline]
    /// Return the raw offsets array that demarcates each vector slice.
    pub fn offsets(&self) -> &[usize] {
        self.offsets.as_ref()
    }

    #[inline]
    /// Return the pooled packed data buffer shared by all vectors.
    pub fn data(&self) -> &[E::PackedDataType] {
        self.data.as_ref()
    }

    #[inline]
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    #[inline]
    /// Translate vector id into a data range.
    pub fn range_from_id(&self, id: VectorId) -> std::ops::Range<usize> {
        let index = id as usize;
        let offsets = self.offsets.as_ref();
        assert!(index + 1 < offsets.len(), "Index out of bounds.");
        offsets[index]..offsets[index + 1]
    }

    #[inline]
    /// Translate a packed data range back into the owning id.
    pub fn id_from_range(&self, range: std::ops::Range<usize>) -> VectorId {
        let offsets = self.offsets.as_ref();
        let idx = offsets.binary_search(&range.start).unwrap();
        assert_eq!(
            offsets[idx + 1],
            range.end,
            "Range does not match vector boundaries."
        );
        idx as VectorId
    }

    /// Parallel iterator over dataset encoded vectors.
    ///
    /// Each item is an `E::EncodedVector<'_>` borrowing its slice from the dataset `data`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rayon::iter::ParallelIterator;
    /// use vectorium::{
    ///     DotVByteFixedU8Encoder, FixedU8Q, FromF32, PackedSparseDataset, PackedSparseDatasetGrowable,
    ///     GrowableDataset, Dataset, SparseVectorView,
    /// };
    ///
    /// let encoder = DotVByteFixedU8Encoder::new(4, 4);
    /// let mut growable = PackedSparseDatasetGrowable::new(encoder);
    /// growable.push(SparseVectorView::new(
    ///     &[0_u16, 1],
    ///     &[
    ///         FixedU8Q::from_f32_saturating(1.0),
    ///         FixedU8Q::from_f32_saturating(2.0),
    ///     ],
    /// ));
    /// let packed: PackedSparseDataset<_> = growable.into();
    /// let collected: Vec<_> = packed.par_iter().collect();
    /// assert_eq!(collected.len(), packed.len());
    /// ```
    #[inline]
    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = E::EncodedVector<'_>> + '_
    where
        for<'a> E::EncodedVector<'a>: Send,
        Offsets: Sync,
        Data: Sync,
    {
        let offsets = self.offsets.as_ref();
        let data = self.data.as_ref();

        // https://github.com/rayon-rs/rayon/pull/789
        offsets.par_windows(2).map(move |window| {
            let start = window[0];
            let end = window[1];
            PackedVectorView::new(&data[start..end])
        })
    }
}

impl<E> GrowableDataset
    for PackedSparseDatasetGeneric<
        E,
        Vec<usize>,
        Vec<<E as PackedSparseVectorEncoder>::PackedDataType>,
    >
where
    E: PackedSparseVectorEncoder,
{
    #[inline]
    fn new(encoder: E) -> Self {
        Self {
            offsets: vec![0],
            data: Vec::new(),
            encoder,
            nnz: 0,
        }
    }

    fn with_capacity(encoder: E, capacity: usize) -> Self {
        let mut offsets = Vec::with_capacity(capacity + 1);
        offsets.push(0);
        Self {
            offsets,
            data: Vec::with_capacity(capacity),
            encoder,
            nnz: 0,
        }
    }

    #[inline]
    /// Append a packed encoding directly into the dataset buffer.
    ///
    /// The implementation relies on `push_encoded` to extend the pooled `data` vector,
    /// keeping allocations at a minimum.
    fn push<'a>(&mut self, vec: E::InputVector<'a>) {
        self.nnz += vec.components().len(); // Capture length before move if needed? Copy view is cheap.

        self.encoder.push_encoded(vec, &mut self.data);
        self.offsets.push(self.data.len());
    }
}

impl<E, Offsets, Data> SpaceUsage for PackedSparseDatasetGeneric<E, Offsets, Data>
where
    E: PackedSparseVectorEncoder,
    E: SpaceUsage,
    Offsets: AsRef<[usize]> + SpaceUsage,
    Data: AsRef<[E::PackedDataType]> + SpaceUsage,
{
    fn space_usage_bytes(&self) -> usize {
        self.encoder.space_usage_bytes()
            + self.offsets.space_usage_bytes()
            + self.data.space_usage_bytes()
            + self.nnz.space_usage_bytes()
    }
}

impl<E, Offsets, Data> Dataset for PackedSparseDatasetGeneric<E, Offsets, Data>
where
    E: PackedSparseVectorEncoder,
    Offsets: AsRef<[usize]>,
    Data: AsRef<[E::PackedDataType]>,
{
    type Encoder = E;

    #[inline]
    fn nnz(&self) -> usize {
        self.nnz
    }

    #[inline]
    fn range_from_id(&self, id: VectorId) -> std::ops::Range<usize> {
        let index = id as usize;
        let offsets = self.offsets.as_ref();
        assert!(index + 1 < offsets.len(), "Index out of bounds.");
        offsets[index]..offsets[index + 1]
    }

    #[inline]
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

    #[inline]
    fn encoder(&self) -> &E {
        &self.encoder
    }

    #[inline]
    fn len(&self) -> usize {
        self.offsets.as_ref().len().saturating_sub(1)
    }

    #[inline]
    fn get(&self, index: VectorId) -> E::EncodedVector<'_> {
        let range = self.range_from_id(index);
        self.get_with_range(range)
    }

    #[inline]
    fn get_with_range<'a>(&'a self, range: std::ops::Range<usize>) -> E::EncodedVector<'a> {
        let slice = &self.data.as_ref()[range];
        PackedVectorView::new(slice)
    }

    #[inline]
    fn prefetch_with_range(&self, range: std::ops::Range<usize>) {
        prefetch_read_slice(&self.data.as_ref()[range]);
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = E::EncodedVector<'a>> {
        let offsets = self.offsets.as_ref();
        let data = self.data.as_ref();
        offsets
            .windows(2)
            .map(move |w| PackedVectorView::new(&data[w[0]..w[1]]))
    }
}

impl<E, Offsets, Data> SparseData for PackedSparseDatasetGeneric<E, Offsets, Data>
where
    E: PackedSparseVectorEncoder,
    Offsets: AsRef<[usize]>,
    Data: AsRef<[E::PackedDataType]>,
{
}

impl<E> From<PackedSparseDatasetGrowable<E>> for PackedSparseDataset<E>
where
    E: PackedSparseVectorEncoder,
{
    fn from(dataset: PackedSparseDatasetGrowable<E>) -> Self {
        PackedSparseDatasetGeneric {
            offsets: dataset.offsets.into_boxed_slice(),
            data: dataset.data.into_boxed_slice(),
            encoder: dataset.encoder,
            nnz: dataset.nnz,
        }
    }
}

impl<E> ConvertFrom<PackedSparseDatasetGrowable<E>> for PackedSparseDataset<E>
where
    E: PackedSparseVectorEncoder,
{
    fn convert_from(dataset: PackedSparseDatasetGrowable<E>) -> Self {
        dataset.into()
    }
}

impl<E> From<PackedSparseDataset<E>> for PackedSparseDatasetGrowable<E>
where
    E: PackedSparseVectorEncoder,
{
    fn from(dataset: PackedSparseDataset<E>) -> Self {
        PackedSparseDatasetGeneric {
            offsets: dataset.offsets.to_vec(),
            data: dataset.data.to_vec(),
            encoder: dataset.encoder,
            nnz: dataset.nnz,
        }
    }
}

impl<E> ConvertFrom<PackedSparseDataset<E>> for PackedSparseDatasetGrowable<E>
where
    E: PackedSparseVectorEncoder,
{
    fn convert_from(dataset: PackedSparseDataset<E>) -> Self {
        dataset.into()
    }
}

impl<EIn, S> From<crate::datasets::sparse_dataset::SparseDatasetGeneric<EIn, S>>
    for PackedSparseDataset<crate::DotVByteFixedU8Encoder>
where
    EIn: crate::SparseVectorEncoder<OutputComponentType = u16>,
    EIn::OutputValueType: crate::ValueType + crate::Float,
    for<'a> EIn::EncodedVector<'a>: crate::VectorView,
    S: crate::core::storage::SparseStorage<EIn>,
{
    fn from(dataset: crate::datasets::sparse_dataset::SparseDatasetGeneric<EIn, S>) -> Self {
        use crate::SparseVectorEncoder;
        use crate::encoders::sparse_scalar::ScalarSparseQuantizer;
        use crate::{DotProduct, FixedU8Q};

        let dim = dataset.output_dim();
        // Use a scalar quantizer to map values from `EIn::OutputValueType` into `FixedU8Q`.
        let scalar =
            ScalarSparseQuantizer::<u16, EIn::OutputValueType, FixedU8Q, DotProduct>::new(dim, dim);

        let mut dotvbyte_encoder = crate::DotVByteFixedU8Encoder::new(dim, dim);

        // Train using the original dataset components.
        // DotVByte reorders components, so it only needs to see the component distribution.

        // Computing the permutation on the whole dataset can be expensive. Do it on a sample with 1/SAMPLE_RATE of the vectors.
        // For simplicity, we just take the first 5% of the dataset here.
        const SAMPLE_RATE: usize = 20;
        let sample_size = if dataset.len() / SAMPLE_RATE < 50_000 {
            dataset.len()
        } else {
            dataset.len() / SAMPLE_RATE
        };

        dotvbyte_encoder.train(dataset.iter().take(sample_size));

        let mut offsets = Vec::with_capacity(dataset.len() + 1);
        offsets.push(0);
        let mut data = Vec::new();

        for v in dataset.iter() {
            // Quantize on the fly
            let q_vec = scalar.encode_vector(v);
            // Encode (pack)
            dotvbyte_encoder.push_encoded(q_vec.as_view(), &mut data);
            offsets.push(data.len());
        }

        PackedSparseDatasetGeneric {
            offsets: offsets.into_boxed_slice(),
            data: data.into_boxed_slice(),
            encoder: dotvbyte_encoder,
            nnz: dataset.nnz(),
        }
    }
}

impl<EIn, S> ConvertFrom<crate::datasets::sparse_dataset::SparseDatasetGeneric<EIn, S>>
    for PackedSparseDataset<crate::DotVByteFixedU8Encoder>
where
    EIn: crate::SparseVectorEncoder<OutputComponentType = u16>,
    EIn::OutputValueType: crate::ValueType + crate::Float,
    for<'a> EIn::EncodedVector<'a>: crate::VectorView,
    S: crate::core::storage::SparseStorage<EIn>,
{
    fn convert_from(
        dataset: crate::datasets::sparse_dataset::SparseDatasetGeneric<EIn, S>,
    ) -> Self {
        dataset.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DotVByteFixedU8Encoder;
    use crate::FixedU8Q;
    use crate::FromF32 as _;
    use crate::core::vector::SparseVectorView;
    use crate::core::vector_encoder::VectorEncoder;

    #[test]
    fn conversion_and_dot_product() {
        use crate::GrowableDataset;
        use crate::QueryEvaluator as _;
        use crate::VectorEncoder as _;
        use crate::core::vector::SparseVectorView;
        use crate::distances::Distance as _;
        use crate::{
            DotProduct, DotVByteFixedU8Encoder, FixedU8Q, FromF32 as _, PlainSparseDataset,
            PlainSparseDatasetGrowable,
        };
        use num_traits::ToPrimitive as _;

        let dim = 505;

        let mut growable: PlainSparseDatasetGrowable<u16, f32, DotProduct> =
            PlainSparseDatasetGrowable::new(
                crate::PlainSparseQuantizer::<u16, f32, DotProduct>::new(dim, dim),
            );

        let v0_components = vec![1_u16, 10, 100];
        let v0_values = vec![1.5_f32, 2.0, 2.5];

        growable.push(SparseVectorView::new(&v0_components, &v0_values));

        let v1_components = vec![2_u16, 11];
        let v1_values = vec![0.5_f32, 1.0];

        growable.push(SparseVectorView::new(&v1_components, &v1_values));

        let frozen: PlainSparseDataset<u16, f32, DotProduct> = growable.into();

        let dataset: PackedSparseDataset<DotVByteFixedU8Encoder> = frozen.into();

        let query = SparseVectorView::new(&[1_u16, 10, 11][..], &[2.0_f32, 3.0, 4.0][..]);
        let evaluator = dataset.encoder().query_evaluator(query);

        let d0 = evaluator.compute_distance(dataset.get(0)).distance();
        let d1 = evaluator.compute_distance(dataset.get(1)).distance();

        let expected0 = FixedU8Q::from_f32_saturating(1.5).to_f32().unwrap() * 2.0
            + FixedU8Q::from_f32_saturating(2.0).to_f32().unwrap() * 3.0;
        let expected1 = FixedU8Q::from_f32_saturating(1.0).to_f32().unwrap() * 4.0;

        assert_eq!(d0, expected0);
        assert_eq!(d1, expected1);
    }

    #[test]
    fn packed_growable_immutable_roundtrip() {
        use crate::PackedSparseDatasetGrowable;
        use crate::core::vector::SparseVectorView;
        use crate::{DotVByteFixedU8Encoder, FixedU8Q, GrowableDataset, PackedSparseDataset};

        let dim = 8;
        let encoder = DotVByteFixedU8Encoder::new(dim, dim);
        let mut growable = PackedSparseDatasetGrowable::new(encoder);

        growable.push(SparseVectorView::new(
            &[1_u16, 4],
            &[
                FixedU8Q::from_f32_saturating(1.0),
                FixedU8Q::from_f32_saturating(2.0),
            ],
        ));
        growable.push(SparseVectorView::new(
            &[2_u16],
            &[FixedU8Q::from_f32_saturating(3.0)],
        ));

        let frozen: PackedSparseDataset<DotVByteFixedU8Encoder> = growable.into();
        assert_eq!(frozen.len(), 2);
        assert_eq!(frozen.nnz(), 3);

        let mut growable_again: PackedSparseDatasetGrowable<DotVByteFixedU8Encoder> = frozen.into();
        growable_again.push(SparseVectorView::new(
            &[7_u16],
            &[FixedU8Q::from_f32_saturating(4.0)],
        ));

        assert_eq!(growable_again.len(), 3);
        assert_eq!(growable_again.nnz(), 4);
    }

    #[test]
    fn packed_dataset_iter_and_convert_from_traits_work() {
        use crate::DotVByteFixedU8Encoder;
        use crate::PackedSparseDataset;
        use crate::PackedSparseDatasetGrowable;
        use crate::PlainSparseDataset;
        use crate::PlainSparseDatasetGrowable;
        use crate::core::vector::PackedVectorView;
        use crate::encoders::sparse_scalar::PlainSparseQuantizer;
        use crate::{DotProduct, SparseVectorView};

        let quantizer = PlainSparseQuantizer::<u16, f32, DotProduct>::new(6, 6);
        let mut growable = PlainSparseDatasetGrowable::new(quantizer);
        growable.push(SparseVectorView::new(&[0_u16, 2], &[1.0_f32, 2.0]));
        growable.push(SparseVectorView::new(&[1_u16, 3], &[3.0_f32, 4.0]));

        let frozen: PlainSparseDataset<u16, f32, DotProduct> = growable.into();
        let packed = PackedSparseDataset::convert_from(frozen.clone());

        assert_eq!(packed.nnz(), 4);
        assert_eq!(packed.len(), 2);

        let range = packed.range_from_id(1);
        let view: PackedVectorView<'_, u64> = packed.get_with_range(range);
        assert!(view.len() > 0);

        let iter_values: Vec<_> = packed.iter().collect();
        assert_eq!(iter_values.len(), 2);

        let par_iter_values: Vec<_> = packed.par_iter().collect();
        assert_eq!(par_iter_values.len(), 2);

        let growable_again: PackedSparseDatasetGrowable<DotVByteFixedU8Encoder> =
            PackedSparseDatasetGrowable::convert_from(packed.clone());
        let packed_again = PackedSparseDataset::convert_from(growable_again);
        assert_eq!(packed_again.nnz(), packed.nnz());
    }

    #[test]
    fn packed_dataset_offsets_data_and_prefetch() {
        use crate::core::vector::SparseVectorView;
        use crate::{DotProduct, DotVByteFixedU8Encoder, FixedU8Q, PackedSparseDatasetGrowable};

        let dim = 5;
        let encoder = DotVByteFixedU8Encoder::new(dim, dim);
        let mut growable = PackedSparseDatasetGrowable::new(encoder);
        growable.push(SparseVectorView::new(
            &[1_u16],
            &[FixedU8Q::from_f32_saturating(1.0)],
        ));

        let dataset: PackedSparseDataset<DotVByteFixedU8Encoder> = growable.into();
        assert_eq!(dataset.nnz(), 1);
        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset.offsets(), &[0, dataset.data().len()]);
        assert_eq!(dataset.encoder().output_dim(), dim);
        dataset.prefetch_with_range(0..dataset.data().len());
        let mut iter = dataset.iter();
        assert!(iter.next().is_some());
        assert!(dataset.par_iter().count() > 0);
        assert!(dataset.space_usage_bytes() > 0);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds.")]
    fn packed_dataset_range_from_id_panics_when_invalid() {
        let encoder = DotVByteFixedU8Encoder::new(3, 3);
        let mut growable = PackedSparseDatasetGrowable::new(encoder);
        growable.push(SparseVectorView::new(
            &[0_u16],
            &[FixedU8Q::from_f32_saturating(0.5)],
        ));
        let dataset: PackedSparseDataset<DotVByteFixedU8Encoder> = growable.into();
        let _ = dataset.range_from_id(1);
    }

    #[test]
    #[should_panic]
    fn packed_dataset_id_from_range_panics_when_mismatch() {
        let encoder = DotVByteFixedU8Encoder::new(3, 3);
        let mut growable = PackedSparseDatasetGrowable::new(encoder);
        growable.push(SparseVectorView::new(
            &[0_u16],
            &[FixedU8Q::from_f32_saturating(0.5)],
        ));
        let dataset: PackedSparseDataset<DotVByteFixedU8Encoder> = growable.into();
        let _ = dataset.id_from_range(1..2);
    }

    #[test]
    fn packed_dataset_with_capacity_pushes_vectors() {
        use crate::{DotVByteFixedU8Encoder, FixedU8Q};

        let encoder = DotVByteFixedU8Encoder::new(3, 3);
        let mut growable = PackedSparseDatasetGrowable::with_capacity(encoder, 2);
        assert_eq!(growable.offsets().len(), 1);
        growable.push(SparseVectorView::new(
            &[0_u16],
            &[FixedU8Q::from_f32_saturating(1.0)],
        ));
        growable.push(SparseVectorView::new(
            &[1_u16],
            &[FixedU8Q::from_f32_saturating(2.0)],
        ));
        assert_eq!(growable.nnz(), 2);
        let frozen: PackedSparseDataset<DotVByteFixedU8Encoder> = growable.into();
        assert_eq!(frozen.len(), 2);
    }
}
