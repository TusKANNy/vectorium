//! Packed sparse dataset helpers keep the encodings in a single concatenated buffer and expose growable/immutable APIs.
//! The module reuses `PackedSparseVectorEncoder` implementations so conversions and queries stay efficient.
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::CDotPacking8FixedU8Encoder;
use crate::CDotPackingDp8FixedU8Encoder;
use crate::CDotPackingDp16FixedU8Encoder;
use crate::CegFixedU8Encoder;
use crate::DotPacking8FixedU8Encoder;
use crate::DotPacking8ScalarU8Encoder;
use crate::DotPackingDp8FixedU8Encoder;
use crate::DotPackingDp16FixedU8Encoder;
use crate::EgFixedU8Encoder;
use crate::PackedSparseVectorEncoder;
use crate::SpaceUsage;
use crate::core::sealed;
use crate::core::storage::SparseStorage;
use crate::dataset::ConvertFrom;
use crate::utils::prefetch_read_slice;
use crate::{Dataset, DatasetGrowable, PackedVectorView, SparseData, VectorId};
use crate::{Float, SparseVectorEncoder, ValueType, VectorView};

use rayon::prelude::*;

/// A growable packed dataset.
///
/// # Examples
///
/// ```
/// use vectorium::{
///     DotVByteFixedU8Encoder, FixedU8Q, FromF32, PackedSparseDatasetGrowable, DatasetGrowable,
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
        crate::DatasetGrowable::new(encoder)
    }

    #[inline]
    pub fn with_capacity(encoder: E, capacity: usize) -> Self {
        crate::DatasetGrowable::with_capacity(encoder, capacity)
    }
}

/// An immutable packed dataset.
///
/// # Examples
///
/// ```
/// use vectorium::{
///     Dataset, DotProduct, DotVByteFixedU8Encoder, DatasetGrowable, PackedSparseDataset,
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
///     Dataset, DotProduct, DotVByteFixedU8Encoder, DatasetGrowable, PackedSparseDataset,
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
    ///     DatasetGrowable, Dataset, SparseVectorView,
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

impl<E> DatasetGrowable
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

impl<E> PackedSparseDataset<E>
where
    E: PackedSparseVectorEncoder,
{
    /// Build an immutable dataset by encoding all sparse vectors in parallel.
    ///
    /// `flat_components`: all input components concatenated in vector order.
    /// `flat_values`: all input values concatenated in vector order.
    /// `vec_nnzs`: number of non-zero elements per vector.
    ///
    /// Each vector is encoded independently on a rayon thread pool, then the
    /// results are assembled into a single flat buffer. This is the preferred
    /// constructor when all input data is available upfront, as it is
    /// significantly faster than sequential `push` for any non-trivial encoder.
    pub fn from_flat_par(
        encoder: E,
        flat_components: &[E::InputComponentType],
        flat_values: &[E::InputValueType],
        vec_nnzs: &[usize],
    ) -> Self
    where
        E: Sync,
        E::InputComponentType: Sync,
        E::InputValueType: Sync,
        E::PackedDataType: Send,
    {
        let n_vecs = vec_nnzs.len();

        assert_eq!(
            flat_components.len(),
            flat_values.len(),
            "flat_components and flat_values must have same length"
        );

        // Build per-vector ranges into flat_components/flat_values (sequential, O(n_vecs)).
        let mut input_offsets = Vec::with_capacity(n_vecs + 1);
        input_offsets.push(0usize);
        for &nnz in vec_nnzs {
            input_offsets.push(input_offsets.last().unwrap() + nnz);
        }

        assert_eq!(
            *input_offsets.last().unwrap(),
            flat_components.len(),
            "sum of vec_nnzs must equal flat_components length"
        );

        let total_nnz: usize = vec_nnzs.iter().sum();

        // Encode each vector on a rayon thread (encoder is Sync, flat inputs are Sync).
        let encoded_vecs: Vec<Vec<E::PackedDataType>> = input_offsets
            .par_windows(2)
            .map(|w| {
                let view = crate::SparseVectorView::new(
                    &flat_components[w[0]..w[1]],
                    &flat_values[w[0]..w[1]],
                );
                let mut buf = Vec::new();
                encoder.push_encoded(view, &mut buf);
                buf
            })
            .collect();

        // Assemble flat data buffer and offsets (sequential, O(total_encoded_len)).
        let total_packed_len: usize = encoded_vecs.iter().map(|v| v.len()).sum();
        let mut data = Vec::with_capacity(total_packed_len);
        let mut offsets = Vec::with_capacity(n_vecs + 1);
        offsets.push(0usize);

        for vec in &encoded_vecs {
            data.extend_from_slice(vec);
            offsets.push(data.len());
        }

        Self {
            offsets: offsets.into_boxed_slice(),
            data: data.into_boxed_slice(),
            encoder,
            nnz: total_nnz,
        }
    }
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

impl<EIn, S> From<crate::datasets::sparse_dataset::SparseDatasetGeneric<EIn, S>>
    for PackedSparseDataset<crate::encoders::dotvbyte_scalaru8::DotVByteScalarU8Encoder>
where
    EIn: crate::SparseVectorEncoder<OutputComponentType = u16>,
    EIn::OutputValueType: crate::ValueType + crate::Float,
    for<'a> EIn::EncodedVector<'a>: crate::VectorView,
    S: crate::core::storage::SparseStorage<EIn>,
{
    fn from(dataset: crate::datasets::sparse_dataset::SparseDatasetGeneric<EIn, S>) -> Self {
        use crate::DatasetGrowable;
        use crate::SquaredEuclideanDistance;
        use crate::core::vector::SparseVectorView;
        use crate::encoders::dotvbyte_scalaru8::DotVByteScalarU8Encoder;
        use crate::encoders::sparse_scalar::PlainSparseQuantizer;
        use num_traits::ToPrimitive as _;

        let dim = dataset.output_dim();

        // Materialize the full dataset as f32 for scalar-quantizer training.
        // `DotVByteScalarU8Encoder::train` already samples internally for the
        // bisection permutation, but it must see all vectors for robust per-
        // component maxima in `train_sparse_scalar_quantizer`.
        let plain_quantizer =
            PlainSparseQuantizer::<u16, f32, SquaredEuclideanDistance>::new(dim, dim);
        let mut growable = crate::PlainSparseDatasetGrowable::new(plain_quantizer);
        for v in dataset.iter() {
            let components: Vec<u16> = v.components().to_vec();
            let values: Vec<f32> = v
                .values()
                .iter()
                .map(|x| x.to_f32().unwrap_or(0.0))
                .collect();
            growable.push(SparseVectorView::new(&components, &values));
        }
        let training_data: crate::PlainSparseDataset<u16, f32, SquaredEuclideanDistance> =
            growable.into();

        let mut encoder = DotVByteScalarU8Encoder::new(dim, dim);
        encoder.train::<f32>(&training_data);

        // Encode all vectors.
        let mut offsets = Vec::with_capacity(dataset.len() + 1);
        offsets.push(0);
        let mut data = Vec::new();
        for v in dataset.iter() {
            let components: Vec<u16> = v.components().to_vec();
            let values: Vec<f32> = v
                .values()
                .iter()
                .map(|x| x.to_f32().unwrap_or(0.0))
                .collect();
            encoder.push_encoded(SparseVectorView::new(&components, &values), &mut data);
            offsets.push(data.len());
        }

        PackedSparseDatasetGeneric {
            offsets: offsets.into_boxed_slice(),
            data: data.into_boxed_slice(),
            encoder,
            nnz: dataset.nnz(),
        }
    }
}

impl<EIn, S>
    crate::dataset::ConvertFrom<crate::datasets::sparse_dataset::SparseDatasetGeneric<EIn, S>>
    for PackedSparseDataset<crate::encoders::dotvbyte_scalaru8::DotVByteScalarU8Encoder>
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

impl<EIn, S> From<crate::datasets::sparse_dataset::SparseDatasetGeneric<EIn, S>>
    for PackedSparseDataset<crate::encoders::dotvbyte_u32_fixedu8::DotVByteU32FixedU8Encoder>
where
    EIn: crate::SparseVectorEncoder<OutputComponentType = u32>,
    EIn::OutputValueType: crate::ValueType + crate::Float,
    for<'a> EIn::EncodedVector<'a>: crate::VectorView,
    S: crate::core::storage::SparseStorage<EIn>,
{
    fn from(dataset: crate::datasets::sparse_dataset::SparseDatasetGeneric<EIn, S>) -> Self {
        use crate::SparseVectorEncoder;
        use crate::encoders::sparse_scalar::ScalarSparseQuantizer;
        use crate::{DotProduct, FixedU8Q};

        let dim = dataset.output_dim();
        let scalar =
            ScalarSparseQuantizer::<u32, EIn::OutputValueType, FixedU8Q, DotProduct>::new(dim, dim);

        let mut dotvbyte_encoder = crate::DotVByteU32FixedU8Encoder::new(dim, dim);

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
            let q_vec = scalar.encode_vector(v);
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
    for PackedSparseDataset<crate::encoders::dotvbyte_u32_fixedu8::DotVByteU32FixedU8Encoder>
where
    EIn: crate::SparseVectorEncoder<OutputComponentType = u32>,
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

impl<EIn, S> From<crate::datasets::sparse_dataset::SparseDatasetGeneric<EIn, S>>
    for PackedSparseDataset<crate::encoders::dotvbyte_u32_scalaru8::DotVByteU32ScalarU8Encoder>
where
    EIn: crate::SparseVectorEncoder<OutputComponentType = u32>,
    EIn::OutputValueType: crate::ValueType + crate::Float,
    for<'a> EIn::EncodedVector<'a>: crate::VectorView,
    S: crate::core::storage::SparseStorage<EIn>,
{
    fn from(dataset: crate::datasets::sparse_dataset::SparseDatasetGeneric<EIn, S>) -> Self {
        use crate::DatasetGrowable;
        use crate::SquaredEuclideanDistance;
        use crate::core::vector::SparseVectorView;
        use crate::encoders::dotvbyte_u32_scalaru8::DotVByteU32ScalarU8Encoder;
        use crate::encoders::sparse_scalar::PlainSparseQuantizer;
        use num_traits::ToPrimitive as _;

        let dim = dataset.output_dim();

        // Match u16 scalaru8 conversion behavior: train on the full dataset.
        let plain_quantizer =
            PlainSparseQuantizer::<u32, f32, SquaredEuclideanDistance>::new(dim, dim);
        let mut growable = crate::PlainSparseDatasetGrowable::new(plain_quantizer);
        for v in dataset.iter() {
            let components: Vec<u32> = v.components().to_vec();
            let values: Vec<f32> = v
                .values()
                .iter()
                .map(|x| x.to_f32().unwrap_or(0.0))
                .collect();
            growable.push(SparseVectorView::new(&components, &values));
        }
        let training_data: crate::PlainSparseDataset<u32, f32, SquaredEuclideanDistance> =
            growable.into();

        let mut encoder = DotVByteU32ScalarU8Encoder::new(dim, dim);
        encoder.train::<f32>(&training_data);

        let mut offsets = Vec::with_capacity(dataset.len() + 1);
        offsets.push(0);
        let mut data = Vec::new();
        for v in dataset.iter() {
            let components: Vec<u32> = v.components().to_vec();
            let values: Vec<f32> = v
                .values()
                .iter()
                .map(|x| x.to_f32().unwrap_or(0.0))
                .collect();
            encoder.push_encoded(SparseVectorView::new(&components, &values), &mut data);
            offsets.push(data.len());
        }

        PackedSparseDatasetGeneric {
            offsets: offsets.into_boxed_slice(),
            data: data.into_boxed_slice(),
            encoder,
            nnz: dataset.nnz(),
        }
    }
}

impl<EIn, S> ConvertFrom<crate::datasets::sparse_dataset::SparseDatasetGeneric<EIn, S>>
    for PackedSparseDataset<crate::encoders::dotvbyte_u32_scalaru8::DotVByteU32ScalarU8Encoder>
where
    EIn: crate::SparseVectorEncoder<OutputComponentType = u32>,
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

impl<EIn, S> From<crate::datasets::sparse_dataset::SparseDatasetGeneric<EIn, S>>
    for PackedSparseDataset<crate::encoders::dotvbyte_u32_fixedu8::OptimisticDotVByteFixedU8Encoder>
where
    EIn: crate::SparseVectorEncoder<OutputComponentType = u32>,
    EIn::OutputValueType: crate::ValueType + crate::Float,
    for<'a> EIn::EncodedVector<'a>: crate::VectorView,
    S: crate::core::storage::SparseStorage<EIn>,
{
    fn from(dataset: crate::datasets::sparse_dataset::SparseDatasetGeneric<EIn, S>) -> Self {
        use crate::SparseVectorEncoder;
        use crate::encoders::sparse_scalar::ScalarSparseQuantizer;
        use crate::{DotProduct, FixedU8Q};

        let dim = dataset.output_dim();
        let scalar =
            ScalarSparseQuantizer::<u32, EIn::OutputValueType, FixedU8Q, DotProduct>::new(dim, dim);

        let mut dotvbyte_encoder = crate::OptimisticDotVByteFixedU8Encoder::new(dim, dim);

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
            let q_vec = scalar.encode_vector(v);
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
    for PackedSparseDataset<crate::encoders::dotvbyte_u32_fixedu8::OptimisticDotVByteFixedU8Encoder>
where
    EIn: crate::SparseVectorEncoder<OutputComponentType = u32>,
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

impl<EIn, S> From<crate::datasets::sparse_dataset::SparseDatasetGeneric<EIn, S>>
    for PackedSparseDataset<
        crate::encoders::dotvbyte_u32_scalaru8::OptimisticDotVByteScalarU8Encoder,
    >
where
    EIn: crate::SparseVectorEncoder<OutputComponentType = u32>,
    EIn::OutputValueType: crate::ValueType + crate::Float,
    for<'a> EIn::EncodedVector<'a>: crate::VectorView,
    S: crate::core::storage::SparseStorage<EIn>,
{
    fn from(dataset: crate::datasets::sparse_dataset::SparseDatasetGeneric<EIn, S>) -> Self {
        use crate::DatasetGrowable;
        use crate::SquaredEuclideanDistance;
        use crate::core::vector::SparseVectorView;
        use crate::encoders::dotvbyte_u32_scalaru8::OptimisticDotVByteScalarU8Encoder;
        use crate::encoders::sparse_scalar::PlainSparseQuantizer;
        use num_traits::ToPrimitive as _;

        let dim = dataset.output_dim();

        // Match scalaru8 behavior: train on the full dataset.
        let plain_quantizer =
            PlainSparseQuantizer::<u32, f32, SquaredEuclideanDistance>::new(dim, dim);
        let mut growable = crate::PlainSparseDatasetGrowable::new(plain_quantizer);
        for v in dataset.iter() {
            let components: Vec<u32> = v.components().to_vec();
            let values: Vec<f32> = v
                .values()
                .iter()
                .map(|x| x.to_f32().unwrap_or(0.0))
                .collect();
            growable.push(SparseVectorView::new(&components, &values));
        }
        let training_data: crate::PlainSparseDataset<u32, f32, SquaredEuclideanDistance> =
            growable.into();

        let mut encoder = OptimisticDotVByteScalarU8Encoder::new(dim, dim);
        encoder.train::<f32>(&training_data);

        let mut offsets = Vec::with_capacity(dataset.len() + 1);
        offsets.push(0);
        let mut data = Vec::new();
        for v in dataset.iter() {
            let components: Vec<u32> = v.components().to_vec();
            let values: Vec<f32> = v
                .values()
                .iter()
                .map(|x| x.to_f32().unwrap_or(0.0))
                .collect();
            encoder.push_encoded(SparseVectorView::new(&components, &values), &mut data);
            offsets.push(data.len());
        }

        PackedSparseDatasetGeneric {
            offsets: offsets.into_boxed_slice(),
            data: data.into_boxed_slice(),
            encoder,
            nnz: dataset.nnz(),
        }
    }
}

impl<EIn, S> ConvertFrom<crate::datasets::sparse_dataset::SparseDatasetGeneric<EIn, S>>
    for PackedSparseDataset<
        crate::encoders::dotvbyte_u32_scalaru8::OptimisticDotVByteScalarU8Encoder,
    >
where
    EIn: crate::SparseVectorEncoder<OutputComponentType = u32>,
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

use crate::datasets::sparse_dataset::SparseDatasetGeneric;

impl<EIn, S> From<SparseDatasetGeneric<EIn, S>>
    for PackedSparseDataset<crate::encoders::blocked_sparse::BlockedSparseEncoder>
where
    EIn: crate::SparseVectorEncoder<OutputComponentType = u16>,
    EIn::OutputValueType: crate::ValueType + crate::Float,
    for<'a> EIn::EncodedVector<'a>: crate::VectorView,
    S: crate::core::storage::SparseStorage<EIn>,
{
    fn from(dataset: SparseDatasetGeneric<EIn, S>) -> Self {
        use crate::DotProduct;
        use crate::SparseVectorEncoder;
        use crate::encoders::blocked_sparse::BlockedSparseEncoder;
        use crate::encoders::sparse_scalar::ScalarSparseQuantizer;
        use half::f16;

        let dim = dataset.output_dim();
        let scalar =
            ScalarSparseQuantizer::<u16, EIn::OutputValueType, f16, DotProduct>::new(dim, dim);
        let encoder = BlockedSparseEncoder::new(dim);

        let mut offsets = Vec::with_capacity(dataset.len() + 1);
        offsets.push(0);
        let mut data = Vec::new();

        for v in dataset.iter() {
            let q_vec = scalar.encode_vector(v);
            encoder.push_encoded(q_vec.as_view(), &mut data);
            offsets.push(data.len());
        }

        PackedSparseDatasetGeneric {
            offsets: offsets.into_boxed_slice(),
            data: data.into_boxed_slice(),
            encoder,
            nnz: dataset.nnz(),
        }
    }
}

fn invalid_data_error(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

pub fn read_cluster_assignments(path: &Path, n_docs: usize) -> io::Result<Vec<usize>> {
    let clusters = BufReader::new(File::open(path)?);
    let initial = (vec![usize::MAX; n_docs], 0usize);

    let parsed_lines = clusters
        .lines()
        .enumerate()
        .filter_map(|(line_no, line_result)| match line_result {
            Err(err) => Some(Err(err)),
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    return None;
                }
                let values = trimmed
                    .split_whitespace()
                    .map(|token| {
                        token.parse::<usize>().map_err(|_| {
                            invalid_data_error(format!(
                                "Cluster file {} contains non-integer token '{}' at line {}",
                                path.display(),
                                token,
                                line_no + 1
                            ))
                        })
                    })
                    .collect::<io::Result<Vec<usize>>>();

                Some(values)
            }
        });

    let (assignments, assigned) = parsed_lines.enumerate().try_fold(
        initial,
        |(mut assignments, mut assigned), (cluster_id, values)| {
            let values = values?;
            let expected_cluster_len = values[0];
            let members = &values[1..];
            if members.len() != expected_cluster_len {
                return Err(invalid_data_error(format!(
                    "Cluster line {} has declared len {} but {} ids provided",
                    cluster_id + 1,
                    expected_cluster_len,
                    members.len()
                )));
            }

            members.iter().try_for_each(|&doc_id| {
                if doc_id >= n_docs {
                    return Err(invalid_data_error(format!(
                        "Cluster line {} references doc_id {} but dataset has {} docs",
                        cluster_id + 1,
                        doc_id,
                        n_docs
                    )));
                }
                if assignments[doc_id] != usize::MAX {
                    return Err(invalid_data_error(format!(
                        "doc_id {} appears in multiple clusters (line {})",
                        doc_id,
                        cluster_id + 1
                    )));
                }

                assignments[doc_id] = cluster_id;
                assigned += 1;
                Ok(())
            })?;

            Ok((assignments, assigned))
        },
    )?;

    if assigned != n_docs {
        return Err(invalid_data_error(format!(
            "Cluster file assignment incomplete: assigned {} docs but dataset has {} docs",
            assigned, n_docs
        )));
    }

    Ok(assignments)
}

pub fn build_cluster_references_from_dataset<EIn, S>(
    dataset: &SparseDatasetGeneric<EIn, S>,
    assignments: &[usize],
    dim: usize,
    max_ref_size: usize,
) -> io::Result<Vec<Vec<u16>>>
where
    EIn: SparseVectorEncoder<OutputComponentType = u16>,
    EIn::OutputValueType: ValueType + Float,
    for<'a> EIn::EncodedVector<'a>: VectorView,
    S: crate::core::storage::SparseStorage<EIn>,
{
    if assignments.len() != dataset.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "assignments len {} does not match dataset len {}",
                assignments.len(),
                dataset.len()
            ),
        ));
    }

    let n_clusters = assignments.iter().max().map_or(0usize, |value| value + 1);
    if n_clusters == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "No clusters found in assignments",
        ));
    }

    let mut occs = vec![vec![0u32; dim]; n_clusters];
    for (doc_id, vector) in dataset.iter().enumerate() {
        let cluster_id = assignments[doc_id];
        for &component in vector.components() {
            let idx = component as usize;
            if idx >= dim {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Component {} exceeds output dim {} for doc {}",
                        idx, dim, doc_id
                    ),
                ));
            }
            occs[cluster_id][idx] = occs[cluster_id][idx].saturating_add(1);
        }
    }

    let mut references = Vec::with_capacity(n_clusters);
    for cluster_occs in occs.into_iter() {
        let mut by_freq: Vec<(u16, u32)> = cluster_occs
            .iter()
            .enumerate()
            .filter_map(|(component, &freq)| {
                if freq > 0 {
                    Some((component as u16, freq))
                } else {
                    None
                }
            })
            .collect();

        if by_freq.is_empty() {
            references.push(vec![0u16]);
            continue;
        }

        by_freq.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let mut reference: Vec<u16> = by_freq
            .into_iter()
            .take(max_ref_size)
            .map(|(component, _)| component)
            .collect();
        reference.sort_unstable();
        references.push(reference);
    }

    Ok(references)
}

impl<EIn, S> crate::dataset::ConvertFrom<SparseDatasetGeneric<EIn, S>>
    for PackedSparseDataset<crate::encoders::blocked_sparse::BlockedSparseEncoder>
where
    EIn: crate::SparseVectorEncoder<OutputComponentType = u16>,
    EIn::OutputValueType: crate::ValueType + crate::Float,
    for<'a> EIn::EncodedVector<'a>: crate::VectorView,
    S: crate::core::storage::SparseStorage<EIn>,
{
    fn convert_from(dataset: SparseDatasetGeneric<EIn, S>) -> Self {
        dataset.into()
    }
}

macro_rules! impl_from_packed_sparse_dataset_clustered {
    ($ClusteredEncoderType:ty) => {
        impl PackedSparseDataset<$ClusteredEncoderType> {
            pub fn from_sparse_with_cluster_file<EIn, S>(
                dataset: &SparseDatasetGeneric<EIn, S>,
                cluster_file: impl AsRef<Path>,
                max_ref_size: usize,
            ) -> io::Result<Self>
            where
                EIn: SparseVectorEncoder<OutputComponentType = u16>,
                EIn::OutputValueType: ValueType + Float,
                for<'a> EIn::EncodedVector<'a>: VectorView,
                S: SparseStorage<EIn>,
            {
                use crate::SparseVectorEncoder;
                use crate::encoders::sparse_scalar::ScalarSparseQuantizer;
                use crate::{DotProduct, FixedU8Q};
                println!("Creating {} packed dataset from cluster file '{}'", stringify!($ClusteredEncoderType), cluster_file.as_ref().display());
                let dim = dataset.output_dim();
                let assignments = read_cluster_assignments(cluster_file.as_ref(), dataset.len())?;

                let references = build_cluster_references_from_dataset(
                    dataset,
                    &assignments,
                    dim,
                    max_ref_size,
                )?;

                let scalar =
                    ScalarSparseQuantizer::<u16, EIn::OutputValueType, FixedU8Q, DotProduct>::new(dim, dim);
                let mut encoder = <$ClusteredEncoderType>::new_with_references(dim, references, max_ref_size);

                const SAMPLE_RATE: usize = 20;
                let sample_size = if dataset.len() / SAMPLE_RATE < 50_000 {
                    dataset.len()
                } else {
                    dataset.len() / SAMPLE_RATE
                };

                encoder.train(dataset.iter().take(sample_size));

                let mut offsets = Vec::with_capacity(dataset.len() + 1);
                offsets.push(0);
                let mut data = Vec::new();

                for (doc_id, vector) in dataset.iter().enumerate() {
                    let q_vec = scalar.encode_vector(vector);
                    let cluster_idx = assignments[doc_id];
                    encoder.push_vector(q_vec.as_view(), cluster_idx as u16, &mut data);
                    offsets.push(data.len());
                }

                Ok(PackedSparseDatasetGeneric {
                    offsets: offsets.into_boxed_slice(),
                    data: data.into_boxed_slice(),
                    encoder,
                    nnz: dataset.nnz(),
                })
            }
        }

        impl<EIn, S> From<SparseDatasetGeneric<EIn, S>> for PackedSparseDataset<$ClusteredEncoderType>
        where
            EIn: crate::SparseVectorEncoder<OutputComponentType = u16>,
            EIn::OutputValueType: crate::ValueType + crate::Float,
            for<'a> EIn::EncodedVector<'a>: crate::VectorView,
            S: crate::core::storage::SparseStorage<EIn>,
        {
            fn from(dataset: SparseDatasetGeneric<EIn, S>) -> Self {
                crate::dataset::ConvertFrom::convert_from(&dataset)
            }
        }

        impl<'ds, EIn, S> ConvertFrom<&'ds SparseDatasetGeneric<EIn, S>> for PackedSparseDataset<$ClusteredEncoderType>
        where
            EIn: crate::SparseVectorEncoder<OutputComponentType = u16>,
            EIn::OutputValueType: crate::ValueType + crate::Float,
            for<'a> EIn::EncodedVector<'a>: crate::VectorView,
            S: crate::core::storage::SparseStorage<EIn>,
        {
            fn convert_from(dataset: &'ds SparseDatasetGeneric<EIn, S>) -> Self {
                let cluster_file = std::env::var("CLUSTER_FILE").unwrap_or_else(|_| {
                    panic!(
                        "CLUSTER_FILE environment variable must be set to the path of the cluster assignments file when converting to PackedSparseDataset<{}>",
                        stringify!($ClusteredEncoderType)
                    )
                });
                let max_ref_size = std::env::var("MAX_REF_SIZE")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(256);

                PackedSparseDataset::<$ClusteredEncoderType>::from_sparse_with_cluster_file(
                    dataset,
                    &cluster_file,
                    max_ref_size,
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "Failed to build {} packed dataset from cluster file '{}': {}",
                        stringify!($ClusteredEncoderType),
                        cluster_file,
                        e
                    )
                })
            }
        }

        impl<EIn, S> ConvertFrom<SparseDatasetGeneric<EIn, S>> for PackedSparseDataset<$ClusteredEncoderType>
        where
            EIn: crate::SparseVectorEncoder<OutputComponentType = u16>,
            EIn::OutputValueType: crate::ValueType + crate::Float,
            for<'a> EIn::EncodedVector<'a>: crate::VectorView,
            S: crate::core::storage::SparseStorage<EIn>,
        {
            fn convert_from(dataset: SparseDatasetGeneric<EIn, S>) -> Self {
                dataset.into()
            }
        }

    };
}

impl_from_packed_sparse_dataset_clustered!(CDotPacking8FixedU8Encoder);
impl_from_packed_sparse_dataset_clustered!(CDotPackingDp8FixedU8Encoder);
impl_from_packed_sparse_dataset_clustered!(CDotPackingDp16FixedU8Encoder);
impl_from_packed_sparse_dataset_clustered!(CegFixedU8Encoder);

macro_rules! impl_from_packed_sparse_dataset_fixedu8 {
    ($EncoderType:ty) => {
        impl<EIn, S> From<SparseDatasetGeneric<EIn, S>> for PackedSparseDataset<$EncoderType>
        where
            EIn: crate::SparseVectorEncoder<OutputComponentType = u16>,
            EIn::OutputValueType: crate::ValueType + crate::Float,
            for<'a> EIn::EncodedVector<'a>: crate::VectorView,
            S: crate::core::storage::SparseStorage<EIn>,
        {
            fn from(dataset: SparseDatasetGeneric<EIn, S>) -> Self {
                crate::dataset::ConvertFrom::convert_from(&dataset)
            }
        }

        impl<EIn, S> ConvertFrom<SparseDatasetGeneric<EIn, S>> for PackedSparseDataset<$EncoderType>
        where
            EIn: crate::SparseVectorEncoder<OutputComponentType = u16>,
            EIn::OutputValueType: crate::ValueType + crate::Float,
            for<'a> EIn::EncodedVector<'a>: crate::VectorView,
            S: crate::core::storage::SparseStorage<EIn>,
        {
            fn convert_from(dataset: SparseDatasetGeneric<EIn, S>) -> Self {
                dataset.into()
            }
        }

        impl<'ds, EIn, S> ConvertFrom<&'ds SparseDatasetGeneric<EIn, S>>
            for PackedSparseDataset<$EncoderType>
        where
            EIn: crate::SparseVectorEncoder<OutputComponentType = u16>,
            EIn::OutputValueType: crate::ValueType + crate::Float,
            for<'a> EIn::EncodedVector<'a>: crate::VectorView,
            S: crate::core::storage::SparseStorage<EIn>,
        {
            fn convert_from(dataset: &'ds SparseDatasetGeneric<EIn, S>) -> Self {
                use crate::SparseVectorEncoder;
                use crate::encoders::sparse_scalar::ScalarSparseQuantizer;
                use crate::{DotProduct, FixedU8Q};

                let dim = dataset.output_dim();
                let scalar =
                    ScalarSparseQuantizer::<u16, EIn::OutputValueType, FixedU8Q, DotProduct>::new(
                        dim, dim,
                    );

                let mut encoder = <$EncoderType>::new(dim);

                const SAMPLE_RATE: usize = 20;
                let sample_size = if dataset.len() / SAMPLE_RATE < 50_000 {
                    dataset.len()
                } else {
                    dataset.len() / SAMPLE_RATE
                };

                encoder.train(dataset.iter().take(sample_size));

                let mut offsets = Vec::with_capacity(dataset.len() + 1);
                offsets.push(0);
                let mut data = Vec::new();

                for v in dataset.iter() {
                    let q_vec = scalar.encode_vector(v);
                    encoder.push_encoded(q_vec.as_view(), &mut data);
                    offsets.push(data.len());
                }

                PackedSparseDatasetGeneric {
                    offsets: offsets.into_boxed_slice(),
                    data: data.into_boxed_slice(),
                    encoder,
                    nnz: dataset.nnz(),
                }
            }
        }
    };
}

macro_rules! impl_from_packed_sparse_dataset_f32 {
    ($EncoderType:ty) => {
        impl<EIn, S> From<SparseDatasetGeneric<EIn, S>> for PackedSparseDataset<$EncoderType>
        where
            EIn: SparseVectorEncoder<OutputComponentType = u16, InputValueType = f32, OutputValueType = f32>,
            EIn::OutputValueType: ValueType + Float,
            for<'a> EIn::EncodedVector<'a>: VectorView,
            S: SparseStorage<EIn>,
        {
            fn from(dataset: SparseDatasetGeneric<EIn, S>) -> Self {
                crate::dataset::ConvertFrom::convert_from(&dataset)
            }
        }

        impl<EIn, S> ConvertFrom<SparseDatasetGeneric<EIn, S>> for PackedSparseDataset<$EncoderType>
        where
            EIn: SparseVectorEncoder<OutputComponentType = u16, InputValueType = f32, OutputValueType = f32>,
            EIn::OutputValueType: ValueType + Float,
            for<'a> EIn::EncodedVector<'a>: VectorView,
            S: SparseStorage<EIn>,
        {
            fn convert_from(dataset: SparseDatasetGeneric<EIn, S>) -> Self {
                dataset.into()
            }
        }

        impl<'ds, EIn, S> ConvertFrom<&'ds SparseDatasetGeneric<EIn, S>>
            for PackedSparseDataset<$EncoderType>
        where
            EIn: SparseVectorEncoder<OutputComponentType = u16, InputValueType = f32, OutputValueType = f32>,
            EIn::OutputValueType: Float + ValueType,
            for<'a> EIn::EncodedVector<'a>: VectorView,
            S: SparseStorage<EIn>,
        {
            fn convert_from(dataset: &'ds SparseDatasetGeneric<EIn, S>) -> Self {
                use crate::SquaredEuclideanDistance;
                use crate::core::vector::SparseVectorView;
                use crate::encoders::sparse_scalar::PlainSparseQuantizer;
                use num_traits::ToPrimitive as _;

                let dim = dataset.output_dim();

                let mut encoder = <$EncoderType>::new(dim);

                let plain_quantizer = PlainSparseQuantizer::<u16, f32, SquaredEuclideanDistance>::new(dim, dim);
                let mut growable = crate::PlainSparseDatasetGrowable::new(plain_quantizer);
                for v in dataset.iter() {
                    let components: Vec<u16> = v.components().to_vec();
                    let values: Vec<f32> = v
                        .values()
                        .iter()
                        .map(|x| x.to_f32().unwrap_or(0.0))
                        .collect();
                    growable.push(SparseVectorView::new(&components, &values));
                }
                let training_data: crate::PlainSparseDataset<u16, f32, SquaredEuclideanDistance> =
                    growable.into();

                encoder.train(&training_data);

                let mut offsets = Vec::with_capacity(dataset.len() + 1);
                offsets.push(0);
                let mut data = Vec::new();

                for v in dataset.iter() {
                    encoder.push_encoded(v, &mut data);
                    offsets.push(data.len());
                }

                PackedSparseDatasetGeneric {
                    offsets: offsets.into_boxed_slice(),
                    data: data.into_boxed_slice(),
                    encoder,
                    nnz: dataset.nnz(),
                }
            }
        }
    };
}

impl_from_packed_sparse_dataset_f32!(DotPacking8FixedU8Encoder);
impl_from_packed_sparse_dataset_f32!(DotPacking8ScalarU8Encoder);
impl_from_packed_sparse_dataset_fixedu8!(DotPackingDp8FixedU8Encoder);
impl_from_packed_sparse_dataset_fixedu8!(DotPackingDp16FixedU8Encoder);
impl_from_packed_sparse_dataset_fixedu8!(EgFixedU8Encoder);

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
        use crate::DatasetGrowable;
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
        use crate::{DatasetGrowable, DotVByteFixedU8Encoder, FixedU8Q, PackedSparseDataset};

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
        assert!(!view.is_empty());

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
        use crate::{DotVByteFixedU8Encoder, FixedU8Q, PackedSparseDatasetGrowable};

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

    macro_rules! packed_dataset_plain_tests {
        ($dot_test:ident, $search_test:ident, $Encoder:ty) => {
            #[test]
            fn $dot_test() {
                use crate::DatasetGrowable;
                use crate::QueryEvaluator as _;
                use crate::VectorEncoder as _;
                use crate::core::vector::SparseVectorView;
                use crate::distances::Distance as _;
                use crate::{
                    DotProduct, FixedU8Q, FromF32 as _, PlainSparseDataset,
                    PlainSparseDatasetGrowable,
                };
                use num_traits::ToPrimitive as _;

                let dim = 505;

                let mut growable: PlainSparseDatasetGrowable<u16, f32, DotProduct> =
                    PlainSparseDatasetGrowable::new(crate::PlainSparseQuantizer::<
                        u16,
                        f32,
                        DotProduct,
                    >::new(dim, dim));

                let v0_components = vec![1_u16, 10, 100];
                let v0_values = vec![1.5_f32, 2.0, 2.5];

                growable.push(SparseVectorView::new(&v0_components, &v0_values));

                let v1_components = vec![2_u16, 11];
                let v1_values = vec![0.5_f32, 1.0];

                growable.push(SparseVectorView::new(&v1_components, &v1_values));

                let frozen: PlainSparseDataset<u16, f32, DotProduct> = growable.into();

                let dataset: PackedSparseDataset<$Encoder> = frozen.into();

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
            fn $search_test() {
                use crate::DatasetGrowable;
                use crate::core::vector::SparseVectorView;
                use crate::{DotProduct, PlainSparseDataset, PlainSparseDatasetGrowable};

                let dim = 505;

                let mut growable: PlainSparseDatasetGrowable<u16, f32, DotProduct> =
                    PlainSparseDatasetGrowable::new(crate::PlainSparseQuantizer::<
                        u16,
                        f32,
                        DotProduct,
                    >::new(dim, dim));

                let v0_components = vec![2_u16, 11];
                let v0_values = vec![0.5_f32, 1.0];

                let v1_components = vec![1_u16, 10, 100];
                let v1_values = vec![1.5_f32, 2.0, 2.5];

                growable.push(SparseVectorView::new(&v0_components, &v0_values));
                growable.push(SparseVectorView::new(&v1_components, &v1_values));

                let frozen: PlainSparseDataset<u16, f32, DotProduct> = growable.into();

                let dataset: PackedSparseDataset<$Encoder> = frozen.into();

                let query = SparseVectorView::new(&[1_u16, 10, 11][..], &[2.0_f32, 3.0, 4.0][..]);
                let results = dataset.search(query, 2);
                assert_eq!(results.len(), 2);
                let first_result = results[0];
                assert_eq!(first_result.vector, 1);
                let second_result = results[1];
                assert_eq!(second_result.vector, 0);
            }
        };
    }

    macro_rules! packed_dataset_clustered_tests {
        ($dot_test:ident, $search_test:ident, $Encoder:ty) => {
            #[test]
            fn $dot_test() {
                use crate::DatasetGrowable;
                use crate::QueryEvaluator as _;
                use crate::VectorEncoder as _;
                use crate::core::vector::SparseVectorView;
                use crate::distances::Distance as _;
                use crate::{
                    DotProduct, FixedU8Q, FromF32 as _, PlainSparseDataset,
                    PlainSparseDatasetGrowable,
                };
                use num_traits::ToPrimitive as _;
                use std::time::{SystemTime, UNIX_EPOCH};

                let dim = 505;

                let mut growable: PlainSparseDatasetGrowable<u16, f32, DotProduct> =
                    PlainSparseDatasetGrowable::new(crate::PlainSparseQuantizer::<
                        u16,
                        f32,
                        DotProduct,
                    >::new(dim, dim));

                let v0_components = vec![1_u16, 10, 100];
                let v0_values = vec![1.5_f32, 2.0, 2.5];

                growable.push(SparseVectorView::new(&v0_components, &v0_values));

                let v1_components = vec![2_u16, 11];
                let v1_values = vec![0.5_f32, 1.0];

                growable.push(SparseVectorView::new(&v1_components, &v1_values));

                let frozen: PlainSparseDataset<u16, f32, DotProduct> = growable.into();

                let nanos = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos();
                let cluster_path =
                    std::env::temp_dir().join(format!("vectorium_cluster_{}.txt", nanos));
                std::fs::write(&cluster_path, "1 0\n1 1\n").unwrap();

                let dataset = PackedSparseDataset::<$Encoder>::from_sparse_with_cluster_file(
                    &frozen,
                    &cluster_path,
                    5,
                )
                .unwrap();

                let query = SparseVectorView::new(&[1_u16, 10, 11][..], &[2.0_f32, 3.0, 4.0][..]);
                let evaluator = dataset.encoder().query_evaluator(query);

                let d0 = evaluator.compute_distance(dataset.get(0)).distance();
                let d1 = evaluator.compute_distance(dataset.get(1)).distance();

                let expected0 = FixedU8Q::from_f32_saturating(1.5).to_f32().unwrap() * 2.0
                    + FixedU8Q::from_f32_saturating(2.0).to_f32().unwrap() * 3.0;
                let expected1 = FixedU8Q::from_f32_saturating(1.0).to_f32().unwrap() * 4.0;

                assert_eq!(d0, expected0);
                assert_eq!(d1, expected1);

                let _ = std::fs::remove_file(cluster_path);
            }

            #[test]
            fn $search_test() {
                use crate::DatasetGrowable;
                use crate::core::vector::SparseVectorView;
                use crate::{DotProduct, PlainSparseDataset, PlainSparseDatasetGrowable};
                use std::time::{SystemTime, UNIX_EPOCH};

                let dim = 505;

                let mut growable: PlainSparseDatasetGrowable<u16, f32, DotProduct> =
                    PlainSparseDatasetGrowable::new(crate::PlainSparseQuantizer::<
                        u16,
                        f32,
                        DotProduct,
                    >::new(dim, dim));

                let v0_components = vec![2_u16, 11];
                let v0_values = vec![0.5_f32, 1.0];

                let v1_components = vec![1_u16, 10, 100];
                let v1_values = vec![1.5_f32, 2.0, 2.5];

                growable.push(SparseVectorView::new(&v0_components, &v0_values));
                growable.push(SparseVectorView::new(&v1_components, &v1_values));

                let frozen: PlainSparseDataset<u16, f32, DotProduct> = growable.into();

                let nanos = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos();
                let cluster_path =
                    std::env::temp_dir().join(format!("vectorium_cluster_{}.txt", nanos));
                std::fs::write(&cluster_path, "1 0\n1 1\n").unwrap();

                let dataset = PackedSparseDataset::<$Encoder>::from_sparse_with_cluster_file(
                    &frozen,
                    &cluster_path,
                    5,
                )
                .unwrap();

                let query = SparseVectorView::new(&[1_u16, 10, 11][..], &[2.0_f32, 3.0, 4.0][..]);
                let results = dataset.search(query, 2);
                assert_eq!(results.len(), 2);
                let first_result = results[0];
                assert_eq!(first_result.vector, 1);
                let second_result = results[1];
                assert_eq!(second_result.vector, 0);

                let _ = std::fs::remove_file(cluster_path);
            }
        };
    }

    packed_dataset_clustered_tests!(
        conversion_and_dot_product_ceg,
        conversion_and_search_ceg,
        CegFixedU8Encoder
    );
    packed_dataset_clustered_tests!(
        conversion_and_dot_product_cdot8,
        conversion_and_search_cdot8,
        CDotPacking8FixedU8Encoder
    );
    packed_dataset_clustered_tests!(
        conversion_and_dot_product_cdotdp8,
        conversion_and_search_cdotdp8,
        CDotPackingDp8FixedU8Encoder
    );
    packed_dataset_clustered_tests!(
        conversion_and_dot_product_cdotdp16,
        conversion_and_search_cdotdp16,
        CDotPackingDp16FixedU8Encoder
    );
    packed_dataset_plain_tests!(
        conversion_and_dot_product_bpdp8,
        conversion_and_search_bpdp8,
        DotPackingDp8FixedU8Encoder
    );
    packed_dataset_plain_tests!(
        conversion_and_dot_product_bpdp16,
        conversion_and_search_bpdp16,
        DotPackingDp16FixedU8Encoder
    );
    packed_dataset_plain_tests!(
        conversion_and_dot_product_bp8,
        conversion_and_search_bp8,
        DotPacking8FixedU8Encoder
    );

    
}
