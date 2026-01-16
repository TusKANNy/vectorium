use serde::{Deserialize, Serialize};

use crate::SpaceUsage;
use crate::core::sealed;
use crate::core::vector_encoder::{DenseVectorEncoder, VectorEncoder};
use crate::{Dataset, GrowableDataset, VectorId};

use rayon::prelude::*;

/// Growable dense dataset backed by a `Vec` buffer.
///
/// This alias is the preferred entry point for building datasets incrementally.
/// The example below shows how to push vectors and freeze the dataset.
///
/// # Examples
///
/// ```
/// use vectorium::datasets::dense_dataset::DenseDatasetGrowable;
/// use vectorium::encoders::dense_scalar::ScalarDenseQuantizer;
/// use vectorium::distances::DotProduct;
/// use vectorium::core::vector::DenseVectorView;
/// use vectorium::GrowableDataset;
/// use vectorium::Dataset;
///
/// let encoder = ScalarDenseQuantizer::<f32, f32, DotProduct>::new(2);
/// let mut growable = DenseDatasetGrowable::new(encoder);
/// growable.push(DenseVectorView::new(&[1.0, 0.0]));
/// growable.push(DenseVectorView::new(&[0.0, 1.0]));
/// assert_eq!(growable.len(), 2);
/// ```
pub type DenseDatasetGrowable<E> =
    DenseDatasetGeneric<E, Vec<<E as DenseVectorEncoder>::OutputValueType>>;

/// Immutable dataset backed by a boxed slice.
///
/// Use this alias for workloads that need a compact, read-only dataset.
///
/// # Examples
///
/// ```
/// use vectorium::datasets::dense_dataset::DenseDatasetGrowable;
/// use vectorium::{DenseDataset, Dataset, GrowableDataset};
/// use vectorium::encoders::dense_scalar::ScalarDenseQuantizer;
/// use vectorium::distances::DotProduct;
/// use vectorium::core::vector::DenseVectorView;
///
/// let encoder = ScalarDenseQuantizer::<f32, f32, DotProduct>::new(2);
/// let mut growable = DenseDatasetGrowable::new(encoder);
/// growable.push(DenseVectorView::new(&[1.0, 0.0]));
/// growable.push(DenseVectorView::new(&[0.0, 1.0]));
/// let dataset: DenseDataset<_> = growable.into();
/// assert_eq!(dataset.len(), 2);
/// ```
pub type DenseDataset<E> =
    DenseDatasetGeneric<E, Box<[<E as DenseVectorEncoder>::OutputValueType]>>;

#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
/// Shared implementation for growable and frozen dense datasets.
/// Wraps a contiguous buffer and the encoder so callers can treat both variants with the same API.
pub struct DenseDatasetGeneric<E, Data>
where
    E: DenseVectorEncoder,
    Data: AsRef<[E::OutputValueType]>,
{
    n_vecs: usize,
    data: Data,
    encoder: E,
}

impl<E, Data> sealed::Sealed for DenseDatasetGeneric<E, Data>
where
    E: DenseVectorEncoder,
    Data: AsRef<[E::OutputValueType]>,
{
}

impl<E, Data> SpaceUsage for DenseDatasetGeneric<E, Data>
where
    E: DenseVectorEncoder,
    Data: AsRef<[E::OutputValueType]> + SpaceUsage,
{
    fn space_usage_bytes(&self) -> usize {
        self.n_vecs.space_usage_bytes()
            + self.encoder.space_usage_bytes()
            + self.data.space_usage_bytes()
    }
}

impl<E, Data> DenseDatasetGeneric<E, Data>
where
    E: DenseVectorEncoder,
    Data: AsRef<[E::OutputValueType]>,
{
    /// Build a dataset from its raw encoded buffer.
    ///
    /// # Examples
    ///
    /// ```
/// use vectorium::DenseDataset;
/// use vectorium::encoders::dense_scalar::ScalarDenseQuantizer;
/// use vectorium::distances::DotProduct;
/// use vectorium::Dataset;
    ///
    /// let encoder = ScalarDenseQuantizer::<f32, f32, DotProduct>::new(2);
    /// let dataset: DenseDataset<_> =
    ///     DenseDataset::from_raw(vec![1.0f32, 2.0, 3.0, 4.0].into_boxed_slice(), 2, encoder);
    /// assert_eq!(dataset.len(), 2);
    /// ```
    #[inline]
    pub fn from_raw(data: Data, n_vecs: usize, encoder: E) -> Self {
        assert_eq!(
            data.as_ref().len(),
            n_vecs * encoder.output_dim(),
            "Data length must equal n_vecs * encoder.output_dim()"
        );
        Self {
            n_vecs,
            data,
            encoder,
        }
    }

    /// Access the contiguous storage backing the dataset.
    ///
    /// This is the same buffer that gets populated by `DenseDatasetGrowable::push`.
    #[inline]
    pub fn values(&self) -> &[E::OutputValueType] {
        self.data.as_ref()
    }

    /// Number of stored elements (all components are considered non-zero here).
    #[inline]
    pub fn nnz(&self) -> usize {
        self.data.as_ref().len()
    }

    // Note: par_iter returns EncodedVector (view)
    /// Iterate the dataset in parallel without reallocating temporary buffers.
    ///
    /// # Examples
    ///
/// ```
/// use vectorium::DenseDataset;
/// use vectorium::encoders::dense_scalar::ScalarDenseQuantizer;
/// use vectorium::distances::DotProduct;
/// use vectorium::Dataset;
/// use rayon::iter::ParallelIterator;
///
/// let encoder = ScalarDenseQuantizer::<f32, f32, DotProduct>::new(2);
/// let dataset: DenseDataset<_> =
///     DenseDataset::from_raw(vec![1.0f32, 2.0, 3.0, 4.0].into_boxed_slice(), 2, encoder);
/// let collected: Vec<_> = dataset.par_iter().map(|view| view.values().to_vec()).collect();
/// assert_eq!(collected.len(), dataset.len());
/// ```
    #[inline]
    pub fn par_iter(&self) -> impl ParallelIterator<Item = E::EncodedVector<'_>> {
        let m = self.encoder.output_dim();
        let data = self.data.as_ref();
        let n = self.n_vecs;

        (0..n).into_par_iter().map(move |i| {
            let start = i * m;
            let end = start + m;
            DenseVectorView::new(&data[start..end])
        })
    }
}

impl<E, Data> Dataset for DenseDatasetGeneric<E, Data>
where
    E: DenseVectorEncoder,
    Data: AsRef<[E::OutputValueType]>,
{
    type Encoder = E;

    #[inline]
    fn encoder(&self) -> &E {
        &self.encoder
    }

    #[inline]
    fn len(&self) -> usize {
        self.n_vecs
    }

    #[inline]
    fn nnz(&self) -> usize {
        self.data.as_ref().len()
    }

    #[inline]
    fn range_from_id(&self, id: VectorId) -> std::ops::Range<usize> {
        let m = self.encoder.output_dim();
        let index = id as usize;
        assert!(index < self.n_vecs, "Index out of bounds.");
        let start = index * m;
        start..start + m
    }

    #[inline]
    fn id_from_range(&self, range: std::ops::Range<usize>) -> VectorId {
        let m = self.encoder.output_dim();

        if m == 0 {
            assert_eq!(
                range.start, range.end,
                "Range does not match vector boundaries."
            );
            return 0;
        }

        assert!(
            range.start % m == 0,
            "Range does not match vector boundaries."
        );
        assert_eq!(
            range.end,
            range.start + m,
            "Range does not match vector boundaries."
        );
        let idx = range.start / m;
        assert!(idx < self.n_vecs, "Index out of bounds.");
        idx as VectorId
    }

    #[inline]
    fn get(&self, index: VectorId) -> E::EncodedVector<'_> {
        assert!(index < self.n_vecs as VectorId, "Index out of bounds.");
        let m = self.encoder.output_dim();
        let start = index as usize * m;
        let end = start + m;
        DenseVectorView::new(&self.data.as_ref()[start..end])
    }

    #[inline]
    fn get_with_range(&self, range: std::ops::Range<usize>) -> E::EncodedVector<'_> {
        DenseVectorView::new(&self.data.as_ref()[range])
    }

    #[inline]
    fn iter(&self) -> impl Iterator<Item = E::EncodedVector<'_>> {
        let m = self.encoder.output_dim();
        let data = self.data.as_ref();
        let n = self.n_vecs;

        (0..n).map(move |i| {
            let start = i * m;
            let end = start + m;
            DenseVectorView::new(&data[start..end])
        })
    }

    #[inline]
    fn prefetch_with_range(&self, range: std::ops::Range<usize>) {
        crate::utils::prefetch_read_slice(&self.data.as_ref()[range]);
    }
}

// GrowableDataset implementation
// We need to implement push which takes EncodedVector?
// No, GrowableDataset takes InputVectorType to push.
// But InputVectorType was removed from VectorEncoder trait.
// It is now defined by `encode_vector` method signature which takes `DenseVectorView`.
// So we should adapt `GrowableDataset` trait or implementation.
// `DenseDatasetGeneric` assumes inputs are compatible with `E`.

use crate::core::vector::DenseVectorView;
use crate::{Float, FromF32, ValueType};

impl<E> GrowableDataset for DenseDatasetGeneric<E, Vec<E::OutputValueType>>
where
    E: DenseVectorEncoder,
{
    fn new(encoder: E) -> Self {
        Self {
            n_vecs: 0,
            data: Vec::new(),
            encoder,
        }
    }

    fn with_capacity(encoder: E, capacity: usize) -> Self {
        Self {
            n_vecs: 0,
            data: Vec::with_capacity(capacity * encoder.output_dim()),
            encoder,
        }
    }

    fn push<'a>(&mut self, vec: E::InputVector<'a>) {
        self.encoder.push_encoded(vec, &mut self.data);
        self.n_vecs += 1;
    }
}

impl<E> DenseDatasetGrowable<E>
where
    E: DenseVectorEncoder,
{
    /// Build a new growable dataset using the provided encoder.
    #[inline]
    pub fn new(encoder: E) -> Self {
        crate::GrowableDataset::new(encoder)
    }

    /// Build a growable dataset with the provided encoder and reserved capacity.
    #[inline]
    pub fn with_capacity(encoder: E, capacity: usize) -> Self {
        crate::GrowableDataset::with_capacity(encoder, capacity)
    }

    /// Return how many vectors can be stored without growing the underlying buffer.
    ///
    /// This mirrors the behavior of `Vec::capacity`, but expressed in vector units instead of scalar components.
    pub fn capacity(&self) -> usize {
        if self.encoder.output_dim() == 0 {
            0
        } else {
            self.data.capacity() / self.encoder.output_dim()
        }
    }

    /// Make room for `additional` vectors without extra reallocations.
    ///
    /// The argument counts vectors, so the method multiplies it by the encoder output dimension.
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional * self.encoder.output_dim());
    }
}

impl<VIn, VOut, D> DenseDatasetGrowable<crate::encoders::dense_scalar::ScalarDenseQuantizer<VIn, VOut, D>>
where
    VIn: ValueType + crate::Float,
    VOut: ValueType + crate::Float + crate::FromF32,
    D: crate::distances::Distance + crate::encoders::dense_scalar::ScalarDenseSupportedDistance,
{
    /// Convenience constructor that creates a quantizer and an empty dataset for any scalar quantizer.
    pub fn with_dim(dim: usize) -> Self {
        let encoder = crate::encoders::dense_scalar::ScalarDenseQuantizer::new(dim);
        crate::GrowableDataset::new(encoder)
    }

    /// Convenience constructor that also preallocates enough space for `capacity` vectors.
    pub fn with_dim_and_capacity(dim: usize, capacity: usize) -> Self {
        let encoder = crate::encoders::dense_scalar::ScalarDenseQuantizer::new(dim);
        Self {
            n_vecs: 0,
            data: Vec::with_capacity(capacity * dim),
            encoder,
        }
    }
}

impl<E> From<DenseDatasetGrowable<E>> for DenseDataset<E>
where
    E: DenseVectorEncoder,
{
    fn from(dataset: DenseDatasetGrowable<E>) -> Self {
        Self {
            n_vecs: dataset.n_vecs,
            data: dataset.data.into_boxed_slice(),
            encoder: dataset.encoder,
        }
    }
}

use crate::dataset::ConvertFrom;
use crate::encoders::dense_scalar::ScalarDenseQuantizer;
use crate::encoders::dense_scalar::ScalarDenseSupportedDistance;

impl<SrcIn, Mid, DstOut, D, SrcStorage, DstStorage>
    ConvertFrom<&DenseDatasetGeneric<ScalarDenseQuantizer<SrcIn, Mid, D>, SrcStorage>>
    for DenseDatasetGeneric<ScalarDenseQuantizer<Mid, DstOut, D>, DstStorage>
where
    SrcIn: ValueType + Float,
    Mid: ValueType + Float + FromF32,
    DstOut: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
    ScalarDenseQuantizer<SrcIn, Mid, D>:
        DenseVectorEncoder<InputValueType = SrcIn, OutputValueType = Mid>,
    ScalarDenseQuantizer<Mid, DstOut, D>:
        DenseVectorEncoder<InputValueType = Mid, OutputValueType = DstOut>,
    SrcStorage: AsRef<[Mid]>,
    DstStorage: From<Box<[DstOut]>> + AsRef<[DstOut]>,
{
    fn convert_from(
        source: &DenseDatasetGeneric<ScalarDenseQuantizer<SrcIn, Mid, D>, SrcStorage>,
    ) -> Self {
        let m = source.encoder.output_dim();
        let encoder = ScalarDenseQuantizer::<Mid, DstOut, D>::new(m);

        // Treat source data as a contiguous array of dense vectors of type Mid.
        let mut new_data = Vec::with_capacity(source.data.as_ref().len());
        let src_data = source.data.as_ref();

        for chunk in src_data.chunks_exact(m) {
            let vec_view = DenseVectorView::new(chunk);
            encoder.push_encoded(vec_view, &mut new_data);
        }

        Self {
            n_vecs: source.n_vecs,
            data: new_data.into_boxed_slice().into(),
            encoder,
        }
    }
}

impl<E, Data> crate::core::dataset::DenseData for DenseDatasetGeneric<E, Data>
where
    E: DenseVectorEncoder,
    Data: AsRef<[E::OutputValueType]>,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vector::DenseVectorView;
    use crate::dataset::ConvertFrom;
    use crate::distances::DotProduct;
    use crate::encoders::dense_scalar::ScalarDenseQuantizer;

    #[test]
    fn dense_dataset_range_and_id_are_consistent() {
        type Encoder = ScalarDenseQuantizer<f32, f32, DotProduct>;
        let encoder = Encoder::new(2);

        let mut growable = DenseDatasetGrowable::new(encoder);
        growable.push(DenseVectorView::new(&[1.0f32, 0.0]));
        growable.push(DenseVectorView::new(&[0.0f32, 1.0]));

        let frozen: DenseDataset<Encoder> = growable.into();
        let range = frozen.range_from_id(1);
        assert_eq!(range, 2..4);
        assert_eq!(frozen.id_from_range(range), 1);
    }

    #[test]
    fn dense_dataset_from_raw_rebuilds_vectors() {
        type Encoder = ScalarDenseQuantizer<f32, f32, DotProduct>;
        let encoder = Encoder::new(2);
        let raw = vec![1.0f32, 2.0, 3.0, 4.0];

        let dataset = DenseDataset::from_raw(raw.into_boxed_slice(), 2, encoder);
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.output_dim(), 2);
        let first = dataset.get(0);
        assert_eq!(first.values(), &[1.0f32, 2.0]);
    }

    #[test]
    fn dense_dataset_values_par_iter_and_space_usage() {
        type Encoder = ScalarDenseQuantizer<f32, f32, DotProduct>;

        let encoder = Encoder::new(2);
        let mut growable = DenseDatasetGrowable::new(encoder);
        growable.push(DenseVectorView::new(&[1.0f32, 2.0]));
        growable.push(DenseVectorView::new(&[3.0f32, 4.0]));

        let dataset: DenseDataset<Encoder> = growable.into();
        assert_eq!(dataset.values(), &[1.0f32, 2.0, 3.0, 4.0]);
        assert_eq!(dataset.nnz(), 4);
        assert!(dataset.space_usage_bytes() > 0);

        let iter_values: Vec<Vec<_>> = dataset.iter().map(|v| v.values().to_vec()).collect();
        assert_eq!(iter_values, vec![vec![1.0f32, 2.0], vec![3.0f32, 4.0]]);

        let par_values: Vec<Vec<_>> = dataset.par_iter().map(|v| v.values().to_vec()).collect();
        assert_eq!(par_values.len(), dataset.len());
        assert_eq!(par_values, iter_values);

        dataset.prefetch_with_range(0..2);
    }

    #[test]
    #[should_panic(expected = "Data length must equal n_vecs * encoder.output_dim()")]
    fn dense_dataset_from_raw_length_mismatch_panics() {
        type Encoder = ScalarDenseQuantizer<f32, f32, DotProduct>;
        let encoder = Encoder::new(2);
        let _ = DenseDataset::from_raw(vec![1.0f32, 2.0].into_boxed_slice(), 2, encoder);
    }

    #[test]
    #[should_panic(expected = "Range does not match vector boundaries.")]
    fn id_from_range_bad_range_panics() {
        type Encoder = ScalarDenseQuantizer<f32, f32, DotProduct>;
        let encoder = Encoder::new(2);
        let dataset =
            DenseDataset::from_raw(vec![1.0f32, 2.0, 3.0, 4.0].into_boxed_slice(), 2, encoder);
        let _ = dataset.id_from_range(1..3);
    }

    #[test]
    fn id_from_range_zero_dim_returns_zero() {
        type Encoder = ScalarDenseQuantizer<f32, f32, DotProduct>;
        let encoder = Encoder::new(0);
        let dataset = DenseDataset::from_raw(vec![].into_boxed_slice(), 5, encoder);
        assert_eq!(dataset.id_from_range(0..0), 0);
    }

    #[test]
    fn get_with_range_returns_expected_view() {
        type Encoder = ScalarDenseQuantizer<f32, f32, DotProduct>;
        let encoder = Encoder::new(2);
        let dataset = DenseDataset::from_raw(
            vec![10.0f32, 20.0, 30.0, 40.0].into_boxed_slice(),
            2,
            encoder,
        );
        let view = dataset.get_with_range(2..4);
        assert_eq!(view.values(), &[30.0, 40.0]);
    }

    #[test]
    fn growable_capacity_and_reserve_affect_data() {
        type Encoder = ScalarDenseQuantizer<f32, f32, DotProduct>;
        let encoder = Encoder::new(2);
        let mut growable =
            DenseDatasetGrowable::<ScalarDenseQuantizer<f32, f32, DotProduct>>::with_dim(2);
        assert_eq!(growable.capacity(), 0);
        growable.reserve(3);
        assert!(growable.capacity() >= 3);
        growable.push(DenseVectorView::new(&[0.0f32, 1.0]));
        assert!(growable.capacity() >= 1);
        growable.push(DenseVectorView::new(&[2.0f32, 3.0]));
        assert!(growable.len() >= 2);
        assert_eq!(growable.encoder.output_dim(), 2);
    }

    #[test]
    fn with_dim_constructors_return_dataset() {
        let growable =
            DenseDatasetGrowable::<ScalarDenseQuantizer<f32, f32, DotProduct>>::with_dim(3);
        assert_eq!(growable.encoder.input_dim(), 3);
        let with_capacity = DenseDatasetGrowable::<ScalarDenseQuantizer<f32, f32, DotProduct>>::with_dim_and_capacity(
            3, 4,
        );
        assert_eq!(with_capacity.encoder.output_dim(), 3);
        assert!(with_capacity.capacity() >= 0);
    }

    #[test]
    fn with_dim_and_capacity_supported_for_generic_quantizers() {
        let growable =
            DenseDatasetGrowable::<ScalarDenseQuantizer<f32, f64, DotProduct>>::with_dim(4);
        assert_eq!(growable.encoder.output_dim(), 4);
        let with_capacity =
            DenseDatasetGrowable::<ScalarDenseQuantizer<f32, f64, DotProduct>>::with_dim_and_capacity(
                4, 8,
            );
        assert_eq!(with_capacity.encoder.output_dim(), 4);
        assert_eq!(with_capacity.capacity(), 8);
    }

    #[test]
    fn convert_from_dense_dataset_preserves_values() {
        type SrcEncoder = ScalarDenseQuantizer<f32, f32, DotProduct>;
        type MidEncoder = ScalarDenseQuantizer<f32, f32, DotProduct>;
        let encoder = SrcEncoder::new(2);
        let dataset =
            DenseDataset::from_raw(vec![5.0f32, 6.0, 7.0, 8.0].into_boxed_slice(), 2, encoder);
        let converted: DenseDatasetGeneric<MidEncoder, Vec<f32>> =
            ConvertFrom::convert_from(&dataset);
        assert_eq!(converted.len(), dataset.len());
        assert_eq!(converted.values(), dataset.values());
    }
}
