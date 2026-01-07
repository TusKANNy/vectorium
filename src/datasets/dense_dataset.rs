use serde::{Deserialize, Serialize};

use crate::SpaceUsage;
use crate::VectorId;
use crate::utils::prefetch_read_slice;
use crate::core::sealed;
use crate::{Dataset, GrowableDataset};
use crate::core::dataset::ConvertFrom;
use crate::numeric_markers::DenseComponent;
use crate::{DenseVectorEncoder, VectorEncoder};
use crate::{DenseVector1D, Vector1D};

use rayon::prelude::*;

// Implementation of a growable dense dataset.
pub type DenseDatasetGrowable<E> =
    DenseDatasetGeneric<E, Vec<<E as VectorEncoder>::OutputValueType>>;

// Implementation of a (immutable) sparse dataset.
pub type DenseDataset<E> = DenseDatasetGeneric<E, Box<[<E as VectorEncoder>::OutputValueType]>>;

/// Dense dataset storing fixed-length vectors in a flat array.
///
/// # Example
/// ```
/// use vectorium::{
///     Dataset, DenseDataset, DenseDatasetGrowable, DenseVector1D, DotProduct, GrowableDataset,
///     PlainDenseQuantizer, Vector1D, VectorEncoder,
/// };
///
/// let quantizer = PlainDenseQuantizer::<f32, DotProduct>::new(3);
/// let mut dataset = DenseDatasetGrowable::new(quantizer);
/// let v0 = vec![1.0, 0.0, 2.0];
/// dataset.push(DenseVector1D::new(v0.as_slice()));
///
/// let frozen: DenseDataset<_> = dataset.into();
/// let range = frozen.range_from_id(0);
/// let v = frozen.get_by_range(range);
/// assert_eq!(v.values_as_slice(), &[1.0, 0.0, 2.0]);
/// ```
#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct DenseDatasetGeneric<E, Data>
where
    E: DenseVectorEncoder,
    Data: AsRef<[E::OutputValueType]>,
{
    n_vecs: usize,
    data: Data,
    quantizer: E,
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
    E: SpaceUsage,
    Data: AsRef<[E::OutputValueType]> + SpaceUsage,
{
    fn space_usage_bytes(&self) -> usize {
        // Use size_of for the encoder to avoid requiring every VectorEncoder to
        // implement `SpaceUsage`.
        self.n_vecs.space_usage_bytes()
            + self.quantizer.space_usage_bytes()
            + self.data.space_usage_bytes()
    }
}

impl<E, Data> DenseDatasetGeneric<E, Data>
where
    E: DenseVectorEncoder,
    Data: AsRef<[E::OutputValueType]>,
{
    /// Creates a DenseDatasetGeneric from raw data.
    ///
    /// # Arguments
    /// * `data` - The raw vector data (flattened)
    /// * `n_vecs` - Number of vectors
    /// * `quantizer` - The encoder to use
    #[inline]
    pub fn from_raw(data: Data, n_vecs: usize, quantizer: E) -> Self {
        assert_eq!(
            data.as_ref().len(),
            n_vecs * quantizer.output_dim(),
            "Data length must equal n_vecs * encoder.output_dim()"
        );
        Self {
            n_vecs,
            data,
            quantizer,
        }
    }

    #[inline]
    pub fn values(&self) -> &[E::OutputValueType] {
        self.data.as_ref()
    }

    /// Parallel iterator over dataset vectors (each item is a `DenseVector1D` borrowing a slice).
    #[inline]
    pub fn par_iter(
        &self,
    ) -> impl ParallelIterator<Item = DenseVector1D<E::OutputValueType, &'_ [E::OutputValueType]>>
    {
        let m = self.quantizer.output_dim();
        let data = self.data.as_ref();
        let n = self.n_vecs;

        (0..n).into_par_iter().map(move |i| {
            let start = i * m;
            let end = start + m;
            DenseVector1D::new(&data[start..end])
        })
    }
}

/// immutable
impl<E, Data> Dataset for DenseDatasetGeneric<E, Data>
where
    E: DenseVectorEncoder,
    Data: AsRef<[E::OutputValueType]>,
    for<'a> E::EncodedVectorType<'a>: Vector1D<Component = DenseComponent, Value = E::OutputValueType>,
{
    type Encoder = E;
    type EncodedVectorType<'a>
        = E::EncodedVectorType<'a>
    where
        Self: 'a;

    #[inline]
    fn encoder(&self) -> &E {
        &self.quantizer
    }

    #[inline]
    fn len(&self) -> usize {
        self.n_vecs
    }

    #[inline]
    fn nnz(&self) -> usize {
        self.n_vecs * self.quantizer.output_dim()
    }

    #[inline]
    fn get_by_range<'a>(
        &'a self,
        range: std::ops::Range<usize>,
    ) -> E::EncodedVectorType<'a> {
        self.quantizer.encoded_from_slice(&self.data.as_ref()[range])
    }

    #[inline]
    fn range_from_id(&self, id: VectorId) -> std::ops::Range<usize> {
        let index = id as usize;
        assert!(index < self.len(), "Index out of bounds.");

        let m = self.quantizer.output_dim();
        let start = index * m;
        let end = start + m;
        start..end
    }

    #[inline]
    fn id_from_range(&self, range: std::ops::Range<usize>) -> VectorId {
        let m = self.quantizer.output_dim();
        assert!(
            range.start.is_multiple_of(m) && range.end == range.start + m,
            "Range does not match vector boundaries."
        );
        (range.start / m) as VectorId
    }

    #[inline]
    fn prefetch(&self, range: std::ops::Range<usize>) {
        prefetch_read_slice(&self.data.as_ref()[range]);
    }

    #[inline]
    fn iter<'a>(&'a self) -> impl Iterator<Item = E::EncodedVectorType<'a>> {
        DenseDatasetIter::new(self)
    }
}

impl<E, Data> crate::core::dataset::DenseDatasetTrait for DenseDatasetGeneric<E, Data>
where
    E: DenseVectorEncoder,
    Data: AsRef<[E::OutputValueType]>,
    for<'a> E::EncodedVectorType<'a>: Vector1D<Component = DenseComponent, Value = E::OutputValueType>,
{
}

// impl<'a, E, B> DenseDatasetGeneric<E, B>
// where
//     E: VectorEncoder,
//     B: AsRef<[E::OutputItem]>,
// {
//     #[inline]
//     pub fn values(&self) -> &[E::OutputItem] {
//         self.data.as_ref()
//     }
// }

#[cfg(test)]
mod tests {
    use crate::{
        dataset::ConvertInto, Dataset, DenseDataset, DenseDatasetGrowable, DenseVector1D,
        DotProduct, GrowableDataset, PlainDenseDataset, PlainDenseDatasetGrowable,
        PlainDenseQuantizer, Vector1D,
    };
    use half::f16;

    #[test]
    fn dense_growable_immutable_roundtrip() {
        let quantizer = PlainDenseQuantizer::<f32, DotProduct>::new(3);
        let mut growable = PlainDenseDatasetGrowable::new(quantizer);

        growable.push(DenseVector1D::new(&[1.0, 2.0, 3.0]));
        growable.push(DenseVector1D::new(&[4.0, 5.0, 6.0]));

        let frozen: PlainDenseDataset<f32, DotProduct> = growable.into();
        assert_eq!(frozen.len(), 2);
        assert_eq!(frozen.nnz(), 6);

        let first = frozen.get(0);
        assert_eq!(first.values_as_slice(), &[1.0, 2.0, 3.0]);

        let mut growable_again: PlainDenseDatasetGrowable<f32, DotProduct> = frozen.into();
        growable_again.push(DenseVector1D::new(&[7.0, 8.0, 9.0]));

        assert_eq!(growable_again.len(), 3);
        assert_eq!(growable_again.nnz(), 9);
    }

    #[test]
    fn dense_scalar_plain_roundtrip_without_reencode() {
        let quantizer = crate::ScalarDenseQuantizer::<f32, f16, DotProduct>::new(3);
        let mut growable = DenseDatasetGrowable::new(quantizer);

        growable.push(DenseVector1D::new(&[1.0_f32, 2.0, 3.0]));
        growable.push(DenseVector1D::new(&[4.0_f32, 5.0, 6.0]));

        let growable_plain: DenseDatasetGrowable<PlainDenseQuantizer<f16, DotProduct>> =
            growable.relabel_as_plain();
        assert_eq!(
            growable_plain.values(),
            &[
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
                f16::from_f32(5.0),
                f16::from_f32(6.0),
            ]
        );

        let frozen_plain: PlainDenseDataset<f16, DotProduct> = growable_plain.into();
        let frozen_scalar: DenseDataset<crate::ScalarDenseQuantizer<f32, f16, DotProduct>> =
            frozen_plain.relabel_as_scalar();
        assert_eq!(
            frozen_scalar.values(),
            &[
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
                f16::from_f32(5.0),
                f16::from_f32(6.0),
            ]
        );
    }

    #[test]
    fn dense_scalar_reencode_changes_value_type() {
        let quantizer = PlainDenseQuantizer::<f32, DotProduct>::new(3);
        let mut growable = PlainDenseDatasetGrowable::new(quantizer);

        growable.push(DenseVector1D::new(&[1.25_f32, 2.5, 3.75]));
        let frozen: PlainDenseDataset<f32, DotProduct> = growable.into();

        let reencoded: DenseDataset<crate::ScalarDenseQuantizer<f32, f16, DotProduct>> =
            (&frozen).convert_into();

        assert_eq!(
            reencoded.values(),
            &[
                f16::from_f32(1.25),
                f16::from_f32(2.5),
                f16::from_f32(3.75),
            ]
        );
    }
}

// impl<'a, E> DenseDatasetGeneric<E, Vec<E::OutputItem>>
// where
//     E: VectorEncoder,
// {
//     #[inline]
//     pub fn with_capacity(quantizer: E, d: usize, capacity: usize) -> Self {
//         Self {
//             data: Vec::with_capacity(capacity * d),
//             n_vecs: 0,
//             d,
//             quantizer,
//         }
//     }

//     #[inline]
//     pub fn with_dim(quantizer: E, d: usize) -> Self {
//         Self {
//             data: Vec::new(),
//             n_vecs: 0,
//             d,
//             quantizer,
//         }
//     }

//     #[inline]
//     pub fn from_vec(data: Vec<E::OutputItem>, d: usize, quantizer: E) -> Self {
//         let n_components = data.len();
//         Self {
//             data,
//             n_vecs: n_components / d,
//             d,
//             quantizer,
//         }
//     }

//     #[inline]
//     pub fn shrink_to_fit(&mut self) {
//         self.data.shrink_to_fit();
//     }
// }

// Growable dataset implementation
impl<E> GrowableDataset for DenseDatasetGeneric<E, Vec<E::OutputValueType>>
where
    E: DenseVectorEncoder,
    E::OutputValueType: Default,
    for<'a> E::EncodedVectorType<'a>: Vector1D<Component = DenseComponent, Value = E::OutputValueType>,
{
    #[inline]
    fn new(quantizer: E) -> Self {
        Self {
            data: Vec::new(),
            n_vecs: 0,
            quantizer,
        }
    }

    #[inline]
    fn push<'a>(&mut self, vec: <Self as Dataset>::InputVectorType<'a>) {
        assert!(
            vec.len() == self.quantizer.input_dim(),
            "Input vector length doesn't match encoder input dimensionality."
        );

        let input = DenseVector1D::new(vec.values_as_slice());
        let before_len = self.data.len();
        self.quantizer.extend_with_encode(input, &mut self.data);
        assert_eq!(
            self.data.len(),
            before_len + self.quantizer.output_dim(),
            "DenseVectorEncoder::extend_with_encode must append exactly output_dim() values"
        );

        self.n_vecs += 1;
    }
}

// impl<E> Extend<E::OutputItem> for DenseDatasetGeneric<E, Vec<E::OutputItem>>
// where
//     E: VectorEncoder,
// {
//     fn extend<I: IntoIterator<Item = E::OutputItem>>(&mut self, iter: I) {
//         for item in iter {
//             self.data.push(item);
//         }
//         self.n_vecs = self.data.len() / self.d;
//     }
// }

impl<E> ConvertFrom<DenseDatasetGrowable<E>> for DenseDataset<E>
where
    E: DenseVectorEncoder,
    for<'a> E::EncodedVectorType<'a>: Vector1D<Component = DenseComponent, Value = E::OutputValueType>,
{
    /// Converts a mutable dense dataset into an immutable one.
    ///
    /// This function consumes the provided `DenseDatasetGrowable<C, V>` and produces
    /// a corresponding immutable `DenseDataset<C, V>` instance. The conversion is performed
    /// by transferring ownership of the internal data structures from the mutable dataset
    /// to the immutable one.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectorium::{Dataset, GrowableDataset};
    /// use vectorium::{DenseDataset, PlainDenseDatasetGrowable};
    /// use vectorium::{DenseVector1D, PlainDenseQuantizer, SquaredEuclideanDistance};
    ///
    /// let quantizer = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(4);
    /// let mut growable_dataset =
    ///     PlainDenseDatasetGrowable::<f32, SquaredEuclideanDistance>::new(quantizer);
    ///
    /// // Populate mutable dataset...
    /// growable_dataset.push(DenseVector1D::new(&[1.0, 2.0, 3.0, 4.0]));
    /// growable_dataset.push(DenseVector1D::new(&[0.0, 4.0, 5.0, 6.0]));
    /// growable_dataset.push(DenseVector1D::new(&[1.0, 2.0, 3.0, 4.0]));
    ///
    /// let immutable_dataset: DenseDataset<_> = growable_dataset.into();
    /// assert_eq!(immutable_dataset.len(), 3);
    /// assert_eq!(immutable_dataset.nnz(), 12);
    /// ```
    fn convert_from(dataset: DenseDatasetGrowable<E>) -> Self {
        Self {
            n_vecs: dataset.n_vecs,
            data: dataset.data.into_boxed_slice(),
            quantizer: dataset.quantizer,
        }
    }
}

impl<E> From<DenseDatasetGrowable<E>> for DenseDataset<E>
where
    E: DenseVectorEncoder,
    for<'a> E::EncodedVectorType<'a>: Vector1D<Component = DenseComponent, Value = E::OutputValueType>,
{
    fn from(dataset: DenseDatasetGrowable<E>) -> Self {
        Self::convert_from(dataset)
    }
}

impl<E> ConvertFrom<DenseDataset<E>> for DenseDatasetGrowable<E>
where
    E: DenseVectorEncoder,
    for<'a> E::EncodedVectorType<'a>: Vector1D<Component = DenseComponent, Value = E::OutputValueType>,
{
    /// Converts an immutable sparse dataset into a mutable one.
    ///
    /// This function consumes the provided `DenseDataset<C, V>` and produces
    /// a corresponding mutable `DenseDatasetGrowable<C, V>` instance. The conversion is performed
    /// by transferring ownership of the internal data structures from the immutable dataset
    /// to the mutable one.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectorium::{Dataset, GrowableDataset};
    /// use vectorium::{
    ///     DenseVector1D, PlainDenseDataset, PlainDenseDatasetGrowable, PlainDenseQuantizer,
    ///     SquaredEuclideanDistance,
    /// };
    ///
    /// let quantizer = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(4);
    /// let mut growable_dataset =
    ///     PlainDenseDatasetGrowable::<f32, SquaredEuclideanDistance>::new(quantizer);
    ///
    /// growable_dataset.push(DenseVector1D::new(&[1.0, 2.0, 3.0, 4.0]));
    /// growable_dataset.push(DenseVector1D::new(&[0.0, 4.0, 5.0, 6.0]));
    /// growable_dataset.push(DenseVector1D::new(&[1.0, 2.0, 3.0, 4.0]));
    ///
    /// let immutable_dataset: PlainDenseDataset<f32, SquaredEuclideanDistance> =
    ///     growable_dataset.into();
    ///
    /// // Convert immutable dataset back to a growable one
    /// let mut growable_dataset_again: PlainDenseDatasetGrowable<f32, SquaredEuclideanDistance> =
    ///     immutable_dataset.into();
    /// growable_dataset_again.push(DenseVector1D::new(&[1.0, 7.0, 8.0, 9.0]));
    ///
    /// assert_eq!(growable_dataset_again.len(), 4);
    /// assert_eq!(growable_dataset_again.nnz(), 16);
    /// ```
    fn convert_from(dataset: DenseDataset<E>) -> Self {
        Self {
            data: dataset.data.to_vec(),
            n_vecs: dataset.n_vecs,
            quantizer: dataset.quantizer,
        }
    }
}

impl<E> From<DenseDataset<E>> for DenseDatasetGrowable<E>
where
    E: DenseVectorEncoder,
    for<'a> E::EncodedVectorType<'a>: Vector1D<Component = DenseComponent, Value = E::OutputValueType>,
{
    fn from(dataset: DenseDataset<E>) -> Self {
        Self::convert_from(dataset)
    }
}

impl<In, V, D, Data> DenseDatasetGeneric<crate::ScalarDenseQuantizer<In, V, D>, Data>
where
    In: crate::ValueType + crate::Float,
    V: crate::ValueType + crate::Float + crate::FromF32,
    D: crate::ScalarDenseSupportedDistance,
    Data: AsRef<[V]>,
{
    /// Relabels a scalar-quantized dataset as plain without re-encoding.
    pub fn relabel_as_plain(self) -> DenseDatasetGeneric<crate::PlainDenseQuantizer<V, D>, Data> {
        let dim = self.quantizer.output_dim();
        DenseDatasetGeneric {
            n_vecs: self.n_vecs,
            data: self.data,
            quantizer: crate::PlainDenseQuantizer::<V, D>::new(dim),
        }
    }
}

impl<V, D, Data> DenseDatasetGeneric<crate::PlainDenseQuantizer<V, D>, Data>
where
    V: crate::ValueType + crate::Float + crate::FromF32,
    D: crate::ScalarDenseSupportedDistance,
    Data: AsRef<[V]>,
{
    /// Relabels a plain-quantized dataset as scalar without re-encoding.
    pub fn relabel_as_scalar<In>(
        self,
    ) -> DenseDatasetGeneric<crate::ScalarDenseQuantizer<In, V, D>, Data>
    where
        In: crate::ValueType + crate::Float,
    {
        let dim = self.quantizer.output_dim();
        DenseDatasetGeneric {
            n_vecs: self.n_vecs,
            data: self.data,
            quantizer: crate::ScalarDenseQuantizer::<In, V, D>::new(dim),
        }
    }
}

// impl<'a, E, B> IntoIterator for &'a DenseDataset<E, B>
// where
//     E: VectorEncoder<DatasetType = DenseDataset<E, B>>,
//     B: AsRef<[E::OutputItem]> + Default,
// {
//     type Item = DenseVector1D<&'a [E::OutputItem]>;
//     type IntoIter = DenseDatasetIter<'a, E>;

//     fn into_iter(self) -> Self::IntoIter {
//         DenseDatasetIter::new(self, 1)
//     }
// }

impl<E, T> AsRef<[E::OutputValueType]> for DenseDatasetGeneric<E, T>
where
    E: DenseVectorEncoder,
    T: AsRef<[E::OutputValueType]>,
{
    fn as_ref(&self) -> &[E::OutputValueType] {
        self.data.as_ref()
    }
}

/// densedataset iterator
pub struct DenseDatasetIter<'a, E>
where
    E: DenseVectorEncoder,
{
    data: &'a [E::OutputValueType],
    encoder: &'a E,
    dim: usize,
    index: usize,
}

impl<'a, E> DenseDatasetIter<'a, E>
where
    E: DenseVectorEncoder,
{
    pub fn new<Data>(dataset: &'a DenseDatasetGeneric<E, Data>) -> Self
    where
        Data: AsRef<[E::OutputValueType]>,
    {
        Self {
            data: dataset.values(),
            encoder: &dataset.quantizer,
            dim: dataset.quantizer.output_dim(),
            index: 0,
        }
    }
}

impl<'a, E> Iterator for DenseDatasetIter<'a, E>
where
    E: DenseVectorEncoder,
    E::EncodedVectorType<'a>: Vector1D<Component = DenseComponent, Value = E::OutputValueType>,
{
    type Item = E::EncodedVectorType<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.data.len() {
            return None;
        }

        let start = self.index;
        let end = std::cmp::min(start + self.dim, self.data.len());
        self.index = end;

        Some(self.encoder.encoded_from_slice(&self.data[start..end]))
    }
}

use crate::encoders::dense_scalar::{ScalarDenseQuantizer, ScalarDenseSupportedDistance};
use crate::{Float, FromF32, ValueType};

/// Converts a dense dataset with scalar quantizer output `In` into one with output `Out`.
impl<In, Out, D, AVOut, SrcIn, SrcData>
    ConvertFrom<&DenseDatasetGeneric<ScalarDenseQuantizer<SrcIn, In, D>, SrcData>>
    for DenseDatasetGeneric<ScalarDenseQuantizer<In, Out, D>, AVOut>
where
    In: ValueType + Float + FromF32,
    Out: ValueType + Float + FromF32,
    AVOut: AsRef<[Out]> + crate::SpaceUsage + From<Vec<Out>>,
    D: ScalarDenseSupportedDistance,
    SrcIn: ValueType + Float,
    SrcData: AsRef<[In]> + crate::SpaceUsage,
    for<'a> <ScalarDenseQuantizer<SrcIn, In, D> as VectorEncoder>::EncodedVectorType<'a>:
        Vector1D<Component = DenseComponent, Value = In>,
    for<'a> <ScalarDenseQuantizer<In, Out, D> as VectorEncoder>::EncodedVectorType<'a>:
        Vector1D<Component = DenseComponent, Value = Out>,
{
    fn convert_from(
        source: &DenseDatasetGeneric<ScalarDenseQuantizer<SrcIn, In, D>, SrcData>,
    ) -> Self {
        let n_vecs = source.len();
        let src_dim = source.quantizer.output_dim();
        let quantizer: ScalarDenseQuantizer<In, Out, D> = ScalarDenseQuantizer::new(src_dim);
        let dst_dim = quantizer.output_dim();

        // Preallocate output buffer
        let mut output_data: Vec<Out> = Vec::with_capacity(n_vecs * dst_dim);

        // Iterate vector by vector and encode
        for src_vec in source.iter() {
            let input = DenseVector1D::new(src_vec.values_as_slice());
            let before_len = output_data.len();
            quantizer.extend_with_encode(input, &mut output_data);
            assert_eq!(
                output_data.len(),
                before_len + dst_dim,
                "DenseVectorEncoder::extend_with_encode must append exactly output_dim() values"
            );
        }

        DenseDatasetGeneric::<ScalarDenseQuantizer<In, Out, D>, AVOut> {
            n_vecs,
            data: output_data.into(),
            quantizer,
        }
    }
}

impl<In, Out, D, SrcIn>
    ConvertFrom<DenseDataset<ScalarDenseQuantizer<SrcIn, In, D>>>
    for DenseDataset<ScalarDenseQuantizer<In, Out, D>>
where
    In: ValueType + Float + FromF32,
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
    SrcIn: ValueType + Float,
    ScalarDenseQuantizer<SrcIn, In, D>: DenseVectorEncoder<
        InputValueType = SrcIn,
        OutputValueType = In,
    >,
    ScalarDenseQuantizer<In, Out, D>: DenseVectorEncoder<
        InputValueType = In,
        OutputValueType = Out,
    >,
    for<'a> <ScalarDenseQuantizer<SrcIn, In, D> as VectorEncoder>::EncodedVectorType<'a>:
        Vector1D<Component = DenseComponent, Value = In>,
    for<'a> <ScalarDenseQuantizer<In, Out, D> as VectorEncoder>::EncodedVectorType<'a>:
        Vector1D<Component = DenseComponent, Value = Out>,
{
    fn convert_from(source: DenseDataset<ScalarDenseQuantizer<SrcIn, In, D>>) -> Self {
        let n_vecs = source.len();
        let src_dim = source.quantizer.output_dim();
        let quantizer: ScalarDenseQuantizer<In, Out, D> = ScalarDenseQuantizer::new(src_dim);
        let dst_dim = quantizer.output_dim();

        let mut output_data: Vec<Out> = Vec::with_capacity(n_vecs * dst_dim);

        for src_vec in source.iter() {
            let input = DenseVector1D::<In, _>::new(src_vec.values_as_slice());
            let before_len = output_data.len();
            quantizer.extend_with_encode(input, &mut output_data);
            assert_eq!(
                output_data.len(),
                before_len + dst_dim,
                "DenseVectorEncoder::extend_with_encode must append exactly output_dim() values"
            );
        }

        DenseDatasetGeneric::<ScalarDenseQuantizer<In, Out, D>, Box<[Out]>> {
            n_vecs,
            data: output_data.into_boxed_slice(),
            quantizer,
        }
    }
}

impl<In, Out, D, SrcIn>
    ConvertFrom<DenseDatasetGrowable<ScalarDenseQuantizer<SrcIn, In, D>>>
    for DenseDatasetGrowable<ScalarDenseQuantizer<In, Out, D>>
where
    In: ValueType + Float + FromF32,
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
    SrcIn: ValueType + Float,
    ScalarDenseQuantizer<SrcIn, In, D>: DenseVectorEncoder<
        InputValueType = SrcIn,
        OutputValueType = In,
    >,
    ScalarDenseQuantizer<In, Out, D>: DenseVectorEncoder<
        InputValueType = In,
        OutputValueType = Out,
    >,
    for<'a> <ScalarDenseQuantizer<SrcIn, In, D> as VectorEncoder>::EncodedVectorType<'a>:
        Vector1D<Component = DenseComponent, Value = In>,
    for<'a> <ScalarDenseQuantizer<In, Out, D> as VectorEncoder>::EncodedVectorType<'a>:
        Vector1D<Component = DenseComponent, Value = Out>,
{
    fn convert_from(source: DenseDatasetGrowable<ScalarDenseQuantizer<SrcIn, In, D>>) -> Self {
        let n_vecs = source.len();
        let src_dim = source.quantizer.output_dim();
        let quantizer: ScalarDenseQuantizer<In, Out, D> = ScalarDenseQuantizer::new(src_dim);
        let dst_dim = quantizer.output_dim();

        let mut output_data: Vec<Out> = Vec::with_capacity(n_vecs * dst_dim);

        for src_vec in source.iter() {
            let input = DenseVector1D::<In, _>::new(src_vec.values_as_slice());
            let before_len = output_data.len();
            quantizer.extend_with_encode(input, &mut output_data);
            assert_eq!(
                output_data.len(),
                before_len + dst_dim,
                "DenseVectorEncoder::extend_with_encode must append exactly output_dim() values"
            );
        }

        DenseDatasetGeneric::<ScalarDenseQuantizer<In, Out, D>, Vec<Out>> {
            n_vecs,
            data: output_data,
            quantizer,
        }
    }
}

// impl<T> PlainDenseDataset<T>
// where
//     T: Float,
// {
//     #[inline]
//     pub fn with_dim_plain(d: usize) -> Self {
//         let quantizer = PlainQuantizer::new(d, DistanceType::Euclidean);
//         Self {
//             data: Vec::new(),
//             n_vecs: 0,
//             d,
//             quantizer,
//         }
//     }

//     #[inline]
//     pub fn from_vec_plain(data: Vec<T>, d: usize) -> Self {
//         let n_components = data.len();
//         let quantizer = PlainQuantizer::new(d, DistanceType::Euclidean);
//         Self {
//             data,
//             n_vecs: n_components / d,
//             d,
//             quantizer,
//         }
//     }

//     #[inline]
//     pub fn with_capacity_plain(capacity: usize, d: usize) -> Self {
//         let quantizer = PlainQuantizer::new(d, DistanceType::Euclidean);
//         Self {
//             data: Vec::with_capacity(capacity * d),
//             n_vecs: 0,
//             d,
//             quantizer,
//         }
//     }

//     #[inline]
//     pub fn from_random_sample(&self, n_vecs: usize) -> Self {
//         use rand::seq::index::sample;

//         let mut rng = rand::thread_rng();
//         let sampled_id = sample(&mut rng, self.len(), n_vecs);
//         let mut sample = Self::with_capacity_plain(n_vecs, self.dim);

//         for id in sampled_id {
//             sample.push(&DenseVector1D::new(
//                 &self.data[id * self.dim..(id + 1) * self.dim],
//             ));
//         }

//         sample
//     }

//     pub fn top1(&self, queries: &[T], batch_size: usize) -> Vec<(f32, usize)>
//     where
//         T: Float,
//     {
//         assert!(
//             queries.len() == batch_size * self.dim(),
//             "Query dimension ({}) does not match centroid dimension ({})!",
//             queries.len() / batch_size,
//             self.dim(),
//         );

//         let mut results = Vec::with_capacity(batch_size);

//         for query in queries.chunks_exact(self.dim()) {
//             let query_array = DenseVector1D::new(query);

//             let mut heap = TopkHeap::new(1);
//             let search_results = self.search(query_array, &mut heap);

//             if let Some((dist, idx)) = search_results.into_iter().next() {
//                 results.push((dist, idx));
//             } else {
//                 results.push((f32::MAX, usize::MAX));
//             }
//         }

//         results
//     }
// }
