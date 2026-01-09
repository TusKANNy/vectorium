use serde::{Deserialize, Serialize};

use crate::SpaceUsage;
use crate::core::sealed;
use crate::core::vector_encoder::{DenseVectorEncoder, VectorEncoder};
use crate::{Dataset, GrowableDataset};
// use crate::{DenseVector1DView};

use rayon::prelude::*;

// Implementation of a growable dense dataset.
pub type DenseDatasetGrowable<E> =
    DenseDatasetGeneric<E, Vec<<E as DenseVectorEncoder>::OutputValueType>>;

// Implementation of a (immutable) sparse dataset.
pub type DenseDataset<E> =
    DenseDatasetGeneric<E, Box<[<E as DenseVectorEncoder>::OutputValueType]>>;

#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
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
    E: DenseVectorEncoder + SpaceUsage,
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
    Data: AsRef<[E::OutputValueType]> + Sync,
    E::OutputValueType: Sync + Send, // For parallel iteration
    for<'a> E::EncodedVector<'a>: Send,
{
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

    #[inline]
    pub fn values(&self) -> &[E::OutputValueType] {
        self.data.as_ref()
    }

    #[inline]
    pub fn nnz(&self) -> usize {
        self.data.as_ref().len()
    }

    // Note: par_iter returns EncodedVector (view)
    #[inline]
    pub fn par_iter(&self) -> impl ParallelIterator<Item = E::EncodedVector<'_>> {
        let m = self.encoder.output_dim();
        let data = self.data.as_ref();
        let n = self.n_vecs;

        (0..n).into_par_iter().map(move |i| {
            let start = i * m;
            let end = start + m;
            DenseVector1DView::new(&data[start..end])
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
    fn get(&self, index: usize) -> E::EncodedVector<'_> {
        let m = self.encoder.output_dim();
        let start = index * m;
        let end = start + m;
        DenseVector1DView::new(&self.data.as_ref()[start..end])
    }

    #[inline]
    fn get_by_range(&self, range: std::ops::Range<usize>) -> E::EncodedVector<'_> {
        DenseVector1DView::new(&self.data.as_ref()[range])
    }

    #[inline]
    fn iter(&self) -> impl Iterator<Item = E::EncodedVector<'_>> {
        let m = self.encoder.output_dim();
        let data = self.data.as_ref();
        let n = self.n_vecs;

        (0..n).map(move |i| {
            let start = i * m;
            let end = start + m;
            DenseVector1DView::new(&data[start..end])
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
// It is now defined by `encode_vector` method signature which takes `DenseVector1DView`.
// So we should adapt `GrowableDataset` trait or implementation.
// `DenseDatasetGeneric` assumes inputs are compatible with `E`.

use crate::core::vector1d::DenseVector1DView;
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

    fn push<'a>(&mut self, vec: E::InputVector<'a>) {
        self.encoder.push_encoded(vec, &mut self.data);
        self.n_vecs += 1;
    }
}

impl<E> DenseDatasetGrowable<E>
where
    E: DenseVectorEncoder,
{
    pub fn capacity(&self) -> usize {
        if self.encoder.output_dim() == 0 {
            0
        } else {
            self.data.capacity() / self.encoder.output_dim()
        }
    }

    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional * self.encoder.output_dim());
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
            let vec_view = DenseVector1DView::new(chunk);
            let encoded = encoder.encode_vector(vec_view);
            new_data.extend_from_slice(encoded.values());
        }

        Self {
            n_vecs: source.n_vecs,
            data: new_data.into_boxed_slice().into(),
            encoder,
        }
    }
}
