use serde::{Deserialize, Serialize};

use crate::SpaceUsage;
use crate::datasets::{Dataset, GrowableDataset};
use crate::quantizers::{DenseQuantizer, Quantizer};
use crate::utils::prefetch_read_slice;
use crate::{DenseVector1D, Vector1D};
use crate::{VectorId, VectorKey};

use rayon::prelude::*;

// Implementation of a growable dense dataset.
pub type DenseDatasetGrowable<Q> = DenseDatasetGeneric<Q, Vec<<Q as Quantizer>::OutputValueType>>;

// Implementation of a (immutable) sparse dataset.
pub type DenseDataset<Q> = DenseDatasetGeneric<Q, Box<[<Q as Quantizer>::OutputValueType]>>;

#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct DenseDatasetGeneric<Q, Data>
where
    Q: DenseQuantizer,
    Data: AsRef<[Q::OutputValueType]>,
{
    dim: usize,
    n_vecs: usize,
    data: Data,
    quantizer: Q,
}

impl<Q, Data> SpaceUsage for DenseDatasetGeneric<Q, Data>
where
    Q: DenseQuantizer + SpaceUsage,
    Data: AsRef<[Q::OutputValueType]> + SpaceUsage,
{
    fn space_usage_byte(&self) -> usize {
        // Use size_of for the quantizer to avoid requiring every Quantizer to
        // implement `SpaceUsage`.
        std::mem::size_of::<Self>()
            + self.quantizer.space_usage_byte()
            + self.data.space_usage_byte()
    }
}

impl<Q, Data> DenseDatasetGeneric<Q, Data>
where
    Q: DenseQuantizer,
    Data: AsRef<[Q::OutputValueType]>,
{
    /// Creates a DenseDatasetGeneric from raw data.
    ///
    /// # Arguments
    /// * `data` - The raw vector data (flattened)
    /// * `n_vecs` - Number of vectors
    /// * `d` - Dimensionality of each vector
    /// * `quantizer` - The quantizer to use
    #[inline]
    pub fn from_raw(data: Data, n_vecs: usize, dim: usize, quantizer: Q) -> Self {
        debug_assert_eq!(
            data.as_ref().len(),
            n_vecs * dim,
            "Data length must equal n_vecs * dim"
        );
        Self {
            dim,
            n_vecs,
            data,
            quantizer,
        }
    }

    #[inline]
    pub fn values(&self) -> &[Q::OutputValueType] {
        self.data.as_ref()
    }

    /// Parallel iterator over dataset vectors (each item is a `DenseVector1D` borrowing a slice).
    #[inline]
    pub fn par_iter(
        &self,
    ) -> impl ParallelIterator<Item = DenseVector1D<Q::OutputValueType, &'_ [Q::OutputValueType]>>
    {
        let m = self.quantizer.m();
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
impl<Q, Data> Dataset<Q> for DenseDatasetGeneric<Q, Data>
where
    Q: DenseQuantizer + SpaceUsage,
    Data: AsRef<[Q::OutputValueType]> + SpaceUsage,
    Q::OutputValueType: SpaceUsage,
{
    #[inline]
    fn quantizer(&self) -> &Q {
        &self.quantizer
    }

    #[inline]
    fn shape(&self) -> (usize, usize) {
        (self.n_vecs, self.dim)
    }

    #[inline]
    fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    fn len(&self) -> usize {
        self.n_vecs
    }

    #[inline]
    fn nnz(&self) -> usize {
        self.n_vecs * self.dim
    }

    #[inline]
    fn get(
        &self,
        key: VectorKey,
    ) -> impl Vector1D<ComponentType = Q::OutputComponentType, ValueType = Q::OutputValueType> {
        let index = key as usize; // indexes and keys coincide in this implementation

        assert!(index < self.len(), "Index out of bounds.");

        let m = self.quantizer.m();

        let start = index * m;
        let end = start + m;

        DenseVector1D::new(&self.data.as_ref()[start..end])
    }

    #[inline]
    fn key_from_id(&self, id: VectorId) -> VectorKey {
        id
    }

    #[inline]
    fn id_from_key(&self, key: VectorKey) -> VectorId {
        key
    }

    #[inline]
    fn prefetch(&self, key: VectorKey) {
        prefetch_read_slice(
            &self.data.as_ref()[(key as usize) * self.dim..(key as usize + 1) * self.dim],
        );
    }

    #[inline]
    fn iter(
        &self,
    ) -> impl Iterator<
        Item = impl Vector1D<ComponentType = Q::OutputComponentType, ValueType = Q::OutputValueType>,
    > {
        DenseDatasetIter::new(self)
    }
}

// impl<'a, Q, B> DenseDatasetGeneric<Q, B>
// where
//     Q: Quantizer,
//     B: AsRef<[Q::OutputItem]>,
// {
//     #[inline]
//     pub fn values(&self) -> &[Q::OutputItem] {
//         self.data.as_ref()
//     }
// }

// impl<'a, Q> DenseDatasetGeneric<Q, Vec<Q::OutputItem>>
// where
//     Q: Quantizer,
// {
//     #[inline]
//     pub fn with_capacity(quantizer: Q, d: usize, capacity: usize) -> Self {
//         Self {
//             data: Vec::with_capacity(capacity * d),
//             n_vecs: 0,
//             d,
//             quantizer,
//         }
//     }

//     #[inline]
//     pub fn with_dim(quantizer: Q, d: usize) -> Self {
//         Self {
//             data: Vec::new(),
//             n_vecs: 0,
//             d,
//             quantizer,
//         }
//     }

//     #[inline]
//     pub fn from_vec(data: Vec<Q::OutputItem>, d: usize, quantizer: Q) -> Self {
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
impl<Q> GrowableDataset<Q> for DenseDatasetGeneric<Q, Vec<Q::OutputValueType>>
where
    Q: DenseQuantizer + SpaceUsage,
    Q::OutputValueType: Default + SpaceUsage,
{
    #[inline]
    fn new(quantizer: Q, dim: usize) -> Self {
        Self {
            data: Vec::new(),
            n_vecs: 0,
            dim,
            quantizer,
        }
    }

    #[inline]
    fn push(
        &mut self,
        vec: impl Vector1D<ComponentType = Q::InputComponentType, ValueType = Q::InputValueType>,
    ) {
        assert!(
            vec.len() == self.dim,
            "Input vector' length doesn't match datasets's dimensionality."
        );

        self.quantizer.extend_with_encode(vec, &mut self.data);

        self.n_vecs += 1;
    }
}

// impl<Q> Extend<Q::OutputItem> for DenseDatasetGeneric<Q, Vec<Q::OutputItem>>
// where
//     Q: Quantizer,
// {
//     fn extend<I: IntoIterator<Item = Q::OutputItem>>(&mut self, iter: I) {
//         for item in iter {
//             self.data.push(item);
//         }
//         self.n_vecs = self.data.len() / self.d;
//     }
// }

impl<Q> From<DenseDatasetGrowable<Q>> for DenseDataset<Q>
where
    Q: DenseQuantizer,
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
    /// ```
    /// use vectorium::{DenseDatasetGrowable, DenseDataset};
    ///
    /// let mut growable_dataset = DenseDatasetGrowable::<u16, f32>::new();
    /// // Populate mutable dataset...
    /// growable_dataset.push(&[0, 2, 4],    &[1.0, 2.0, 3.0]);
    /// growable_dataset.push(&[1, 3],       &[4.0, 5.0]);
    /// growable_dataset.push(&[0, 1, 2, 3], &[1.0, 2.0, 3.0, 4.0]);
    ///
    /// let immutable_dataset: DenseDataset<u16, f32> = growable_dataset.into();
    ///
    /// assert_eq!(immutable_dataset.nnz(), 9); // Total non-zero components across all vectors
    /// ```
    fn from(dataset: DenseDatasetGrowable<Q>) -> Self {
        Self {
            dim: dataset.dim,
            n_vecs: dataset.n_vecs,
            data: dataset.data.into_boxed_slice(),
            quantizer: dataset.quantizer,
        }
    }
}

impl<Q> From<DenseDataset<Q>> for DenseDatasetGrowable<Q>
where
    Q: DenseQuantizer,
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
    /// ```
    /// use vectorium::{DenseDatasetGrowable, DenseDataset};
    ///
    /// let mut growable_dataset = DenseDatasetGrowable::<u16, f32>::new();
    /// // Populate mutable dataset...
    /// growable_dataset.push(&[0, 2, 4],    &[1.0, 2.0, 3.0]);
    /// growable_dataset.push(&[1, 3],       &[4.0, 5.0]);
    /// growable_dataset.push(&[0, 1, 2, 3], &[1.0, 2.0, 3.0, 4.0]);
    ///
    /// let immutable_dataset: DenseDataset<u16, f32> = growable_dataset.into();
    ///
    /// // Convert immutable dataset back to mutable
    /// let mut growable_dataset_again: DenseDatasetGrowable<u16, f32> = immutable_dataset.into();
    ///
    /// growable_dataset_again.push(&[1, 7], &[1.0, 3.0]);
    ///
    /// assert_eq!(growable_dataset_again.nnz(), 11); // Total non-zero components across all vectors
    /// ```
    fn from(dataset: DenseDataset<Q>) -> Self {
        Self {
            dim: dataset.dim,
            data: dataset.data.to_vec(),
            n_vecs: dataset.n_vecs,
            quantizer: dataset.quantizer,
        }
    }
}

// impl<'a, Q, B> IntoIterator for &'a DenseDataset<Q, B>
// where
//     Q: Quantizer<DatasetType = DenseDataset<Q, B>>,
//     B: AsRef<[Q::OutputItem]> + Default,
// {
//     type Item = DenseVector1D<&'a [Q::OutputItem]>;
//     type IntoIter = DenseDatasetIter<'a, Q>;

//     fn into_iter(self) -> Self::IntoIter {
//         DenseDatasetIter::new(self, 1)
//     }
// }

impl<Q, T> AsRef<[Q::OutputValueType]> for DenseDatasetGeneric<Q, T>
where
    Q: DenseQuantizer,
    T: AsRef<[Q::OutputValueType]>,
{
    fn as_ref(&self) -> &[Q::OutputValueType] {
        self.data.as_ref()
    }
}

/// densedataset iterator
pub struct DenseDatasetIter<'a, Q>
where
    Q: DenseQuantizer,
{
    data: &'a [Q::OutputValueType],
    dim: usize,
    index: usize,
}

impl<'a, Q> DenseDatasetIter<'a, Q>
where
    Q: DenseQuantizer,
{
    pub fn new<Data>(dataset: &'a DenseDatasetGeneric<Q, Data>) -> Self
    where
        Data: AsRef<[Q::OutputValueType]> + SpaceUsage,
        Q: SpaceUsage,
    {
        Self {
            data: dataset.values(),
            dim: dataset.dim(),
            index: 0,
        }
    }
}

impl<'a, Q> Iterator for DenseDatasetIter<'a, Q>
where
    Q: DenseQuantizer,
    Q: Quantizer,
{
    type Item = DenseVector1D<Q::OutputValueType, &'a [Q::OutputValueType]>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.data.len() {
            return None;
        }

        let start = self.index;
        let end = std::cmp::min(start + self.dim, self.data.len());
        self.index = end;

        Some(DenseVector1D::new(&self.data[start..end]))
    }
}

use crate::quantizers::dense_scalar::{ScalarDenseQuantizer, ScalarDenseSupportedDistance};
use crate::{Float, ValueType};

/// Conversion methods for DenseDataset
///
/// Convert a DenseDataset ith ScalarDenseQuantizer<SrcIn, SrcOut, D> to
/// DenseDataset<ScalarDenseQuantizer<SrcOut, Out, D>, Vec<Out>>.
///
/// The source dataset's output type (SrcOut) must match the target quantizer's
/// input type (In).
impl<In, Out, D, AVOut> DenseDatasetGeneric<ScalarDenseQuantizer<In, Out, D>, AVOut>
where
    In: ValueType + Float,
    Out: ValueType + Float + crate::FromF32,
    AVOut: AsRef<[Out]> + crate::SpaceUsage + From<Vec<Out>>,
    D: ScalarDenseSupportedDistance,
{
    pub fn quantize<SrcIn, SrcData>(
        source: &DenseDatasetGeneric<ScalarDenseQuantizer<SrcIn, In, D>, SrcData>,
    ) -> Self
    where
        SrcIn: ValueType + Float,
        SrcData: AsRef<[In]> + crate::SpaceUsage,
    {
        let (n_vecs, dim) = source.shape();
        let quantizer: ScalarDenseQuantizer<In, Out, D> = ScalarDenseQuantizer::new(dim);
        let dst_dim = quantizer.m();

        // Preallocate output buffer
        let mut output_data: Vec<Out> = Vec::with_capacity(n_vecs * dst_dim);

        // Iterate vector by vector and encode
        for src_vec in source.iter() {
            quantizer.extend_with_encode(src_vec, &mut output_data);
        }

        DenseDatasetGeneric::<ScalarDenseQuantizer<In, Out, D>, AVOut> {
            dim: dst_dim,
            n_vecs,
            data: output_data.into(),
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
//         T: Float + EuclideanDistance<T> + DotProduct<T>,
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
