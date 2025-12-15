use serde::{Deserialize, Serialize};

use crate::SpaceUsage;
use crate::datasets::{Dataset, GrowableDataset};
use crate::quantizers::Quantizer;
use crate::utils::prefetch_read_slice;
use crate::{DenseVector1D, Vector1D};
use crate::{VectorId, VectorKey};

use rayon::prelude::*;

#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct DenseDataset<Q, Data>
where
    Q: Quantizer,
    Data: AsRef<[Q::OutputValueType]>,
{
    data: Data,
    n_vecs: usize,
    d: usize,
    quantizer: Q,
}

impl<Q, Data> SpaceUsage for DenseDataset<Q, Data>
where
    Q: Quantizer<OutputComponentType = crate::DenseComponent> + SpaceUsage,
    Data: AsRef<[Q::OutputValueType]> + SpaceUsage,
{
    fn space_usage_byte(&self) -> usize {
        std::mem::size_of::<Self>()
            // + std::mem::size_of::<Q::OutputValueType>() * self.data.as_ref().len()
            + self.quantizer.space_usage_byte()
            + self.data.space_usage_byte()
    }
}

impl<Q, Data> DenseDataset<Q, Data>
where
    Q: Quantizer<OutputComponentType = crate::DenseComponent>,
    Data: AsRef<[Q::OutputValueType]>,
{
    /// Creates a DenseDataset from raw data.
    ///
    /// # Arguments
    /// * `data` - The raw vector data (flattened)
    /// * `n_vecs` - Number of vectors
    /// * `d` - Dimensionality of each vector
    /// * `quantizer` - The quantizer to use
    #[inline]
    pub fn from_raw(data: Data, n_vecs: usize, d: usize, quantizer: Q) -> Self {
        debug_assert_eq!(
            data.as_ref().len(),
            n_vecs * d,
            "Data length must equal n_vecs * d"
        );
        Self {
            data,
            n_vecs,
            d,
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
impl<Q, Data> Dataset<Q> for DenseDataset<Q, Data>
where
    Q: Quantizer<OutputComponentType = crate::DenseComponent>,
    Data: AsRef<[Q::OutputValueType]>,
{
    #[inline]
    fn quantizer(&self) -> &Q {
        &self.quantizer
    }

    #[inline]
    fn shape(&self) -> (usize, usize) {
        (self.n_vecs, self.d)
    }

    #[inline]
    fn dim(&self) -> usize {
        self.d
    }

    #[inline]
    fn len(&self) -> usize {
        self.n_vecs
    }

    #[inline]
    fn nnz(&self) -> usize {
        self.n_vecs * self.d
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
            &self.data.as_ref()[(key as usize) * self.d..(key as usize + 1) * self.d],
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

// impl<'a, Q, B> DenseDataset<Q, B>
// where
//     Q: Quantizer,
//     B: AsRef<[Q::OutputItem]>,
// {
//     #[inline]
//     pub fn values(&self) -> &[Q::OutputItem] {
//         self.data.as_ref()
//     }
// }

// impl<'a, Q> DenseDataset<Q, Vec<Q::OutputItem>>
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
impl<Q> GrowableDataset<Q> for DenseDataset<Q, Vec<Q::OutputValueType>>
where
    Q: Quantizer<OutputComponentType = crate::DenseComponent>,
    Q::OutputValueType: Default,
{
    #[inline]
    fn new(quantizer: Q, d: usize) -> Self {
        Self {
            data: Vec::new(),
            n_vecs: 0,
            d,
            quantizer,
        }
    }

    #[inline]
    fn push(
        &mut self,
        vec: impl Vector1D<ComponentType = Q::InputComponentType, ValueType = Q::InputValueType>,
    ) {
        assert_eq!(
            vec.len() % self.d,
            0,
            "Input vectors' length is not divisible by the dimensionality."
        );

        let old_size = self.data.len();
        let new_size = self.quantizer.m() + old_size;

        self.data.resize(new_size, Default::default());

        let mut output_vec = DenseVector1D::new(&mut self.data[old_size..new_size]);
        self.quantizer.encode(vec, &mut output_vec);

        self.n_vecs += 1;
    }
}

// impl<Q> Extend<Q::OutputItem> for DenseDataset<Q, Vec<Q::OutputItem>>
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

impl<Q> From<DenseDataset<Q, Vec<Q::OutputValueType>>>
    for DenseDataset<Q, Box<[Q::OutputValueType]>>
where
    Q: Quantizer<OutputComponentType = crate::DenseComponent>,
{
    fn from(mutable_dataset: DenseDataset<Q, Vec<Q::OutputValueType>>) -> Self {
        DenseDataset {
            data: mutable_dataset.data.into_boxed_slice(),
            n_vecs: mutable_dataset.n_vecs,
            d: mutable_dataset.d,
            quantizer: mutable_dataset.quantizer,
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

impl<Q, T> AsRef<[Q::OutputValueType]> for DenseDataset<Q, T>
where
    Q: Quantizer<OutputComponentType = crate::DenseComponent>,
    T: AsRef<[Q::OutputValueType]>,
{
    fn as_ref(&self) -> &[Q::OutputValueType] {
        self.data.as_ref()
    }
}

/// densedataset iterator
pub struct DenseDatasetIter<'a, Q>
where
    Q: Quantizer<OutputComponentType = crate::DenseComponent>,
{
    data: &'a [Q::OutputValueType],
    d: usize,
    index: usize,
}

impl<'a, Q> DenseDatasetIter<'a, Q>
where
    Q: Quantizer<OutputComponentType = crate::DenseComponent>,
{
    pub fn new<Data>(dataset: &'a DenseDataset<Q, Data>) -> Self
    where
        Data: AsRef<[Q::OutputValueType]>,
    {
        Self {
            data: dataset.values(),
            d: dataset.dim(),
            index: 0,
        }
    }
}

impl<'a, Q> Iterator for DenseDatasetIter<'a, Q>
where
    Q: Quantizer<OutputComponentType = crate::DenseComponent>,
    Q: Quantizer,
{
    type Item = DenseVector1D<Q::OutputValueType, &'a [Q::OutputValueType]>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.data.len() {
            return None;
        }

        let start = self.index;
        let end = std::cmp::min(start + self.d, self.data.len());
        self.index = end;

        Some(DenseVector1D::new(&self.data[start..end]))
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
//         let mut sample = Self::with_capacity_plain(n_vecs, self.d);

//         for id in sampled_id {
//             sample.push(&DenseVector1D::new(
//                 &self.data[id * self.d..(id + 1) * self.d],
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
