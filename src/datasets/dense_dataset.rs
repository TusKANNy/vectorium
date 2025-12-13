use serde::{Deserialize, Serialize};

use crate::DenseComponent;
use crate::DenseVector1D;
use crate::SpaceUsage;
use crate::Vector1D;
use crate::datasets::Dataset;
use crate::datasets::GrowableDataset;
use crate::quantizers::Quantizer;

#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct DenseDataset<Q, D>
where
    Q: Quantizer,
    D: AsRef<[Q::OutputValueType]>,
{
    data: D,
    n_vecs: usize,
    d: usize,
    quantizer: Q,
}

impl<Q, D> SpaceUsage for DenseDataset<Q, D>
where
    Q: Quantizer<OutputComponentType = DenseComponent>,
    D: AsRef<[Q::OutputValueType]>,
{
    fn space_usage_byte(&self) -> usize {
        std::mem::size_of::<Self>()
            + std::mem::size_of::<Q::OutputValueType>() * self.data.as_ref().len()
    }
}

impl<Q, D> DenseDataset<Q, D>
where
    Q: Quantizer<OutputComponentType = DenseComponent>,
    D: AsRef<[Q::OutputValueType]>,
{
    #[inline]
    pub fn values(&self) -> &[Q::OutputValueType] {
        self.data.as_ref()
    }
}

/// immutable
impl<Q, D> Dataset<Q> for DenseDataset<Q, D>
where
    Q: Quantizer<OutputComponentType = DenseComponent>,
    D: AsRef<[Q::OutputValueType]>,
{
    #[inline]
    fn quantizer(&self) -> &Q {
        &self.quantizer
    }

    #[inline]
    fn shape(&self) -> (usize, usize) {
        (self.n_vecs, self.d)
    }

    // #[inline]
    // fn data<'a>(&'a self) -> Self::DataType<'a> {
    //     DenseVector1D::new(self.data.as_ref())
    // }

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

    // fn get_space_usage_bytes(&self) -> usize {
    //     self.len() * self.dim() * std::mem::size_of::<Q::OutputItem>()
    //         + self.quantizer.get_space_usage_bytes()
    // }

    #[inline]
    fn get(
        &self,
        index: usize,
    ) -> impl Vector1D<ComponentType = Q::OutputComponentType, ValueType = Q::OutputValueType> {
        assert!(index < self.len(), "Index out of bounds.");

        let m = self.quantizer.m();

        let start = index * m;
        let end = start + m;

        DenseVector1D::new(&self.data.as_ref()[start..end])
    }

    // fn compute_distance_by_id(&self, idx1: usize, idx2: usize) -> f32
    // where
    //     Q::OutputItem: Float,
    // {
    //     let document1_slice = self.get(idx1);
    //     let document1_slice = document1_slice.values_as_slice();
    //     let document2_slice = self.get(idx2);
    //     let document2_slice = document2_slice.values_as_slice();
    //     match self.quantizer().distance() {
    //         DistanceType::Euclidean => {
    //             dense_euclidean_distance_unrolled(document1_slice, document2_slice)
    //         }
    //         DistanceType::DotProduct => {
    //             -dense_dot_product_unrolled(document1_slice, document2_slice)
    //         }
    //     }
    // }

    #[inline]
    fn iter(
        &self,
    ) -> impl Iterator<
        Item = impl Vector1D<ComponentType = Q::OutputComponentType, ValueType = Q::OutputValueType>,
    > {
        DenseDatasetIter::new(self)
    }

    // #[inline]
    // fn search<'a, H: OnlineTopKSelector>(
    //     &self,
    //     query: <Q::Evaluator<'a> as QueryEvaluator<'a>>::QueryType,
    //     heap: &mut H,
    // ) -> Vec<(f32, usize)>
    // where
    //     Q::InputItem: Float + EuclideanDistance<Q::InputItem> + DotProduct<Q::InputItem>,
    // {
    //     assert_eq!(
    //         query.len(),
    //         self.dim(),
    //         "Query dimension ({}) does not match the vector dimension ({}).",
    //         query.len(),
    //         self.dim()
    //     );

    //     if self.data().values_as_slice().is_empty() {
    //         return Vec::new();
    //     }

    //     let evaluator = self.query_evaluator(query);
    //     let distances = evaluator.compute_distances(self, 0..self.len());
    //     evaluator.topk_retrieval(distances, heap)
    // }
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

// mutable
impl<Q> GrowableDataset<Q> for DenseDataset<Q, Vec<Q::OutputValueType>>
where
    Q: Quantizer<OutputComponentType = DenseComponent, InputComponentType = DenseComponent>,
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
    Q: Quantizer,
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

// impl<Q, T> AsRef<[Q::OutputItem]> for DenseDataset<Q, T>
// where
//     Q: Quantizer,
//     T: AsRef<[Q::OutputItem]>,
// {
//     fn as_ref(&self) -> &[Q::OutputItem] {
//         self.data.as_ref()
//     }
// }

// impl<Q, T> AsMut<[Q::OutputItem]> for DenseDataset<Q, T>
// where
//     Q: Quantizer,
//     T: AsMut<[Q::OutputItem]>,
// {
//     fn as_mut(&mut self) -> &mut [Q::OutputItem] {
//         self.data.as_mut()
//     }
// }

/// densedataset iterator
pub struct DenseDatasetIter<'a, Q>
where
    Q: Quantizer<OutputComponentType = DenseComponent>,
{
    data: &'a [Q::OutputValueType],
    d: usize,
    index: usize,
}

impl<'a, Q> DenseDatasetIter<'a, Q>
where
    Q: Quantizer<OutputComponentType = DenseComponent>,
{
    pub fn new<D>(dataset: &'a DenseDataset<Q, D>) -> Self
    where
        Q: Quantizer<OutputComponentType = DenseComponent>,
        D: AsRef<[Q::OutputValueType]>,
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
    Q: Quantizer<OutputComponentType = DenseComponent>,
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
