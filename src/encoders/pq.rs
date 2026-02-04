use std::marker::PhantomData;

use crate::PlainDenseDataset;
use crate::PlainDenseQuantizer;
use crate::SpaceUsage;
use crate::clustering::KMeansBuilder;
use crate::core::vector::{DenseVectorOwned, DenseVectorView};
use crate::core::vector_encoder::{DenseVectorEncoder, QueryEvaluator, VectorEncoder};
use crate::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use crate::{Dataset, DatasetGrowable, PlainDenseDatasetGrowable};

/// Minimum dataset size below which no sampling is performed for PQ training
const PQ_TRAIN_SAMPLING_MIN_SIZE: usize = 1_000_000;
/// Fraction of dataset to sample when dataset exceeds PQ_TRAIN_SAMPLING_MIN_SIZE
const PQ_TRAIN_SAMPLE_RATE: f64 = 0.10;

pub trait ProductQuantizerDistance: Distance + From<f32> {
    fn compute_query_distance_table<const M: usize>(
        encoder: &ProductQuantizer<M, Self>,
        query: DenseVectorView<'_, f32>,
    ) -> Vec<f32>;
}

impl ProductQuantizerDistance for SquaredEuclideanDistance {
    fn compute_query_distance_table<const M: usize>(
        encoder: &ProductQuantizer<M, Self>,
        query: DenseVectorView<'_, f32>,
    ) -> Vec<f32> {
        encoder.compute_euclidean_distance_table(query)
    }
}

impl ProductQuantizerDistance for DotProduct {
    fn compute_query_distance_table<const M: usize>(
        encoder: &ProductQuantizer<M, Self>,
        query: DenseVectorView<'_, f32>,
    ) -> Vec<f32> {
        encoder.compute_dot_product_table(query)
    }
}

/// Number of centroids per subspace (always 256 = 2^8)
const KSUB: usize = 256;

#[derive(Clone, Debug, PartialEq)]
pub struct ProductQuantizer<const M: usize, D>
where
    D: ProductQuantizerDistance,
{
    d: usize,
    dsub: usize,
    centroids: [PlainDenseDataset<f32, SquaredEuclideanDistance>; M],
    _distance: PhantomData<D>,
}

impl<const M: usize, D> ProductQuantizer<M, D>
where
    D: ProductQuantizerDistance,
{
    #[inline]
    pub fn train(training_data: &PlainDenseDataset<f32, SquaredEuclideanDistance>) -> Self {
        let d = training_data.output_dim();
        assert_eq!(M % 4, 0, "M ({}) is not divisible by 4", M);
        assert_eq!(d % M, 0, "d ({}) is not divisible by M ({})", d, M);
        let dsub = d / M;
        let centroids = Self::train_centroids(training_data, dsub);
        Self {
            d,
            dsub,
            centroids,
            _distance: PhantomData,
        }
    }

    fn train_centroids(
        training_data: &PlainDenseDataset<f32, SquaredEuclideanDistance>,
        dsub: usize,
    ) -> [PlainDenseDataset<f32, SquaredEuclideanDistance>; M] {
        use rayon::prelude::*;

        println!("Running K-Means for {} subspaces", M);

        // Use par_iter().map().collect() which preserves order
        // (unlike reduce() which combines results in arbitrary order)
        let datasets: Vec<PlainDenseDataset<f32, SquaredEuclideanDistance>> = (0..M)
            .into_par_iter()
            .map(|m| {
                let quantizer = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(dsub);
                let mut sub_dataset =
                    PlainDenseDatasetGrowable::<f32, SquaredEuclideanDistance>::with_capacity(
                        quantizer,
                        training_data.len(),
                    );

                for vector in training_data.iter() {
                    let start = m * dsub;
                    let end = start + dsub;
                    let sub_vector = DenseVectorView::new(&vector.values()[start..end]);
                    sub_dataset.push(sub_vector);
                }

                let kmeans = KMeansBuilder::new().build();
                kmeans.train(&sub_dataset, KSUB, None)
            })
            .collect();

        println!("K-Means finished");

        datasets
            .try_into()
            .unwrap_or_else(|_| panic!("Expected exactly {} datasets", M))
    }

    fn sample_random_dataset(
        dataset: &PlainDenseDataset<f32, SquaredEuclideanDistance>,
        sample_size: usize,
    ) -> PlainDenseDataset<f32, SquaredEuclideanDistance> {
        use rand::seq::SliceRandom;
        let d = dataset.output_dim();
        let dataset_size = dataset.len();

        // Create random indices
        let mut indices: Vec<usize> = (0..dataset_size).collect();
        let mut rng = rand::thread_rng();
        indices.partial_shuffle(&mut rng, sample_size);

        // Extract sampled vectors
        let mut values = Vec::with_capacity(sample_size * d);
        for &idx in &indices[..sample_size] {
            let start = idx * d;
            values.extend_from_slice(&dataset.values()[start..start + d]);
        }

        PlainDenseDataset::<f32, SquaredEuclideanDistance>::from_raw(
            values.into_boxed_slice(),
            sample_size,
            dataset.encoder().clone(),
        )
    }

    /// Compute the training sample size based on the automatic sampling strategy.
    ///
    /// - If dataset size ≤ PQ_SAMPLING_MIN_SIZE: no sampling (returns None)
    /// - If dataset size > PQ_SAMPLING_MIN_SIZE: sample max(PQ_SAMPLE_RATE * size, PQ_SAMPLING_MIN_SIZE)
    fn compute_training_sample_size(dataset_len: usize) -> Option<usize> {
        if dataset_len <= PQ_TRAIN_SAMPLING_MIN_SIZE {
            None
        } else {
            let sample_by_rate = (dataset_len as f64 * PQ_TRAIN_SAMPLE_RATE) as usize;
            Some(sample_by_rate.max(PQ_TRAIN_SAMPLING_MIN_SIZE))
        }
    }

    /// Train PQ encoder and encode an entire PlainDenseDataset into a PQ dataset.
    ///
    /// Uses automatic sampling strategy for training:
    /// - If dataset size ≤ 1,000,000: train on full dataset
    /// - If dataset size > 1,000,000: sample max(10% of dataset, 1,000,000)
    ///
    /// # Arguments
    /// * `dataset` - Source dataset with f32 values and SquaredEuclideanDistance
    pub fn encode_dataset(
        dataset: &PlainDenseDataset<f32, SquaredEuclideanDistance>,
    ) -> crate::DenseDataset<Self> {
        let sample_size = Self::compute_training_sample_size(dataset.len());

        // Sample if needed
        let training_dataset = match sample_size {
            Some(size) => {
                println!(
                    "Sampling {} vectors from {} for PQ training",
                    size,
                    dataset.len()
                );
                Self::sample_random_dataset(dataset, size)
            }
            None => dataset.clone(),
        };

        // Train the encoder
        let pq_encoder = Self::train(&training_dataset);

        // Encode all vectors from the original dataset
        let encoded_vector_len = pq_encoder.output_dim();
        let mut encoded_data = Vec::with_capacity(dataset.len() * encoded_vector_len);
        for vector in dataset.iter() {
            pq_encoder.push_encoded(vector, &mut encoded_data);
        }

        let encoded_data = encoded_data.into_boxed_slice();
        crate::DenseDataset::<ProductQuantizer<M, D>>::from_raw(
            encoded_data,
            dataset.len(),
            pq_encoder,
        )
    }

    /// Train PQ encoder and encode an entire PlainDenseDataset with DotProduct distance.
    ///
    /// Converts to SquaredEuclideanDistance for training (zero-copy distance change),
    /// then encodes with target distance D.
    ///
    /// Uses automatic sampling strategy for training:
    /// - If dataset size ≤ 1,000,000: train on full dataset
    /// - If dataset size > 1,000,000: sample max(10% of dataset, 1,000,000)
    ///
    /// # Arguments
    /// * `dataset` - Source dataset with f32 values and DotProduct distance
    pub fn encode_dataset_from_dotproduct(
        dataset: &PlainDenseDataset<f32, DotProduct>,
    ) -> crate::DenseDataset<Self> {
        // Convert to SquaredEuclideanDistance for training (zero-copy distance change)
        let euclidean_dataset: PlainDenseDataset<f32, SquaredEuclideanDistance> =
            dataset.clone().into();

        let sample_size = Self::compute_training_sample_size(euclidean_dataset.len());

        // Sample if needed
        let training_dataset = match sample_size {
            Some(size) => {
                println!(
                    "Sampling {} vectors from {} for PQ training",
                    size,
                    euclidean_dataset.len()
                );
                Self::sample_random_dataset(&euclidean_dataset, size)
            }
            None => euclidean_dataset,
        };

        // Train the encoder
        let pq_encoder = Self::train(&training_dataset);

        // Encode all vectors from the original dataset
        let encoded_vector_len = pq_encoder.output_dim();
        let mut encoded_data = Vec::with_capacity(dataset.len() * encoded_vector_len);
        for vector in dataset.iter() {
            pq_encoder.push_encoded(vector, &mut encoded_data);
        }

        let encoded_data = encoded_data.into_boxed_slice();
        crate::DenseDataset::<ProductQuantizer<M, D>>::from_raw(
            encoded_data,
            dataset.len(),
            pq_encoder,
        )
    }

    #[inline]
    pub fn from_pretrained(
        d: usize,
        centroids: [PlainDenseDataset<f32, SquaredEuclideanDistance>; M],
    ) -> Self {
        assert_eq!(M % 4, 0, "M ({}) is not divisible by 4", M);
        assert_eq!(d % M, 0, "d ({}) is not divisible by M ({})", d, M);
        let dsub = d / M;
        // Verify each dataset has KSUB centroids of dimension dsub
        for (i, dataset) in centroids.iter().enumerate() {
            assert_eq!(
                dataset.len(),
                KSUB,
                "Subspace {} has {} centroids, expected {}",
                i,
                dataset.len(),
                KSUB
            );
            assert_eq!(
                dataset.output_dim(),
                dsub,
                "Subspace {} has dimension {}, expected {}",
                i,
                dataset.output_dim(),
                dsub
            );
        }
        Self {
            d,
            dsub,
            centroids,
            _distance: PhantomData,
        }
    }

    #[inline]
    pub fn input_dim(&self) -> usize {
        self.d
    }

    #[inline]
    pub fn output_dim(&self) -> usize {
        self.code_size()
    }

    #[inline]
    pub fn ksub(&self) -> usize {
        KSUB
    }

    #[inline]
    pub fn dsub(&self) -> usize {
        self.dsub
    }

    #[inline]
    pub fn d(&self) -> usize {
        self.d
    }

    /// Code size in bytes (one byte per subspace)
    #[inline]
    pub fn code_size(&self) -> usize {
        M
    }

    fn compute_euclidean_distance_table(&self, query: DenseVectorView<'_, f32>) -> Vec<f32> {
        assert_eq!(query.len(), self.d());
        let mut table = vec![0.0_f32; KSUB * M];

        for (sub_id, (sub_distance_table, query_sub)) in table
            .chunks_mut(KSUB)
            .zip(query.values().chunks(self.dsub))
            .enumerate()
        {
            let centroids = &self.centroids[sub_id];

            let query = DenseVectorView::new(query_sub);
            for (i, slot) in sub_distance_table.iter_mut().enumerate() {
                let centroid = centroids.get(i as crate::VectorId);
                *slot = unsafe {
                    crate::distances::squared_euclidean_distance_dense_unchecked(query, centroid)
                }
                .distance();
            }
        }
        table
    }

    fn compute_dot_product_table(&self, query: DenseVectorView<'_, f32>) -> Vec<f32> {
        assert_eq!(query.len(), self.d());
        let mut table = vec![0.0_f32; KSUB * M];

        for (sub_id, (sub_distance_table, query_sub)) in table
            .chunks_mut(KSUB)
            .zip(query.values().chunks(self.dsub))
            .enumerate()
        {
            let centroids = &self.centroids[sub_id];
            let query = DenseVectorView::new(query_sub);

            // For dot product, we need to compute q · c for each centroid
            // We use the raw dot product computation since the centroids dataset
            // is stored with SquaredEuclideanDistance but we need dot products
            for (i, slot) in sub_distance_table.iter_mut().enumerate() {
                let centroid = centroids.get(i as crate::VectorId);
                *slot = unsafe {
                    crate::distances::dot_product_dense_unchecked(query, centroid).distance()
                };
            }
        }
        table
    }

    fn compute_distance(&self, distance_table: &[f32], code: DenseVectorView<'_, u8>) -> f32 {
        assert_eq!(code.len(), M);
        let code_bytes = code.values();
        let mut total = 0.0_f32;
        for (sub_id, code_byte) in code_bytes.iter().enumerate() {
            unsafe {
                total = total.algebraic_add(
                    *distance_table.get_unchecked(sub_id * KSUB + *code_byte as usize),
                );
            }
        }
        total
    }

    fn decode_vector(&self, vector: DenseVectorView<'_, u8>) -> DenseVectorOwned<f32> {
        assert_eq!(vector.len(), M);
        let code_bytes = vector.values();
        let mut decoded = Vec::with_capacity(self.d());
        for (sub_id, code_byte) in code_bytes.iter().enumerate() {
            let centroids = &self.centroids[sub_id as usize];
            let centroid = centroids.get(*code_byte as crate::VectorId);
            decoded.extend_from_slice(centroid.values());
        }
        DenseVectorOwned::new(decoded)
    }
}

impl<const M: usize, D> SpaceUsage for ProductQuantizer<M, D>
where
    D: ProductQuantizerDistance,
{
    fn space_usage_bytes(&self) -> usize {
        2 * std::mem::size_of::<usize>()
            + self
                .centroids
                .iter()
                .map(|ds| ds.space_usage_bytes())
                .sum::<usize>()
    }
}

impl<const M: usize, D> DenseVectorEncoder for ProductQuantizer<M, D>
where
    D: ProductQuantizerDistance,
{
    type InputValueType = f32;
    type OutputValueType = u8;

    fn decode_vector<'a>(
        &self,
        encoded: DenseVectorView<'a, Self::OutputValueType>,
    ) -> DenseVectorOwned<f32> {
        self.decode_vector(encoded)
    }

    #[inline]
    fn push_encoded<'a, OutputContainer>(
        &self,
        input: DenseVectorView<'a, Self::InputValueType>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<Self::OutputValueType>,
    {
        assert_eq!(input.len(), self.d());
        let values = input.values();
        for (sub_id, query_sub) in values.chunks(self.dsub).enumerate() {
            let centroid_idx = self.centroids[sub_id]
                .search_nearest(DenseVectorView::new(query_sub))
                .map(|scored| scored.vector as u8)
                .unwrap_or(0);
            output.extend(std::iter::once(centroid_idx));
        }
    }
}

impl<const M: usize, D> VectorEncoder for ProductQuantizer<M, D>
where
    D: ProductQuantizerDistance,
{
    type Distance = D;
    type InputVector<'a> = DenseVectorView<'a, f32>;
    type QueryVector<'q> = DenseVectorView<'q, f32>;
    type EncodedVector<'a> = DenseVectorView<'a, u8>;
    type Evaluator<'e>
        = ProductQuantizerQueryEvaluator<'e, M, D>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        assert_eq!(query.len(), self.d());
        ProductQuantizerQueryEvaluator::new(self, query)
    }

    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = self.decode_vector(vector);
        ProductQuantizerQueryEvaluator::new(self, decoded.as_view())
    }

    fn input_dim(&self) -> usize {
        self.d()
    }

    fn output_dim(&self) -> usize {
        self.code_size()
    }
}

pub struct ProductQuantizerQueryEvaluator<'a, const M: usize, D>
where
    D: ProductQuantizerDistance,
{
    encoder: &'a ProductQuantizer<M, D>,
    distance_table: Vec<f32>,
}

impl<'a, const M: usize, D> ProductQuantizerQueryEvaluator<'a, M, D>
where
    D: ProductQuantizerDistance,
{
    fn new(encoder: &'a ProductQuantizer<M, D>, query: DenseVectorView<'_, f32>) -> Self {
        let table = D::compute_query_distance_table(encoder, query);
        Self {
            encoder,
            distance_table: table,
        }
    }
}

impl<'a, const M: usize, D> QueryEvaluator<DenseVectorView<'_, u8>>
    for ProductQuantizerQueryEvaluator<'a, M, D>
where
    D: ProductQuantizerDistance,
{
    type Distance = D;

    fn compute_distance(&self, vector: DenseVectorView<'_, u8>) -> Self::Distance {
        let raw = self.encoder.compute_distance(&self.distance_table, vector);
        D::from(raw)
    }
}
