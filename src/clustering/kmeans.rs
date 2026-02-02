use crate::core::dataset::{DatasetGrowable, VectorId};
use crate::core::distances::{Distance, SquaredEuclideanDistance, dot_product_dense};
use crate::core::vector::DenseVectorView;
use crate::core::vector_encoder::{QueryEvaluator, VectorEncoder};
use crate::datasets::dense_dataset::DenseDatasetGeneric;
use crate::encoders::dense_scalar::ScalarDenseQuantizer;
use crate::{Dataset, Float, FromF32, PlainDenseDataset, PlainDenseDatasetGrowable, ValueType};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::time::Instant;

pub struct KMeans {
    n_iter: usize,
    n_redo: usize,
    verbose: bool,
    // min_points_per_centroid: usize,
    // sample_size: Option<usize>,
    // Threshold for number of centroids above which HNSW index is used instead of flat search
    // index_threshold: usize,
}

impl KMeans {
    /// Computes the imbalance factor of the clustering.
    /// Smaller unfairness factor means more balanced clusters.
    ///
    /// # Arguments
    ///
    /// * `histograms`: vector storing how many vectors are assigned to each cluster
    /// * `k`: the number of clusters
    ///
    /// returns: the imbalance factor as f32
    ///
    #[inline]
    fn imbalance_factor(histograms: &[f32], k: usize) -> f32 {
        let hist_view = DenseVectorView::new(histograms);
        let unfairness_factor = dot_product_dense(hist_view, hist_view).distance();
        let total: f32 = histograms.iter().sum();
        unfairness_factor * k as f32 / (total * total)
    }

    /// Assignment: for each vector, find nearest centroid by parallelizing over vectors
    /// Internally decides whether to use HNSW index or flat search based on the number of centroids.
    /// For small k (< index_threshold), uses flat dataset search. For large k, builds HNSW index.
    pub fn compute_assignments<VIn, VOut>(
        dataset: &DenseDatasetGeneric<
            ScalarDenseQuantizer<VIn, VOut, SquaredEuclideanDistance>,
            impl AsRef<[VOut]>,
        >,
        centroids: &DenseDatasetGeneric<
            ScalarDenseQuantizer<VOut, VOut, SquaredEuclideanDistance>,
            impl AsRef<[VOut]> + Sync,
        >,
        _index_threshold: usize,
    ) -> Vec<(f32, usize)>
    // sum of distances and assignments
    where
        VIn: Float + ValueType + FromF32,
        VOut: Float + ValueType + FromF32,
    {
        // HNSW path disabled: always fall back to flat dataset search.
        /*
        if _k >= index_threshold {
            // Use HNSW index for large k
            let hnsw_params = HNSWBuildParams {
                num_neighbors_per_vec: 16,
                ef_construction: 600,
                initial_build_batch_size: 4,
                max_build_batch_size: 64,
            };
            let centroids_hnsw = HNSW::<PlainDenseDataset<T>, PlainQuantizer<T>, GraphFixedDegree>::build_index(
                centroids,
                PlainQuantizer::<T>::new(Dataset::dim(centroids), DistanceType::Euclidean),
                &hnsw_params,
            );
            let search_params = HNSWSearchParams { ef_search: 600 };

            (0..dataset.len())
                .into_par_iter()
                .map(|i| {
                    let query = dataset.get(i);
                    let res = centroids_hnsw.search::<PlainDenseDataset<T>, PlainQuantizer<T>>(
                        query,
                        1,
                        &search_params,
                    );
                    if let Some((dist, idx)) = res.into_iter().next() {
                        (dist, idx)
                    } else {
                        (f32::MAX, usize::MAX)
                    }
                })
                .collect()
        } else {
        */
        let centroid_encoder = centroids.encoder();

        dataset
            .par_iter()
            .map(|vector| {
                let evaluator = centroid_encoder.vector_evaluator(vector);

                centroids
                    .iter()
                    .enumerate()
                    .map(|(ci, centroid_view)| {
                        let distance = evaluator.compute_distance(centroid_view).distance();
                        (distance, ci)
                    })
                    .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                    .unwrap()
            })
            .collect()
    }

    /// Computes the centroids of the clustering as the mean of vectors assigned to each cluster.
    /// Then, splits empty clusters
    ///
    /// # Arguments
    ///
    /// * `dataset`: the input dataset
    /// * `weights`: the weight of each vector, optionally
    /// * `k`: the number of clusters
    /// * `assignments`: the latest assignment vector in the dataset - cluster
    ///
    /// returns: the number of splits, a vector storing how many vectors are assigned to each cluster and the new centroids in a Dataset.

    fn update_and_split<VIn, VOut, Data>(
        dataset: &DenseDatasetGeneric<
            ScalarDenseQuantizer<VIn, VOut, SquaredEuclideanDistance>,
            Data,
        >,
        weights: Option<&[f32]>,
        k: usize,
        assignments: &Vec<(f32, usize)>,
        rng: &mut StdRng,
    ) -> (
        usize,
        Vec<f32>,
        PlainDenseDataset<VOut, SquaredEuclideanDistance>,
    )
    where
        VIn: Float + ValueType + FromF32,
        VOut: Float + ValueType + FromF32 + num_traits::ToPrimitive + num_traits::FromPrimitive,
        Data: AsRef<[VOut]> + Sync,
    {
        let n = dataset.len();
        let d = dataset.output_dim();

        // TODO: This code is terrible. group by centroids and avoid iterating over all data k times.
        // However, its cost is just a fraction of the cost of the assignment step, so leaving it as is for now.
        let (centroids, histograms) = {
            let results: Vec<(Vec<f32>, f32)> = (0..k)
                .into_par_iter()
                .map(|ci| {
                    let mut centroid = vec![0.0; d];
                    let mut count = 0.0;

                    for i in 0..n {
                        if assignments[i].1 == ci {
                            let vec = dataset.get(i as VectorId);
                            match weights {
                                Some(w) => {
                                    count += w[i];
                                    centroid
                                        .iter_mut()
                                        .zip(vec.values().iter())
                                        .for_each(|(c, x)| *c += x.to_f32().unwrap() * w[i]);
                                }
                                None => {
                                    count += 1.0;
                                    centroid
                                        .iter_mut()
                                        .zip(vec.values().iter())
                                        .for_each(|(c, x)| *c += x.to_f32().unwrap());
                                }
                            }
                        }
                    }

                    // Normalize
                    if count > 0.0 {
                        centroid.iter_mut().for_each(|c| *c /= count);
                    }

                    (centroid, count)
                })
                .collect();

            // Flatten results
            let mut centroids = Vec::with_capacity(k * d);
            let mut histograms = Vec::with_capacity(k);
            for (centroid, count) in results {
                centroids.extend(centroid);
                histograms.push(count);
            }
            (centroids, histograms)
        };

        let mut centroids = centroids;
        let mut histograms = histograms;

        // Splits clusters
        let mut n_splits = 0;
        let mut cj;
        let epsilon = 1.0 / 1024.;

        for ci in 0..k {
            if histograms[ci] != 0.0 {
                continue;
            }
            cj = 0;
            loop {
                let p = (histograms[cj] - 1.0) / (n - k) as f32;
                let r = Rng::r#gen::<f32>(rng);
                if r < p {
                    break;
                }
                cj = (cj + 1) % k;
            }

            let tmp = centroids[cj * d..(cj + 1) * d].to_owned();
            centroids[ci * d..(ci + 1) * d].copy_from_slice(&tmp);

            for j in 0..d {
                if j % 2 == 0 {
                    centroids[ci * d + j] *= 1.0 + epsilon;
                    centroids[cj * d + j] *= 1.0 - epsilon;
                } else {
                    centroids[ci * d + j] *= 1.0 - epsilon;
                    centroids[cj * d + j] *= 1.0 + epsilon;
                }
            }

            histograms[ci] = histograms[cj] / 2.0;
            histograms[cj] /= 2.0;
            n_splits += 1;
        }

        // Convert f32 centroids back to VOut
        let centroids_vout: Vec<VOut> = centroids
            .iter()
            .map(|&x| VOut::from_f32(x).unwrap())
            .collect();

        (
            n_splits,
            histograms,
            PlainDenseDataset::<VOut, SquaredEuclideanDistance>::from_raw(
                centroids_vout.into_boxed_slice(),
                k,
                ScalarDenseQuantizer::new(dataset.encoder().output_dim()),
            ),
        )
    }

    /// Runs K-Means training on a dataset with k clusters.
    /// If the user has provided input weights, the computation of centroids is the weighted mean
    /// of every vector assigned to their cluster. Otherwise the computation is just the mean.
    ///
    /// If a sample_size is specified, a random sample of that size is used for training.
    /// Otherwise, the full dataset is used for training.
    ///
    /// # Arguments
    ///
    /// * `dataset`: the dataset (can be f32 or f16)
    /// * `k`: the desired number of clusters
    /// * `weights`: optionally weights of the same length of the dataset
    ///
    /// returns: the best computed centroids in the training.
    ///
    pub fn train<VIn, VOut, Data>(
        &self,
        training_dataset: &DenseDatasetGeneric<
            ScalarDenseQuantizer<VIn, VOut, SquaredEuclideanDistance>,
            Data,
        >,
        k: usize,
        weights: Option<Vec<f32>>,
    ) -> PlainDenseDataset<VOut, SquaredEuclideanDistance>
    where
        VIn: Float + ValueType + FromF32,
        VOut: Float
            + ValueType
            + FromF32
            + num_traits::ToPrimitive
            + num_traits::FromPrimitive
            + Clone,
        Data: AsRef<[VOut]> + Sync,
    {
        let n = training_dataset.len();

        if n == k {
            if self.verbose {
                println!("WARNING: number of training data is equal to the number of clusters.");
            }
            // Convert generic dataset to PlainDenseDataset for return
            return PlainDenseDataset::<VOut, SquaredEuclideanDistance>::from_raw(
                training_dataset.values().to_vec().into_boxed_slice(),
                n,
                ScalarDenseQuantizer::new(training_dataset.encoder().output_dim()),
            );
        }

        let d = training_dataset.output_dim();

        if self.verbose {
            println!(
                "Clustering {} points in {}D to {} clusters, redo {} times, {} iterations",
                n, d, k, self.n_redo, self.n_iter
            );
        }

        let mut best_obj = f32::MAX;

        // clustering-related
        let output_dim = training_dataset.output_dim();
        let mut best_centroids = PlainDenseDataset::<VOut, SquaredEuclideanDistance>::from_raw(
            Vec::new().into_boxed_slice(),
            0,
            ScalarDenseQuantizer::new(output_dim),
        );

        let w = weights.as_deref();

        // HNSW logging disabled while index usage is deferred.
        let mut rng = StdRng::from_entropy();

        for redo in 0..self.n_redo {
            let mut centroids_builder =
                PlainDenseDatasetGrowable::with_capacity(ScalarDenseQuantizer::new(output_dim), k);

            // Randomly select k vectors for initial centroids
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(&mut rng);

            for i in indices.iter().take(k) {
                let vector = training_dataset.get(*i as VectorId);
                centroids_builder.push(vector);
            }

            let mut centroids: PlainDenseDataset<VOut, SquaredEuclideanDistance> =
                centroids_builder.into();

            let mut obj;
            let mut average_imbalance_factor = 0.0;
            let mut total_splits = 0;

            for i in 0..self.n_iter {
                let t0 = Instant::now();

                // Assignment: find nearest centroid for each vector
                let assignments = Self::compute_assignments(&training_dataset, &centroids, 0);

                let search_time = t0.elapsed();
                let t0 = Instant::now();
                obj = assignments.iter().map(|&(value, _)| value).sum();

                // Update: recompute centroids
                let (n_split, histograms, new_centroids) =
                    Self::update_and_split(&training_dataset, w, k, &assignments, &mut rng);

                let imbalance_factor = Self::imbalance_factor(&histograms, k);
                let split_time = t0.elapsed();

                average_imbalance_factor += imbalance_factor;
                total_splits += n_split;

                if obj < best_obj {
                    if self.verbose {
                        println!("New best objective: {} (keep new clusters)", obj);
                    }
                    best_obj = obj;
                    best_centroids = new_centroids.clone();
                }

                centroids = new_centroids;

                if self.verbose {
                    println!(
                        "Iteration {}, imbalance: {}, splits: {}, search time: {:.2?} split time: {:.2?} ",
                        i, imbalance_factor, n_split, search_time, split_time
                    );
                }
            }

            if self.verbose {
                println!(
                    "Outer iteration {} -- average imbalance: {}, splits: {}",
                    redo,
                    average_imbalance_factor / (self.n_iter + 1) as f32,
                    total_splits
                );
            }
        }
        best_centroids
    }
}

pub struct KMeansBuilder {
    n_iter: usize,
    n_redo: usize,
    verbose: bool,
    // min_points_per_centroid: usize,
    max_points_per_centroid: usize,
    // sample_size: Option<usize>,
}

impl Default for KMeansBuilder {
    fn default() -> Self {
        KMeansBuilder {
            n_iter: 10,
            n_redo: 1,
            verbose: false,
            // min_points_per_centroid: 39,
            max_points_per_centroid: 256,
            // sample_size: None,
        }
    }
}

impl KMeansBuilder {
    pub fn new() -> Self {
        KMeansBuilder::default()
    }

    pub fn n_iter(mut self, n_iter: usize) -> KMeansBuilder {
        self.n_iter = n_iter;
        self
    }

    pub fn n_redo(mut self, n_redo: usize) -> KMeansBuilder {
        self.n_redo = n_redo;
        self
    }

    pub fn verbose(mut self, verbose: bool) -> KMeansBuilder {
        self.verbose = verbose;
        self
    }

    // pub fn min_points_per_centroid(mut self, min_points_per_centroid: usize) -> KMeansBuilder {
    //     self.min_points_per_centroid = min_points_per_centroid;
    //     self
    // }

    pub fn max_points_per_centroid(mut self, max_points_per_centroid: usize) -> KMeansBuilder {
        self.max_points_per_centroid = max_points_per_centroid;
        self
    }

    // pub fn sample_size(mut self, sample_size: usize) -> KMeansBuilder {
    //     self.sample_size = Some(sample_size);
    //     self
    // }

    pub fn build(self) -> KMeans {
        KMeans {
            n_iter: self.n_iter,
            n_redo: self.n_redo,
            verbose: self.verbose,
            // min_points_per_centroid: self.min_points_per_centroid,
            // sample_size: self.sample_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vector::DenseVectorView;
    use crate::distances::SquaredEuclideanDistance;
    use crate::encoders::dense_scalar::PlainDenseQuantizer;
    use crate::{PlainDenseDataset, PlainDenseDatasetGrowable};

    #[test]
    fn compute_assignments_picks_nearest_centroid() {
        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(2);
        let mut dataset_builder = PlainDenseDatasetGrowable::new(encoder.clone());
        dataset_builder.push(DenseVectorView::new(&[0.0f32, 0.0]));
        dataset_builder.push(DenseVectorView::new(&[10.0f32, 10.0]));

        let dataset: PlainDenseDataset<f32, SquaredEuclideanDistance> = dataset_builder.into();
        let centroids = vec![0.0f32, 0.0, 9.0, 9.0];
        let centroids_data = centroids.into_boxed_slice();
        let centroids_len = centroids_data.len() / dataset.output_dim();
        let centroids_dataset = PlainDenseDataset::<f32, SquaredEuclideanDistance>::from_raw(
            centroids_data,
            centroids_len,
            encoder.clone(),
        );
        let assignments: Vec<(f32, usize)> =
            KMeans::compute_assignments(&dataset, &centroids_dataset, 0)
                .into_iter()
                .collect();

        assert_eq!(assignments.len(), dataset.len());
        assert_eq!(assignments[0].1, 0);
        assert_eq!(assignments[1].1, 1);
    }

    #[test]
    fn train_builds_two_distinct_clusters() {
        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(2);
        let mut dataset_builder = PlainDenseDatasetGrowable::new(encoder.clone());
        dataset_builder.push(DenseVectorView::new(&[0.0f32, 0.0]));
        dataset_builder.push(DenseVectorView::new(&[0.0f32, 1.0]));
        dataset_builder.push(DenseVectorView::new(&[9.0f32, 9.0]));
        dataset_builder.push(DenseVectorView::new(&[10.0f32, 10.0]));

        let dataset: PlainDenseDataset<f32, SquaredEuclideanDistance> = dataset_builder.into();
        let kmeans = KMeansBuilder::new().n_iter(5).n_redo(2).build();
        let centroids = kmeans.train(&dataset, 2, None);

        assert_eq!(centroids.len(), 2);
        let assignments = KMeans::compute_assignments(&dataset, &centroids, 0);
        let assignments: Vec<(f32, usize)> = assignments.into_iter().collect();
        let cluster_ids: std::collections::HashSet<usize> =
            assignments.iter().map(|&(_, idx)| idx).collect();
        assert_eq!(cluster_ids.len(), 2);
        assert_eq!(assignments.len(), dataset.len());
    }
}
