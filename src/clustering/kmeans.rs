use crate::core::dataset::{DatasetGrowable, VectorId};
use crate::core::distances::{Distance, SquaredEuclideanDistance, dot_product_dense};
use crate::core::index::Index;
use crate::core::vector::DenseVectorView;
use crate::core::vector_encoder::{QueryEvaluator, VectorEncoder};
use crate::datasets::dense_dataset::DenseDatasetGeneric;
use crate::encoders::dense_scalar::ScalarDenseQuantizer;
use crate::{Dataset, Float, FromF32, PlainDenseDataset, PlainDenseDatasetGrowable, ValueType};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::index;
use rayon::prelude::*;
use std::time::Instant;

/// Plain dense dataset over value type `V`, used for both training storage and
/// centroids in [`KMeans::train_with_index`]. Means always accumulate in `f32`
/// and queries are always `f32`; `T`/`C` only choose the stored layout.
///
/// Precision guidance: `T = f16` halves the training-corpus footprint at equal
/// clustering quality. For `C`, it depends on how the binary is compiled: with
/// hardware f16 conversion enabled (e.g. `-C target-cpu=native` on x86 with
/// F16C), an f16 index is both smaller and faster than f32 (distance kernels
/// are memory-bound, and half the bytes means half the traffic); without those
/// target features the f16 -> f32 conversion falls back to slow scalar code and
/// an f16 index regresses badly, so portable builds should keep `C = f32`.
///
/// # Range precondition on `C`
///
/// Means are written into `C` via [`FromF32::from_f32_saturating`]. For IEEE
/// half types that name is misleading: `f16`/`bf16` use `from_f32` rounding, so
/// magnitudes above the type's finite max become `±inf` rather than clamping.
/// With `T = f32` and `C = f16`, an unbounded training corpus can therefore
/// produce `inf` centroids, an `inf` objective, and — because `best_obj` starts
/// at `f32::MAX` — a silently empty return instead of a failure. Callers must
/// ensure stored means stay in the finite range of `C` (or keep `C = f32`).
/// The fixed-point types satisfy the bounds too, but `FixedU8`/`FixedU16` are
/// *unsigned*: their `from_f32_saturating` really does clamp, yet every negative
/// component clamps to `0`, so they silently destroy centred data as `T` or `C`.
type Centroids<V> = PlainDenseDataset<V, SquaredEuclideanDistance>;

pub struct KMeans {
    n_iter: usize,
    n_redo: usize,
    verbose: bool,
    /// Optional fixed seed for the RNG.  `None` = `from_entropy()` (non-deterministic).
    /// Setting a seed makes clustering fully reproducible across runs and thread counts.
    seed: Option<u64>,
    /// Spherical k-means: L2-normalize each centroid after every update step.
    spherical: bool,
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

    fn renorm_l2(centroids: &mut [f32], d: usize) {
        for sl in centroids.chunks_exact_mut(d) {
            let norm: f32 = sl.iter().map(|x| x * x).sum::<f32>().sqrt();
            // Take inverse of norm, take 0 if norm is 0
            let inv_norm = f32::from(norm > 0.0) * (1.0 / norm.max(f32::MIN_POSITIVE));
            sl.iter_mut().for_each(|x| *x *= inv_norm);
        }
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
    ///
    /// `VOut` is the training-vector storage type (read from `dataset`); `VCent`
    /// is the centroid storage type written from the f32 means. They are usually
    /// the same (`train`); [`train_with_index`] may choose `VCent = f32` while
    /// training vectors stay `f16`.
    fn update_and_split<VIn, VOut, VCent, Data>(
        dataset: &DenseDatasetGeneric<
            ScalarDenseQuantizer<VIn, VOut, SquaredEuclideanDistance>,
            Data,
        >,
        weights: Option<&[f32]>,
        k: usize,
        assignments: &[(f32, usize)],
        rng: &mut StdRng,
        spherical: bool,
    ) -> (
        usize,
        Vec<f32>,
        PlainDenseDataset<VCent, SquaredEuclideanDistance>,
    )
    where
        VIn: Float + ValueType + FromF32,
        VOut: Float + ValueType + FromF32 + num_traits::ToPrimitive + num_traits::FromPrimitive,
        VCent: Float + ValueType + FromF32 + num_traits::ToPrimitive + num_traits::FromPrimitive,
        Data: AsRef<[VOut]> + Sync,
    {
        let n = dataset.len();
        let d = dataset.output_dim();

        // Counting-sort + cluster-parallel mean: two serial O(n) passes over the
        // assignments (the histogram and the scatter) followed by a single parallel
        // O(n*d) pass over the points.
        // Scratch is one u32 per point (the grouping permutation) plus O(k) offsets
        // — independent of thread count — instead of n_threads * k * d private
        // accumulators. Clusters are processed in parallel into one shared (sums, counts)
        // output; each cluster's members are visited in ascending point order from the
        // stable counting sort, so the float summation order is fixed and results are
        // bit-identical across thread counts.
        let (centroids, histograms) = {
            let mut cluster_sizes = vec![0usize; k];
            for &(_, ci) in assignments {
                cluster_sizes[ci] += 1;
            }

            let mut offsets = vec![0usize; k + 1];
            for c in 0..k {
                offsets[c + 1] = offsets[c] + cluster_sizes[c];
            }

            let mut grouped = vec![0u32; n];
            let mut cursor = offsets.clone();
            for (i, &(_, ci)) in assignments.iter().enumerate().take(n) {
                grouped[cursor[ci]] = i as u32;
                cursor[ci] += 1;
            }

            let mut sums = vec![0.0f32; k * d];
            let mut counts = vec![0.0f32; k];
            sums.par_chunks_mut(d)
                .zip(counts.par_iter_mut())
                .enumerate()
                .for_each(|(ci, (centroid, cnt))| {
                    for &i in &grouped[offsets[ci]..offsets[ci + 1]] {
                        let i = i as usize;
                        let w = weights.map_or(1.0, |w| w[i]);
                        *cnt += w;
                        let vec = dataset.get(i as VectorId);
                        for (c, x) in centroid.iter_mut().zip(vec.values().iter()) {
                            *c += x.to_f32().unwrap() * w;
                        }
                    }
                    if *cnt > 0.0 {
                        let inv = 1.0 / *cnt;
                        for x in centroid.iter_mut() {
                            *x *= inv;
                        }
                    }
                });

            (sums, counts)
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

        if spherical {
            Self::renorm_l2(&mut centroids, d);
        }

        // Convert f32 means into the caller-chosen centroid storage type.
        let centroids_out: Vec<VCent> = centroids
            .iter()
            .map(|&x| VCent::from_f32_saturating(x))
            .collect();

        (
            n_splits,
            histograms,
            PlainDenseDataset::<VCent, SquaredEuclideanDistance>::from_raw(
                centroids_out.into_boxed_slice(),
                k,
                ScalarDenseQuantizer::new(dataset.encoder().output_dim()),
            ),
        )
    }

    /// Cast a dense vector from storage type `T` to centroid type `C` via f32.
    #[inline]
    fn cast_dense_values<T, C>(values: &[T]) -> Vec<C>
    where
        T: ValueType + num_traits::ToPrimitive,
        C: FromF32,
    {
        values
            .iter()
            .map(|x| {
                C::from_f32_saturating(x.to_f32().expect("value type is not representable as f32"))
            })
            .collect()
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

        // RNG used only for empty-cluster splitting.
        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        for redo in 0..self.n_redo {
            let mut centroids_builder =
                PlainDenseDatasetGrowable::with_capacity(ScalarDenseQuantizer::new(output_dim), k);

            // Offset by `redo` so each restart draws a different initial sample
            // (a fixed `s + 1` would make every redo identical under a fixed seed).
            let mut init_rng = match self.seed {
                Some(s) => StdRng::seed_from_u64(s + 1 + redo as u64),
                None => StdRng::from_entropy(),
            };
            for i in index::sample(&mut init_rng, n, k).into_iter() {
                let vector = training_dataset.get(i as VectorId);
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
                let assignments = Self::compute_assignments(training_dataset, &centroids, 0);

                let search_time = t0.elapsed();
                let t0 = Instant::now();
                obj = assignments.iter().map(|&(value, _)| value).sum();

                // Update: recompute centroids
                let (n_split, histograms, new_centroids) = Self::update_and_split(
                    training_dataset,
                    w,
                    k,
                    &assignments,
                    &mut rng,
                    self.spherical,
                );

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

    /// Like [`train`], but uses a generic [`Index`] for centroid assignment instead of exhaustive
    /// flat search. At each k-means iteration the index is rebuilt from the current centroids and
    /// then searched (top-1) for every training vector.
    ///
    /// # Type parameters
    /// * `T` — storage precision of the **training** vectors (e.g. `f16` to halve the corpus).
    /// * `C` — storage precision of the **centroids** and of the dataset the index is built on
    ///   (`f32` is the safe portable default; `f16` is faster and smaller when the binary is
    ///   built with hardware f16 conversion, e.g. `-C target-cpu=native` — see [`Centroids`]).
    /// * `Q` — centroid index type; must accept `f32` queries.
    ///
    /// Means still accumulate in `f32` inside [`update_and_split`]. Training vectors of type `T`
    /// are upconverted to `f32` per query in [`assign_with_index`]. `T = C = f32` is the IVF /
    /// kannolo call shape (turbofish `train_with_index::<HNSW<..>, f32, f32>` or let `T`/`C`
    /// infer from the dataset and closure).
    ///
    /// This is a breaking change vs the previous single-generic `train_with_index::<Q>`: callers
    /// must supply or infer `T` and `C`.
    ///
    /// # Range precondition on `C`
    ///
    /// Means are cast into `C` with [`FromF32::from_f32_saturating`]. For `f16`/`bf16` that
    /// method does **not** clamp: it is IEEE `from_f32`, so values outside the finite range
    /// become `±inf`. With `T = f32` and `C = f16`, an unbounded training corpus can therefore
    /// produce `inf` centroids and an `inf` objective; because `best_obj` starts at `f32::MAX`
    /// and `inf < f32::MAX` is false, this method then returns an **empty** centroid set instead
    /// of failing. Callers must keep stored means inside the finite range of `C`, or use
    /// `C = f32`.
    ///
    /// The fixed-point scalar types satisfy `T`/`C`'s bounds as well, and for them
    /// `from_f32_saturating` really does clamp — but `FixedU8`/`FixedU16` are *unsigned*,
    /// so every negative component clamps to `0`. Centred data must not be stored in them.
    ///
    /// **Reproducibility caveat:** with a fixed [`KMeansBuilder::seed`], initialization and the
    /// update step are deterministic, but end-to-end reproducibility additionally requires the
    /// index produced by `build_centroid_index` to be deterministic in build and search. A
    /// parallel HNSW build generally is not, so assignments (and thus centroids) may still vary
    /// run to run. [`train`] (flat assignment) does not have this caveat.
    pub fn train_with_index<Q, T, C>(
        &self,
        training_dataset: &Centroids<T>,
        k: usize,
        weights: Option<Vec<f32>>,
        build_centroid_index: impl Fn(Centroids<C>) -> Q,
        search_params: &Q::SearchParams,
    ) -> Centroids<C>
    where
        T: Float
            + ValueType
            + FromF32
            + num_traits::ToPrimitive
            + num_traits::FromPrimitive
            + Clone,
        C: Float
            + ValueType
            + FromF32
            + num_traits::ToPrimitive
            + num_traits::FromPrimitive
            + Clone,
        Q: Index + Sync,
        for<'q> Q: Index<Query<'q> = DenseVectorView<'q, f32>>,
        Q::SearchParams: Sync,
    {
        let n = training_dataset.len();
        let d = training_dataset.output_dim();

        if n == k {
            if self.verbose {
                println!("WARNING: number of training data is equal to the number of clusters.");
            }
            let cast = Self::cast_dense_values::<T, C>(training_dataset.values());
            return Centroids::<C>::from_raw(
                cast.into_boxed_slice(),
                n,
                ScalarDenseQuantizer::new(d),
            );
        }
        assert!(
            k < n,
            "k ({k}) must not exceed the number of training vectors ({n})"
        );

        if self.verbose {
            println!(
                "Clustering {} points in {}D to {} clusters (ANN assignment), redo {} times, {} iterations",
                n, d, k, self.n_redo, self.n_iter
            );
        }

        let mut best_obj = f32::MAX;
        let mut best_centroids = Centroids::<C>::from_raw(
            Vec::new().into_boxed_slice(),
            0,
            ScalarDenseQuantizer::new(d),
        );

        let w = weights.as_deref();
        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        for redo in 0..self.n_redo {
            let mut centroids_builder: PlainDenseDatasetGrowable<C, SquaredEuclideanDistance> =
                PlainDenseDatasetGrowable::with_capacity(ScalarDenseQuantizer::new(d), k);
            // Offset by `redo` so each restart draws a different initial sample
            // (a fixed `s + 1` would make every redo identical under a fixed seed).
            let mut init_rng = match self.seed {
                Some(s) => StdRng::seed_from_u64(s + 1 + redo as u64),
                None => StdRng::from_entropy(),
            };
            for i in index::sample(&mut init_rng, n, k).into_iter() {
                let src = training_dataset.get(i as VectorId);
                let cast = Self::cast_dense_values::<T, C>(src.values());
                centroids_builder.push(DenseVectorView::new(&cast));
            }
            let mut centroids: Centroids<C> = centroids_builder.into();

            let mut obj;
            let mut average_imbalance_factor = 0.0;
            let mut total_splits = 0;

            for i in 0..self.n_iter {
                let t0 = Instant::now();
                let centroid_index = build_centroid_index(centroids.clone());
                let build_idx_time = t0.elapsed();

                let t0 = Instant::now();
                let assignments =
                    Self::assign_with_index(training_dataset, &centroid_index, search_params);
                let search_time = t0.elapsed();

                let t0 = Instant::now();
                obj = assignments.iter().map(|&(v, _)| v).sum();

                let (n_split, histograms, new_centroids) = Self::update_and_split::<_, _, C, _>(
                    training_dataset,
                    w,
                    k,
                    &assignments,
                    &mut rng,
                    self.spherical,
                );
                let imbalance_factor = Self::imbalance_factor(&histograms, k);
                let split_time = t0.elapsed();

                average_imbalance_factor += imbalance_factor;
                total_splits += n_split;

                if obj < best_obj {
                    if self.verbose {
                        println!("New best objective: {obj} (keep new clusters)");
                    }
                    best_obj = obj;
                    best_centroids = new_centroids.clone();
                }
                centroids = new_centroids;

                if self.verbose {
                    println!(
                        "Iteration {i}, imbalance: {imbalance_factor:.4}, splits: {n_split}, \
                         build_idx: {build_idx_time:.2?}, search: {search_time:.2?}, split: {split_time:.2?}"
                    );
                }
            }

            if self.verbose {
                println!(
                    "Outer iteration {redo} -- average imbalance: {:.4}, splits: {total_splits}",
                    average_imbalance_factor / (self.n_iter + 1) as f32,
                );
            }
        }

        best_centroids
    }

    /// Assigns each vector in `dataset` to its nearest centroid using `index` (top-1 search).
    ///
    /// The dataset stores vectors in precision `T`; the centroid index is searched with `f32`
    /// queries, so each stored vector is upconverted to `f32` before the search. The scratch
    /// buffer is reused per rayon worker (via `map_init`) rather than allocated per query, so the
    /// `T = f32` path costs one `dim`-length copy per query instead of the previous zero-copy view
    /// — negligible next to a single HNSW search, and it keeps the hot loop allocation-free.
    pub fn assign_with_index<Q, T>(
        dataset: &Centroids<T>,
        index: &Q,
        search_params: &Q::SearchParams,
    ) -> Vec<(f32, usize)>
    where
        T: Float
            + ValueType
            + FromF32
            + num_traits::ToPrimitive
            + num_traits::FromPrimitive
            + Clone,
        Q: Index + Sync,
        for<'q> Q: Index<Query<'q> = DenseVectorView<'q, f32>>,
        Q::SearchParams: Sync,
    {
        let values = dataset.values();
        let dim = dataset.output_dim();
        let n = dataset.len();

        (0..n)
            .into_par_iter()
            .map_init(
                || Vec::<f32>::with_capacity(dim),
                |query_buf, i| {
                    query_buf.clear();
                    query_buf.extend(
                        values[i * dim..(i + 1) * dim]
                            .iter()
                            .map(|x| x.to_f32().expect("value type is not representable as f32")),
                    );
                    let query = DenseVectorView::new(query_buf.as_slice());
                    let results = index.search(query, 1, search_params);
                    let best = results
                        .into_iter()
                        .next()
                        .expect("centroid index returned no result during k-means assignment");
                    (best.distance.distance(), best.vector as usize)
                },
            )
            .collect()
    }
}

pub struct KMeansBuilder {
    n_iter: usize,
    n_redo: usize,
    verbose: bool,
    max_points_per_centroid: usize,
    seed: Option<u64>,
    spherical: bool,
}

impl Default for KMeansBuilder {
    fn default() -> Self {
        KMeansBuilder {
            n_iter: 10,
            n_redo: 1,
            verbose: false,
            max_points_per_centroid: 256,
            seed: None,
            spherical: false,
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

    /// Fix the RNG seed for reproducible clustering.
    /// `None` (default) uses `from_entropy()` — different results every run.
    /// `Some(s)` uses `seed_from_u64(s)` — identical results for the same data and seed.
    pub fn seed(mut self, seed: Option<u64>) -> KMeansBuilder {
        self.seed = seed;
        self
    }

    /// Spherical k-means: L2-normalize centroids after every update step.
    pub fn spherical(mut self, spherical: bool) -> KMeansBuilder {
        self.spherical = spherical;
        self
    }

    pub fn build(self) -> KMeans {
        KMeans {
            n_iter: self.n_iter,
            n_redo: self.n_redo,
            verbose: self.verbose,
            seed: self.seed,
            spherical: self.spherical,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::flat_index::FlatIndex;
    use crate::core::vector::DenseVectorView;
    use crate::distances::SquaredEuclideanDistance;
    use crate::encoders::dense_scalar::PlainDenseQuantizer;
    use crate::{PlainDenseDataset, PlainDenseDatasetGrowable};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

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

    // ---- train_with_index: dual storage precision (T train, C centroids) -----
    //
    // ANN-assignment k-means via in-crate `FlatIndex` (no kannolo). Two well-separated
    // blobs with integer coords (exact in f16). Covers T=C=f32, T=C=f16, and the
    // recommended hybrid T=f16 / C=f32.

    /// Two blobs: points 0..3 near the origin, points 3..6 near (10, 10).
    const BLOB_POINTS: [[f32; 2]; 6] = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [10.0, 10.0],
        [10.0, 11.0],
        [11.0, 10.0],
    ];

    /// Cluster `BLOB_POINTS` into 2 via `train_with_index` with training storage `T`
    /// and centroid / index storage `C`.
    fn blob_partition<T, C>() -> Vec<usize>
    where
        T: Float
            + ValueType
            + FromF32
            + num_traits::ToPrimitive
            + num_traits::FromPrimitive
            + Clone,
        C: Float
            + ValueType
            + FromF32
            + num_traits::ToPrimitive
            + num_traits::FromPrimitive
            + Clone,
    {
        let encoder = PlainDenseQuantizer::<T, SquaredEuclideanDistance>::new(2);
        let mut builder = PlainDenseDatasetGrowable::new(encoder);
        for pt in BLOB_POINTS.iter() {
            let v: Vec<T> = pt.iter().map(|&x| T::from_f32_saturating(x)).collect();
            builder.push(DenseVectorView::new(&v[..]));
        }
        let dataset: Centroids<T> = builder.into();

        let kmeans = KMeansBuilder::new()
            .n_iter(10)
            .n_redo(3)
            .seed(Some(42))
            .build();
        let centroids = kmeans.train_with_index::<FlatIndex<Centroids<C>>, T, C>(
            &dataset,
            2,
            None,
            FlatIndex::from,
            &(),
        );
        assert_eq!(centroids.len(), 2, "expected exactly 2 centroids");

        // Final labels via the same ANN path (works for T != C; compute_assignments
        // requires matching storage types).
        let index = FlatIndex::from(centroids);
        KMeans::assign_with_index(&dataset, &index, &())
            .into_iter()
            .map(|(_, cluster)| cluster)
            .collect()
    }

    /// A correct partition puts {0,1,2} in one cluster and {3,4,5} in the other.
    fn assert_recovers_blobs(p: &[usize]) {
        assert_eq!(p.len(), 6);
        assert_eq!(p[0], p[1], "blob A points 0,1 must co-cluster");
        assert_eq!(p[1], p[2], "blob A points 1,2 must co-cluster");
        assert_eq!(p[3], p[4], "blob B points 3,4 must co-cluster");
        assert_eq!(p[4], p[5], "blob B points 4,5 must co-cluster");
        assert_ne!(p[0], p[3], "the two blobs must be in different clusters");
    }

    fn assert_same_partition(a: &[usize], b: &[usize]) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            for j in 0..a.len() {
                assert_eq!(
                    a[i] == a[j],
                    b[i] == b[j],
                    "co-assignment of points {i},{j} differs"
                );
            }
        }
    }

    #[test]
    fn train_with_index_f32_recovers_blobs() {
        assert_recovers_blobs(&blob_partition::<f32, f32>());
    }

    #[test]
    fn train_with_index_f16_recovers_blobs() {
        assert_recovers_blobs(&blob_partition::<half::f16, half::f16>());
    }

    #[test]
    fn train_with_index_f16_train_f32_centroids_recovers_blobs() {
        // Recommended scale path: f16 training corpus, f32 centroid index.
        assert_recovers_blobs(&blob_partition::<half::f16, f32>());
    }

    #[test]
    fn train_with_index_precisions_agree_on_partition() {
        let f32p = blob_partition::<f32, f32>();
        let f16p = blob_partition::<half::f16, half::f16>();
        let hybrid = blob_partition::<half::f16, f32>();
        assert_same_partition(&f32p, &f16p);
        assert_same_partition(&f32p, &hybrid);
    }

    // ---- centroid update: correctness ----------------------------------------

    #[test]
    fn update_and_split_matches_reference_means() {
        // Two non-empty clusters => no splits, so the returned centroids are
        // exactly the per-cluster means. Compare against hand-computed means.
        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(2);
        let mut builder = PlainDenseDatasetGrowable::new(encoder);
        let pts = [
            [0.0f32, 0.0],
            [2.0, 0.0],
            [1.0, 3.0], // cluster 0 -> mean [1, 1]
            [10.0, 10.0],
            [12.0, 10.0],
            [11.0, 13.0], // cluster 1 -> mean [11, 11]
        ];
        for p in pts.iter() {
            builder.push(DenseVectorView::new(&p[..]));
        }
        let dataset: PlainDenseDataset<f32, SquaredEuclideanDistance> = builder.into();

        let assignments = vec![
            (0.0, 0usize),
            (0.0, 0),
            (0.0, 0),
            (0.0, 1),
            (0.0, 1),
            (0.0, 1),
        ];
        let mut rng = StdRng::seed_from_u64(0);
        let (n_splits, hist, centroids): (_, _, Centroids<f32>) =
            KMeans::update_and_split(&dataset, None, 2, &assignments, &mut rng, false);

        assert_eq!(n_splits, 0, "no empty clusters => no splits");
        assert_eq!(hist, vec![3.0, 3.0]);

        let c0 = centroids.get(0 as VectorId).values().to_vec();
        let c1 = centroids.get(1 as VectorId).values().to_vec();
        for (a, b) in c0.iter().zip([1.0f32, 1.0].iter()) {
            assert!((a - b).abs() < 1e-5, "cluster 0 mean {a} != {b}");
        }
        for (a, b) in c1.iter().zip([11.0f32, 11.0].iter()) {
            assert!((a - b).abs() < 1e-5, "cluster 1 mean {a} != {b}");
        }
    }

    #[test]
    fn update_and_split_fires_split_on_empty_cluster() {
        // k=3 but assignments only ever use clusters 0 and 1 => cluster 2 is empty
        // and must be split off a populated one.
        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(2);
        let mut builder = PlainDenseDatasetGrowable::new(encoder);
        for p in [[0.0f32, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]].iter() {
            builder.push(DenseVectorView::new(&p[..]));
        }
        let dataset: PlainDenseDataset<f32, SquaredEuclideanDistance> = builder.into();
        let assignments = vec![(0.0, 0usize), (0.0, 0), (0.0, 1), (0.0, 1)];
        let mut rng = StdRng::seed_from_u64(0);
        let (n_splits, _hist, centroids): (_, _, Centroids<f32>) =
            KMeans::update_and_split(&dataset, None, 3, &assignments, &mut rng, false);
        assert_eq!(n_splits, 1, "one empty cluster => one split");
        assert_eq!(centroids.len(), 3);
    }

    #[test]
    fn update_deterministic_across_thread_counts() {
        // Non-integer random data so float reassociation would surface as bit diffs
        // (integer-valued data is exactly representable in f32 and hides reordering).
        let n = 2000usize;
        let d = 16usize;
        let k = 32usize;
        let mut data_rng = StdRng::seed_from_u64(42);
        let mut raw = vec![0.0f32; n * d];
        for x in raw.iter_mut() {
            *x = data_rng.gen_range(-1.0f32..1.0);
        }
        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(d);
        let mut builder = PlainDenseDatasetGrowable::new(encoder);
        for i in 0..n {
            builder.push(DenseVectorView::new(&raw[i * d..(i + 1) * d]));
        }
        let dataset: PlainDenseDataset<f32, SquaredEuclideanDistance> = builder.into();
        let assignments: Vec<(f32, usize)> = (0..n).map(|i| (0.0f32, i % k)).collect();

        let mut reference: Option<Vec<u32>> = None;
        for t in [1usize, 2, 4, 8] {
            let bits = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
                .install(|| {
                    let mut rng = StdRng::seed_from_u64(0);
                    let (_, _, centroids): (_, _, Centroids<f32>) =
                        KMeans::update_and_split(&dataset, None, k, &assignments, &mut rng, false);
                    (0..k)
                        .flat_map(|ci| {
                            centroids
                                .get(ci as VectorId)
                                .values()
                                .iter()
                                .map(|x| x.to_bits())
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<u32>>()
                });
            match &reference {
                None => reference = Some(bits),
                Some(ref_bits) => {
                    assert_eq!(&bits, ref_bits, "centroid bits differ at thread count {t}")
                }
            }
        }
    }

    #[test]
    fn update_weighted_means_match_reference() {
        // 4 points, d=2, k=2 with non-uniform weights.
        // cluster 0: pts 0,1 with weights 1,3 -> mean (1*[0,0] + 3*[2,0]) / 4 = [1.5, 0]
        // cluster 1: pts 2,3 with weights 2,2 -> mean (2*[10,10] + 2*[12,10]) / 4 = [11, 10]
        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(2);
        let mut builder = PlainDenseDatasetGrowable::new(encoder);
        for p in [[0.0f32, 0.0], [2.0, 0.0], [10.0, 10.0], [12.0, 10.0]].iter() {
            builder.push(DenseVectorView::new(&p[..]));
        }
        let dataset: PlainDenseDataset<f32, SquaredEuclideanDistance> = builder.into();
        let weights = [1.0f32, 3.0, 2.0, 2.0];
        let assignments = vec![(0.0, 0usize), (0.0, 0), (0.0, 1), (0.0, 1)];
        let mut rng = StdRng::seed_from_u64(0);
        let (n_splits, hist, centroids): (_, _, Centroids<f32>) =
            KMeans::update_and_split(&dataset, Some(&weights), 2, &assignments, &mut rng, false);

        assert_eq!(n_splits, 0);
        assert_eq!(hist, vec![4.0, 4.0]);
        let c0 = centroids.get(0 as VectorId).values().to_vec();
        let c1 = centroids.get(1 as VectorId).values().to_vec();
        for (a, b) in c0.iter().zip([1.5f32, 0.0].iter()) {
            assert!((a - b).abs() < 1e-5, "cluster 0 weighted mean {a} != {b}");
        }
        for (a, b) in c1.iter().zip([11.0f32, 10.0].iter()) {
            assert!((a - b).abs() < 1e-5, "cluster 1 weighted mean {a} != {b}");
        }
    }
}
