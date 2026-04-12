use clap::Parser;
use std::time::Instant;

use indicatif::{ParallelProgressIterator, ProgressStyle};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use vectorium::dataset::{ConvertInto, ScoredVector};
use vectorium::distances::DotProduct;
use vectorium::readers;
use vectorium::{
    CentroidSparseQuantizer, Dataset, DatasetGrowable, FixedU8Q, PlainSparseDataset,
    ScalarSparseDataset, SpaceUsage, SparseDatasetGrowable, UniformSparseQuantizer,
};

#[derive(Parser, Debug)]
#[clap(
    author,
    version,
    about = "Compare quantization methods (uniform, reverse-exp, FixedU8) for sparse dot product"
)]
struct Args {
    /// Sparse dataset in Seismic binary format
    #[clap(short, long)]
    input_file: String,

    /// Sparse queries in Seismic binary format
    #[clap(short, long)]
    query_file: String,

    /// Number of top results
    #[clap(short, long, default_value_t = 10)]
    k: usize,

    /// Number of queries to use
    #[clap(long, default_value_t = 1000)]
    n_queries: usize,
}

fn recall_at_k(
    gt: &[Vec<ScoredVector<DotProduct>>],
    approx: &[Vec<ScoredVector<DotProduct>>],
    k: usize,
) -> f64 {
    let mut total_recall = 0.0;
    for (gt_row, approx_row) in gt.iter().zip(approx.iter()) {
        let gt_ids: std::collections::HashSet<u64> =
            gt_row.iter().take(k).map(|s| s.vector).collect();
        let found = approx_row
            .iter()
            .take(k)
            .filter(|s| gt_ids.contains(&s.vector))
            .count();
        total_recall += found as f64 / k as f64;
    }
    total_recall / gt.len() as f64
}

fn main() {
    let args = Args::parse();

    // Load dataset and queries as f32 (ground truth)
    println!("Loading dataset...");
    let dataset_f32: PlainSparseDataset<u16, f32, DotProduct> =
        readers::read_seismic_format(&args.input_file).expect("failed to read dataset");
    let queries: PlainSparseDataset<u16, f32, DotProduct> =
        readers::read_seismic_format(&args.query_file).expect("failed to read queries");

    let n = dataset_f32.len();
    let dim = dataset_f32.input_dim();
    let nnz = dataset_f32.nnz();
    let n_queries = args.n_queries;
    let total_n_queries = queries.len();
    let n_queries = n_queries.min(total_n_queries);

    println!("Using {n_queries} queries out of {total_n_queries}");

    println!("Dataset: {n} docs, dim={dim}, nnz={nnz}");
    println!("Queries: {total_n_queries}");
    println!(
        "Dataset size (f32): {:.3} GiB",
        dataset_f32.space_usage_GiB()
    );

    let pb_style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}, ETA: {eta})")
        .unwrap()
        .progress_chars("=>-");

    // ── Ground truth: f32 ──────────────────────────────────────────
    println!("\n--- f32 ground truth ---");
    let start = Instant::now();
    let gt: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
        .into_par_iter()
        .progress_count(n_queries as u64)
        .with_style(pb_style.clone())
        .map(|qi| dataset_f32.search(queries.get(qi as u64), args.k))
        .collect();
    println!("f32 search: {:.3}s", start.elapsed().as_secs_f64());

    // ── FixedU8 scalar quantization ────────────────────────────────
    println!("\n--- FixedU8 scalar quantization ---");
    let start = Instant::now();
    let dataset_fixedu8: ScalarSparseDataset<u16, f32, FixedU8Q, DotProduct> =
        (&dataset_f32).convert_into();
    println!("FixedU8 build: {:.3}s", start.elapsed().as_secs_f64());
    println!("FixedU8 size: {:.3} GiB", dataset_fixedu8.space_usage_GiB());

    let start = Instant::now();
    let results_fixedu8: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
        .into_par_iter()
        .progress_count(n_queries as u64)
        .with_style(pb_style.clone())
        .map(|qi| dataset_fixedu8.search(queries.get(qi as u64), args.k))
        .collect();
    println!("FixedU8 search: {:.3}s", start.elapsed().as_secs_f64());

    // ── Training data (shared by uniform and reverse-exp) ──────────
    let training_data: PlainSparseDataset<u16, f32, vectorium::SquaredEuclideanDistance> =
        readers::read_seismic_format(&args.input_file).expect("failed to re-read for training");

    let start = Instant::now();
    let quantizer = UniformSparseQuantizer::<u16, DotProduct>::train(&training_data, 0.0, 1.0);
    println!("  train: {:.3}s", start.elapsed().as_secs_f64());

    let start = Instant::now();
    let mut growable: SparseDatasetGrowable<UniformSparseQuantizer<u16, DotProduct>> =
        SparseDatasetGrowable::new(quantizer);
    for vec in dataset_f32.iter() {
        growable.push(vec);
    }
    let dataset_uniform: vectorium::UniformSparseDataset<u16, DotProduct> = growable.into();
    println!("  build: {:.3}s", start.elapsed().as_secs_f64());
    println!("  size:  {:.3} GiB", dataset_uniform.space_usage_GiB());

    let start = Instant::now();
    let results_uniform: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
        .into_par_iter()
        .progress_count(n_queries as u64)
        .with_style(pb_style.clone())
        .map(|qi| dataset_uniform.search(queries.get(qi as u64), args.k))
        .collect();
    println!("  search: {:.3}s", start.elapsed().as_secs_f64());

    // ── Centroid-based quantization (uniform centroids) ─────────────
    println!("\n--- Centroid quantization (Greedy KMeans) ---");
    for n_iter in [1, 5, 10, 20, 100] {
        let start = Instant::now();
        let centroid_quantizer =
            CentroidSparseQuantizer::<u16, DotProduct>::train(&training_data, 0.0, 1.0, 8, n_iter);
        println!("Centroid train: {:.3}s", start.elapsed().as_secs_f64());

        let start = Instant::now();
        let mut growable: SparseDatasetGrowable<CentroidSparseQuantizer<u16, DotProduct>> =
            SparseDatasetGrowable::new(centroid_quantizer);
        for vec in dataset_f32.iter() {
            growable.push(vec);
        }
        let dataset_centroid: vectorium::CentroidSparseDataset<u16, DotProduct> = growable.into();
        println!(
            "Centroid build (MultiThread): {:.3}s",
            start.elapsed().as_secs_f64()
        );
        println!(
            "Centroid size: {:.3} GiB",
            dataset_centroid.space_usage_GiB()
        );

        let start = Instant::now();
        let results_centroid: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
            .into_par_iter()
            .progress_count(n_queries as u64)
            .with_style(pb_style.clone())
            .map(|qi| dataset_centroid.search(queries.get(qi as u64), args.k))
            .collect();
        println!("Centroid search: {:.3}s", start.elapsed().as_secs_f64());
        let recall_centroid = recall_at_k(&gt, &results_centroid, args.k);
        println!("\nCentroid:                   {:.4}", recall_centroid);
        println!("Centroid: {:.3}", dataset_centroid.space_usage_GiB());
    }
    // ── Reverse exponential quantization ────────────────────────────
    // println!("\n--- Reverse exponential quantization ---");

    // let start = Instant::now();
    // let rev_quantizer = ReverseExpSparseQuantizer::<u16, DotProduct>::train(&training_data);
    // println!("RevExp train: {:.3}s", start.elapsed().as_secs_f64());
    // drop(training_data);

    // let start = Instant::now();
    // let mut growable: SparseDatasetGrowable<ReverseExpSparseQuantizer<u16, DotProduct>> =
    //     SparseDatasetGrowable::new(rev_quantizer);
    // for vec in dataset_f32.iter() {
    //     growable.push(vec);
    // }
    // let dataset_revexp: vectorium::ReverseExpSparseDataset<u16, DotProduct> = growable.into();
    // println!("RevExp build: {:.3}s", start.elapsed().as_secs_f64());
    // println!("RevExp size: {:.3} GiB", dataset_revexp.space_usage_GiB());

    // let start = Instant::now();
    // let results_revexp: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
    //     .into_par_iter()
    //     .progress_count(n_queries as u64)
    //     .with_style(pb_style.clone())
    //     .map(|qi| dataset_revexp.search(queries.get(qi as u64), args.k))
    //     .collect();
    // println!("RevExp search: {:.3}s", start.elapsed().as_secs_f64());

    // ── Results ────────────────────────────────────────────────────
    println!("\n=== Recall@{} ===", args.k);
    let recall_fixedu8 = recall_at_k(&gt, &results_fixedu8, args.k);
    println!("FixedU8:                    {:.4}", recall_fixedu8);
    let recall_uniform = recall_at_k(&gt, &results_uniform, args.k);
    println!("Uniform:                    {:.4}", recall_uniform);
    // println!("\n  -- Varying lower percentile (upper=1.0) --");
    // for (pct, results) in &uniform_lower_results {
    //     let recall = recall_at_k(&gt, results, args.k);
    //     println!("  Uniform lo={pct:.2} up=1.00:  {:.4}", recall);
    // }
    // println!("\n  -- Varying upper percentile (lower=0.0) --");
    // for (pct, results) in &uniform_upper_results {
    //     let recall = recall_at_k(&gt, results, args.k);
    //     println!("  Uniform lo=0.00 up={pct:.3}: {:.4}", recall);
    // }

    //let recall_revexp = recall_at_k(&gt, &results_revexp, args.k);
    //println!("RevExp:                     {:.4}", recall_revexp);

    println!("\n=== Space (GiB) ===");
    println!("f32:      {:.3}", dataset_f32.space_usage_GiB());
    println!("FixedU8:  {:.3}", dataset_fixedu8.space_usage_GiB());
    println!("Uniform:  {:.3}", dataset_uniform.space_usage_GiB());

    //println!("RevExp:   {:.3}", dataset_revexp.space_usage_GiB());
}
