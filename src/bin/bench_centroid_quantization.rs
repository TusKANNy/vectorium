use clap::{Parser, ValueEnum};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::time::Instant;

use indicatif::{ParallelProgressIterator, ProgressStyle};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use vectorium::Distance;
use vectorium::dataset::{ConvertInto, ScoredVector};
use vectorium::distances::DotProduct;
use vectorium::readers;
use vectorium::{
    CentroidSparseQuantizer, Dataset, DatasetGrowable, FixedU8Q, PlainSparseDataset,
    ScalarSparseDataset, SpaceUsage, SparseDatasetGrowable,
};

#[derive(Debug, Clone, ValueEnum)]
enum Mode {
    /// Compute f32 ground truth and save to file
    F32,
    /// Run FixedU8 scalar quantization search
    Fixedu8,
    /// Run centroid-based quantization search
    Centroid,
}

#[derive(Parser, Debug)]
#[clap(
    author,
    version,
    about = "Benchmark individual quantization methods for perf profiling"
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

    /// Which method to benchmark
    #[clap(short, long)]
    mode: Mode,

    /// Ground truth file path (written by f32 mode, read by others)
    #[clap(short, long, default_value = "gt.txt")]
    gt_file: String,
}

/// Write ground truth results to a text file.
/// Format: one query per line, space-separated "vector_id:distance" pairs.
fn save_gt(path: &str, gt: &[Vec<ScoredVector<DotProduct>>]) {
    let file = File::create(path).expect("failed to create gt file");
    let mut writer = BufWriter::new(file);
    for row in gt {
        let line: Vec<String> = row
            .iter()
            .map(|s| format!("{}:{}", s.vector, s.distance.distance()))
            .collect();
        writeln!(writer, "{}", line.join(" ")).expect("failed to write gt");
    }
}

/// Load ground truth results from a text file.
fn load_gt(path: &str) -> Vec<Vec<ScoredVector<DotProduct>>> {
    let file = File::open(path).expect("failed to open gt file");
    let reader = BufReader::new(file);
    reader
        .lines()
        .map(|line| {
            let line = line.expect("failed to read gt line");
            line.split_whitespace()
                .map(|token| {
                    let mut parts = token.split(':');
                    let vector: u64 = parts.next().unwrap().parse().unwrap();
                    let dist: f32 = parts.next().unwrap().parse().unwrap();
                    ScoredVector {
                        vector,
                        distance: DotProduct::from(dist),
                    }
                })
                .collect()
        })
        .collect()
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

fn pb_style() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}, ETA: {eta})")
        .unwrap()
        .progress_chars("=>-")
}

fn main() {
    let args = Args::parse();

    // Load dataset and queries
    println!("Loading dataset...");
    let dataset_f32: PlainSparseDataset<u16, f32, DotProduct> =
        readers::read_seismic_format(&args.input_file).expect("failed to read dataset");
    let queries: PlainSparseDataset<u16, f32, DotProduct> =
        readers::read_seismic_format(&args.query_file).expect("failed to read queries");

    let n = dataset_f32.len();
    let dim = dataset_f32.input_dim();
    let nnz = dataset_f32.nnz();
    let total_n_queries = queries.len();
    let n_queries = args.n_queries.min(total_n_queries);

    println!("Dataset: {n} docs, dim={dim}, nnz={nnz}");
    println!("Using {n_queries} queries out of {total_n_queries}");
    println!(
        "Dataset size (f32): {:.3} GiB",
        dataset_f32.space_usage_GiB()
    );

    let style = pb_style();

    match args.mode {
        Mode::F32 => {
            println!("\n=== f32 ground truth ===");
            let start = Instant::now();
            let gt: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
                .into_par_iter()
                .progress_count(n_queries as u64)
                .with_style(style)
                .map(|qi| dataset_f32.search(queries.get(qi as u64), args.k))
                .collect();
            println!("f32 search: {:.3}s", start.elapsed().as_secs_f64());

            save_gt(&args.gt_file, &gt);
            println!("Ground truth saved to {}", args.gt_file);
        }

        Mode::Fixedu8 => {
            println!("\n=== FixedU8 scalar quantization ===");
            let start = Instant::now();
            let dataset_fixedu8: ScalarSparseDataset<u16, f32, FixedU8Q, DotProduct> =
                (&dataset_f32).convert_into();
            println!("FixedU8 build: {:.3}s", start.elapsed().as_secs_f64());
            println!(
                "FixedU8 size: {:.3} GiB",
                dataset_fixedu8.space_usage_GiB()
            );

            let start = Instant::now();
            let results: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
                .into_par_iter()
                .progress_count(n_queries as u64)
                .with_style(style)
                .map(|qi| dataset_fixedu8.search(queries.get(qi as u64), args.k))
                .collect();
            println!("FixedU8 search: {:.3}s", start.elapsed().as_secs_f64());

            let gt = load_gt(&args.gt_file);
            let recall = recall_at_k(&gt, &results, args.k);
            println!("Recall@{}: {:.4}", args.k, recall);
        }

        Mode::Centroid => {
            println!("\n=== Centroid quantization (uniform centroids) ===");

            let training_data: PlainSparseDataset<u16, f32, vectorium::SquaredEuclideanDistance> =
                readers::read_seismic_format(&args.input_file)
                    .expect("failed to re-read for training");

            let start = Instant::now();
            let quantizer =
                CentroidSparseQuantizer::<u16, DotProduct>::train(&training_data, 0.0, 1.0);
            println!("Centroid train: {:.3}s", start.elapsed().as_secs_f64());
            drop(training_data);

            let start = Instant::now();
            let mut growable: SparseDatasetGrowable<CentroidSparseQuantizer<u16, DotProduct>> =
                SparseDatasetGrowable::new(quantizer);
            for vec in dataset_f32.iter() {
                growable.push(vec);
            }
            let dataset_centroid: vectorium::CentroidSparseDataset<u16, DotProduct> =
                growable.into();
            println!("Centroid build: {:.3}s", start.elapsed().as_secs_f64());
            println!(
                "Centroid size: {:.3} GiB",
                dataset_centroid.space_usage_GiB()
            );

            let start = Instant::now();
            let results: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
                .into_par_iter()
                .progress_count(n_queries as u64)
                .with_style(style)
                .map(|qi| dataset_centroid.search(queries.get(qi as u64), args.k))
                .collect();
            println!("Centroid search: {:.3}s", start.elapsed().as_secs_f64());

            let gt = load_gt(&args.gt_file);
            let recall = recall_at_k(&gt, &results, args.k);
            println!("Recall@{}: {:.4}", args.k, recall);
        }
    }
}
