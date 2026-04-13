use clap::Parser;
use ndarray::Array1;
use ndarray_npy::write_npy;
use std::io::Write;
use std::time::Instant;

use indicatif::{ParallelProgressIterator, ProgressStyle};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use vectorium::dataset::ScoredVector;
use vectorium::distances::DotProduct;
use vectorium::readers;
use vectorium::{
    Dataset, DatasetGrowable, PlainSparseDataset, SpaceUsage, SparseDatasetGrowable,
    VariableBitUniformSparseQuantizer,
};

#[derive(Parser, Debug)]
#[clap(
    author,
    version,
    about = "Benchmark 4-bit uniform sparse quantization at various lower percentile values"
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

    /// Output TSV file path
    #[clap(short, long, default_value = "4bit_lower_percentile_results.tsv")]
    output: String,

    /// Directory where per-query recall .npy files are written
    #[clap(long, default_value = ".")]
    recall_dir: String,
}

/// Returns (mean_recall, per_query_recall).
fn recall_at_k(
    gt: &[Vec<ScoredVector<DotProduct>>],
    approx: &[Vec<ScoredVector<DotProduct>>],
    k: usize,
) -> (f64, Vec<f32>) {
    let per_query: Vec<f32> = gt
        .iter()
        .zip(approx.iter())
        .map(|(gt_row, approx_row)| {
            let gt_ids: std::collections::HashSet<u64> =
                gt_row.iter().take(k).map(|s| s.vector).collect();
            let found = approx_row
                .iter()
                .take(k)
                .filter(|s| gt_ids.contains(&s.vector))
                .count();
            found as f32 / k as f32
        })
        .collect();
    let mean = per_query.iter().map(|&r| r as f64).sum::<f64>() / per_query.len() as f64;
    (mean, per_query)
}

struct BenchResult {
    lower_percentile: f32,
    size_gib: f64,
    train_time: f64,
    build_time: f64,
    search_time: f64,
    recall: f64,
}

fn main() {
    let args = Args::parse();

    let pb_style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}, ETA: {eta})")
        .unwrap()
        .progress_chars("=>-");

    // ── Load dataset and queries ────────────────────────────────────
    println!("Loading dataset (u16 components)...");
    let dataset_f32: PlainSparseDataset<u16, f32, DotProduct> =
        readers::read_seismic_format(&args.input_file).expect("failed to read dataset");

    println!("Loading queries (u16 components)...");
    let queries: PlainSparseDataset<u16, f32, DotProduct> =
        readers::read_seismic_format(&args.query_file).expect("failed to read queries");

    let n = dataset_f32.len();
    let dim = dataset_f32.input_dim();
    let nnz = dataset_f32.nnz();
    let f32_size = dataset_f32.space_usage_GiB();
    let total_n_queries = queries.len();
    let n_queries = args.n_queries.min(total_n_queries);

    println!("Dataset: {n} docs, dim={dim}, nnz={nnz}");
    println!("Using {n_queries} queries out of {total_n_queries}");

    // ── Ground truth: <u16, f32> ────────────────────────────────────
    println!("\n=== <u16, f32> ground truth ===");
    println!("Size: {f32_size:.3} GiB");
    let start = Instant::now();
    let gt: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
        .into_par_iter()
        .progress_count(n_queries as u64)
        .with_style(pb_style.clone())
        .map(|qi| dataset_f32.search(queries.get(qi as u64), args.k))
        .collect();
    let search_time_f32 = start.elapsed().as_secs_f64();
    println!("Search: {search_time_f32:.3}s");

    // ── Training data ───────────────────────────────────────────────
    let training_data: PlainSparseDataset<u16, f32, vectorium::SquaredEuclideanDistance> =
        readers::read_seismic_format(&args.input_file).expect("failed to re-read for training");

    // ── Benchmark 4-bit with various lower percentile values ────────
    let lower_percentiles = vec![0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9];
    let mut results: Vec<BenchResult> = Vec::new();

    for lower_pct in lower_percentiles {
        println!(
            "\n=== 4-bit uniform quantization, lower_percentile={} ===",
            lower_pct
        );

        let start = Instant::now();
        let quantizer = VariableBitUniformSparseQuantizer::<u16, DotProduct>::train(
            &training_data,
            lower_pct,
            1.0,
            4,
        );
        let train_time = start.elapsed().as_secs_f64();
        println!("Train:  {train_time:.3}s");

        let start = Instant::now();
        let mut growable: SparseDatasetGrowable<
            VariableBitUniformSparseQuantizer<u16, DotProduct>,
        > = SparseDatasetGrowable::new(quantizer);
        for vec in dataset_f32.iter() {
            growable.push(vec);
        }
        let dataset_quantized: vectorium::VariableBitUniformSparseDataset<u16, DotProduct> =
            growable.into();
        let build_time = start.elapsed().as_secs_f64();
        let size_gib = dataset_quantized.space_usage_GiB();
        println!("Build:  {build_time:.3}s");
        println!("Size:   {size_gib:.3} GiB");

        let start = Instant::now();
        let approx: Vec<Vec<ScoredVector<DotProduct>>> = (0..n_queries)
            .into_par_iter()
            .progress_count(n_queries as u64)
            .with_style(pb_style.clone())
            .map(|qi| dataset_quantized.search(queries.get(qi as u64), args.k))
            .collect();
        let search_time = start.elapsed().as_secs_f64();
        println!("Search: {search_time:.3}s");

        let (recall, per_query_recall) = recall_at_k(&gt, &approx, args.k);
        println!("Recall@{}: {recall:.4}", args.k);

        // Save per-query recall as a numpy array.
        let npy_path = format!(
            "{}/recalls_4bit_lower{}.npy",
            args.recall_dir,
            lower_pct.to_string().replace(".", "_")
        );
        let arr = Array1::from(per_query_recall);
        write_npy(&npy_path, &arr).expect("failed to write per-query recall npy");
        println!("Per-query recall written to {npy_path}");

        results.push(BenchResult {
            lower_percentile: lower_pct as f32,
            size_gib,
            train_time,
            build_time,
            search_time,
            recall,
        });
    }

    // ── Summary table (stdout) ──────────────────────────────────────
    println!("\n=== Summary ===");
    println!(
        "{:<18} {:>10} {:>12} {:>12} {:>12} {:>10}",
        "lower_percentile", "Size GiB", "Train (s)", "Build (s)", "Search (s)", "Recall@k"
    );
    println!("{:-<80}", "");
    println!(
        "{:<18} {:>10.3} {:>12} {:>12} {:>12.3} {:>10}",
        "f32", f32_size, "-", "-", search_time_f32, "1.0000"
    );
    for r in &results {
        println!(
            "{:<18} {:>10.3} {:>12.3} {:>12.3} {:>12.3} {:>10.4}",
            r.lower_percentile, r.size_gib, r.train_time, r.build_time, r.search_time, r.recall
        );
    }

    // ── Write TSV ───────────────────────────────────────────────────
    let mut f = std::fs::File::create(&args.output).expect("failed to create output TSV");
    writeln!(
        f,
        "lower_percentile\tsize_gib\ttrain_time_s\tbuild_time_s\tsearch_time_s\trecall_at_k"
    )
    .unwrap();
    writeln!(f, "f32\t{f32_size:.6}\t\t\t{search_time_f32:.6}\t1.000000").unwrap();
    for r in &results {
        writeln!(
            f,
            "{:.2}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            r.lower_percentile, r.size_gib, r.train_time, r.build_time, r.search_time, r.recall
        )
        .unwrap();
    }
    println!("\nResults written to {}", args.output);
}
