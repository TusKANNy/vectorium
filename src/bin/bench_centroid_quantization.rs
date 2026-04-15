use clap::Parser;
use std::io::Write;
use std::time::Instant;

use indicatif::{ParallelProgressIterator, ProgressStyle};
use ndarray::Array2;
use ndarray_npy::write_npy;
use num_traits::AsPrimitive;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use vectorium::dataset::ScoredVector;
use vectorium::distances::DotProduct;
use vectorium::readers;
use vectorium::{
    CentroidSparseQuantizer, Dataset, DatasetGrowable, PlainSparseDataset, SpaceUsage,
    SparseDatasetGrowable, VectorEncoder,
};

#[derive(Parser, Debug)]
#[clap(
    author,
    version,
    about = "Benchmark centroid-based sparse quantization (1..=8 bits)"
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

    /// Number of k-means iterations for centroid training
    #[clap(long, default_value_t = 10)]
    n_iterations: usize,

    /// Number of bits for quantization (1-8). If not specified, runs all bits from 1 to 8
    #[clap(long)]
    nbits: Option<u8>,

    /// Output TSV file path
    #[clap(short, long, default_value = "centroid_quantization_results.tsv")]
    output: String,
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

struct BenchResult {
    nbits: u8,
    size_gib: f64,
    train_time: f64,
    build_time: f64,
    search_time: f64,
    recall: f64,
}

fn save_centroids(
    quantizer: &CentroidSparseQuantizer<u16, DotProduct>,
    stage: &str,
    nbits: u8,
) {
    let num_centroids = quantizer.num_centroids();
    let dim = quantizer.output_dim();

    // Create a 2D array: (dim, num_centroids)
    let mut centroids_data = vec![0.0f32; dim * num_centroids];
    for d in 0..dim {
        let dim_centroids = quantizer.centroids_for_dim(d);
        for (idx, &val) in dim_centroids.iter().enumerate() {
            centroids_data[d * num_centroids + idx] = val;
        }
    }

    let centroids_array = Array2::from_shape_vec((dim, num_centroids), centroids_data)
        .expect("shape error");

    let filename = format!("centroids_{}_{}.npy", stage, nbits);
    write_npy(&filename, &centroids_array).expect("failed to write npy file");
    println!("Saved centroids to {}", filename);
}

fn save_initial_uniform_centroids(
    training_data: &PlainSparseDataset<u16, f32, vectorium::SquaredEuclideanDistance>,
    nbits: u8,
) {
    let num_centroids = 1usize << nbits;
    let dim = training_data.output_dim();

    // Collect per-component min/max
    let mut min_vals = vec![f32::MAX; dim];
    let mut max_vals = vec![f32::MIN; dim];

    for doc in training_data.iter() {
        for (&c, &v) in doc.components().iter().zip(doc.values()) {
            let idx: usize = c.as_();
            min_vals[idx] = min_vals[idx].min(v);
            max_vals[idx] = max_vals[idx].max(v);
        }
    }

    // Create uniform centroids
    let mut centroids_data = vec![0.0f32; dim * num_centroids];
    for d in 0..dim {
        let span = (max_vals[d] - min_vals[d]) / (num_centroids as f32);
        for i in 0..num_centroids {
            centroids_data[d * num_centroids + i] = min_vals[d] + span * i as f32;
        }
    }

    let centroids_array = Array2::from_shape_vec((dim, num_centroids), centroids_data)
        .expect("shape error");

    let filename = format!("centroids_initial_{}.npy", nbits);
    write_npy(&filename, &centroids_array).expect("failed to write npy file");
    println!("Saved initial centroids to {}", filename);
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

    // ── Benchmark each nbits from 1 to 8 ───────────────────────────
    let mut results: Vec<BenchResult> = Vec::new();

    let nbits_range: Box<dyn Iterator<Item = u8>> = if let Some(nbits_val) = args.nbits {
        Box::new(std::iter::once(nbits_val))
    } else {
        Box::new(1..=8u8)
    };

    for nbits in nbits_range {
        println!(
            "\n=== centroid quantization, nbits={nbits} (2^{nbits}={} centroids) ===",
            1u32 << nbits
        );

        // Save initial uniform centroids
        save_initial_uniform_centroids(&training_data, nbits);

        let start = Instant::now();
        let quantizer = CentroidSparseQuantizer::<u16, DotProduct>::train(
            &training_data,
            0.0,
            1.0,
            nbits,
            args.n_iterations,
        );
        let train_time = start.elapsed().as_secs_f64();
        println!("Train:  {train_time:.3}s");

        // Save final centroids
        save_centroids(&quantizer, "final", nbits);

        let start = Instant::now();
        let mut growable: SparseDatasetGrowable<CentroidSparseQuantizer<u16, DotProduct>> =
            SparseDatasetGrowable::new(quantizer);
        for vec in dataset_f32.iter() {
            growable.push(vec);
        }
        let dataset_quantized: vectorium::CentroidSparseDataset<u16, DotProduct> = growable.into();
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

        let recall = recall_at_k(&gt, &approx, args.k);
        println!("Recall@{}: {recall:.4}", args.k);

        results.push(BenchResult {
            nbits,
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
        "{:<10} {:>10} {:>12} {:>12} {:>12} {:>10}",
        "nbits", "Size GiB", "Train (s)", "Build (s)", "Search (s)", "Recall@k"
    );
    println!("{:-<68}", "");
    println!(
        "{:<10} {:>10.3} {:>12} {:>12} {:>12.3} {:>10}",
        "f32", f32_size, "-", "-", search_time_f32, "1.0000"
    );
    for r in &results {
        println!(
            "{:<10} {:>10.3} {:>12.3} {:>12.3} {:>12.3} {:>10.4}",
            r.nbits, r.size_gib, r.train_time, r.build_time, r.search_time, r.recall
        );
    }

    // ── Write TSV ───────────────────────────────────────────────────
    let mut f = std::fs::File::create(&args.output).expect("failed to create output TSV");
    writeln!(
        f,
        "nbits\tsize_gib\ttrain_time_s\tbuild_time_s\tsearch_time_s\trecall_at_k"
    )
    .unwrap();
    writeln!(f, "f32\t{f32_size:.6}\t\t\t{search_time_f32:.6}\t1.000000").unwrap();
    for r in &results {
        writeln!(
            f,
            "{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            r.nbits, r.size_gib, r.train_time, r.build_time, r.search_time, r.recall
        )
        .unwrap();
    }
    println!("\nResults written to {}", args.output);
}
