use clap::Parser;
use ndarray::Array1;
use ndarray_npy::write_npy;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

use indicatif::{ParallelProgressIterator, ProgressStyle};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use vectorium::dataset::ScoredVector;
use vectorium::distances::DotProduct;
use vectorium::encoders::packed_variable_bit_uniform_quantization_sparse_scalar::PackedVariableBitUniformSparseQuantizer;
use vectorium::readers;
use vectorium::{
    Dataset, DatasetGrowable, PackedSparseDataset, PackedSparseDatasetGrowable,
    PlainSparseDataset, SpaceUsage, SparseDatasetGrowable, UniformSparseDataset,
    UniformSparseQuantizer,
};

#[derive(Parser, Debug)]
#[clap(
    author,
    version,
    about = "Benchmark variable-bit uniform sparse quantization (1..=8 bits)"
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
    #[clap(short, long, default_value = "variable_bit_quantization_results.tsv")]
    output: String,

    /// Directory where per-query recall .npy files are written (recalls_Xbits.npy).
    /// If omitted, per-query recall arrays are not saved.
    #[clap(long)]
    recall_dir: Option<String>,

    /// Optional precomputed f32 ground-truth TSV (format: query_id\tdoc_id\trank\tscore).
    /// When provided, the f32 search is skipped and recall is computed against this file.
    #[clap(long)]
    groundtruth_tsv: Option<String>,
}

/// Parse a ground-truth TSV (`query_id<TAB>doc_id<TAB>rank<TAB>score`) into a per-query
/// list of doc-ids. The order in which rows appear within a query is preserved, so callers
/// must take the first `k` entries to compare top-k.
fn load_groundtruth_tsv(path: &str, n_queries: usize) -> std::io::Result<Vec<Vec<u64>>> {
    let f = std::fs::File::open(path)?;
    let reader = BufReader::new(f);
    let mut gt: Vec<Vec<u64>> = vec![Vec::new(); n_queries];
    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let mut it = line.split('\t');
        let qid: usize = it
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("malformed query_id at line {}: {line}", line_no + 1));
        let doc_id: u64 = it
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("malformed doc_id at line {}: {line}", line_no + 1));
        if qid < n_queries {
            gt[qid].push(doc_id);
        }
    }
    Ok(gt)
}

/// Returns (mean_recall, per_query_recall). `gt` and `approx` are per-query top-k doc-id lists.
fn recall_at_k(gt: &[Vec<u64>], approx: &[Vec<u64>], k: usize) -> (f64, Vec<f32>) {
    let per_query: Vec<f32> = gt
        .iter()
        .zip(approx.iter())
        .map(|(gt_row, approx_row)| {
            let gt_ids: std::collections::HashSet<u64> = gt_row.iter().take(k).copied().collect();
            let found = approx_row
                .iter()
                .take(k)
                .filter(|d| gt_ids.contains(d))
                .count();
            found as f32 / k as f32
        })
        .collect();
    let mean = per_query.iter().map(|&r| r as f64).sum::<f64>() / per_query.len() as f64;
    (mean, per_query)
}

struct BenchResult {
    nbits: u8,
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
    // Either load a precomputed TSV (`--groundtruth-tsv`) or run the f32 exhaustive search.
    let (gt, search_time_f32): (Vec<Vec<u64>>, f64) = if let Some(gt_path) = &args.groundtruth_tsv {
        println!("\n=== Loading f32 ground truth from TSV ===");
        println!("Path: {gt_path}");
        let gt_full = load_groundtruth_tsv(gt_path, n_queries).expect("failed to read GT TSV");
        let missing: Vec<usize> = gt_full
            .iter()
            .enumerate()
            .filter_map(|(qi, row)| if row.len() < args.k { Some(qi) } else { None })
            .collect();
        if !missing.is_empty() {
            panic!(
                "Ground-truth TSV is short for {} queries (expected ≥ k={}). First missing: q{}",
                missing.len(),
                args.k,
                missing[0]
            );
        }
        println!("Loaded GT for {n_queries} queries (size: {f32_size:.3} GiB f32 index).");
        (gt_full, f64::NAN)
    } else {
        println!("\n=== <u16, f32> ground truth ===");
        println!("Size: {f32_size:.3} GiB");
        let start = Instant::now();
        let gt: Vec<Vec<u64>> = (0..n_queries)
            .into_par_iter()
            .progress_count(n_queries as u64)
            .with_style(pb_style.clone())
            .map(|qi| {
                dataset_f32
                    .search(queries.get(qi as u64), args.k)
                    .into_iter()
                    .map(|s: ScoredVector<DotProduct>| s.vector)
                    .collect()
            })
            .collect();
        let search_time_f32 = start.elapsed().as_secs_f64();
        println!("Search: {search_time_f32:.3}s");
        (gt, search_time_f32)
    };

    // ── Training data ───────────────────────────────────────────────
    let training_data: PlainSparseDataset<u16, f32, vectorium::SquaredEuclideanDistance> =
        readers::read_seismic_format(&args.input_file).expect("failed to re-read for training");

    // ── Benchmark each nbits from 1 to 8 ───────────────────────────
    let mut results: Vec<BenchResult> = Vec::new();

    for nbits in [1, 2, 3, 4, 5, 6, 7, 8] {
        println!("\n=== uniform quantization, nbits={nbits} ===");

        // Dispatch on bit width: nbits=8 → byte-stored UniformSparseQuantizer;
        // nbits ∈ [1,7] → PackedVariableBitUniformSparseQuantizer (bitpacked codes).
        let (size_gib, train_time, build_time, search_time, approx): (
            f64,
            f64,
            f64,
            f64,
            Vec<Vec<u64>>,
        ) = if nbits == 8 {
            let start = Instant::now();
            let quantizer = UniformSparseQuantizer::<u16, DotProduct>::train(
                &training_data,
                0.0,
                1.0,
            );
            let train_time = start.elapsed().as_secs_f64();
            println!("Train:  {train_time:.3}s");

            let start = Instant::now();
            let mut growable: SparseDatasetGrowable<UniformSparseQuantizer<u16, DotProduct>> =
                SparseDatasetGrowable::new(quantizer);
            for vec in dataset_f32.iter() {
                growable.push(vec);
            }
            let dataset_quantized: UniformSparseDataset<u16, DotProduct> = growable.into();
            let build_time = start.elapsed().as_secs_f64();
            let size_gib = dataset_quantized.space_usage_GiB();
            println!("Build:  {build_time:.3}s");
            println!("Size:   {size_gib:.3} GiB");

            let start = Instant::now();
            let approx: Vec<Vec<u64>> = (0..n_queries)
                .into_par_iter()
                .progress_count(n_queries as u64)
                .with_style(pb_style.clone())
                .map(|qi| {
                    dataset_quantized
                        .search(queries.get(qi as u64), args.k)
                        .into_iter()
                        .map(|s: ScoredVector<DotProduct>| s.vector)
                        .collect()
                })
                .collect();
            let search_time = start.elapsed().as_secs_f64();
            println!("Search: {search_time:.3}s");
            (size_gib, train_time, build_time, search_time, approx)
        } else {
            let start = Instant::now();
            let quantizer = PackedVariableBitUniformSparseQuantizer::train(
                &training_data,
                0.0,
                1.0,
                nbits,
            );
            let train_time = start.elapsed().as_secs_f64();
            println!("Train:  {train_time:.3}s");

            let start = Instant::now();
            let mut growable: PackedSparseDatasetGrowable<
                PackedVariableBitUniformSparseQuantizer,
            > = PackedSparseDatasetGrowable::new(quantizer);
            for vec in dataset_f32.iter() {
                growable.push(vec);
            }
            let dataset_quantized: PackedSparseDataset<PackedVariableBitUniformSparseQuantizer> =
                growable.into();
            let build_time = start.elapsed().as_secs_f64();
            let size_gib = dataset_quantized.space_usage_GiB();
            println!("Build:  {build_time:.3}s");
            println!("Size:   {size_gib:.3} GiB");

            let start = Instant::now();
            let approx: Vec<Vec<u64>> = (0..n_queries)
                .into_par_iter()
                .progress_count(n_queries as u64)
                .with_style(pb_style.clone())
                .map(|qi| {
                    dataset_quantized
                        .search(queries.get(qi as u64), args.k)
                        .into_iter()
                        .map(|s: ScoredVector<DotProduct>| s.vector)
                        .collect()
                })
                .collect();
            let search_time = start.elapsed().as_secs_f64();
            println!("Search: {search_time:.3}s");
            (size_gib, train_time, build_time, search_time, approx)
        };

        let (recall, per_query_recall) = recall_at_k(&gt, &approx, args.k);
        println!("Recall@{}: {recall:.4}", args.k);

        // Save per-query recall as a numpy array (only if --recall-dir is set).
        if let Some(recall_dir) = &args.recall_dir {
            let npy_path = format!("{}/recalls_{}bits.npy", recall_dir, nbits);
            let arr = Array1::from(per_query_recall);
            write_npy(&npy_path, &arr).expect("failed to write per-query recall npy");
            println!("Per-query recall written to {npy_path}");
        }

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
    let f32_search_str = if search_time_f32.is_nan() {
        "loaded".to_string()
    } else {
        format!("{:.3}", search_time_f32)
    };
    println!("\n=== Summary ===");
    println!(
        "{:<10} {:>10} {:>12} {:>12} {:>12} {:>10}",
        "nbits", "Size GiB", "Train (s)", "Build (s)", "Search (s)", "Recall@k"
    );
    println!("{:-<68}", "");
    println!(
        "{:<10} {:>10.3} {:>12} {:>12} {:>12} {:>10}",
        "f32", f32_size, "-", "-", f32_search_str, "1.0000"
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
    let f32_search_tsv = if search_time_f32.is_nan() {
        "loaded".to_string()
    } else {
        format!("{:.6}", search_time_f32)
    };
    writeln!(f, "f32\t{f32_size:.6}\t\t\t{f32_search_tsv}\t1.000000").unwrap();
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
