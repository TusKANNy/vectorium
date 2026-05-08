//! Single-thread search-efficiency benchmark for the 4-bit *centroid* (k-means)
//! sparse quantization kernel with the **f16-compressed query LUT**
//! (`PackedCentroidSparseQuantizerF16Lut`). Doc-side encoding and packed bytes
//! are byte-identical to the f32-LUT variant; only the query LUT and scoring
//! kernel differ. This bin exists so we can A/B against
//! `bench_4bit_efficiency_kmeans` (the f32-LUT baseline) on the same workload.
//!
//! Same CLI as the baseline so runs are directly comparable.

use clap::Parser;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

use vectorium::dataset::ScoredVector;
use vectorium::distances::DotProduct;
use vectorium::encoders::packed_centroid_based_quantization_sparse_scalar::{
    PackedCentroidSparseQuantizer, PackedCentroidSparseQuantizerF16Lut,
};
use vectorium::readers;
use vectorium::{
    Dataset, DatasetGrowable, PackedSparseDataset, PackedSparseDatasetGrowable, PlainSparseDataset,
    SpaceUsage, SquaredEuclideanDistance,
};

const NBITS: u8 = 4;

#[derive(Parser, Debug)]
#[clap(
    author,
    version,
    about = "4-bit centroid (k-means) sparse quantization search efficiency, f16 LUT (single-thread)"
)]
struct Args {
    #[clap(short, long)]
    input_file: String,

    #[clap(short, long)]
    query_file: String,

    #[clap(short, long, default_value_t = 10)]
    k: usize,

    #[clap(long, default_value_t = 1000)]
    n_queries: usize,

    #[clap(long, default_value_t = 100)]
    warmup: usize,

    #[clap(long, default_value_t = 3)]
    repeats: usize,

    #[clap(long, default_value_t = 10)]
    n_iterations: usize,

    #[clap(short, long, default_value = "kmeans_4bit_f16lut_efficiency.tsv")]
    output: String,

    #[clap(long)]
    groundtruth_tsv: String,

    #[clap(long, default_value_t = 0.0)]
    lower_percentile: f32,

    #[clap(long, default_value_t = 1.0)]
    upper_percentile: f32,
}

fn load_groundtruth_tsv(path: &str, n_queries: usize, k: usize) -> Vec<Vec<u64>> {
    let f = std::fs::File::open(path).expect("failed to open groundtruth TSV");
    let reader = BufReader::new(f);
    let mut gt: Vec<Vec<u64>> = vec![Vec::new(); n_queries];
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.expect("failed to read groundtruth line");
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
    let missing: Option<usize> = gt.iter().position(|row| row.len() < k);
    if let Some(qi) = missing {
        panic!(
            "Ground-truth TSV is short for query {qi} (expected ≥ k={k}, got {})",
            gt[qi].len()
        );
    }
    gt
}

fn recall_at_k(gt: &[Vec<u64>], approx: &[Vec<u64>], k: usize) -> f64 {
    let total: f64 = gt
        .iter()
        .zip(approx.iter())
        .map(|(g, a)| {
            let gset: std::collections::HashSet<u64> = g.iter().take(k).copied().collect();
            let found = a.iter().take(k).filter(|d| gset.contains(d)).count();
            found as f64 / k as f64
        })
        .sum();
    total / gt.len() as f64
}

fn time_search_serial(
    dataset: &PackedSparseDataset<PackedCentroidSparseQuantizerF16Lut>,
    queries: &PlainSparseDataset<u16, f32, DotProduct>,
    n_queries: usize,
    k: usize,
) -> (f64, Vec<Vec<u64>>) {
    let mut approx: Vec<Vec<u64>> = Vec::with_capacity(n_queries);
    let t0 = Instant::now();
    for qi in 0..n_queries {
        let res: Vec<u64> = dataset
            .search(queries.get(qi as u64), k)
            .into_iter()
            .map(|s: ScoredVector<DotProduct>| s.vector)
            .collect();
        approx.push(res);
    }
    (t0.elapsed().as_secs_f64(), approx)
}

fn main() {
    let args = Args::parse();

    println!("Loading dataset (u16 components)...");
    let dataset_f32: PlainSparseDataset<u16, f32, DotProduct> =
        readers::read_seismic_format(&args.input_file).expect("failed to read dataset");

    println!("Loading queries (u16 components)...");
    let queries: PlainSparseDataset<u16, f32, DotProduct> =
        readers::read_seismic_format(&args.query_file).expect("failed to read queries");

    let n = dataset_f32.len();
    let dim = dataset_f32.input_dim();
    let nnz = dataset_f32.nnz();
    let total_n_queries = queries.len();
    let n_queries = args.n_queries.min(total_n_queries);
    let warmup = args.warmup.min(n_queries);

    println!("Dataset: {n} docs, dim={dim}, nnz={nnz}");
    println!("Queries: timed={n_queries}, warmup={warmup} (of {total_n_queries} available)");
    println!(
        "Repeats: {} | single-threaded | Lloyd iters: {} | LUT precision: f16",
        args.repeats, args.n_iterations
    );

    println!(
        "\n=== Training PackedCentroidSparseQuantizer @ nbits={NBITS} \
         (Lloyd iters={}) ===",
        args.n_iterations
    );
    let training_data: PlainSparseDataset<u16, f32, SquaredEuclideanDistance> =
        readers::read_seismic_format(&args.input_file).expect("failed to re-read for training");
    let train_t = Instant::now();
    let inner = PackedCentroidSparseQuantizer::train(
        &training_data,
        args.lower_percentile,
        args.upper_percentile,
        args.n_iterations,
    );
    let quantizer = PackedCentroidSparseQuantizerF16Lut::from_inner(inner);
    let train_time = train_t.elapsed().as_secs_f64();
    drop(training_data);
    println!("Train: {train_time:.3}s");

    let build_t = Instant::now();
    let mut growable: PackedSparseDatasetGrowable<PackedCentroidSparseQuantizerF16Lut> =
        PackedSparseDatasetGrowable::new(quantizer);
    for vec in dataset_f32.iter() {
        growable.push(vec);
    }
    let dataset_quantized: PackedSparseDataset<PackedCentroidSparseQuantizerF16Lut> =
        growable.into();
    let build_time = build_t.elapsed().as_secs_f64();
    let size_gib = dataset_quantized.space_usage_GiB();
    println!("Build: {build_time:.3}s | Index size: {size_gib:.3} GiB");

    println!("\nLoading ground truth from {}...", args.groundtruth_tsv);
    let gt = load_groundtruth_tsv(&args.groundtruth_tsv, n_queries, args.k);

    if warmup > 0 {
        println!("\n=== Warmup ({warmup} queries) ===");
        let _ = time_search_serial(&dataset_quantized, &queries, warmup, args.k);
    }

    println!(
        "\n=== Timing (search top-{}, repeats={}) ===",
        args.k, args.repeats
    );
    let mut best_total: Option<f64> = None;
    let mut best_approx: Option<Vec<Vec<u64>>> = None;
    for r in 0..args.repeats {
        let (total_s, approx) = time_search_serial(&dataset_quantized, &queries, n_queries, args.k);
        println!(
            "  repeat {r}: total={total_s:.3}s  mean={mean:.1}µs  QPS={qps:.1}",
            mean = (total_s / n_queries as f64) * 1e6,
            qps = n_queries as f64 / total_s,
        );
        if best_total.map(|t| total_s < t).unwrap_or(true) {
            best_total = Some(total_s);
            best_approx = Some(approx);
        }
    }

    let best_total = best_total.unwrap();
    let best_approx = best_approx.unwrap();
    let mean_us = (best_total / n_queries as f64) * 1e6;
    let qps = n_queries as f64 / best_total;

    let recall = recall_at_k(&gt, &best_approx, args.k);

    println!(
        "\n=== Summary (best of {} repeats, single-thread, f16 LUT) ===",
        args.repeats
    );
    println!("  total          {best_total:.3} s");
    println!("  mean latency   {mean_us:.1} µs / query");
    println!("  QPS            {qps:.1}");
    println!("  Recall@{}      {recall:.4}", args.k);

    let mut f = std::fs::File::create(&args.output).expect("failed to create output TSV");
    writeln!(
        f,
        "encoder\tnbits\tn_iterations\tn_docs\tdim\tnnz\tindex_size_gib\ttrain_time_s\tbuild_time_s\t\
         n_queries\twarmup\trepeats\tk\ttotal_s\tmean_us\tqps\trecall_at_k"
    )
    .unwrap();
    writeln!(
        f,
        "kmeans_4bit_f16lut\t{NBITS}\t{niter}\t{n}\t{dim}\t{nnz}\t{size_gib:.6}\t{train_time:.6}\t{build_time:.6}\t\
         {nq}\t{warmup}\t{rep}\t{k}\t{best_total:.6}\t{mean_us:.3}\t{qps:.3}\t{recall:.6}",
        niter = args.n_iterations,
        nq = n_queries,
        rep = args.repeats,
        k = args.k,
    )
    .unwrap();
    println!("\nResults written to {}", args.output);
}
