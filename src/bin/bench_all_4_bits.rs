//! Single-thread search-efficiency comparison of the two `dotpacking8` 4-bit encoders
//! against the `<u16, f16>` blocked-sparse baseline, in one run over the same dataset,
//! queries, and ground truth.
//!
//! Encoders compared (all exhaustive top-k via [`Dataset::search`], single-threaded):
//!   - `blocked_f16`      — baseline [`BlockedSparseDataset`]: u16 components + f16 values
//!                          packed into cache-line `DataBlock`s (no component compression).
//!   - `dot_scalar4bit`   — [`DotPacking8Scalar4BitEncoder`]: dotpacking8 gap-coded
//!                          components + 4-bit uniform-scalar nibble values.
//!   - `dot_centroid4bit` — [`DotPacking8Centroid4BitEncoder`]: same components + 4-bit
//!                          per-component centroid (codebook) nibble values.
//!
//! For each encoder we report train/build time, index size, mean query latency, QPS, and
//! recall@k vs a precomputed f32 ground truth. The timed loop is repeated `--repeats`
//! times and the best (smallest total) repeat is kept. Mirrors the warmup / repeats /
//! recall / TSV conventions of `bench_4bit_efficiency_scalar.rs` and
//! `bench_blocked_sparse_vs_f16.rs` so numbers are directly comparable.

use clap::Parser;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

use vectorium::dataset::ScoredVector;
use vectorium::distances::DotProduct;
use vectorium::readers;
use vectorium::{
    BlockedSparseDataset, Dataset, DatasetGrowable, DotPacking8Centroid4BitEncoder,
    DotPacking8Scalar4BitEncoder, PackedSparseDataset, PackedSparseDatasetGrowable,
    PlainSparseDataset, SpaceUsage, SquaredEuclideanDistance,
};

#[derive(Parser, Debug)]
#[clap(
    author,
    version,
    about = "Compare dotpacking8 scalar-4bit and centroid-4bit vs the <u16,f16> blocked baseline (single-thread)"
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

    /// Number of queries actually timed
    #[clap(long, default_value_t = 1000)]
    n_queries: usize,

    /// Queries used to warm caches/TLB before any timing starts.
    #[clap(long, default_value_t = 100)]
    warmup: usize,

    /// Number of times the timed loop is repeated. We keep the best repeat.
    #[clap(long, default_value_t = 3)]
    repeats: usize,

    /// Output TSV file path
    #[clap(short, long, default_value = "all_4bit_efficiency.tsv")]
    output: String,

    /// Precomputed f32 ground-truth TSV (format: query_id\tdoc_id\trank\tscore).
    /// Required: recall@k is computed against this file.
    #[clap(long)]
    groundtruth_tsv: String,

    /// Lower percentile clip used during centroid training.
    #[clap(long, default_value_t = 0.0)]
    lower_percentile: f32,

    /// Upper percentile clip used during centroid training.
    #[clap(long, default_value_t = 1.0)]
    upper_percentile: f32,

    /// Lloyd iterations for the centroid (k-means) quantizer.
    #[clap(long, default_value_t = 10)]
    n_iterations: usize,
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

/// One encoder's measured results.
struct Row {
    label: &'static str,
    train_s: f64,
    build_s: f64,
    size_gib: f64,
    total_s: f64,
    mean_us: f64,
    qps: f64,
    recall: f64,
}

fn main() {
    let args = Args::parse();

    // ── Load search dataset, queries, ground truth ──────────────────
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

    println!(
        "Dataset: {n} docs, dim={dim}, nnz={nnz} (mean {:.1} nnz/doc)",
        nnz as f64 / n as f64
    );
    println!("Queries: timed={n_queries}, warmup={warmup} (of {total_n_queries} available)");
    println!("Repeats: {} | single-threaded | k={}", args.repeats, args.k);

    println!("\nLoading ground truth from {}...", args.groundtruth_tsv);
    let gt = load_groundtruth_tsv(&args.groundtruth_tsv, n_queries, args.k);

    let mut rows: Vec<Row> = Vec::new();

    // Build a dataset, warm it, time `repeats` runs (keep best), record a `Row`, then drop
    // the index before building the next one (keeps peak memory to one index at a time).
    // `$ds` is consumed; `train_s`/`build_s` are measured by the caller during build.
    macro_rules! eval {
        ($label:expr, $ds:expr, $train_s:expr, $build_s:expr) => {{
            let ds = $ds;
            let size_gib = ds.space_usage_GiB();
            let run = |nq: usize| -> (f64, Vec<Vec<u64>>) {
                let mut approx: Vec<Vec<u64>> = Vec::with_capacity(nq);
                let t0 = Instant::now();
                for qi in 0..nq {
                    let res: Vec<u64> = ds
                        .search(queries.get(qi as u64), args.k)
                        .into_iter()
                        .map(|s: ScoredVector<DotProduct>| s.vector)
                        .collect();
                    approx.push(res);
                }
                (t0.elapsed().as_secs_f64(), approx)
            };

            if warmup > 0 {
                let _ = run(warmup);
            }

            let mut best_total = f64::INFINITY;
            let mut best_approx: Vec<Vec<u64>> = Vec::new();
            for r in 0..args.repeats {
                let (t, a) = run(n_queries);
                println!(
                    "  [{}] repeat {r}: total={t:.3}s  mean={:.1}µs  QPS={:.1}",
                    $label,
                    (t / n_queries as f64) * 1e6,
                    n_queries as f64 / t
                );
                if t < best_total {
                    best_total = t;
                    best_approx = a;
                }
            }

            let recall = recall_at_k(&gt, &best_approx, args.k);
            let mean_us = (best_total / n_queries as f64) * 1e6;
            let qps = n_queries as f64 / best_total;
            println!(
                "  [{}] best: mean={mean_us:.1}µs  QPS={qps:.1}  recall@{}={recall:.4}  size={size_gib:.3}GiB",
                $label, args.k
            );
            rows.push(Row {
                label: $label,
                train_s: $train_s,
                build_s: $build_s,
                size_gib,
                total_s: best_total,
                mean_us,
                qps,
                recall,
            });
        }};
    }

    // ── Re-read training data (SquaredEuclideanDistance) for the packed encoders ──
    println!("\nLoading training data...");
    let training_data: PlainSparseDataset<u16, f32, SquaredEuclideanDistance> =
        readers::read_seismic_format(&args.input_file).expect("failed to re-read for training");

    // ── dotpacking8 scalar-4bit ─────────────────────────────────────
    println!("\n=== dot_scalar4bit: DotPacking8Scalar4BitEncoder ===");
    let train_t = Instant::now();
    let mut enc_s = DotPacking8Scalar4BitEncoder::new(dim);
    enc_s.train(&training_data);
    let train_s_scalar = train_t.elapsed().as_secs_f64();
    let build_t = Instant::now();
    let mut g_s = PackedSparseDatasetGrowable::new(enc_s);
    for v in dataset_f32.iter() {
        g_s.push(v);
    }
    let packed_s: PackedSparseDataset<DotPacking8Scalar4BitEncoder> = g_s.into();
    let build_s_scalar = build_t.elapsed().as_secs_f64();
    println!("Train: {train_s_scalar:.3}s | Build: {build_s_scalar:.3}s");
    eval!("dot_scalar4bit", packed_s, train_s_scalar, build_s_scalar);

    // ── dotpacking8 centroid-4bit ───────────────────────────────────
    println!(
        "\n=== dot_centroid4bit: DotPacking8Centroid4BitEncoder (Lloyd iters={}) ===",
        args.n_iterations
    );
    let train_t = Instant::now();
    let mut enc_c = DotPacking8Centroid4BitEncoder::new(dim);
    enc_c.train_with_params(
        &training_data,
        args.lower_percentile,
        args.upper_percentile,
        args.n_iterations,
    );
    let train_s_centroid = train_t.elapsed().as_secs_f64();
    let build_t = Instant::now();
    let mut g_c = PackedSparseDatasetGrowable::new(enc_c);
    for v in dataset_f32.iter() {
        g_c.push(v);
    }
    let packed_c: PackedSparseDataset<DotPacking8Centroid4BitEncoder> = g_c.into();
    let build_s_centroid = build_t.elapsed().as_secs_f64();
    println!("Train: {train_s_centroid:.3}s | Build: {build_s_centroid:.3}s");
    eval!("dot_centroid4bit", packed_c, train_s_centroid, build_s_centroid);

    drop(training_data);

    // ── baseline blocked <u16, f16> (consumes dataset_f32) ──────────
    println!("\n=== blocked_f16: BlockedSparseDataset (<u16, f16> baseline) ===");
    let build_t = Instant::now();
    let blocked: BlockedSparseDataset = dataset_f32.into();
    let build_s_blocked = build_t.elapsed().as_secs_f64();
    println!("Build: {build_s_blocked:.3}s");
    eval!("blocked_f16", blocked, 0.0_f64, build_s_blocked);

    // ── Comparison table (baseline first) ───────────────────────────
    let order = ["blocked_f16", "dot_scalar4bit", "dot_centroid4bit"];
    println!(
        "\n=== Comparison (best of {} repeats, single-thread, k={}) ===",
        args.repeats, args.k
    );
    println!(
        "{:<18} {:>9} {:>9} {:>10} {:>10} {:>10} {:>10}",
        "encoder", "train_s", "build_s", "size_GiB", "mean_us", "QPS", "recall@k"
    );
    let baseline_size = rows
        .iter()
        .find(|r| r.label == "blocked_f16")
        .map(|r| r.size_gib);
    for label in order {
        if let Some(r) = rows.iter().find(|r| r.label == label) {
            let size_str = match baseline_size {
                Some(b) if b > 0.0 && r.label != "blocked_f16" => {
                    format!("{:.3} ({:.2}x)", r.size_gib, r.size_gib / b)
                }
                _ => format!("{:.3}", r.size_gib),
            };
            println!(
                "{:<18} {:>9.3} {:>9.3} {:>10} {:>10.1} {:>10.1} {:>10.4}",
                r.label, r.train_s, r.build_s, size_str, r.mean_us, r.qps, r.recall
            );
        }
    }

    // ── Write TSV (one row per encoder) ─────────────────────────────
    let mut f = std::fs::File::create(&args.output).expect("failed to create output TSV");
    writeln!(
        f,
        "encoder\tn_docs\tdim\tnnz\ttrain_s\tbuild_s\tindex_size_gib\t\
         n_queries\twarmup\trepeats\tk\ttotal_s\tmean_us\tqps\trecall_at_k"
    )
    .unwrap();
    for label in order {
        if let Some(r) = rows.iter().find(|r| r.label == label) {
            writeln!(
                f,
                "{}\t{n}\t{dim}\t{nnz}\t{:.6}\t{:.6}\t{:.6}\t{nq}\t{warmup}\t{rep}\t{k}\t\
                 {:.6}\t{:.3}\t{:.3}\t{:.6}",
                r.label,
                r.train_s,
                r.build_s,
                r.size_gib,
                r.total_s,
                r.mean_us,
                r.qps,
                r.recall,
                nq = n_queries,
                rep = args.repeats,
                k = args.k,
            )
            .unwrap();
        }
    }
    println!("\nResults written to {}", args.output);
}
