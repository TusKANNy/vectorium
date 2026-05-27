//! Single-thread search-efficiency benchmark for the **blocked sparse** dot-product
//! kernel introduced in `src/encoders/blocked_sparse.rs`.
//!
//! Mirrors `bench_f16_efficiency.rs` (same warmup / repeats / recall TSV schema) so
//! numbers are directly comparable to the plain `<u16, f16>` baseline (which itself
//! has two internal paths: dense gather vs merge, gated by `small_dim` in
//! `sparse_scalar.rs`).
//!
//! Encoder: [`BlockedSparseEncoder`] — values narrowed to `f16` and packed into
//! cache-line-aligned `DataBlock`s of 16 (component, value) pairs. At query time
//! the evaluator picks one of two paths based on `SPARSE_QUERY_THRESHOLD`:
//!   - dense block-gather  (query nnz ≥ threshold; query densified to `Vec<f32>`)
//!   - v1 AVX2 block-skip (query nnz < threshold; sorted-merge over blocks)

use clap::Parser;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

use vectorium::dataset::ScoredVector;
use vectorium::distances::DotProduct;
use vectorium::readers;
use vectorium::{
    BlockedSparseDataset, Dataset, PlainSparseDataset, SpaceUsage, SPARSE_QUERY_THRESHOLD,
};

#[derive(Parser, Debug)]
#[clap(
    author,
    version,
    about = "Blocked sparse <u16, f16> search efficiency (single-thread)"
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
    #[clap(short, long, default_value = "blocked_sparse_efficiency.tsv")]
    output: String,

    /// Precomputed f32 ground-truth TSV (format: query_id\tdoc_id\trank\tscore).
    /// Required: recall@k is computed against this file.
    #[clap(long)]
    groundtruth_tsv: String,
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

/// Sequential timed search over `n_queries` consecutive queries.
fn time_search_serial(
    dataset: &BlockedSparseDataset,
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
    let total_n_queries = queries.len();
    let n_queries = args.n_queries.min(total_n_queries);
    let warmup = args.warmup.min(n_queries);

    let q_nnz = queries.nnz();
    let mean_q_nnz = q_nnz as f64 / total_n_queries as f64;
    let mean_d_nnz = nnz as f64 / n as f64;
    println!("Dataset: {n} docs, dim={dim}, nnz={nnz} (mean {mean_d_nnz:.1} nnz/doc)");
    println!(
        "Queries: timed={n_queries}, warmup={warmup} (of {total_n_queries} available, total nnz={q_nnz}, mean {mean_q_nnz:.2} nnz/query)"
    );
    println!(
        "SPARSE_QUERY_THRESHOLD={SPARSE_QUERY_THRESHOLD} → mean-query path: {path}",
        path = if mean_q_nnz < SPARSE_QUERY_THRESHOLD as f64 {
            "v1 (block-skip, AVX2)"
        } else {
            "dense gather"
        }
    );
    println!("Repeats: {} | single-threaded", args.repeats);

    // ── Build blocked dataset ──────────────────────────────────────
    println!("\n=== Building blocked sparse dataset ===");
    let build_t = Instant::now();
    let dataset_blocked: BlockedSparseDataset = dataset_f32.into();
    let build_time = build_t.elapsed().as_secs_f64();
    let size_gib = dataset_blocked.space_usage_GiB();
    println!("Build: {build_time:.3}s | Index size: {size_gib:.3} GiB");

    // ── Load ground truth ──────────────────────────────────────────
    println!("\nLoading ground truth from {}...", args.groundtruth_tsv);
    let gt = load_groundtruth_tsv(&args.groundtruth_tsv, n_queries, args.k);

    // ── Warmup ─────────────────────────────────────────────────────
    if warmup > 0 {
        println!("\n=== Warmup ({warmup} queries) ===");
        let _ = time_search_serial(&dataset_blocked, &queries, warmup, args.k);
    }

    // ── Timed loop ─────────────────────────────────────────────────
    println!(
        "\n=== Timing (search top-{}, repeats={}) ===",
        args.k, args.repeats
    );
    let mut best_total: Option<f64> = None;
    let mut best_approx: Option<Vec<Vec<u64>>> = None;
    for r in 0..args.repeats {
        let (total_s, approx) = time_search_serial(&dataset_blocked, &queries, n_queries, args.k);
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

    // ── Summary ────────────────────────────────────────────────────
    println!(
        "\n=== Summary (best of {} repeats, single-thread) ===",
        args.repeats
    );
    println!("  total          {best_total:.3} s");
    println!("  mean latency   {mean_us:.1} µs / query");
    println!("  QPS            {qps:.1}");
    println!("  Recall@{}      {recall:.4}", args.k);

    // ── Write TSV ──────────────────────────────────────────────────
    let mut f = std::fs::File::create(&args.output).expect("failed to create output TSV");
    writeln!(
        f,
        "encoder\tn_docs\tdim\tnnz\tindex_size_gib\tbuild_time_s\t\
         n_queries\twarmup\trepeats\tk\ttotal_s\tmean_us\tqps\trecall_at_k"
    )
    .unwrap();
    writeln!(
        f,
        "blocked_sparse_f16\t{n}\t{dim}\t{nnz}\t{size_gib:.6}\t{build_time:.6}\t\
         {nq}\t{warmup}\t{rep}\t{k}\t{best_total:.6}\t{mean_us:.3}\t{qps:.3}\t{recall:.6}",
        nq = n_queries,
        rep = args.repeats,
        k = args.k,
    )
    .unwrap();
    println!("\nResults written to {}", args.output);
}
