//! Recall + speed harness for the 1-bit-per-component [`BinaryQuantizer`].
//!
//! Binary-encodes a dense f32 dataset, runs exhaustive search for a set of
//! queries, and reports:
//!   * **effectiveness** — recall@k against a pre-computed exact `groundtruth.tsv`,
//!     including oversampling (`k'` > k) to estimate the ceiling a perfect exact
//!     reranker over the top-`k'` binary candidates could reach; and
//!   * **efficiency** — parallel search throughput (wall time + QPS).
//!
//! Ground-truth convention: each `doc_id` in the TSV is compared directly against
//! the search-result `VectorId` (row index) — no external id mapping.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use clap::Parser;
use indicatif::{ParallelProgressIterator, ProgressStyle};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use vectorium::dataset::{ConvertInto, ScoredVector as DatasetResult};
use vectorium::distances::DotProduct;
use vectorium::readers;
use vectorium::{
    BinaryQuantizer, Dataset, DenseDataset, FlatIndex, Index, SpaceUsage, VectorEncoder, VectorId,
};

#[derive(Parser, Debug)]
#[clap(author, version, about = "Evaluate the binary encoder: recall vs. groundtruth + search speed", long_about = None)]
struct Args {
    /// Documents `.npy` file (dense, f32).
    #[clap(short, long, value_parser)]
    input_file: String,

    /// Queries `.npy` file (dense, f32).
    #[clap(short, long, value_parser)]
    query_file: String,

    /// Exact-search ground truth TSV (`query_id \t doc_id \t rank \t score`).
    #[clap(short, long, value_parser)]
    groundtruth: String,

    /// Evaluate only the first N queries (by query id). Default: all queries in the file.
    #[clap(short, long, value_parser)]
    num_queries: Option<usize>,

    /// Comma-separated oversampling values k' (each must be >= k), e.g. "10,100,200".
    #[clap(long, value_parser, default_value = "10,100,200")]
    oversample: String,

    /// True-nearest-neighbor cutoff for recall@k.
    #[clap(short, long, value_parser, default_value_t = 10)]
    k: usize,
}

fn main() {
    let args = Args::parse();

    // --- Parse and validate the oversampling list ------------------------------
    let mut oversample: Vec<usize> = match parse_oversample(&args.oversample) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Invalid --oversample '{}': {}", args.oversample, e);
            return;
        }
    };
    if oversample.is_empty() {
        eprintln!("--oversample must contain at least one value.");
        return;
    }
    if let Some(&bad) = oversample.iter().find(|&&kp| kp < args.k) {
        eprintln!(
            "Every oversample value must be >= k ({}); found {}.",
            args.k, bad
        );
        return;
    }
    oversample.sort_unstable();
    oversample.dedup();

    // --- Binary-encode the dataset --------------------------------------------
    println!("Reading dataset from {}...", args.input_file);
    let dataset_f32 =
        readers::read_npy_f32::<DotProduct>(&args.input_file).expect("failed to read dataset");

    let start = Instant::now();
    let dataset: DenseDataset<BinaryQuantizer> = dataset_f32.convert_into();
    println!(
        "Binary encoding completed in {:.3}s",
        start.elapsed().as_secs_f64()
    );

    let n_docs = dataset.len();

    // --- Read queries ----------------------------------------------------------
    let queries =
        readers::read_npy_f32::<DotProduct>(&args.query_file).expect("failed to read queries");
    let n_eval = args.num_queries.unwrap_or(queries.len()).min(queries.len());

    // Clamp k' to the number of documents (search can't return more than n_docs).
    let kmax = oversample.iter().copied().max().unwrap().min(n_docs);
    if oversample.iter().any(|&kp| kp > n_docs) {
        eprintln!(
            "Warning: some oversample values exceed the {} documents; they are clamped to {}.",
            n_docs, n_docs
        );
    }

    // --- Load ground truth (ranks 1..=k, doc_id == VectorId) -------------------
    let gt = load_groundtruth(&args.groundtruth, n_eval, args.k);

    // --- Report configuration --------------------------------------------------
    println!("N documents: {}", n_docs);
    println!("N dims: {}", dataset.input_dim());
    println!("N u64 words per vector: {}", dataset.encoder().output_dim());
    println!("Dataset size: {:.3} GiB", dataset.space_usage_GiB());
    println!(
        "Bits per vector: {:.1}",
        (dataset.space_usage_bytes() * 8) as f64 / n_docs.max(1) as f64
    );
    println!("N queries available: {}", queries.len());
    println!("N queries to evaluate: {}", n_eval);
    println!("N queries with ground truth: {}", gt.len());
    println!("k (recall cutoff): {}", args.k);
    println!("Oversampling k': {:?}", oversample);

    // --- Search (parallel throughput) -----------------------------------------
    println!("Searching {} queries (retrieving top-{})...", n_eval, kmax);
    let pb_style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}, ETA: {eta})")
        .unwrap()
        .progress_chars("=>-");

    let start = Instant::now();
    let results: Vec<Vec<u64>> = (0..n_eval)
        .into_par_iter()
        .progress_count(n_eval as u64)
        .with_style(pb_style)
        .map(|qid| {
            let qvec = queries.get(qid as VectorId);
            let res: Vec<DatasetResult<DotProduct>> =
                FlatIndex::from(&dataset).search(qvec, kmax, &());
            res.into_iter().map(|r| r.vector).collect()
        })
        .collect();
    let elapsed = start.elapsed();

    let secs = elapsed.as_secs_f64();
    println!();
    println!("Search completed in {:.3}s", secs);
    println!(
        "Throughput: {:.1} queries/s ({} rayon threads)",
        n_eval as f64 / secs.max(f64::EPSILON),
        rayon::current_num_threads()
    );

    // --- Effectiveness: recall@k for each oversampling k' ----------------------
    // Macro-average over queries that have ground truth.
    println!();
    println!("Recall@{} (macro-averaged over {} queries):", args.k, gt.len());
    println!("  {:>8}  {:>10}", "k'", format!("recall@{}", args.k));
    for &kp in &oversample {
        let mut sum = 0.0f64;
        let mut counted = 0usize;
        for (qid, truth) in &gt {
            let Some(retrieved) = results.get(*qid) else {
                continue;
            };
            if truth.is_empty() {
                continue;
            }
            let candidate: HashSet<u64> = retrieved.iter().take(kp).copied().collect();
            let hits = truth.iter().filter(|id| candidate.contains(id)).count();
            sum += hits as f64 / truth.len() as f64;
            counted += 1;
        }
        let recall = if counted > 0 { sum / counted as f64 } else { 0.0 };
        println!("  {:>8}  {:>10.4}", kp, recall);
    }
}

/// Parse a comma-separated list of positive integers.
fn parse_oversample(s: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<usize>()
                .map_err(|_| format!("'{}' is not a non-negative integer", t))
                .and_then(|v| {
                    if v == 0 {
                        Err("values must be > 0".to_string())
                    } else {
                        Ok(v)
                    }
                })
        })
        .collect()
}

/// Load ground truth into `query_id -> [doc_id; up to k]`, keeping only queries
/// with `query_id < n_eval` and rows with `rank <= k`. `doc_id` is parsed as the
/// raw `VectorId` (row index) and compared directly to search results.
fn load_groundtruth(path: &str, n_eval: usize, k: usize) -> HashMap<usize, Vec<u64>> {
    let content = std::fs::read_to_string(path).expect("failed to read groundtruth file");
    let mut gt: HashMap<usize, Vec<u64>> = HashMap::new();
    for (lineno, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut fields = line.split('\t');
        let query_id = fields.next().and_then(|s| s.trim().parse::<usize>().ok());
        let doc_id = fields.next().and_then(|s| s.trim().parse::<u64>().ok());
        let rank = fields.next().and_then(|s| s.trim().parse::<usize>().ok());
        let (Some(query_id), Some(doc_id), Some(rank)) = (query_id, doc_id, rank) else {
            eprintln!("Skipping malformed groundtruth line {}: {}", lineno + 1, line);
            continue;
        };
        if query_id >= n_eval || rank > k {
            continue;
        }
        gt.entry(query_id).or_default().push(doc_id);
    }
    gt
}
