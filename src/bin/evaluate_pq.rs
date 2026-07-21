//! Recall + speed harness for the Product Quantizer ([`ProductQuantizer`]).
//!
//! Full-scan analogue of `evaluate_rabitq_ext`: PQ-encodes a dense f32 dataset (M subspaces,
//! 8 bits / 256 centroids per subspace → one `u8` per subspace), builds an exhaustive
//! [`FlatIndex`] over the codes, runs the same per-query search loop, and reports:
//!   * **effectiveness** — recall@k against a pre-computed exact `groundtruth.tsv`
//!     (with oversampling k' >= k); and
//!   * **efficiency** — parallel search throughput (wall time + QPS).
//!
//! This is the apples-to-apples PQ baseline for the RaBitQ evaluators: identical reader,
//! identical `FlatIndex` full scan, identical recall computation — only the encoder differs.
//! The document budget is `M` bytes/vector = `M*8/dim` bits per element (e.g. M=192 on 1536
//! dims = 1.0 bit/element).

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use clap::{Parser, ValueEnum};
use indicatif::{ParallelProgressIterator, ProgressStyle};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use vectorium::core::dataset::ConvertFrom;
use vectorium::dataset::ScoredVector as DatasetResult;
use vectorium::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use vectorium::encoders::pq::ProductQuantizer;
use vectorium::readers;
use vectorium::{
    Dataset, DenseDataset, DenseVectorEncoder, FlatIndex, Index, PlainDenseDataset, SpaceUsage,
    VectorEncoder, VectorId,
};

#[derive(Parser, Debug)]
#[clap(author, version, about = "Evaluate the Product Quantizer: recall vs. groundtruth + search speed", long_about = None)]
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
    #[clap(long, value_parser, default_value = "10")]
    oversample: String,

    /// True-nearest-neighbor cutoff for recall@k.
    #[clap(short, long, value_parser, default_value_t = 10)]
    k: usize,

    /// Number of PQ subspaces (one u8 code / 8 bits each). Bits-per-element = m_pq*8/dim.
    /// Constraints: m_pq % 4 == 0 and dim % m_pq == 0. On 1536 dims: 192 = 1.0 bit/elem,
    /// 384 = 2.0, 512 = 2.67, 768 = 4.0.
    #[clap(long, default_value_t = 192)]
    m_pq: usize,

    /// The metric to score with. `ip` = inner product, `l2` = squared Euclidean.
    #[clap(long, value_enum, default_value_t = Metric::Ip)]
    metric: Metric,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Metric {
    L2,
    Ip,
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
        eprintln!("Every oversample value must be >= k ({}); found {}.", args.k, bad);
        return;
    }
    oversample.sort_unstable();
    oversample.dedup();

    let n_eval_arg = args.num_queries;
    let k = args.k;

    // --- Read the queries once (metric-independent raw f32) --------------------
    // The metric lives in the encoder type parameter, so each arm builds a distinct dataset
    // type; the macro instantiates the const-generic M and the distance and calls `run`.
    macro_rules! run_pq {
        ($m:literal, $dist:ty, $name:literal) => {{
            println!("Reading dataset from {}...", args.input_file);
            let base = readers::read_npy_f32::<$dist>(&args.input_file).expect("failed to read dataset");
            let queries =
                readers::read_npy_f32::<DotProduct>(&args.query_file).expect("failed to read queries");
            let n_eval = n_eval_arg.unwrap_or(queries.len()).min(queries.len());
            let gt = load_groundtruth(&args.groundtruth, n_eval, k);

            let start = Instant::now();
            let dataset: DenseDataset<ProductQuantizer<$m, $dist>> = ConvertFrom::convert_from(base);
            println!(
                "PQ encoding (metric={}, m_pq={}) completed in {:.3}s",
                $name, $m, start.elapsed().as_secs_f64()
            );
            run(&dataset, &queries, &gt, n_eval, &oversample, k);
        }};
    }

    macro_rules! dispatch_m {
        ($dist:ty, $name:literal) => {
            match args.m_pq {
                4 => run_pq!(4, $dist, $name),
                8 => run_pq!(8, $dist, $name),
                16 => run_pq!(16, $dist, $name),
                32 => run_pq!(32, $dist, $name),
                48 => run_pq!(48, $dist, $name),
                64 => run_pq!(64, $dist, $name),
                96 => run_pq!(96, $dist, $name),
                128 => run_pq!(128, $dist, $name),
                192 => run_pq!(192, $dist, $name),
                384 => run_pq!(384, $dist, $name),
                512 => run_pq!(512, $dist, $name),
                768 => run_pq!(768, $dist, $name),
                m => {
                    eprintln!(
                        "Unsupported --m-pq {m}. Supported: 4,8,16,32,48,64,96,128,192,384,512,768"
                    );
                    return;
                }
            }
        };
    }

    match args.metric {
        Metric::Ip => dispatch_m!(DotProduct, "ip"),
        Metric::L2 => dispatch_m!(SquaredEuclideanDistance, "l2"),
    }
}

/// Report config, run the parallel search, and print the recall@k table for `dataset`.
fn run<E>(
    dataset: &DenseDataset<E>,
    queries: &PlainDenseDataset<f32, DotProduct>,
    gt: &HashMap<usize, Vec<u64>>,
    n_eval: usize,
    oversample: &[usize],
    k: usize,
) where
    E: DenseVectorEncoder<InputValueType = f32, OutputValueType = u8>,
    E: VectorEncoder,
    <E as VectorEncoder>::Distance: Distance,
    for<'q> <E as VectorEncoder>::QueryVector<'q>:
        From<vectorium::core::vector::DenseVectorView<'q, f32>>,
{
    let n_docs = dataset.len();
    let kmax = oversample.iter().copied().max().unwrap().min(n_docs);
    if oversample.iter().any(|&kp| kp > n_docs) {
        eprintln!(
            "Warning: some oversample values exceed the {} documents; they are clamped to {}.",
            n_docs, n_docs
        );
    }

    println!("N documents: {}", n_docs);
    println!("N dims: {}", dataset.input_dim());
    println!("Code bytes per vector: {}", dataset.encoder().output_dim());
    println!("Dataset size: {:.3} GiB", dataset.space_usage_GiB());
    println!(
        "Bits per vector: {:.1}  ({:.3} bits/element)",
        (dataset.space_usage_bytes() * 8) as f64 / n_docs.max(1) as f64,
        (dataset.encoder().output_dim() * 8) as f64 / dataset.input_dim().max(1) as f64
    );
    println!("N queries to evaluate: {}", n_eval);
    println!("N queries with ground truth: {}", gt.len());
    println!("k (recall cutoff): {}", k);
    println!("Oversampling k': {:?}", oversample);

    println!("Searching {} queries (retrieving top-{})...", n_eval, kmax);
    let pb_style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}, ETA: {eta})")
        .unwrap()
        .progress_chars("=>-");

    let index = FlatIndex::from(dataset);
    let start = Instant::now();
    let results: Vec<Vec<u64>> = (0..n_eval)
        .into_par_iter()
        .progress_count(n_eval as u64)
        .with_style(pb_style)
        .map(|qid| {
            let qvec = queries.get(qid as VectorId);
            let res: Vec<DatasetResult<E::Distance>> = index.search(qvec.into(), kmax, &());
            res.into_iter().map(|r| r.vector).collect()
        })
        .collect();
    let secs = start.elapsed().as_secs_f64();
    println!();
    println!("Search completed in {:.3}s", secs);
    println!(
        "Throughput: {:.1} queries/s ({} rayon threads)",
        n_eval as f64 / secs.max(f64::EPSILON),
        rayon::current_num_threads()
    );

    println!();
    println!("Recall@{} (macro-averaged over {} queries):", k, gt.len());
    println!("  {:>8}  {:>10}", "k'", format!("recall@{}", k));
    for &kp in oversample {
        let mut sum = 0.0f64;
        let mut counted = 0usize;
        for (qid, truth) in gt {
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
