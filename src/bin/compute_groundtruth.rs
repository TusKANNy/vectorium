use clap::Parser;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use half::{bf16, f16};
use indicatif::{ParallelProgressIterator, ProgressStyle};
use rayon::iter::ParallelIterator;

use vectorium::dataset::{ConvertInto, ScoredVector as DatasetResult};
use vectorium::distances;
use vectorium::encoders::pq::{ProductQuantizer, ProductQuantizerDistance};
use vectorium::readers;
use vectorium::{
    Dataset, FixedU8Q, FixedU16Q, FlatIndex, Index, PlainDenseDataset, PlainSparseDataset,
    ScalarDenseDataset, SpaceUsage,
};
use vectorium::{Distance, Float};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The binary file with dataset vectors
    #[clap(short, long, value_parser)]
    input_file: Option<String>,

    /// The binary file with query vectors
    #[clap(short, long, value_parser)]
    query_file: Option<String>,

    /// The number of results to report for each query
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 10)]
    k: usize,

    /// The output file to write the results.
    #[clap(short, long, value_parser)]
    output_path: Option<String>,

    /// Distance metric: 'euclidean' or 'dotproduct' (default: euclidean)
    #[clap(short, long, value_parser)]
    #[arg(default_value = "euclidean")]
    distance: String,

    /// Value type for dataset: 'f32', 'f16', 'bf16', 'fixedu8', or 'fixedu16' (default: f32)
    #[clap(short, long, value_parser)]
    #[arg(default_value = "f32")]
    value_type: String,

    /// Dataset format: 'dense' (NPY) or 'sparse' (Seismic binary) (default: dense)
    #[clap(long, value_parser)]
    #[arg(default_value = "dense")]
    dataset_type: String,

    /// Component type for sparse datasets: 'u16' or 'u32' (default: u32)
    #[clap(long, value_parser)]
    #[arg(default_value = "u32")]
    component_type: String,

    /// Encoder: 'plain', 'pq' (dense only), 'dotpacking8' or 'blocked' (default: plain)
    #[clap(long, value_parser)]
    #[arg(default_value = "plain")]
    encoder: String,

    /// PQ subspace count (must divide the dataset dimension). 0 auto-selects from {128,96,64,32,16,8,4}.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 0)]
    pq_subspaces: usize,

    /// Optional reference ground-truth TSV. When given, recall@k of the computed
    /// results against this file is reported after the run.
    #[clap(long, value_parser)]
    groundtruth: Option<String>,

    /// Values stored by the dotpacking8 encoder: 'fixedu8', 'scalaru8', 'f16' or 'scalar4bit'
    /// with component_type='u16'; 'scalaru8' (bridged encoder) with component_type='u32'.
    #[clap(long, value_parser)]
    #[arg(default_value = "fixedu8")]
    dotpacking8_values: String,

    /// Skip component reordering for encoders that support it (dotpacking8 with
    /// 'fixedu8' or 'f16' values, whose quantization needs no training). Reordering only
    /// permutes components, so it never changes scores -- it trades build time for a
    /// smaller, faster index.
    #[clap(long, action)]
    no_train: bool,
}

fn main() {
    let args = Args::parse();
    let output_path = args.output_path.clone().expect("output_path is required");
    let groundtruth = args.groundtruth.clone();
    let k = args.k;

    if !dispatch(args) {
        std::process::exit(2);
    }

    if let Some(gt_path) = groundtruth {
        report_recall(&output_path, &gt_path, k);
    }
}

/// Reads two result TSVs (`query_id\tdoc_id\trank\tscore`) and reports the mean
/// recall@k of `results_path` against `gt_path`, over the queries present in both.
fn report_recall(results_path: &str, gt_path: &str, k: usize) {
    fn load(path: &str, k: usize) -> std::collections::HashMap<usize, Vec<u64>> {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("failed to read '{path}': {e}"));
        let mut map: std::collections::HashMap<usize, Vec<u64>> = std::collections::HashMap::new();
        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 2 {
                continue;
            }
            let (Ok(qid), Ok(did)) = (f[0].parse::<usize>(), f[1].parse::<u64>()) else {
                continue;
            };
            let e = map.entry(qid).or_default();
            if e.len() < k {
                e.push(did);
            }
        }
        map
    }

    let got = load(results_path, k);
    let gt = load(gt_path, k);

    let mut sum = 0.0f64;
    let mut n = 0usize;
    let mut exact = 0usize;
    for (qid, gt_ids) in gt.iter() {
        if gt_ids.is_empty() {
            continue;
        }
        let empty = Vec::new();
        let got_ids = got.get(qid).unwrap_or(&empty);
        let hits = vectorium::utils::intersection(got_ids, gt_ids);
        if hits == gt_ids.len() {
            exact += 1;
        }
        sum += hits as f64 / gt_ids.len() as f64;
        n += 1;
    }

    if n == 0 {
        println!("Recall: no overlapping query ids between results and ground truth.");
        return;
    }
    println!(
        "Recall@{k} vs '{gt_path}': {:.4} over {n} queries ({exact} exact, {:.1}%)",
        sum / n as f64,
        100.0 * exact as f64 / n as f64
    );
}

/// Runs the requested computation. Returns false if the configuration was rejected.
fn dispatch(args: Args) -> bool {
    let input_path = args.input_file.expect("input_file is required");
    let query_path = args.query_file.expect("query_file is required");
    let output_path = args.output_path.expect("output_path is required");
    let k = args.k;
    let value_type = args.value_type.to_lowercase();
    let distance = args.distance.to_lowercase();
    let dataset_type = args.dataset_type.to_lowercase();
    let component_type = args.component_type.to_lowercase();
    let encoder = args.encoder.to_lowercase();
    let pq_subspaces = args.pq_subspaces;

    // Print chosen configuration
    println!("Dataset type: {}", dataset_type);
    println!("Value type: {}", value_type);
    println!("Encoder: {}", encoder);
    if value_type != "f32" {
        println!("Converting from f32 to {}...", value_type);
    }
    if encoder == "dotpacking8" {
        println!(
            "Dotpacking8 encoder: {} components, {} values.",
            component_type, args.dotpacking8_values
        );
        println!(
            "Component reordering (decreasing frequency): {}",
            if args.no_train { "disabled" } else { "enabled" }
        );
    }
    if encoder == "blocked" {
        println!(
            "Blocked encoder: u16 components, f16 values in 64-byte blocks; jump_dot for \
             queries with fewer than {} components.",
            vectorium::SPARSE_QUERY_THRESHOLD
        );
    }
    if !matches!(encoder.as_str(), "plain" | "dotpacking8" | "blocked" | "pq") {
        eprintln!(
            "Unknown encoder='{}'. Use encoder='plain'|'dotpacking8'|'blocked'|'pq' (dense only).",
            encoder
        );
        return false;
    }

    match dataset_type.as_str() {
        "dense" => match encoder.as_str() {
            "plain" => match (distance.as_str(), value_type.as_str()) {
                ("euclidean", "f32") => compute_dense_groundtruth::<
                    f32,
                    distances::SquaredEuclideanDistance,
                >(input_path, query_path, output_path, k),
                ("euclidean", "f16") => compute_dense_groundtruth::<
                    f16,
                    distances::SquaredEuclideanDistance,
                >(input_path, query_path, output_path, k),
                ("euclidean", "bf16") => compute_dense_groundtruth::<
                    bf16,
                    distances::SquaredEuclideanDistance,
                >(input_path, query_path, output_path, k),
                ("euclidean", "fixedu8") => compute_dense_groundtruth::<
                    FixedU8Q,
                    distances::SquaredEuclideanDistance,
                >(
                    input_path, query_path, output_path, k
                ),
                ("euclidean", "fixedu16") => compute_dense_groundtruth::<
                    FixedU16Q,
                    distances::SquaredEuclideanDistance,
                >(
                    input_path, query_path, output_path, k
                ),
                ("dotproduct", "f32") => compute_dense_groundtruth::<f32, distances::DotProduct>(
                    input_path,
                    query_path,
                    output_path,
                    k,
                ),
                ("dotproduct", "f16") => compute_dense_groundtruth::<f16, distances::DotProduct>(
                    input_path,
                    query_path,
                    output_path,
                    k,
                ),
                ("dotproduct", "bf16") => compute_dense_groundtruth::<bf16, distances::DotProduct>(
                    input_path,
                    query_path,
                    output_path,
                    k,
                ),
                ("dotproduct", "fixedu8") => compute_dense_groundtruth::<
                    FixedU8Q,
                    distances::DotProduct,
                >(
                    input_path, query_path, output_path, k
                ),
                ("dotproduct", "fixedu16") => compute_dense_groundtruth::<
                    FixedU16Q,
                    distances::DotProduct,
                >(
                    input_path, query_path, output_path, k
                ),
                _ => {
                    eprintln!(
                        "Unknown combination: distance='{}', value_type='{}'. Use distance='euclidean'|'dotproduct' and value_type='f32'|'f16'|'bf16'|'fixedu8'|'fixedu16'.",
                        distance, value_type
                    );
                    return false;
                }
            },
            "pq" => {
                if value_type != "f32" {
                    eprintln!("Encoder 'pq' requires value_type='f32'.");
                    return false;
                }
                if let Err(err) = compute_dense_groundtruth_pq(
                    input_path,
                    query_path,
                    output_path,
                    k,
                    pq_subspaces,
                    &distance,
                ) {
                    eprintln!("Failed to compute PQ groundtruth: {}", err);
                    return false;
                }
            }
            _ => {
                eprintln!(
                    "Encoder '{}' is not supported for dense datasets. Use 'plain' or 'pq'.",
                    encoder
                );
                return false;
            }
        },
        "sparse" => {
            if encoder == "pq" {
                eprintln!("Encoder 'pq' is only supported with dataset_type='dense'.");
                return false;
            }
            // Sparse dataset logic
            if encoder == "dotpacking8" {
                if distance != "dotproduct" {
                    eprintln!("Dotpacking8 encoder supports only distance='dotproduct'.");
                    return false;
                }
                if value_type != "f32" {
                    eprintln!(
                        "Dotpacking8 encoder reads f32 inputs and quantizes internally; \
                         use value_type='f32' (found '{}').",
                        value_type
                    );
                    return false;
                }
                return compute_sparse_groundtruth_dotpacking8(
                    input_path,
                    query_path,
                    output_path,
                    k,
                    &component_type,
                    &args.dotpacking8_values.to_lowercase(),
                    !args.no_train,
                );
            }
            if encoder == "blocked" {
                if component_type != "u16" {
                    eprintln!(
                        "Blocked encoder requires component_type='u16' (found '{}').",
                        component_type
                    );
                    return false;
                }
                if distance != "dotproduct" {
                    eprintln!("Blocked encoder supports only distance='dotproduct'.");
                    return false;
                }
                if value_type != "f32" {
                    eprintln!(
                        "Blocked encoder reads f32 inputs and stores f16 values; \
                         use value_type='f32' (found '{}').",
                        value_type
                    );
                    return false;
                }
                compute_sparse_groundtruth_blocked(input_path, query_path, output_path, k);
                return true;
            }
            match (
                component_type.as_str(),
                distance.as_str(),
                value_type.as_str(),
            ) {
                ("u16", "dotproduct", "f32") => compute_sparse_groundtruth::<
                    u16,
                    f32,
                    distances::DotProduct,
                >(
                    input_path, query_path, output_path, k
                ),
                ("u16", "dotproduct", "f16") => compute_sparse_groundtruth::<
                    u16,
                    f16,
                    distances::DotProduct,
                >(
                    input_path, query_path, output_path, k
                ),
                ("u16", "dotproduct", "bf16") => compute_sparse_groundtruth::<
                    u16,
                    bf16,
                    distances::DotProduct,
                >(
                    input_path, query_path, output_path, k
                ),
                ("u16", "dotproduct", "fixedu8") => compute_sparse_groundtruth::<
                    u16,
                    FixedU8Q,
                    distances::DotProduct,
                >(
                    input_path, query_path, output_path, k
                ),
                ("u16", "dotproduct", "fixedu16") => compute_sparse_groundtruth::<
                    u16,
                    FixedU16Q,
                    distances::DotProduct,
                >(
                    input_path, query_path, output_path, k
                ),
                ("u32", "dotproduct", "f32") => compute_sparse_groundtruth::<
                    u32,
                    f32,
                    distances::DotProduct,
                >(
                    input_path, query_path, output_path, k
                ),
                ("u32", "dotproduct", "f16") => compute_sparse_groundtruth::<
                    u32,
                    f16,
                    distances::DotProduct,
                >(
                    input_path, query_path, output_path, k
                ),
                ("u32", "dotproduct", "bf16") => compute_sparse_groundtruth::<
                    u32,
                    bf16,
                    distances::DotProduct,
                >(
                    input_path, query_path, output_path, k
                ),
                ("u32", "dotproduct", "fixedu8") => compute_sparse_groundtruth::<
                    u32,
                    FixedU8Q,
                    distances::DotProduct,
                >(
                    input_path, query_path, output_path, k
                ),
                ("u32", "dotproduct", "fixedu16") => compute_sparse_groundtruth::<
                    u32,
                    FixedU16Q,
                    distances::DotProduct,
                >(
                    input_path, query_path, output_path, k
                ),
                _ => {
                    eprintln!(
                        "Unknown combination for sparse: component_type='{}', distance='{}', value_type='{}'. \
                        Use component_type='u16'|'u32', distance='dotproduct' (euclidean not supported for sparse), \
                        and value_type='f32'|'f16'|'bf16'|'fixedu8'|'fixedu16'.",
                        component_type, distance, value_type
                    );
                    return false;
                }
            }
        }
        _ => {
            eprintln!(
                "Unknown dataset_type='{}'. Use dataset_type='dense' or 'sparse'.",
                dataset_type
            );
            return false;
        }
    }

    true
}

fn compute_dense_groundtruth<V, D>(
    input_path: String,
    query_path: String,
    output_path: String,
    k: usize,
) where
    V: vectorium::ValueType + Float + vectorium::FromF32 + vectorium::SpaceUsage,
    D: vectorium::ScalarDenseSupportedDistance + 'static,
{
    // Read dataset and queries as f32
    let dataset_f32 = readers::read_npy_f32::<D>(&input_path).expect("failed to read dataset");
    let queries = readers::read_npy_f32::<D>(&query_path).expect("failed to read queries");

    // Convert dataset from f32 to target value type V using ConvertInto
    let dataset: ScalarDenseDataset<f32, V, D> = (&dataset_f32).convert_into();

    // Print dataset size in GiB
    let dataset_gib = dataset.space_usage_GiB();

    println!("N documents: {}", dataset.len());
    println!("N dims: {}", dataset.input_dim());
    println!("N non-zeroes: {}", dataset.nnz());
    println!("N queries: {}", queries.len());
    println!("N dims: {}", queries.input_dim());

    println!("Dataset size: {:.3} GiB", dataset_gib);
    println!("Computing groundtruth for {} queries...", queries.len());

    // Measure computation time
    let start_time = Instant::now();

    // For each query in parallel, use Dataset::search to compute k-nearest neighbors.
    let pb_style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}, ETA: {eta})")
        .unwrap()
        .progress_chars("=>-");

    let results: Vec<Vec<(f32, u64)>> = queries
        .par_iter()
        .progress_count(queries.len() as u64)
        .with_style(pb_style)
        .map(|qvec| {
            let res: Vec<DatasetResult<D>> = FlatIndex::from(&dataset).search(qvec, k, &());
            res.into_iter()
                .map(|r| (r.distance.distance(), r.vector))
                .collect()
        })
        .collect();

    let elapsed = start_time.elapsed();
    println!("Computation completed in {:.3}s", elapsed.as_secs_f64());

    let mut output_file = File::create(output_path).expect("failed to create output file");

    for (query_id, result) in results.iter().enumerate() {
        for (idx, (score, doc_id)) in result.iter().enumerate() {
            writeln!(
                &mut output_file,
                "{query_id}\t{doc_id}\t{}\t{score}",
                idx + 1
            )
            .expect("failed to write result");
        }
    }
}

const PQ_SUPPORTED_SUBSPACES: [usize; 7] = [128, 96, 64, 32, 16, 8, 4];

#[derive(Copy, Clone, Debug)]
enum PqDistanceKind {
    Euclidean,
    DotProduct,
}

fn choose_pq_subspaces(dim: usize, requested: usize) -> Result<usize, String> {
    if requested != 0 {
        if !PQ_SUPPORTED_SUBSPACES.contains(&requested) {
            let supported = PQ_SUPPORTED_SUBSPACES
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "pq_subspaces={} unsupported; supported values: {}",
                requested, supported
            ));
        }
        if requested > dim {
            return Err(format!(
                "pq_subspaces={} exceeds dataset dimension {}",
                requested, dim
            ));
        }
        if !dim.is_multiple_of(requested) {
            return Err(format!(
                "dataset dimension {} is not divisible by pq_subspaces={}",
                dim, requested
            ));
        }
        return Ok(requested);
    }

    PQ_SUPPORTED_SUBSPACES
        .iter()
        .copied()
        .find(|&m| m <= dim && dim.is_multiple_of(m))
        .ok_or_else(|| {
            let supported = PQ_SUPPORTED_SUBSPACES
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "dataset dimension {} has no compatible pq_subspaces; try one of {}",
                dim, supported
            )
        })
}

fn compute_dense_groundtruth_pq(
    input_path: String,
    query_path: String,
    output_path: String,
    k: usize,
    pq_subspaces: usize,
    distance: &str,
) -> Result<(), String> {
    let dataset = readers::read_npy_f32::<distances::SquaredEuclideanDistance>(&input_path)
        .expect("failed to read dataset");
    let queries = readers::read_npy_f32::<distances::SquaredEuclideanDistance>(&query_path)
        .expect("failed to read queries");

    let selected_m = choose_pq_subspaces(dataset.input_dim(), pq_subspaces)?;
    let distance_kind = match distance {
        "euclidean" => PqDistanceKind::Euclidean,
        "dotproduct" => PqDistanceKind::DotProduct,
        _ => unreachable!("distance validated before"),
    };

    let mut dataset = Some(dataset);
    let mut queries = Some(queries);
    let mut output_path = Some(output_path);

    match selected_m {
        128 => match distance_kind {
            PqDistanceKind::Euclidean => {
                run_dense_groundtruth_pq::<128, distances::SquaredEuclideanDistance>(
                    dataset.take().unwrap(),
                    queries.take().unwrap(),
                    k,
                    output_path.take().unwrap(),
                )
            }
            PqDistanceKind::DotProduct => run_dense_groundtruth_pq::<128, distances::DotProduct>(
                dataset.take().unwrap(),
                queries.take().unwrap(),
                k,
                output_path.take().unwrap(),
            ),
        },
        96 => match distance_kind {
            PqDistanceKind::Euclidean => {
                run_dense_groundtruth_pq::<96, distances::SquaredEuclideanDistance>(
                    dataset.take().unwrap(),
                    queries.take().unwrap(),
                    k,
                    output_path.take().unwrap(),
                )
            }
            PqDistanceKind::DotProduct => run_dense_groundtruth_pq::<96, distances::DotProduct>(
                dataset.take().unwrap(),
                queries.take().unwrap(),
                k,
                output_path.take().unwrap(),
            ),
        },
        64 => match distance_kind {
            PqDistanceKind::Euclidean => {
                run_dense_groundtruth_pq::<64, distances::SquaredEuclideanDistance>(
                    dataset.take().unwrap(),
                    queries.take().unwrap(),
                    k,
                    output_path.take().unwrap(),
                )
            }
            PqDistanceKind::DotProduct => run_dense_groundtruth_pq::<64, distances::DotProduct>(
                dataset.take().unwrap(),
                queries.take().unwrap(),
                k,
                output_path.take().unwrap(),
            ),
        },
        32 => match distance_kind {
            PqDistanceKind::Euclidean => {
                run_dense_groundtruth_pq::<32, distances::SquaredEuclideanDistance>(
                    dataset.take().unwrap(),
                    queries.take().unwrap(),
                    k,
                    output_path.take().unwrap(),
                )
            }
            PqDistanceKind::DotProduct => run_dense_groundtruth_pq::<32, distances::DotProduct>(
                dataset.take().unwrap(),
                queries.take().unwrap(),
                k,
                output_path.take().unwrap(),
            ),
        },
        16 => match distance_kind {
            PqDistanceKind::Euclidean => {
                run_dense_groundtruth_pq::<16, distances::SquaredEuclideanDistance>(
                    dataset.take().unwrap(),
                    queries.take().unwrap(),
                    k,
                    output_path.take().unwrap(),
                )
            }
            PqDistanceKind::DotProduct => run_dense_groundtruth_pq::<16, distances::DotProduct>(
                dataset.take().unwrap(),
                queries.take().unwrap(),
                k,
                output_path.take().unwrap(),
            ),
        },
        8 => match distance_kind {
            PqDistanceKind::Euclidean => {
                run_dense_groundtruth_pq::<8, distances::SquaredEuclideanDistance>(
                    dataset.take().unwrap(),
                    queries.take().unwrap(),
                    k,
                    output_path.take().unwrap(),
                )
            }
            PqDistanceKind::DotProduct => run_dense_groundtruth_pq::<8, distances::DotProduct>(
                dataset.take().unwrap(),
                queries.take().unwrap(),
                k,
                output_path.take().unwrap(),
            ),
        },
        4 => match distance_kind {
            PqDistanceKind::Euclidean => {
                run_dense_groundtruth_pq::<4, distances::SquaredEuclideanDistance>(
                    dataset.take().unwrap(),
                    queries.take().unwrap(),
                    k,
                    output_path.take().unwrap(),
                )
            }
            PqDistanceKind::DotProduct => run_dense_groundtruth_pq::<4, distances::DotProduct>(
                dataset.take().unwrap(),
                queries.take().unwrap(),
                k,
                output_path.take().unwrap(),
            ),
        },
        _ => unreachable!("select_pq_subspaces returned unsupported value"),
    }

    Ok(())
}

fn run_dense_groundtruth_pq<const M: usize, D>(
    dataset: PlainDenseDataset<f32, distances::SquaredEuclideanDistance>,
    queries: PlainDenseDataset<f32, distances::SquaredEuclideanDistance>,
    k: usize,
    output_path: String,
) where
    D: ProductQuantizerDistance + 'static,
{
    let start_time = Instant::now();

    // Convert dataset using ConvertFrom trait with automatic sampling
    let pq_dataset: vectorium::DenseDataset<ProductQuantizer<M, D>> = dataset.convert_into();

    let elapsed = start_time.elapsed();
    println!(
        "Product quantization completed in {:.3}s",
        elapsed.as_secs_f64()
    );

    println!("N documents: {}", pq_dataset.len());
    println!("N dims: {}", pq_dataset.input_dim());
    println!("N non-zeroes: {}", pq_dataset.nnz());
    println!("N queries: {}", queries.len());
    println!("N dims: {}", queries.input_dim());
    println!("Dataset size: {:.3} GiB", pq_dataset.space_usage_GiB());
    println!(
        "PQ codes: {} bytes per vector ({} subspaces)",
        pq_dataset.encoder().output_dim(),
        M
    );
    println!("Computing ground truth for {} queries...", queries.len());

    let start_time = Instant::now();

    let pb_style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}, ETA: {eta})")
        .unwrap()
        .progress_chars("=>-");

    let results: Vec<Vec<(f32, u64)>> = queries
        .par_iter()
        .progress_count(queries.len() as u64)
        .with_style(pb_style)
        .map(|qvec| {
            let res: Vec<DatasetResult<D>> = FlatIndex::from(&pq_dataset).search(qvec, k, &());
            res.into_iter()
                .map(|r| (r.distance.distance(), r.vector))
                .collect()
        })
        .collect();

    let elapsed = start_time.elapsed();
    println!("Groundtruth computed in in {:.3}s", elapsed.as_secs_f64());

    let mut output_file = File::create(output_path).expect("failed to create output file");

    for (query_id, result) in results.iter().enumerate() {
        for (idx, (score, doc_id)) in result.iter().enumerate() {
            writeln!(
                &mut output_file,
                "{query_id}\t{doc_id}\t{}\t{score}",
                idx + 1
            )
            .expect("failed to write result");
        }
    }
}

fn compute_sparse_groundtruth<C, V, D>(
    input_path: String,
    query_path: String,
    output_path: String,
    k: usize,
) where
    C: vectorium::ComponentType
        + vectorium::SpaceUsage
        + std::fmt::Debug
        + num_traits::FromPrimitive,
    V: vectorium::ValueType + Float + vectorium::FromF32 + vectorium::SpaceUsage + std::fmt::Debug,
    D: vectorium::ScalarSparseSupportedDistance + 'static,
{
    // Read sparse dataset from seismic binary format
    let dataset: PlainSparseDataset<C, V, D> =
        readers::read_seismic_format(&input_path).expect("failed to read sparse dataset");

    // Queries must use f32 value type for sparse quantizers
    let queries: PlainSparseDataset<C, f32, D> =
        readers::read_seismic_format(&query_path).expect("failed to read sparse queries");

    // Print dataset info=
    let dataset_gib = dataset.space_usage_GiB();

    println!("N documents: {}", dataset.len());
    println!("N dims: {}", dataset.input_dim());
    println!("N non-zeroes: {}", dataset.nnz());
    println!("N queries: {}", queries.len());
    println!("N dims: {}", queries.input_dim());

    println!("Dataset size: {:.3} GiB", dataset_gib);
    println!("Computing ground truth for {} queries...", queries.len());

    // Measure computation time
    let start_time = Instant::now();

    // For each query in parallel, use Dataset::search to compute k-nearest neighbors.
    let pb_style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}, ETA: {eta})")
        .unwrap()
        .progress_chars("=>-");

    let results: Vec<Vec<(f32, u64)>> = queries
        .par_iter()
        .progress_count(queries.len() as u64)
        .with_style(pb_style)
        .map(|qvec| {
            let res: Vec<DatasetResult<D>> = FlatIndex::from(&dataset).search(qvec, k, &());
            res.into_iter()
                .map(|r| (r.distance.distance(), r.vector))
                .collect()
        })
        .collect();

    let elapsed = start_time.elapsed();
    println!("Computation completed in {:.3}s", elapsed.as_secs_f64());

    let mut output_file = File::create(output_path).expect("failed to create output file");

    for (query_id, result) in results.iter().enumerate() {
        for (idx, (score, doc_id)) in result.iter().enumerate() {
            writeln!(
                &mut output_file,
                "{query_id}\t{doc_id}\t{}\t{score}",
                idx + 1
            )
            .expect("failed to write result");
        }
    }
}

fn compute_sparse_groundtruth_dotpacking8(
    input_path: String,
    query_path: String,
    output_path: String,
    k: usize,
    component_type: &str,
    values: &str,
    train: bool,
) -> bool {
    use vectorium::{
        BridgedDotPacking8U32ScalarU8Encoder, DotPacking8U16F16Encoder,
        DotPacking8U16FixedU8Encoder, DotPacking8U16Scalar4BitEncoder,
        DotPacking8U16ScalarU8Encoder,
    };

    // Value quantizers with per-component steps cannot skip training: their `train` both
    // reorders components and fits the steps.
    let needs_training = !matches!(values, "fixedu8" | "f16");
    if !train && needs_training {
        eprintln!(
            "--no-train is only supported with dotpacking8 values 'fixedu8' or 'f16' \
             ('{values}' needs training to fit its quantization steps)."
        );
        return false;
    }

    fn timed(label: &str, f: impl FnOnce()) {
        let start = Instant::now();
        f();
        println!("{label} in {:.3}s", start.elapsed().as_secs_f64());
    }
    const TRAIN_LABEL: &str = "Training (component reordering + value quantizer) computed";

    match component_type {
        "u16" => {
            let dataset_plain: PlainSparseDataset<u16, f32, distances::DotProduct> =
                readers::read_seismic_format(&input_path).expect("failed to read sparse dataset");
            let queries: PlainSparseDataset<u16, f32, distances::DotProduct> =
                readers::read_seismic_format(&query_path).expect("failed to read sparse queries");
            let dim = dataset_plain.output_dim();
            match values {
                "fixedu8" => {
                    let mut encoder = DotPacking8U16FixedU8Encoder::new(dim);
                    if train {
                        timed(TRAIN_LABEL, || encoder.train(&dataset_plain));
                    }
                    run_packed_sparse_groundtruth(encoder, &dataset_plain, &queries, output_path, k);
                }
                "scalaru8" => {
                    let mut encoder = DotPacking8U16ScalarU8Encoder::new(dim);
                    timed(TRAIN_LABEL, || encoder.train(&dataset_plain));
                    run_packed_sparse_groundtruth(encoder, &dataset_plain, &queries, output_path, k);
                }
                "f16" => {
                    let mut encoder = DotPacking8U16F16Encoder::new(dim);
                    if train {
                        timed(TRAIN_LABEL, || encoder.train(&dataset_plain));
                    }
                    run_packed_sparse_groundtruth(encoder, &dataset_plain, &queries, output_path, k);
                }
                "scalar4bit" => {
                    let mut encoder = DotPacking8U16Scalar4BitEncoder::new(dim);
                    timed(TRAIN_LABEL, || encoder.train(&dataset_plain));
                    run_packed_sparse_groundtruth(encoder, &dataset_plain, &queries, output_path, k);
                }
                _ => {
                    eprintln!(
                        "Unknown dotpacking8 values='{values}' for component_type='u16'. \
                         Use 'fixedu8'|'scalaru8'|'f16'|'scalar4bit'."
                    );
                    return false;
                }
            }
        }
        "u32" => {
            if values != "scalaru8" {
                eprintln!(
                    "Dotpacking8 with component_type='u32' (bridged encoder) supports only \
                     values='scalaru8' (found '{values}')."
                );
                return false;
            }
            let dataset_plain: PlainSparseDataset<u32, f32, distances::DotProduct> =
                readers::read_seismic_format(&input_path).expect("failed to read sparse dataset");
            let queries: PlainSparseDataset<u32, f32, distances::DotProduct> =
                readers::read_seismic_format(&query_path).expect("failed to read sparse queries");
            let mut encoder = BridgedDotPacking8U32ScalarU8Encoder::new(dataset_plain.output_dim());
            timed(TRAIN_LABEL, || encoder.train(&dataset_plain));
            run_packed_sparse_groundtruth(encoder, &dataset_plain, &queries, output_path, k);
        }
        _ => {
            eprintln!(
                "Dotpacking8 encoder requires component_type='u16' or 'u32' (found '{component_type}')."
            );
            return false;
        }
    }
    true
}

/// Converts the dataset into a `BlockedSparseDataset` and runs the exhaustive search.
fn compute_sparse_groundtruth_blocked(input_path: String, query_path: String, output_path: String, k: usize) {
    use vectorium::BlockedSparseDataset;

    let dataset_plain: PlainSparseDataset<u16, f32, distances::DotProduct> =
        readers::read_seismic_format(&input_path).expect("failed to read sparse dataset");
    let queries: PlainSparseDataset<u16, f32, distances::DotProduct> =
        readers::read_seismic_format(&query_path).expect("failed to read sparse queries");
    let plain_bytes = dataset_plain.space_usage_bytes();
    let nnz = dataset_plain.nnz();

    let start = Instant::now();
    let dataset: BlockedSparseDataset = dataset_plain.into();
    println!("Encoding completed in {:.3}s", start.elapsed().as_secs_f64());

    search_packed_sparse_and_report(dataset, plain_bytes, nnz, &queries, output_path, k);
}

/// Encodes `dataset_plain` with an already trained packed `encoder`, runs the exhaustive
/// top-k search for every query, prints size/timing statistics and writes the results.
fn run_packed_sparse_groundtruth<E>(
    encoder: E,
    dataset_plain: &PlainSparseDataset<E::InputComponentType, f32, distances::DotProduct>,
    queries: &PlainSparseDataset<E::InputComponentType, f32, distances::DotProduct>,
    output_path: String,
    k: usize,
) where
    E: vectorium::PackedSparseVectorEncoder<InputValueType = f32>
        + vectorium::VectorEncoder<Distance = distances::DotProduct>
        + SpaceUsage
        + Sync,
    PlainSparseDataset<E::InputComponentType, f32, distances::DotProduct>: SpaceUsage,
{
    use vectorium::{DatasetGrowable, PackedSparseDataset, PackedSparseDatasetGrowable};

    let plain_bytes = dataset_plain.space_usage_bytes();
    let nnz = dataset_plain.nnz();

    let start = Instant::now();
    let mut growable = PackedSparseDatasetGrowable::new(encoder);
    for v in dataset_plain.iter() {
        growable.push(v);
    }
    let dataset: PackedSparseDataset<E> = growable.into();
    println!("Encoding completed in {:.3}s", start.elapsed().as_secs_f64());

    search_packed_sparse_and_report(dataset, plain_bytes, nnz, queries, output_path, k);
}

/// Prints size statistics of an encoded `dataset` (against the plain `f32` sparse dataset of
/// `plain_bytes` bytes and `nnz` entries), runs the exhaustive top-k search for every query
/// and writes the results.
fn search_packed_sparse_and_report<E>(
    dataset: vectorium::PackedSparseDataset<E>,
    plain_bytes: usize,
    nnz: usize,
    queries: &PlainSparseDataset<E::InputComponentType, f32, distances::DotProduct>,
    output_path: String,
    k: usize,
) where
    E: vectorium::PackedSparseVectorEncoder
        + vectorium::VectorEncoder<Distance = distances::DotProduct>
        + SpaceUsage
        + Sync,
{
    println!("N documents: {}", dataset.len());
    println!("N dims: {}", dataset.input_dim());
    println!("N packed words: {}", dataset.nnz());
    println!("N queries: {}", queries.len());

    println!("Dataset size: {:.3} GiB", dataset.space_usage_GiB());
    println!(
        "Bits per entry: {:.3} (plain f32 sparse: {:.3}, {:.2}x smaller)",
        (dataset.space_usage_bytes() * 8) as f32 / nnz as f32,
        (plain_bytes * 8) as f32 / nnz as f32,
        plain_bytes as f32 / dataset.space_usage_bytes() as f32
    );
    println!("Computing ground truth for {} queries...", queries.len());

    let start_time = Instant::now();

    let pb_style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}, ETA: {eta})")
        .unwrap()
        .progress_chars("=>-");

    let results: Vec<Vec<(f32, u64)>> = queries
        .par_iter()
        .progress_count(queries.len() as u64)
        .with_style(pb_style)
        .map(|qvec| {
            let res: Vec<DatasetResult<distances::DotProduct>> =
                FlatIndex::from(&dataset).search(qvec, k, &());
            res.into_iter()
                .map(|r| (r.distance.distance(), r.vector))
                .collect()
        })
        .collect();

    let elapsed = start_time.elapsed();
    println!(
        "Computation completed in {:.3}s ({:.3} ms/query)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / queries.len() as f64
    );

    let mut output_file = File::create(output_path).expect("failed to create output file");

    for (query_id, result) in results.iter().enumerate() {
        for (idx, (score, doc_id)) in result.iter().enumerate() {
            writeln!(
                &mut output_file,
                "{query_id}\t{doc_id}\t{}\t{score}",
                idx + 1
            )
            .expect("failed to write result");
        }
    }
}
