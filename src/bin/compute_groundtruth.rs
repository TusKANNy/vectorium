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
    Dataset, DenseDataset, FixedU8Q, FixedU16Q, FlatIndex, Index, PlainDenseDataset,
    PlainSparseDataset, RabitqConfig, RabitqExtConfig, RabitqExtQuantizer, RabitqQuantizer,
    RabitqQueryParams, RabitqSupportedDistance, ScalarDenseDataset, SpaceUsage,
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

    /// Encoder: 'plain', 'pq', 'rabitq', 'rabitq-ext' (dense only), or 'dotvbyte' (sparse only)
    /// (default: plain)
    #[clap(long, value_parser)]
    #[arg(default_value = "plain")]
    encoder: String,

    /// PQ subspace count (must divide the dataset dimension). 0 auto-selects from {128,96,64,32,16,8,4}.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 0)]
    pq_subspaces: usize,

    /// Bits per component for the query, for `--encoder rabitq` only (1..=8). Documents are always
    /// 1 bit. `rabitq-ext` scores against an unquantized query and ignores this.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 1)]
    rabitq_query_bits: u32,

    /// Bits per component for the document code, for `--encoder rabitq-ext` only: 2, 4 or 8.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 4)]
    rabitq_total_bits: u32,

    /// Seed for the RaBitQ random orthogonal rotation.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 42)]
    rabitq_seed: u64,

    /// Apply the RaBitQ random orthogonal rotation. Pass `--rabitq-rotate false` to skip the
    /// O(d log d) transform at encode and query time; recall on real data is expected to drop.
    #[clap(long, default_value_t = true, action = clap::ArgAction::Set)]
    rabitq_rotate: bool,

    /// For `--encoder rabitq-ext`: estimate the rescale factor once at train time instead of
    /// searching per vector. Pass `--rabitq-faster-quant false` for the exact per-vector search,
    /// which costs a heap sweep per document for a small accuracy gain.
    #[clap(long, default_value_t = true, action = clap::ArgAction::Set)]
    rabitq_faster_quant: bool,
}

fn main() {
    let args = Args::parse();

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
    if encoder == "dotvbyte" {
        println!("Dotvbyte encoder: quantizing values using FixedU8Q.");
    }
    if !matches!(
        encoder.as_str(),
        "plain" | "dotvbyte" | "pq" | "rabitq" | "rabitq-ext"
    ) {
        eprintln!(
            "Unknown encoder='{}'. Use encoder='plain'|'pq'|'rabitq'|'rabitq-ext' (dense) or 'dotvbyte' (sparse).",
            encoder
        );
        return;
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
                }
            },
            "pq" => {
                if value_type != "f32" {
                    eprintln!("Encoder 'pq' requires value_type='f32'.");
                    return;
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
                }
            }
            "rabitq" | "rabitq-ext" => {
                if value_type != "f32" {
                    eprintln!("Encoder '{}' requires value_type='f32'.", encoder);
                    return;
                }
                let config = RabitqCliConfig {
                    query_bits: args.rabitq_query_bits,
                    total_bits: args.rabitq_total_bits,
                    seed: args.rabitq_seed,
                    rotate: args.rabitq_rotate,
                    faster_quant: args.rabitq_faster_quant,
                };
                if let Err(err) = compute_dense_groundtruth_rabitq(
                    input_path,
                    query_path,
                    output_path,
                    k,
                    &encoder,
                    &distance,
                    config,
                ) {
                    eprintln!("Failed to compute RaBitQ groundtruth: {}", err);
                }
            }
            _ => {
                eprintln!(
                    "Encoder '{}' is not supported for dense datasets. Use 'plain', 'pq', 'rabitq' or 'rabitq-ext'.",
                    encoder
                );
            }
        },
        "sparse" => {
            if matches!(encoder.as_str(), "pq" | "rabitq" | "rabitq-ext") {
                eprintln!(
                    "Encoder '{}' is only supported with dataset_type='dense'.",
                    encoder
                );
                return;
            }
            // Sparse dataset logic
            if encoder == "dotvbyte" {
                if component_type != "u16" {
                    eprintln!(
                        "Dotvbyte encoder requires component_type='u16' (found '{}').",
                        component_type
                    );
                    return;
                }
                if distance != "dotproduct" {
                    eprintln!("Dotvbyte encoder supports only distance='dotproduct'.");
                    return;
                }
                match value_type.as_str() {
                    "f32" => compute_sparse_groundtruth_dotvbyte::<f32>(
                        input_path,
                        query_path,
                        output_path,
                        k,
                    ),
                    "f16" => compute_sparse_groundtruth_dotvbyte::<f16>(
                        input_path,
                        query_path,
                        output_path,
                        k,
                    ),
                    "bf16" => compute_sparse_groundtruth_dotvbyte::<bf16>(
                        input_path,
                        query_path,
                        output_path,
                        k,
                    ),
                    "fixedu8" => compute_sparse_groundtruth_dotvbyte::<FixedU8Q>(
                        input_path,
                        query_path,
                        output_path,
                        k,
                    ),
                    "fixedu16" => compute_sparse_groundtruth_dotvbyte::<FixedU16Q>(
                        input_path,
                        query_path,
                        output_path,
                        k,
                    ),
                    _ => {
                        eprintln!(
                            "Unknown value_type='{}'. Use value_type='f32'|'f16'|'bf16'|'fixedu8'|'fixedu16'.",
                            value_type
                        );
                    }
                }
                return;
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
                }
            }
        }
        _ => {
            eprintln!(
                "Unknown dataset_type='{}'. Use dataset_type='dense' or 'sparse'.",
                dataset_type
            );
        }
    }
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
    let dataset: ScalarDenseDataset<f32, V, D> = (&dataset_f32).convert_into(());

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
    let pq_dataset: vectorium::DenseDataset<ProductQuantizer<M, D>> = (&dataset).convert_into(());
    // The plain f32 vectors are not needed past this point, and on a large collection they dwarf
    // the codes; release them before the scan rather than holding both.
    drop(dataset);

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

/// RaBitQ knobs collected from the command line. `query_bits` applies to `rabitq` only,
/// `total_bits` and `faster_quant` to `rabitq-ext` only; the rest are shared.
#[derive(Copy, Clone, Debug)]
struct RabitqCliConfig {
    query_bits: u32,
    total_bits: u32,
    seed: u64,
    rotate: bool,
    faster_quant: bool,
}

/// Validate the RaBitQ options against the dataset, then dispatch on metric and encoder.
///
/// Unlike PQ, neither encoder needs a const-generic ladder here: `total_bits` is a runtime field,
/// so the only type-level choice is the metric.
fn compute_dense_groundtruth_rabitq(
    input_path: String,
    query_path: String,
    output_path: String,
    k: usize,
    encoder: &str,
    distance: &str,
    config: RabitqCliConfig,
) -> Result<(), String> {
    let dataset = readers::read_npy_f32::<distances::SquaredEuclideanDistance>(&input_path)
        .expect("failed to read dataset");
    let queries = readers::read_npy_f32::<distances::SquaredEuclideanDistance>(&query_path)
        .expect("failed to read queries");

    let dim = dataset.input_dim();
    if !dim.is_multiple_of(64) {
        return Err(format!(
            "RaBitQ requires a dataset dimension that is a multiple of 64 (found {dim}); \
             there is no bit-padding"
        ));
    }
    match encoder {
        "rabitq" if !(1..=8).contains(&config.query_bits) => {
            return Err(format!(
                "rabitq_query_bits must be in 1..=8 (found {})",
                config.query_bits
            ));
        }
        "rabitq-ext" if !matches!(config.total_bits, 2 | 4 | 8) => {
            return Err(format!(
                "rabitq_total_bits must be 2, 4 or 8 (found {}); for 1-bit codes use --encoder rabitq",
                config.total_bits
            ));
        }
        _ => {}
    }

    match (encoder, distance) {
        ("rabitq", "euclidean") => run_dense_groundtruth_rabitq::<
            distances::SquaredEuclideanDistance,
        >(dataset, queries, k, output_path, config),
        ("rabitq", "dotproduct") => run_dense_groundtruth_rabitq::<distances::DotProduct>(
            dataset,
            queries,
            k,
            output_path,
            config,
        ),
        ("rabitq-ext", "euclidean") => run_dense_groundtruth_rabitq_ext::<
            distances::SquaredEuclideanDistance,
        >(dataset, queries, k, output_path, config),
        ("rabitq-ext", "dotproduct") => run_dense_groundtruth_rabitq_ext::<distances::DotProduct>(
            dataset,
            queries,
            k,
            output_path,
            config,
        ),
        _ => unreachable!("encoder and distance validated before"),
    }

    Ok(())
}

fn run_dense_groundtruth_rabitq<D>(
    dataset: PlainDenseDataset<f32, distances::SquaredEuclideanDistance>,
    queries: PlainDenseDataset<f32, distances::SquaredEuclideanDistance>,
    k: usize,
    output_path: String,
    config: RabitqCliConfig,
) where
    D: RabitqSupportedDistance + 'static,
{
    let start_time = Instant::now();
    let encoded: DenseDataset<RabitqQuantizer<D>> = (&dataset).convert_into(RabitqConfig {
        seed: config.seed,
        rotate: config.rotate,
    });
    // The plain f32 vectors are not needed past this point, and on a large collection they dwarf
    // the codes; release them before the scan rather than holding both.
    drop(dataset);
    println!(
        "RaBitQ encoding (query_bits={}, seed={}, rotate={}) completed in {:.3}s",
        config.query_bits,
        config.seed,
        config.rotate,
        start_time.elapsed().as_secs_f64()
    );
    scan_dense_and_write(
        &encoded,
        &queries,
        k,
        output_path,
        &RabitqQueryParams::new(config.query_bits),
    );
}

fn run_dense_groundtruth_rabitq_ext<D>(
    dataset: PlainDenseDataset<f32, distances::SquaredEuclideanDistance>,
    queries: PlainDenseDataset<f32, distances::SquaredEuclideanDistance>,
    k: usize,
    output_path: String,
    config: RabitqCliConfig,
) where
    D: RabitqSupportedDistance + 'static,
{
    let start_time = Instant::now();
    let encoded: DenseDataset<RabitqExtQuantizer<D>> = (&dataset).convert_into(RabitqExtConfig {
        total_bits: config.total_bits,
        seed: config.seed,
        rotate: config.rotate,
        faster_quant: config.faster_quant,
    });
    drop(dataset);
    println!(
        "Extended RaBitQ encoding (total_bits={}, seed={}, rotate={}, faster_quant={}) completed in {:.3}s",
        config.total_bits,
        config.seed,
        config.rotate,
        config.faster_quant,
        start_time.elapsed().as_secs_f64()
    );
    scan_dense_and_write(&encoded, &queries, k, output_path, &());
}

/// Exhaustively score every query against an encoded dense dataset and write the ranked results.
fn scan_dense_and_write<E>(
    dataset: &DenseDataset<E>,
    queries: &PlainDenseDataset<f32, distances::SquaredEuclideanDistance>,
    k: usize,
    output_path: String,
    search_params: &E::QueryParams,
) where
    E: vectorium::DenseVectorEncoder + SpaceUsage,
    E::Distance: Distance,
    E::OutputValueType: SpaceUsage,
{
    println!("N documents: {}", dataset.len());
    println!("N dims: {}", dataset.input_dim());
    println!("N queries: {}", queries.len());
    println!("Dataset size: {:.3} GiB", dataset.space_usage_GiB());
    println!(
        "Encoded: {} u64 words per vector",
        dataset.encoder().output_dim()
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
            let res: Vec<DatasetResult<E::Distance>> =
                FlatIndex::from(dataset).search(qvec, k, search_params);
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

fn compute_sparse_groundtruth_dotvbyte<V>(
    input_path: String,
    query_path: String,
    output_path: String,
    k: usize,
) where
    V: vectorium::ValueType + Float + vectorium::FromF32 + vectorium::SpaceUsage,
{
    use vectorium::DotVByteFixedU8Encoder;
    use vectorium::PackedSparseDataset;

    let dataset_plain: PlainSparseDataset<u16, V, distances::DotProduct> =
        readers::read_seismic_format(&input_path).expect("failed to read sparse dataset");
    let queries: PlainSparseDataset<u16, f32, distances::DotProduct> =
        readers::read_seismic_format(&query_path).expect("failed to read sparse queries");

    let dataset: PackedSparseDataset<DotVByteFixedU8Encoder> = dataset_plain.into();

    let dataset_gib = dataset.space_usage_GiB();

    println!("N documents: {}", dataset.len());
    println!("N dims: {}", dataset.input_dim());
    println!("N packed words: {}", dataset.nnz());
    println!("N queries: {}", queries.len());
    println!("N dims: {}", queries.input_dim());

    println!("Dataset size: {:.3} GiB", dataset_gib);
    println!(
        "Bits per entry: {:.3}",
        (dataset.space_usage_bytes() * 8) as f32 / dataset.nnz() as f32
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
