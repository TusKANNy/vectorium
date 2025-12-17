use clap::Parser;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use half::{bf16, f16};
use indicatif::{ParallelProgressIterator, ProgressStyle};
use rayon::iter::ParallelIterator;

use vectorium::datasets::Result as DatasetResult;
use vectorium::distances;
use vectorium::readers;
use vectorium::{Dataset, FixedU8Q, FixedU16Q, PlainSparseDataset, ScalarDenseDataset, SpaceUsage};
use vectorium::{DenseDataset, Float};

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

    // Print chosen configuration
    println!("Dataset type: {}", dataset_type);
    println!("Value type: {}", value_type);
    if value_type != "f32" {
        println!("Converting from f32 to {}...", value_type);
    }

    match dataset_type.as_str() {
        "dense" => {
            // Dense dataset logic
            match (distance.as_str(), value_type.as_str()) {
                ("euclidean", "f32") => compute_dense_groundtruth::<
                    f32,
                    distances::EuclideanDistance,
                >(input_path, query_path, output_path, k),
                ("euclidean", "f16") => compute_dense_groundtruth::<
                    f16,
                    distances::EuclideanDistance,
                >(input_path, query_path, output_path, k),
                ("euclidean", "bf16") => compute_dense_groundtruth::<
                    bf16,
                    distances::EuclideanDistance,
                >(input_path, query_path, output_path, k),
                ("euclidean", "fixedu8") => compute_dense_groundtruth::<
                    FixedU8Q,
                    distances::EuclideanDistance,
                >(
                    input_path, query_path, output_path, k
                ),
                ("euclidean", "fixedu16") => compute_dense_groundtruth::<
                    FixedU16Q,
                    distances::EuclideanDistance,
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
            }
        }
        "sparse" => {
            // Sparse dataset logic
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
    V: vectorium::ValueType + Float,
    D: vectorium::ScalarDenseSupportedDistance,
{
    // Read dataset and queries as f32
    let dataset_f32 = readers::read_npy_f32::<D>(&input_path).expect("failed to read dataset");
    let queries = readers::read_npy_f32::<D>(&query_path).expect("failed to read queries");

    // Convert dataset from f32 to target value type V using DenseDataset::quantize
    let dataset: ScalarDenseDataset<f32, V, D> = DenseDataset::quantize(&dataset_f32);

    // Print dataset size in GiB
    let dataset_gib = dataset.space_usage_GiB();

    println!("N documents: {}", dataset.len());
    println!("N dims: {}", dataset.dim());
    println!("N non-zeroes: {}", dataset.nnz());
    println!("N queries: {}", queries.len());
    println!("N dims: {}", queries.dim());

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
            let res: Vec<DatasetResult<D>> = dataset.search(qvec, k);
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
    C: vectorium::ComponentType + Default,
    V: vectorium::ValueType + Float,
    D: vectorium::ScalarSparseSupportedDistance + Default,
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
    println!("N dims: {}", dataset.dim());
    println!("N non-zeroes: {}", dataset.nnz());
    println!("N queries: {}", queries.len());
    println!("N dims: {}", queries.dim());

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
            let res: Vec<DatasetResult<D>> = dataset.search(qvec, k);
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
