use clap::Parser;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use half::{bf16, f16};
use indicatif::{ParallelProgressIterator, ProgressIterator, ProgressStyle};
use rayon::iter::ParallelIterator;

use vectorium::SpaceUsage;
use vectorium::datasets::{Dataset, Result as DatasetResult};
use vectorium::distances;
use vectorium::readers;
use vectorium::{DenseDataset, Float, ScalarDenseQuantizer};

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

    /// Value type for dataset: 'f32', 'f16', or 'bf16' (default: f32)
    #[clap(short, long, value_parser)]
    #[arg(default_value = "f32")]
    value_type: String,
}

fn main() {
    let args = Args::parse();

    let input_path = args.input_file.expect("input_file is required");
    let query_path = args.query_file.expect("query_file is required");
    let output_path = args.output_path.expect("output_path is required");
    let k = args.k;
    let value_type = args.value_type.to_lowercase();
    let distance = args.distance.to_lowercase();

    // Print chosen value type
    println!("Value type: {}", value_type);
    if value_type != "f32" {
        println!("Converting from f32 to {}...", value_type);
    }

    match (distance.as_str(), value_type.as_str()) {
        ("euclidean", "f32") => compute_groundtruth::<f32, distances::EuclideanDistance>(
            input_path,
            query_path,
            output_path,
            k,
        ),
        ("euclidean", "f16") => compute_groundtruth::<f16, distances::EuclideanDistance>(
            input_path,
            query_path,
            output_path,
            k,
        ),
        ("euclidean", "bf16") => compute_groundtruth::<bf16, distances::EuclideanDistance>(
            input_path,
            query_path,
            output_path,
            k,
        ),
        ("dotproduct", "f32") => compute_groundtruth::<f32, distances::DotProduct>(
            input_path,
            query_path,
            output_path,
            k,
        ),
        ("dotproduct", "f16") => compute_groundtruth::<f16, distances::DotProduct>(
            input_path,
            query_path,
            output_path,
            k,
        ),
        ("dotproduct", "bf16") => compute_groundtruth::<bf16, distances::DotProduct>(
            input_path,
            query_path,
            output_path,
            k,
        ),
        _ => {
            eprintln!(
                "Unknown combination: distance='{}', value_type='{}'. Use distance='euclidean'|'dotproduct' and value_type='f32'|'f16'|'bf16'.",
                distance, value_type
            );
        }
    }
}

fn compute_groundtruth<V, D>(input_path: String, query_path: String, output_path: String, k: usize)
where
    V: vectorium::ValueType + Float,
    D: vectorium::ScalarDenseSupportedDistance,
{
    // Read dataset and queries as f32
    let dataset_f32 = readers::read_npy_f32::<D>(&input_path).expect("failed to read dataset");
    let queries = readers::read_npy_f32::<D>(&query_path).expect("failed to read queries");

    // Convert dataset from f32 to target value type V using DenseDataset::convert
    let dataset: DenseDataset<ScalarDenseQuantizer<f32, V, D>, Vec<V>> =
        DenseDataset::convert(&dataset_f32);

    // Print dataset size in GiB
    let dataset_bytes = dataset.space_usage_byte();
    let dataset_gib = dataset_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    println!("N documents: {}", dataset.len());
    println!("N dims: {}", dataset.dim());
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
