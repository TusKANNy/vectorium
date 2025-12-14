use clap::Parser;
use std::fs::File;
use std::io::Write;

use indicatif::ParallelProgressIterator;
use rayon::prelude::*;

use vectorium::Distance;

use vectorium::datasets::{Dataset, Result as DatasetResult};
use vectorium::distances;
use vectorium::readers;

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
}

fn main() {
    let args = Args::parse();

    let input_path = args.input_file.expect("input_file is required");
    let query_path = args.query_file.expect("query_file is required");
    let output_path = args.output_path.expect("output_path is required");
    let k = args.k;

    // Using EuclideanDistance by default. Change type here to DotProduct if needed.
    type D = distances::EuclideanDistance;

    let dataset = readers::read_npy_f32::<D>(input_path).expect("failed to read dataset");
    let queries = readers::read_npy_f32::<D>(query_path).expect("failed to read queries");

    // For each query, use the Dataset::search helper (already brute-force) and map results.
    let results: Vec<Vec<(f32, usize)>> = (0..queries.len())
        .into_par_iter()
        .progress_count(queries.len() as u64)
        .map(|qi| {
            let qvec = queries.get(qi);
            let res: Vec<DatasetResult<D>> = dataset.search(qvec, k);
            res.into_iter()
                .map(|r| (r.distance.distance(), r.id))
                .collect()
        })
        .collect();

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
