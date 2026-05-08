//! Train [`PackedCentroidSparseQuantizer`] at `nbits = 4` and dump the resulting
//! per-component codebook as a `(dim, 16) f32` npy array. Empty components stay
//! as a row of zeros (the trainer never touches them).
//!
//! Companion to `bench_4bit_efficiency_kmeans` — same training arguments, same
//! defaults.

use clap::Parser;
use ndarray::Array2;
use ndarray_npy::write_npy;

use vectorium::encoders::packed_centroid_based_quantization_sparse_scalar::PackedCentroidSparseQuantizer;
use vectorium::readers;
use vectorium::{Dataset, PlainSparseDataset, SquaredEuclideanDistance};

#[derive(Parser, Debug)]
#[clap(
    author,
    version,
    about = "Dump 4-bit k-means centroids as (dim, 16) f32 npy"
)]
struct Args {
    /// Sparse dataset in Seismic binary format
    #[clap(short, long)]
    input_file: String,

    /// Output npy path (shape: dim × 16, dtype f32)
    #[clap(short, long, default_value = "kmeans_centroids_nbits4.npy")]
    output: String,

    /// Number of k-means (Lloyd) iterations per dimension
    #[clap(long, default_value_t = 10)]
    n_iterations: usize,

    /// Lower percentile clip used during training
    #[clap(long, default_value_t = 0.0)]
    lower_percentile: f32,

    /// Upper percentile clip used during training
    #[clap(long, default_value_t = 1.0)]
    upper_percentile: f32,
}

fn main() {
    let args = Args::parse();

    println!("Loading dataset (u16 components)...");
    let training_data: PlainSparseDataset<u16, f32, SquaredEuclideanDistance> =
        readers::read_seismic_format(&args.input_file).expect("failed to read dataset");

    let dim = training_data.output_dim();
    let nnz = training_data.nnz();
    let n_docs = training_data.len();
    println!("Dataset: {n_docs} docs, dim={dim}, nnz={nnz}");

    println!(
        "\n=== Training PackedCentroidSparseQuantizer @ nbits=4 \
         (Lloyd iters={}, percentiles=[{}, {}]) ===",
        args.n_iterations, args.lower_percentile, args.upper_percentile,
    );
    let quantizer = PackedCentroidSparseQuantizer::train(
        &training_data,
        args.lower_percentile,
        args.upper_percentile,
        args.n_iterations,
    );

    let num_centroids = quantizer.num_centroids();
    assert_eq!(num_centroids, 16);
    let centroids = quantizer.centroids();
    assert_eq!(centroids.len(), dim * num_centroids);

    let arr = Array2::from_shape_vec((dim, num_centroids), centroids.to_vec())
        .expect("centroids array shape error");

    write_npy(&args.output, &arr).expect("failed to write npy file");
    println!(
        "\nWrote centroids to {} (shape: {} × {}, dtype: f32)",
        args.output, dim, num_centroids,
    );
}
