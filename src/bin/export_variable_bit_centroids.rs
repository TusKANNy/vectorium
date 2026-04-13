use clap::Parser;
use ndarray::{Array1, Array2};
use ndarray_npy::write_npy;
use std::error::Error;
use std::path::{Path, PathBuf};

use vectorium::distances::DotProduct;
use vectorium::readers;
use vectorium::{Dataset, PlainSparseDataset, VariableBitUniformSparseQuantizer};

#[derive(Parser, Debug)]
#[clap(
    author,
    version,
    about = "Export per-component variable-bit centroids and absolute reconstruction error as .npy"
)]
struct Args {
    /// Sparse dataset in Seismic binary format
    #[clap(short, long)]
    input_file: String,

    /// Output directory for .npy files
    #[clap(short, long, default_value = ".")]
    output_dir: String,

    /// Prefix used for generated file names
    #[clap(long, default_value = "variable_bit_uniform")]
    output_prefix: String,

    /// Lower percentile used during quantizer training
    #[clap(long, default_value_t = 0.0)]
    lower_percentile: f32,

    /// Upper percentile used during quantizer training
    #[clap(long, default_value_t = 1.0)]
    upper_percentile: f32,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let output_dir = PathBuf::from(&args.output_dir);
    std::fs::create_dir_all(&output_dir)?;

    println!("Loading dataset (u16 components)...");
    let dataset: PlainSparseDataset<u16, f32, DotProduct> =
        readers::read_seismic_format(&args.input_file)?;

    println!("Re-loading dataset for training...");
    let training_data: PlainSparseDataset<u16, f32, vectorium::SquaredEuclideanDistance> =
        readers::read_seismic_format(&args.input_file)?;

    let dim = dataset.input_dim();
    println!(
        "Dataset loaded: {} docs, dim={}, nnz={}",
        dataset.len(),
        dim,
        dataset.nnz()
    );

    for nbits in [8u8, 4u8] {
        export_for_nbits(
            &dataset,
            &training_data,
            nbits,
            args.lower_percentile,
            args.upper_percentile,
            &output_dir,
            &args.output_prefix,
        )?;
    }

    println!("Done.");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn export_for_nbits(
    dataset: &PlainSparseDataset<u16, f32, DotProduct>,
    training_data: &PlainSparseDataset<u16, f32, vectorium::SquaredEuclideanDistance>,
    nbits: u8,
    lower_percentile: f32,
    upper_percentile: f32,
    output_dir: &Path,
    output_prefix: &str,
) -> Result<(), Box<dyn Error>> {
    let levels = 1usize << nbits;
    let max_code = ((1u32 << nbits) - 1) as f32;

    println!("\\nTraining variable-bit quantizer for nbits={}...", nbits);
    let quantizer = VariableBitUniformSparseQuantizer::<u16, DotProduct>::train(
        training_data,
        lower_percentile,
        upper_percentile,
        nbits,
    );
    let quants = quantizer.quants();

    let dim = dataset.input_dim();

    // Centroid table: shape (dim, levels), where centroids[c, k] = k * quant[c].
    let mut centroids = Array2::<f32>::zeros((dim, levels));
    for c in 0..dim {
        let q = quants[c];
        for k in 0..levels {
            centroids[(c, k)] = (k as f32) * q;
        }
    }

    // Per-component mean absolute reconstruction error (MAE).
    let mut abs_err_sum = vec![0.0_f64; dim];
    let mut count = vec![0_u64; dim];

    for vec in dataset.iter() {
        for (&c, &v) in vec.components().iter().zip(vec.values()) {
            let idx = c as usize;
            let q = quants[idx];
            let code = if q > 0.0 {
                (v / q).clamp(0.0, max_code) as u8
            } else {
                0u8
            };
            let reconstructed = (code as f32) * q;
            abs_err_sum[idx] += (v - reconstructed).abs() as f64;
            count[idx] += 1;
        }
    }

    let mut per_component_mae = Array1::<f32>::zeros(dim);
    for i in 0..dim {
        if count[i] > 0 {
            per_component_mae[i] = (abs_err_sum[i] / count[i] as f64) as f32;
        }
    }

    let centroids_path = output_dir.join(format!("{}_nbits{}_centroids.npy", output_prefix, nbits));
    let error_path =
        output_dir.join(format!("{}_nbits{}_per_component_abs_error.npy", output_prefix, nbits));

    write_npy(&centroids_path, &centroids)?;
    write_npy(&error_path, &per_component_mae)?;

    let non_empty = count.iter().filter(|&&n| n > 0).count();
    println!(
        "nbits={}: saved centroids {:?} with shape ({}, {})",
        nbits, centroids_path, dim, levels
    );
    println!(
        "nbits={}: saved per-component MAE {:?} with shape ({}) [non-empty components: {}/{}]",
        nbits, error_path, dim, non_empty, dim
    );

    Ok(())
}
