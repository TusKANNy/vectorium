use clap::Parser;
use std::fs::File;
use std::io::{BufWriter, Write};

use vectorium::distances::DotProduct;
use vectorium::readers;
use vectorium::{Dataset, PlainSparseDataset};

#[derive(Parser, Debug)]
#[clap(author, version, about = "Dump per-component value distributions to a binary file")]
struct Args {
    /// Sparse dataset in Seismic binary format
    #[clap(short, long)]
    input_file: String,

    /// Output .npy file (one flat array: [dim, offsets..., values...])
    #[clap(short, long)]
    output_file: String,
}

/// Output: a single .npy file containing a flat f32 array with the layout:
///   [dim, offset_0, offset_1, ..., offset_{dim}, v_0, v_1, ..., v_{total-1}]
///
/// - dim (as f32, cast to u32 in Python)
/// - offsets: dim+1 values. Component i's values are at indices offset_i..offset_{i+1}
///   in the values section (which starts right after the offsets).
/// - values: all f32 values concatenated.
///
/// Python usage:
///   import numpy as np
///   data = np.load("component_values.npy")
///   dim = int(data[0])
///   offsets = data[1:dim+2].astype(np.int64)
///   values = data[dim+2:]
///   # Component i:
///   comp_i_values = values[offsets[i]:offsets[i+1]]
fn main() {
    let args = Args::parse();

    println!("Loading dataset...");
    let dataset: PlainSparseDataset<u16, f32, DotProduct> =
        readers::read_seismic_format(&args.input_file).expect("failed to read dataset");

    let dim = dataset.input_dim();
    let n = dataset.len();
    let nnz = dataset.nnz();
    println!("Dataset: {n} docs, dim={dim}, nnz={nnz}");

    // Collect values per component
    println!("Collecting values per component...");
    let mut per_component: Vec<Vec<f32>> = vec![Vec::new(); dim];
    for vec in dataset.iter() {
        for (&c, &v) in vec.components().iter().zip(vec.values()) {
            per_component[c as usize].push(v);
        }
    }

    // Build offsets (CSR-style)
    let mut offsets = Vec::with_capacity(dim + 1);
    offsets.push(0u64);
    for vals in &per_component {
        offsets.push(offsets.last().unwrap() + vals.len() as u64);
    }
    let total_values = *offsets.last().unwrap() as usize;

    // Total elements in the flat array: 1 (dim) + (dim+1) (offsets) + total_values
    let total_elements = 1 + (dim + 1) + total_values;

    // Write as .npy (NumPy format v1.0, float32, 1-D)
    println!("Writing to {}...", args.output_file);
    let file = File::create(&args.output_file).expect("failed to create output file");
    let mut w = BufWriter::new(file);

    // .npy header
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}",
        total_elements
    );
    // Pad header to align to 64 bytes (magic=6 + version=2 + header_len=2 + header + \n)
    let prefix_len = 10; // magic(6) + version(2) + header_len(2)
    let padding = 64 - ((prefix_len + header.len() + 1) % 64);
    let padded_header_len = header.len() + padding + 1; // +1 for \n

    // Magic + version 1.0
    w.write_all(b"\x93NUMPY").unwrap();
    w.write_all(&[1u8, 0u8]).unwrap();
    w.write_all(&(padded_header_len as u16).to_le_bytes())
        .unwrap();
    w.write_all(header.as_bytes()).unwrap();
    for _ in 0..padding {
        w.write_all(b" ").unwrap();
    }
    w.write_all(b"\n").unwrap();

    // Data: dim as f32
    w.write_all(&(dim as f32).to_le_bytes()).unwrap();

    // Offsets as f32 (safe for up to ~16M values; use f64 for larger datasets)
    for &off in &offsets {
        w.write_all(&(off as f32).to_le_bytes()).unwrap();
    }

    // Values
    for vals in &per_component {
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 4) };
        w.write_all(bytes).unwrap();
    }
    w.flush().unwrap();

    let non_empty = per_component.iter().filter(|v| !v.is_empty()).count();
    let file_size_mib = (total_elements * 4) as f64 / (1024.0 * 1024.0);
    println!("Done. {total_values} values across {non_empty}/{dim} non-empty components");
    println!("File size: ~{:.1} MiB", file_size_mib);
}
