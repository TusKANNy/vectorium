//! File readers for loading datasets from various formats.

use std::io;
use std::path::Path;

use ndarray::Array2;
use ndarray_npy::ReadNpyExt;

use crate::datasets::dense_dataset::DenseDataset;

use crate::quantizers::dense_scalar::{PlainDenseQuantizer, ScalarDenseSupportedDistance};

/// Error type for dataset reading operations.
#[derive(Debug)]
pub enum ReaderError {
    Io(io::Error),
    Npy(ndarray_npy::ReadNpyError),
}

impl From<io::Error> for ReaderError {
    fn from(e: io::Error) -> Self {
        ReaderError::Io(e)
    }
}

impl From<ndarray_npy::ReadNpyError> for ReaderError {
    fn from(e: ndarray_npy::ReadNpyError) -> Self {
        ReaderError::Npy(e)
    }
}

impl std::fmt::Display for ReaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReaderError::Io(e) => write!(f, "IO error: {}", e),
            ReaderError::Npy(e) => write!(f, "NPY error: {}", e),
        }
    }
}

impl std::error::Error for ReaderError {}

/// Reads a dense dataset from a `.npy` file containing f32 values.
///
/// The file should contain a 2D array where:
/// - Each row is a vector
/// - Each column is a dimension
///
/// The distance type `D` must implement `ScalarDenseSupportedDistance`.
///
/// # Type Parameters
/// * `D` - The distance type (e.g., `EuclideanDistance` or `DotProduct`)
///
/// # Arguments
/// * `filename` - Path to the `.npy` file
///
/// # Example
/// ```ignore
/// use vectorium::readers::read_npy_f32;
/// use vectorium::distances::EuclideanDistance;
///
/// let dataset = read_npy_f32::<EuclideanDistance>("vectors.npy")?;
/// let evaluator = dataset.quantizer().get_query_evaluator(query);
/// ```
pub fn read_npy_f32<D>(
    filename: impl AsRef<Path>,
) -> Result<DenseDataset<PlainDenseQuantizer<f32, D>, Vec<f32>>, ReaderError>
where
    D: ScalarDenseSupportedDistance,
{
    let file = std::fs::File::open(filename)?;
    let reader = io::BufReader::new(file);

    let array: Array2<f32> = Array2::read_npy(reader)?;

    let (n_vecs, d) = array.dim();
    let data: Vec<f32> = array.into_raw_vec_and_offset().0;

    let quantizer = PlainDenseQuantizer::new(d);

    Ok(DenseDataset::from_raw(data, n_vecs, d, quantizer))
}
