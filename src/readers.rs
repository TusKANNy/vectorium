//! File readers for loading datasets from various formats.

use std::io;
use std::path::Path;

use std::fs::File;
use std::io::{BufReader, Read, Result as IoResult};

use ndarray::Array2;
use ndarray_npy::ReadNpyExt;

use crate::datasets::GrowableDataset;
use crate::quantizers::dense_scalar::{PlainDenseQuantizer, ScalarDenseSupportedDistance};
use crate::quantizers::sparse_scalar::{PlainSparseQuantizer, ScalarSparseSupportedDistance};
use crate::{ComponentType, Float, ValueType};
use crate::{PlainDenseDataset, PlainSparseDataset, PlainSparseDatasetGrowable};

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

/// TODO: make it generic over the AsRef(f32) to return a static dataset or a growable one dataset

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
pub fn read_npy_f32<D>(filename: impl AsRef<Path>) -> Result<PlainDenseDataset<f32, D>, ReaderError>
where
    D: ScalarDenseSupportedDistance,
{
    let file = std::fs::File::open(filename)?;
    let reader = io::BufReader::new(file);

    let array: Array2<f32> = Array2::read_npy(reader)?;

    let (n_vecs, d) = array.dim();
    let data: Vec<f32> = array.into_raw_vec_and_offset().0;

    let quantizer = PlainDenseQuantizer::new(d);

    Ok(PlainDenseDataset::from_raw(data, n_vecs, d, quantizer))
}

/// Read a sparse dataset from the Seismic binary file format.
///
/// The format of this binary file is the following:
/// - Number of vectors `n_vecs` in 4 bytes, followed by `n_vecs` sparse vectors.
/// - For each vector:
///   - Its length (number of non-zero components) in 4 bytes
///   - A sorted sequence of n components in 4 bytes each (u32)
///   - Corresponding n values in 4 bytes each (f32)
///
/// # Type Parameters
/// * `C` - Component type (must be `ComponentType`)
/// * `V` - Value type (must be `ValueType + Float`)
/// * `D` - Distance type (must implement `ScalarSparseSupportedDistance`)
///
/// # Errors
/// Returns `IoResult` error if:
/// - File cannot be read
/// - Component values overflow the type C (e.g., reading a value > 2^16 when C is u16)
///
/// # Example
/// ```ignore
/// use vectorium::readers::read_seismic_format;
/// use vectorium::distances::DotProduct;
///
/// let dataset = read_seismic_format::<u32, f32, DotProduct>("vectors.bin")?;
/// ```
pub fn read_seismic_format<C, V, D>(
    filename: impl AsRef<Path>,
) -> IoResult<PlainSparseDataset<C, V, D>>
where
    C: ComponentType + Default,
    V: ValueType + Float,
    D: ScalarSparseSupportedDistance + Default,
{
    read_seismic_format_limit(filename, None)
}

/// Read a sparse dataset from the Seismic binary file format with an optional limit on the number of vectors.
///
/// # Type Parameters
/// * `C` - Component type (must be `ComponentType`)
/// * `V` - Value type (must be `ValueType + Float`)
/// * `D` - Distance type (must implement `ScalarSparseSupportedDistance`)
///
/// # Arguments
/// * `filename` - Path to the binary file
/// * `limit` - Maximum number of vectors to read (if `None`, reads all vectors)
///
/// # Errors
/// Returns `IoResult` error if:
/// - File cannot be read
/// - Component values overflow the type C (e.g., reading a value > 2^16 when C is u16)
pub fn read_seismic_format_limit<C, V, D>(
    filename: impl AsRef<Path>,
    limit: Option<usize>,
) -> IoResult<PlainSparseDataset<C, V, D>>
where
    C: ComponentType + Default,
    V: ValueType + Float,
    D: ScalarSparseSupportedDistance + Default,
{
    let path = Path::new(filename.as_ref());
    let f = File::open(path)?;

    let mut br = BufReader::new(f);

    let mut buffer_d = [0_u8; std::mem::size_of::<u32>()];
    let mut buffer = [0_u8; std::mem::size_of::<f32>()];

    br.read_exact(&mut buffer_d)?;
    let mut n_vecs = u32::from_le_bytes(buffer_d) as usize;

    if let Some(n) = limit {
        n_vecs = n.min(n_vecs);
    }

    let quantizer = PlainSparseQuantizer::default();
    let mut data = PlainSparseDatasetGrowable::<C, V, D>::new(quantizer, 0);

    use crate::SparseVector1D;

    for _ in 0..n_vecs {
        br.read_exact(&mut buffer_d)?;
        let n = u32::from_le_bytes(buffer_d) as usize;

        let mut components = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);

        // Read component indices
        for _ in 0..n {
            br.read_exact(&mut buffer_d)?;
            let comp_val = u32::from_le_bytes(buffer_d);

            // Check for overflow when converting u32 to C
            let c = C::from_u32(comp_val).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Component value {} exceeds maximum for type (possible overflow)",
                        comp_val
                    ),
                )
            })?;
            components.push(c);
        }

        // Read component values
        for _ in 0..n {
            br.read_exact(&mut buffer)?;
            let v = V::from_f32_saturating(f32::from_le_bytes(buffer));
            values.push(v);
        }

        let sparse_vec = SparseVector1D::new(components, values);
        data.push(sparse_vec);
    }

    // Convert growable dataset to immutable dataset
    Ok(data.into())
}
