#![feature(portable_simd)]
#![allow(internal_features)]
#![feature(core_intrinsics)]
#![feature(float_algebraic)]
#![feature(gen_blocks)]
#![feature(yield_expr)]
#![feature(associated_type_defaults)]
#![doc = include_str!("../README.md")]
#![allow(non_snake_case)]

use num_traits::{AsPrimitive, ToPrimitive, Zero};
use serde::{de::DeserializeOwned, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

/// Type aliases for quantized fixed-point types. You can change FRAC in the `fixed` crate to adjust the precision.
/// The `FixedU8Q` type uses 6 fractional bits, while `FixedU16Q` uses 13 fractional bits.
use fixed::FixedU8;
use fixed::FixedU16;
use fixed::types::extra::U6;
use fixed::types::extra::U13;
pub type FixedU8Q = FixedU8<U6>;
pub type FixedU16Q = FixedU16<U13>;

pub mod clustering;
pub mod core;
pub mod datasets;
pub mod encoders;
pub mod utils;

pub use core::dataset;
pub use core::distances;
pub use core::storage;
pub use core::vector;
pub use core::vector_encoder;
pub use utils::numeric_markers;
pub use utils::space_usage;

pub use utils::numeric_markers::{Float, FromF32};
pub use utils::space_usage::SpaceUsage;

pub use core::vector::{
    DenseVectorOwned, DenseVectorView, PackedVectorOwned, PackedVectorView, SparseVectorOwned,
    SparseVectorView, VectorView,
};

pub use core::distances::{Distance, DotProduct, SquaredEuclideanDistance};

pub use core::vector_encoder::{
    DenseVectorEncoder, PackedSparseVectorEncoder, QueryEvaluator, SparseVectorEncoder,
    VectorEncoder,
};

pub use clustering::{KMeans, KMeansBuilder};

pub use encoders::dense_scalar::{
    PlainDenseQuantizer, PlainDenseQuantizerDotProduct, PlainDenseQuantizerSquaredEuclidean,
    ScalarDenseQuantizer, ScalarDenseQuantizerDotProduct, ScalarDenseQuantizerSame,
    ScalarDenseQuantizerSquaredEuclidean, ScalarDenseQueryEvaluator, ScalarDenseSupportedDistance,
};
pub use encoders::dotvbyte_fixedu8::{DotVByteFixedU8Encoder, DotVByteFixedU8QueryEvaluator};
pub use encoders::pq::ProductQuantizer;
pub use encoders::sparse_scalar::{
    PlainSparseQuantizer, PlainSparseQuantizerDotProduct, ScalarSparseQuantizer,
    ScalarSparseQuantizerDotProduct, ScalarSparseQueryEvaluator, ScalarSparseSupportedDistance,
};

pub use core::dataset::{Dataset, DenseData, DatasetGrowable, SparseData, VectorId};
pub use core::storage::{
    GrowableSparseStorage, ImmutableSparseStorage, SparseStorage, SparseStorageMut,
};
pub use datasets::dense_dataset::{DenseDataset, DenseDatasetGeneric, DenseDatasetGrowable};
pub use datasets::packed_dataset::{
    PackedSparseDataset, PackedSparseDatasetGeneric, PackedSparseDatasetGrowable,
};
pub use datasets::sparse_dataset::{SparseDataset, SparseDatasetGrowable};

// Useful type aliases for dense dataset types
pub type ScalarDenseDataset<VIn, VOut, D> = DenseDataset<ScalarDenseQuantizer<VIn, VOut, D>>;
pub type ScalarDenseDatasetGrowable<VIn, VOut, D> =
    DenseDatasetGrowable<ScalarDenseQuantizer<VIn, VOut, D>>;

pub type PlainDenseDataset<V, D> = ScalarDenseDataset<V, V, D>;
pub type PlainDenseDatasetGrowable<V, D> = ScalarDenseDatasetGrowable<V, V, D>;

// Useful type aliases for sparse dataset types
pub type ScalarSparseDataset<C, W, V, D> = SparseDataset<ScalarSparseQuantizer<C, W, V, D>>;
pub type ScalarSparseDatasetGrowable<C, W, V, D> =
    SparseDatasetGrowable<ScalarSparseQuantizer<C, W, V, D>>;

pub type PlainSparseDataset<C, V, D> = SparseDataset<PlainSparseQuantizer<C, V, D>>;
pub type PlainSparseDatasetGrowable<C, V, D> = SparseDatasetGrowable<PlainSparseQuantizer<C, V, D>>;

pub use core::dataset::{ScoredRange, ScoredVector};

pub type ScoredVectorDotProduct = ScoredVector<DotProduct>;
pub type ScoredRangeDotProduct = ScoredRange<DotProduct>;
pub type ScoredVectorEuclidean = ScoredVector<SquaredEuclideanDistance>;
pub type ScoredRangeEuclidean = ScoredRange<SquaredEuclideanDistance>;

pub use datasets::readers;
pub use datasets::readers::{read_npy_f32, read_seismic_format};

/// Marker for types used as values in a dataset.
pub trait ValueType: Copy + Send + Sync + 'static + ToPrimitive + PartialOrd + Zero {}
impl<T> ValueType for T where T: Copy + Send + Sync + 'static + ToPrimitive + PartialOrd + Zero {}

/// Marker for types used as components in a dataset.
pub trait ComponentType: AsPrimitive<usize> + Copy + Send + Sync + 'static + Ord {}
impl<T> ComponentType for T where T: AsPrimitive<usize> + Copy + Send + Sync + 'static + Ord {}

/// Support trait for index types so they can save and load with a common interface.
#[derive(Debug)]
pub enum IndexIoError {
    Io(std::io::Error),
    Encode(bincode::error::EncodeError),
    Decode(bincode::error::DecodeError),
}

impl From<std::io::Error> for IndexIoError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<bincode::error::EncodeError> for IndexIoError {
    fn from(value: bincode::error::EncodeError) -> Self {
        Self::Encode(value)
    }
}

impl From<bincode::error::DecodeError> for IndexIoError {
    fn from(value: bincode::error::DecodeError) -> Self {
        Self::Decode(value)
    }
}

pub trait IndexSerializer: Sized {
    fn save_index(&self, filename: &str) -> Result<(), IndexIoError>
    where
        Self: Serialize,
    {
        let file = File::create(filename)?;
        let mut writer = BufWriter::new(file);
        let config = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian();
        bincode::serde::encode_into_std_write(self, &mut writer, config)?;
        Ok(())
    }

    fn load_index(filename: &str) -> Result<Self, IndexIoError>
    where
        Self: DeserializeOwned,
    {
        let file = File::open(filename)?;
        let mut reader = BufReader::new(file);
        let config = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian();
        let decoded = bincode::serde::decode_from_std_read::<Self, _, _>(&mut reader, config)?;
        Ok(decoded)
    }
}

impl<T> IndexSerializer for T where T: Dataset {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoders::dense_scalar::ScalarDenseQuantizer;
    use crate::distances::DotProduct;
    use crate::core::vector::DenseVectorView;
    use crate::datasets::dense_dataset::DenseDatasetGrowable;
    use crate::{DatasetGrowable, DenseDataset};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();
        let mut path = std::env::temp_dir();
        path.push(format!(
            "vectorium_{}_{}_{}.bin",
            name,
            std::process::id(),
            nanos
        ));
        path
    }

    #[test]
    fn index_serializer_round_trip_dense_dataset() {
        let encoder = ScalarDenseQuantizer::<f32, f32, DotProduct>::new(2);
        let mut growable = DenseDatasetGrowable::new(encoder);
        growable.push(DenseVectorView::new(&[1.0, 0.0]));
        growable.push(DenseVectorView::new(&[0.0, 1.0]));
        let dataset: DenseDataset<_> = growable.into();

        let path = temp_path("dense_round_trip");
        dataset.save_index(path.to_str().unwrap()).unwrap();
        let loaded = DenseDataset::<_>::load_index(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(dataset, loaded);
    }

    #[test]
    fn index_serializer_fails_on_corrupt_file() {
        let path = temp_path("corrupt");
        std::fs::write(&path, b"not a valid index").unwrap();

        let result = DenseDataset::<ScalarDenseQuantizer<f32, f32, DotProduct>>::load_index(
            path.to_str().unwrap(),
        );
        std::fs::remove_file(&path).unwrap();

        assert!(result.is_err());
    }
}
