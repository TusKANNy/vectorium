#![feature(portable_simd)]
#![allow(internal_features)]
#![feature(core_intrinsics)]
#![feature(gen_blocks)]
#![feature(yield_expr)]
#![doc = include_str!("../README.md")]
#![allow(non_snake_case)]

use num_traits::{AsPrimitive, ToPrimitive, Zero};
use serde::{Serialize, de::DeserializeOwned};
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

#[cfg(feature = "multivec")]
pub use core::vector::{DenseMultiVectorOwned, DenseMultiVectorView};
pub use core::vector::{
    DenseVectorOwned, DenseVectorView, PackedVectorOwned, PackedVectorView, SparseVectorOwned,
    SparseVectorView, VectorView,
};

pub use core::distances::{Distance, DotProduct, SquaredEuclideanDistance};

#[cfg(feature = "multivec")]
pub use core::vector_encoder::MultiVecEncoder;
pub use core::vector_encoder::{
    DenseVectorEncoder, PackedSparseVectorEncoder, QueryEvaluator, SparseVectorEncoder,
    VectorEncoder,
};

pub use clustering::{KMeans, KMeansBuilder};

pub use encoders::blocked_sparse::{
    BlockedSparseEncoder, BlockedSparseQueryEvaluator, SPARSE_QUERY_THRESHOLD,
};
pub use core::data_block::{DataBlock, BLOCK_SIZE};
pub use encoders::sparse_scalar_step::{
    ScalarStepSparseQuantizer, ScalarStepSparseQueryEvaluator, ScalarU8SparseDataset,
    ScalarU8SparseDatasetGrowable, ScalarU8SparseQuantizer, ScalarU16SparseDataset,
    ScalarU16SparseDatasetGrowable, ScalarU16SparseQuantizer, StepCode,
};
pub use encoders::dense_scalar::{
    PlainDenseQuantizer, PlainDenseQuantizerDotProduct, PlainDenseQuantizerSquaredEuclidean,
    ScalarDenseQuantizer, ScalarDenseQuantizerDotProduct, ScalarDenseQuantizerSame,
    ScalarDenseQuantizerSquaredEuclidean, ScalarDenseQueryEvaluator, ScalarDenseSupportedDistance,
};
pub use encoders::dotpacking8_u16_fixedu8::{
    DotPacking8U16FixedU8Encoder, DotPacking8U16FixedU8QueryEvaluator,
};
pub use encoders::dotpacking8_u16_f16::{DotPacking8U16F16Encoder, DotPacking8U16F16QueryEvaluator};
pub use encoders::dotpacking8_u16_scalar4bit::{
    DotPacking8U16Scalar4BitEncoder, DotPacking8U16Scalar4BitQueryEvaluator,
};
pub use encoders::dotpacking8_u16_scalaru8::{
    DotPacking8U16ScalarU8Encoder, DotPacking8U16ScalarU8QueryEvaluator,
};
pub use encoders::dotpacking8_u16_scalaru16::{
    DotPacking8U16ScalarU16Encoder, DotPacking8U16ScalarU16QueryEvaluator,
};
pub use encoders::dotpacking8_u32_scalaru16::{
    BridgedDotPacking8U32ScalarU16Encoder, BridgedDotPacking8U32ScalarU16QueryEvaluator,
};
pub use encoders::dotpacking8_u32_scalaru8::{
    BridgedDotPacking8U32ScalarU8Encoder, BridgedDotPacking8U32ScalarU8QueryEvaluator,
};
#[cfg(feature = "multivec")]
pub use encoders::multivec_pq::{MultiVecPQQueryEvaluator, MultiVecProductQuantizer};
#[cfg(feature = "multivec")]
pub use encoders::multivec_scalar::{
    PlainMultiVecQuantizer, ScalarMultiVecQuantizer, ScalarMultiVecQueryEvaluator,
};
#[cfg(feature = "multivec")]
pub use encoders::multivec_two_level_pq::{
    MultiVecTwoLevelPQQueryEvaluator, MultiVecTwoLevelProductQuantizer,
};
pub use encoders::pq::ProductQuantizer;
pub use encoders::sparse_scalar::{
    PlainSparseQuantizer, PlainSparseQuantizerDotProduct, ScalarSparseQuantizer,
    ScalarSparseQuantizerDotProduct, ScalarSparseQueryEvaluator, ScalarSparseSupportedDistance,
};

pub use core::dataset::{Dataset, DatasetGrowable, DenseData, SparseData, VectorId};
pub use core::flat_index::FlatIndex;
pub use core::index::{Index, IndexStats};
pub use core::storage::{
    GrowableSparseStorage, ImmutableSparseStorage, SparseStorage, SparseStorageMut,
};
pub use datasets::dense_dataset::{DenseDataset, DenseDatasetGeneric, DenseDatasetGrowable};
#[cfg(feature = "multivec")]
pub use datasets::multivec_dataset::{
    MultiVecData, MultiVectorDataset, MultiVectorDatasetGeneric, MultiVectorDatasetGrowable,
};
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

// Useful type aliases for packed sparse dataset types
pub type BlockedSparseDataset = PackedSparseDataset<BlockedSparseEncoder>;
pub type BlockedSparseDatasetGrowable = PackedSparseDatasetGrowable<BlockedSparseEncoder>;
pub type DotPacking8U16FixedU8Dataset = PackedSparseDataset<DotPacking8U16FixedU8Encoder>;
pub type DotPacking8U16FixedU8DatasetGrowable =
    PackedSparseDatasetGrowable<DotPacking8U16FixedU8Encoder>;
pub type DotPacking8U16ScalarU8Dataset = PackedSparseDataset<DotPacking8U16ScalarU8Encoder>;
pub type DotPacking8U16ScalarU8DatasetGrowable =
    PackedSparseDatasetGrowable<DotPacking8U16ScalarU8Encoder>;
pub type DotPacking8U16F16Dataset = PackedSparseDataset<DotPacking8U16F16Encoder>;
pub type DotPacking8U16F16DatasetGrowable = PackedSparseDatasetGrowable<DotPacking8U16F16Encoder>;
pub type DotPacking8U16Scalar4BitDataset = PackedSparseDataset<DotPacking8U16Scalar4BitEncoder>;
pub type DotPacking8U16Scalar4BitDatasetGrowable =
    PackedSparseDatasetGrowable<DotPacking8U16Scalar4BitEncoder>;
pub type DotPacking8U16ScalarU16Dataset = PackedSparseDataset<DotPacking8U16ScalarU16Encoder>;
pub type DotPacking8U16ScalarU16DatasetGrowable =
    PackedSparseDatasetGrowable<DotPacking8U16ScalarU16Encoder>;
pub type BridgedDotPacking8U32ScalarU16Dataset =
    PackedSparseDataset<BridgedDotPacking8U32ScalarU16Encoder>;
pub type BridgedDotPacking8U32ScalarU16DatasetGrowable =
    PackedSparseDatasetGrowable<BridgedDotPacking8U32ScalarU16Encoder>;
pub type BridgedDotPacking8U32ScalarU8Dataset =
    PackedSparseDataset<BridgedDotPacking8U32ScalarU8Encoder>;
pub type BridgedDotPacking8U32ScalarU8DatasetGrowable =
    PackedSparseDatasetGrowable<BridgedDotPacking8U32ScalarU8Encoder>;

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
    use crate::core::vector::DenseVectorView;
    use crate::datasets::dense_dataset::DenseDatasetGrowable;
    use crate::distances::DotProduct;
    use crate::encoders::dense_scalar::ScalarDenseQuantizer;
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
