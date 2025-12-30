#![feature(trait_alias)]
#![feature(portable_simd)]
#![allow(internal_features)]
#![feature(core_intrinsics)]
#![feature(float_algebraic)]
#![feature(gen_blocks)]
#![feature(yield_expr)]
#![doc = include_str!("../README.md")]

use fixed::FixedU8;
use fixed::FixedU16;
use num_traits::{AsPrimitive, FromPrimitive, ToPrimitive};

/// Type aliases for quantized fixed-point types. You can change FRAC in the `fixed` crate to adjust the precision.
/// The `FixedU8Q` type uses 6 fractional bits, while `FixedU16Q` uses 13 fractional bits.
use fixed::types::extra::U6;
use fixed::types::extra::U13;
pub type FixedU8Q = FixedU8<U6>;
pub type FixedU16Q = FixedU16<U13>;

pub mod core;
pub mod datasets;
pub mod encoders;
pub mod utils;

pub use core::dataset;
pub use core::distances;
pub use core::storage;
pub use core::vector_encoder;
pub use core::vector1d;
pub use utils::numeric_markers;
pub use utils::space_usage;

pub use utils::numeric_markers::{Float, FromF32};
pub use utils::space_usage::SpaceUsage;

#[allow(non_snake_case)]
pub use core::vector1d::{DenseVector1D, SparseVector1D, Vector1D};

pub use core::vector1d::{PackedEncoded, PackedVector};

pub use core::distances::{Distance, DotProduct, SquaredEuclideanDistance};

pub use core::vector_encoder::{
    DenseVectorEncoder, PackedVectorEncoder, QueryEvaluator, QueryVectorFor, SparseVectorEncoder,
    VectorEncoder,
};

pub use encoders::dense_scalar::{
    PlainDenseQuantizer, PlainDenseQuantizerDotProduct, PlainDenseQuantizerSquaredEuclidean,
    ScalarDenseQuantizer, ScalarDenseQuantizerDotProduct, ScalarDenseQuantizerSame,
    ScalarDenseQuantizerSquaredEuclidean, ScalarDenseQueryEvaluator, ScalarDenseSupportedDistance,
};
pub use encoders::dotvbyte_fixedu8::{DotVByteFixedU8Quantizer, DotVByteFixedU8QueryEvaluator};
pub use encoders::sparse_scalar::{
    PlainSparseQuantizer, PlainSparseQuantizerDotProduct, ScalarSparseQuantizer,
    ScalarSparseQuantizerDotProduct, ScalarSparseQuantizerSame, ScalarSparseQueryEvaluator,
    ScalarSparseSupportedDistance,
};

pub use core::dataset::{Dataset, GrowableDataset, VectorId};
pub use core::storage::{
    GrowableSparseStorage, ImmutableSparseStorage, SparseStorage, SparseStorageMut,
};
pub use datasets::dense_dataset::{DenseDataset, DenseDatasetGeneric, DenseDatasetGrowable};
pub use datasets::packed_dataset::{PackedDataset, PackedDatasetGeneric, PackedDatasetGrowable};
pub use datasets::sparse_dataset::{SparseDataset, SparseDatasetGrowable};

// Useful type aliases for dense dataset types
pub type ScalarDenseDataset<VIn, VOut, D> = DenseDataset<ScalarDenseQuantizer<VIn, VOut, D>>;
pub type ScalarDenseDatasetGrowable<VIn, VOut, D> =
    DenseDatasetGrowable<ScalarDenseQuantizer<VIn, VOut, D>>;

pub type PlainDenseDataset<V, D> = ScalarDenseDataset<V, V, D>;
pub type PlainDenseDatasetGrowable<V, D> = ScalarDenseDatasetGrowable<V, V, D>;

// Useful type aliases for sparse dataset types
pub type ScalarSparseDataset<C, V, D> = SparseDataset<ScalarSparseQuantizer<C, V, V, D>>;
pub type ScalarSparseDatasetGrowable<C, V, D> =
    SparseDatasetGrowable<ScalarSparseQuantizer<C, V, V, D>>;

pub type PlainSparseDataset<C, V, D> = SparseDataset<PlainSparseQuantizer<C, V, D>>;
pub type PlainSparseDatasetGrowable<C, V, D> = SparseDatasetGrowable<PlainSparseQuantizer<C, V, D>>;

pub use datasets::readers;
pub use datasets::readers::{read_npy_f32, read_seismic_format};

/// Marker for types used as values in a dataset
pub trait ValueType = Copy + Send + Sync + 'static + ToPrimitive;

pub trait ComponentType = AsPrimitive<usize>
    + FromPrimitive
    + Copy
    + Send
    + Sync
    + std::hash::Hash
    + Eq
    + Ord
    + std::convert::TryFrom<usize>
    + 'static;

pub trait PackedType = Copy + Send + Sync + Default;
