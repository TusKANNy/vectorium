#![feature(trait_alias)]
#![feature(portable_simd)]
#![allow(internal_features)]
#![feature(core_intrinsics)]
#![feature(float_algebraic)]
#![doc = include_str!("../README.md")]

use fixed::FixedU8;
use fixed::FixedU16;
use num_traits::{AsPrimitive, FromPrimitive, ToPrimitive, Zero};

/// Type aliases for quantized fixed-point types. You can change FRAC in the `fixed` crate to adjust the precision.
/// The `FixedU8Q` type uses 6 fractional bits, while `FixedU16Q` uses 8 fractional bits.
use fixed::types::extra::U6;
use fixed::types::extra::U8;
pub type FixedU8Q = FixedU8<U6>;
pub type FixedU16Q = FixedU16<U8>;

pub mod space_usage;
pub use space_usage::SpaceUsage;

pub mod num_marker;
pub use num_marker::{Float, FromF32};

#[allow(non_snake_case)]
pub mod vector1d;
pub use vector1d::{DenseVector1D, MutableVector1D, SparseVector1D, Vector1D};

pub mod distances;
pub use distances::{Distance, DotProduct, EuclideanDistance};

pub mod quantizers;
pub use quantizers::Quantizer;
pub use quantizers::QueryEvaluator;
pub use quantizers::dense_scalar::{
    PlainDenseQuantizer, PlainDenseQuantizerDotProduct, PlainDenseQuantizerEuclidean,
    ScalarDenseQuantizer, ScalarDenseQuantizerDotProduct, ScalarDenseQuantizerEuclidean,
    ScalarDenseQuantizerSame, ScalarDenseQueryEvaluator, ScalarDenseSupportedDistance,
};
pub use quantizers::sparse_scalar::{
    PlainSparseQuantizer, PlainSparseQuantizerDotProduct, ScalarSparseQuantizer,
    ScalarSparseQuantizerDotProduct, ScalarSparseQuantizerSame, ScalarSparseQueryEvaluator,
    ScalarSparseSupportedDistance,
};

pub mod datasets;
pub use datasets::Dataset;
pub use datasets::VectorId;
pub use datasets::VectorKey;
pub use datasets::dense_dataset::{DenseDataset, DenseDatasetGeneric, DenseDatasetGrowable};
pub use datasets::sparse_dataset::{SparseDataset, SparseDatasetGrowable};

// Usefull type aliases for dense dataset types
pub type ScalarDenseDataset<VIn, VOut, D> = DenseDataset<ScalarDenseQuantizer<VIn, VOut, D>>;
pub type ScalarDenseDatasetGrowable<VIn, VOut, D> =
    DenseDatasetGrowable<ScalarDenseQuantizer<VIn, VOut, D>>;

pub type PlainDenseDataset<V, D> = ScalarDenseDatasetGrowable<V, V, D>;
pub type PlainDenseDatasetGrowable<V, D> = ScalarDenseDatasetGrowable<V, V, D>;

// Usefull type aliases for sparse dataset types
pub type ScalarSparseDataset<C, V, D> = SparseDataset<ScalarSparseQuantizer<C, V, V, D>>;
pub type ScalarSparseDatasetGrowable<C, V, D> =
    SparseDatasetGrowable<ScalarSparseQuantizer<C, V, V, D>>;

pub type PlainSparseDataset<C, V, D> = SparseDataset<PlainSparseQuantizer<C, V, D>>;
pub type PlainSparseDatasetGrowable<C, V, D> = SparseDatasetGrowable<PlainSparseQuantizer<C, V, D>>;

pub mod readers;
pub use readers::read_npy_f32;
pub use readers::read_seismic_format;

pub mod utils;

/// Marker for types used as values in a dataset
pub trait ValueType = SpaceUsage
    + Copy
    + ToPrimitive
    + Zero
    + Send
    + Sync
    + PartialOrd
    + FromPrimitive
    + FromF32
    + SpaceUsage
    + Default;

pub trait ComponentType = AsPrimitive<usize>
    + FromPrimitive
    + SpaceUsage
    + Copy
    + Send
    + Sync
    + std::hash::Hash
    + Eq
    + Ord
    + std::convert::TryFrom<usize>;
