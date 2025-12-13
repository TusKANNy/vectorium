#![feature(trait_alias)]
#![feature(float_algebraic)]

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
pub use num_marker::{DenseComponent, FromF32};

#[allow(non_snake_case)]
pub mod vector1d;
pub use vector1d::{DenseVector1D, MutableVector1D, SparseVector1D, Vector1D};

pub mod distances;
pub use distances::{Distance, DotProduct, EuclideanDistance};

pub mod quantizers;
pub use quantizers::QueryEvaluator;

pub mod datasets;
pub use datasets::dense_dataset::DenseDataset;

/// Marker for types used as values in a dataset
pub trait ValueType =
    SpaceUsage + Copy + ToPrimitive + Zero + Send + Sync + PartialOrd + FromPrimitive + FromF32;

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
