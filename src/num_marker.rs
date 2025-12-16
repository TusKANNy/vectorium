// This whole file exists as a workaround to Rust's deficiencies regarding trait implementation conflicts.
// It's very ugly.

use fixed::{
    FixedU8, FixedU16, FixedU32, FixedU64, FixedU128,
    traits::{FixedUnsigned, ToFixed},
};
use half::{bf16, f16};
use num_traits::AsPrimitive;

use crate::ValueType;

pub(crate) trait MarkerFixedSigned: FixedUnsigned + ValueType {}
impl<T: fixed::types::extra::LeEqU8 + Send + Sync> MarkerFixedSigned for FixedU8<T> {}
impl<T: fixed::types::extra::LeEqU16 + Send + Sync> MarkerFixedSigned for FixedU16<T> {}
impl<T: fixed::types::extra::LeEqU32 + Send + Sync> MarkerFixedSigned for FixedU32<T> {}
impl<T: fixed::types::extra::LeEqU64 + Send + Sync> MarkerFixedSigned for FixedU64<T> {}
impl<T: fixed::types::extra::LeEqU128 + Send + Sync> MarkerFixedSigned for FixedU128<T> {}

pub trait FromF32 {
    fn from_f32_saturating(n: f32) -> Self;
}

impl<T> FromF32 for T
where
    T: MarkerFixedSigned,
{
    fn from_f32_saturating(n: f32) -> Self {
        n.saturating_to_fixed()
    }
}

macro_rules! impl_from_f32_saturating {
    ($($t:ty),*) => {
        $(impl FromF32 for $t {
            fn from_f32_saturating(n: f32) -> Self {
                n.as_()
            }
        })*
    }
}

impl_from_f32_saturating![f64, f32, f16, bf16];

pub trait Float {}

macro_rules! impl_from_float {
    ($($t:ty),*) => {
        $(impl Float for $t {
        })*
    }
}

impl_from_float![f64, f32, f16, bf16];

// Implement Float for fixed-point types
impl<T: fixed::types::extra::LeEqU8 + Send + Sync> Float for FixedU8<T> {}
impl<T: fixed::types::extra::LeEqU16 + Send + Sync> Float for FixedU16<T> {}

// ============================================================================
// DenseComponent - marker type for dense vectors (no component indices)
// ============================================================================

use crate::SpaceUsage;
use num_traits::FromPrimitive;
use std::hash::Hash;

/// Marker type used as `ComponentType` for dense vectors.
/// Dense vectors don't have explicit component indices, so this is a placeholder.
///
/// We use `DenseComponent` instead of the unit type `()` because of Rust's orphan rule.
/// The orphan rule prevents implementing foreign traits (such as `ComponentType`, `ValueType`,
/// `SpaceUsage`, etc.) for foreign types. Since `()` is a built-in type, we cannot implement
/// these custom traits on it within this crate. By defining `DenseComponent` as a local marker
/// type, we can implement all required trait bounds locally without violating the orphan rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct DenseComponent;

impl SpaceUsage for DenseComponent {
    fn space_usage_byte(&self) -> usize {
        0
    }
}

impl AsPrimitive<usize> for DenseComponent {
    fn as_(self) -> usize {
        0
    }
}

impl FromPrimitive for DenseComponent {
    fn from_i64(_: i64) -> Option<Self> {
        Some(DenseComponent)
    }
    fn from_u64(_: u64) -> Option<Self> {
        Some(DenseComponent)
    }
}

impl TryFrom<usize> for DenseComponent {
    type Error = std::convert::Infallible;
    fn try_from(_: usize) -> Result<Self, Self::Error> {
        Ok(DenseComponent)
    }
}
