use half::f16;
use num_traits::{ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use std::ops::Add;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::{DotProduct, SpaceUsage};

pub const BLOCK_SIZE: usize = 16;

/// A cache-line-aligned block of 16 (component, value) pairs.
///
/// Layout (64 bytes total):
///   - components: [u16; 16] = 32 bytes
///   - values:     [f16; 16] = 32 bytes
///
/// The `align(64)` ensures the block starts on a cache-line boundary.
/// Padding slots use `component = last_real_component` and `value = f16::ZERO`.
///
/// # Invariant
///
/// Sparse vectors never contain zero values — a zero component is simply absent
/// from the representation. This means `f16::ZERO` is safe to use as a padding
/// sentinel: any slot with `value == f16::ZERO` is guaranteed to be padding,
/// not a real entry.
#[repr(C, align(64))]
#[derive(Copy, PartialEq, Default, Debug, Clone, Serialize, Deserialize)]
pub struct DataBlock {
    pub(crate) components: [u16; BLOCK_SIZE],
    pub(crate) values: [f16; BLOCK_SIZE],
}

impl SpaceUsage for DataBlock {
    fn space_usage_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

impl DataBlock {
    /// Number of real (non-padding) entries in this block.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.iter().take_while(|&&v| v != f16::ZERO).count()
    }

    /// True when the block has no real entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values[0] == f16::ZERO
    }

    /// Compute dot product with a dense f32 query vector.
    ///
    /// For each (component, value) pair in the block, gathers `query[component]`
    /// and multiplies by value. Padding slots have `value = f16::ZERO`, so they
    /// contribute `0.0` and do not affect the result (guaranteed by IEEE 754).
    #[inline]
    pub fn dot_product_dense_query(&self, dense_query: &[f32]) -> DotProduct {
        let base = dense_query.as_ptr();
        let comps = &self.components;
        let vals = &self.values;

        let mut lanes0 = [0.0f32; 8];
        let mut lanes1 = [0.0f32; 8];

        // First half: components[0..8]
        for k in 0..8 {
            let idx = comps[k] as usize;
            // SAFETY: component indices are within [0, dim) and dense_query.len() == dim.
            let qk = unsafe { *base.add(idx) };
            let vk = vals[k].to_f32();
            lanes0[k] = qk.algebraic_mul(vk);
        }

        // Second half: components[8..16]
        for k in 0..8 {
            let idx = comps[8 + k] as usize;
            // SAFETY: same as above.
            let qk = unsafe { *base.add(idx) };
            let vk = vals[8 + k].to_f32();
            lanes1[k] = qk.algebraic_mul(vk);
        }

        let mut block_sum = 0.0f32;
        for &x in &lanes0 {
            block_sum = block_sum.algebraic_add(x);
        }
        for &x in &lanes1 {
            block_sum = block_sum.algebraic_add(x);
        }
        DotProduct::from(block_sum)
    }

    /// Returns the last (largest) component in this block.
    ///
    /// Because `push_encoded` pads unused component slots with the last real
    /// component, `components[BLOCK_SIZE - 1]` is always the maximum component
    /// in the block.
    #[inline]
    pub fn last_component(&self) -> u16 {
        self.components[BLOCK_SIZE - 1]
    }

    /// Aligned AVX2 load of the block's components and values into registers.
    ///
    /// # Safety
    ///
    /// Requires AVX2 support. `DataBlock` is `#[repr(C, align(64))]` so the
    /// pointers are always at least 32-byte aligned (as required by `_mm256_load_si256`).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    pub unsafe fn load_in_registers(&self) -> (__m256i, __m256i) {
        unsafe {
            let comps = _mm256_load_si256(self.components.as_ptr() as *const __m256i);
            let vbits = _mm256_load_si256(self.values.as_ptr() as *const __m256i);
            (comps, vbits)
        }
    }
}

// ---------------------------------------------------------------------------
// SIMD helper functions for the v1 block-skipping algorithm
// ---------------------------------------------------------------------------

/// Compare 16 u16 components against a single query component, mask the f16
/// values, collapse 256→128 bits with OR (assumes at most one match per 16 lanes),
/// convert f16→f32, and FMA-accumulate into `acc`.
///
/// # Safety
///
/// Requires AVX2, F16C, and FMA support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "f16c")]
#[target_feature(enable = "fma")]
#[inline]
pub(crate) unsafe fn match_and_fma_16(
    query_component: u16,
    query_value: f32,
    comps: __m256i,
    vbits: __m256i,
    acc: &mut __m256,
) {
    // 16×u16 compare → 0xFFFF on match, 0 elsewhere
    let q16: __m256i = _mm256_set1_epi16(query_component as i16);
    let cmp: __m256i = _mm256_cmpeq_epi16(comps, q16);

    // Mask f16 values by the comparison result
    let vbits_masked: __m256i = _mm256_and_si256(vbits, cmp);

    // Split 256→2×128 and collapse with OR (at most one match across 16 lanes)
    let lo: __m128i = _mm256_castsi256_si128(vbits_masked);
    let hi: __m128i = _mm256_extracti128_si256(vbits_masked, 1);
    let merged: __m128i = _mm_or_si128(lo, hi);

    // Convert 8×f16 → 8×f32 and FMA accumulate
    let f: __m256 = _mm256_cvtph_ps(merged);
    let scale: __m256 = _mm256_set1_ps(query_value);
    *acc = _mm256_fmadd_ps(f, scale, *acc);
}

/// Horizontal sum of an `__m256` (8×f32) → single `f32`.
///
/// # Safety
///
/// Requires AVX support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline]
pub(crate) unsafe fn hsum256_ps(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(sums2)
}

// ---------------------------------------------------------------------------
// Numeric trait impls required by `PackedSparseVectorEncoder::PackedDataType: ValueType`
// ---------------------------------------------------------------------------

impl Add for DataBlock {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        for i in 0..BLOCK_SIZE {
            self.values[i] = f16::from_f32(self.values[i].to_f32() + rhs.values[i].to_f32());
        }
        self
    }
}

impl Zero for DataBlock {
    fn zero() -> Self {
        Self::default()
    }
    fn is_zero(&self) -> bool {
        *self == Self::default()
    }
}

impl PartialOrd for DataBlock {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.components.partial_cmp(&other.components)
    }
}

impl ToPrimitive for DataBlock {
    fn to_i64(&self) -> Option<i64> { None }
    fn to_u64(&self) -> Option<u64> { None }
}
