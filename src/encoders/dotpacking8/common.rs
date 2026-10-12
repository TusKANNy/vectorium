//! Shared block-8 gap codec of the `dotpacking8` encoders: block packing (`encode_blocks`),
//! SIMD decoding (`DotPacking8Iter`, `DotPacking8F16Iter`), prefix sums and the gather kernel.
//!
//! # Safety invariants
//!
//! Decoding is unchecked for speed. It is sound because every encoded vector is produced by
//! the matching encoder's `push_encoded`, which guarantees:
//!
//! 1. **Layout.** `[n: u16][selectors: ceil(ceil(n/8)/2) bytes][gap payload][padding][values]`,
//!    padded to whole `u64` words. Values are the last `n` entries (`n` bytes, `2n` for `f16`,
//!    `nibble_bytes(n)` for 4-bit), so the views locate them from the end of the slice.
//! 2. **Payload over-read.** The decoder reads 16 bytes from the start of every block
//!    (`load128_and_broadcast_to_256`), more than the block occupies. `compute_safe_simd_padding`
//!    sizes the padding so that at least 16 bytes are readable from the start of the last block,
//!    counting the padding and the value bytes that follow.
//! 3. **Selectors.** Block `i` uses the nibble `i % 2` of selector byte `i / 2`; the iterators
//!    never go past block `ceil(n/8) - 1`, so `get_unchecked` on the selectors stays in bounds.
//! 4. **Values.** The value cursor advances by 8 per full block and by `n % 8` in the tail:
//!    exactly `n` values in total.
//! 5. **Gather indices.** Components are mapped ids `< dim`, and the dense query has `dim`
//!    entries. Evaluators gather at `prefix_sum(gaps)` relative to a query slice that starts at
//!    the previous block's last component, then advance the slice with `split_at_unchecked` by
//!    the block's last relative component. Every index therefore stays below the remaining
//!    slice length (`debug_assert!`ed at each site).
//!
//! Violating (1)-(4) means the blob was not produced by `push_encoded` (or was corrupted):
//! this is why the view constructors are `unsafe`.

use half::f16;
use std::simd::Simd;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{__m256, __m256i};

use crate::{
    encoders::dotpacking8::swizzle::fast_lane_swizzle, utils::load128_and_broadcast_to_256,
};

const N: usize = 8;

#[repr(C, align(32))]
#[derive(Copy, Clone)]
pub struct Block8Constants {
    pub shuffle: Simd<u8, 32>, // 32 byte
    pub shifts: Simd<u32, 8>,  // 32 byte
    pub masks: Simd<u32, 8>,   // 32 byte
}

pub static BLOCK8_TABLE: [Block8Constants; 16] = gen_block8_table();

const fn gen_block8_table() -> [Block8Constants; 16] {
    let mut all = [Block8Constants {
        shuffle: Simd::from_array([0u8; 32]),
        shifts: Simd::from_array([0u32; 8]),
        masks: Simd::from_array([0u32; 8]),
    }; 16];

    let mut b_idx = 0;
    while b_idx < 16 {
        let b = b_idx + 1;

        // 1. Generate Shuffle Mask
        let mut m_arr = [0xFFu8; 32];
        let mut i = 0;
        while i < 8 {
            let pos_bits = i * b;
            let byte_off = (pos_bits / 8) as u8;

            let dest_idx = i * 4;

            m_arr[dest_idx] = byte_off;
            m_arr[dest_idx + 1] = byte_off + 1;
            m_arr[dest_idx + 2] = byte_off + 2;
            m_arr[dest_idx + 3] = byte_off + 3;
            i += 1;
        }

        // 2. Generate Bit Shifts e Width Masks
        let mut s_arr = [0u32; 8];
        let mut i = 0;
        while i < 8 {
            s_arr[i] = (i * b % 8) as u32;
            i += 1;
        }
        let mask = (1u32 << b) - 1;
        all[b_idx].shuffle = Simd::from_array(m_arr);
        all[b_idx].shifts = Simd::from_array(s_arr);
        all[b_idx].masks = Simd::splat(mask);

        b_idx += 1;
    }
    all
}

pub fn encode_blocks(gaps: &[u32]) -> (Vec<u8>, Vec<u8>) {
    const { assert!(N <= 8) }; // below we assume N gaps fit in a u128.

    let n_blocks = gaps.len().div_ceil(N);
    let mut selectors = vec![0u8; n_blocks.div_ceil(2)];
    let mut payloads = Vec::new();
    for (idx, chunk) in gaps.chunks(N).enumerate() {
        let max_val = chunk.iter().cloned().max().unwrap_or(0);
        let b = (u32::BITS - max_val.max(1).leading_zeros()) as u8;

        let selector_b = b - 1;
        if idx.is_multiple_of(2) {
            selectors[idx / 2] |= selector_b & 0x0F;
        } else {
            selectors[idx / 2] |= (selector_b & 0x0F) << 4;
        }

        // Here we assume that b <= 16, this way N (<= 8) values fit in a u128
        let mut acc = 0u128;
        for (i, &g) in chunk.iter().enumerate() {
            acc |= (g as u128) << (i as u8 * b);
        }
        let bytes_to_write = (chunk.len() * (b as usize)).div_ceil(8);
        payloads.extend_from_slice(&acc.to_le_bytes()[..bytes_to_write]);
    }
    (selectors, payloads)
}

#[inline(always)]
pub fn simd_prefix_sum(mut n: Simd<u32, N>) -> Simd<u32, N> {
    n += n.shift_elements_right::<1>(0);
    n += n.shift_elements_right::<2>(0);
    n += n.shift_elements_right::<4>(0);
    n
}

/// Gather `query_ptr[components[i]]` for the 8 lanes.
///
/// # Safety
///
/// Every lane of `components` must be `< query_ptr.len()`: the gather is unchecked.
#[inline(always)]
pub unsafe fn fast_lane_gather(query_ptr: &[f32], components: Simd<u32, N>) -> Simd<f32, N> {
    debug_assert!(
        components.to_array().iter().all(|&c| (c as usize) < query_ptr.len()),
        "gather index out of bounds"
    );
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
        let q_vals: __m256 =
            _mm256_i32gather_ps(query_ptr.as_ptr(), components.into(), 4);
        Simd::<f32, N>::from(q_vals)
    }

    #[cfg(not(target_arch = "x86_64"))]
    unsafe {
        use std::simd::Mask;
        Simd::gather_select_unchecked(
            query_ptr,
            Mask::splat(true),
            components.cast(),
            Simd::splat(0.0),
        )
    }
}

#[inline(always)]
pub fn compute_safe_simd_padding(
    n_elem: usize,
    selectors: &[u8],
    alignment_padding: usize,
    bytes_after_components: usize,
) -> usize {
    if n_elem == 0 {
        return alignment_padding;
    }
    let n_blocks = n_elem.div_ceil(N);
    let idx = n_blocks - 1;
    let sel = selectors[idx / 2];
    let b = (if idx.is_multiple_of(2) {
        sel & 0x0F
    } else {
        (sel >> 4) & 0x0F
    }) as usize
        + 1;
    let rem = n_elem % N;
    let elems = if rem == 0 { N } else { rem };
    let last_read_bytes = (elems * b).div_ceil(8);

    // SIMD Padding Logic:
    // The decoder performs 128-bit (16 bytes) unaligned loads.
    // To prevent Out-of-Bounds (OOB) access, we must ensure that the distance between
    // the start of the last SIMD read and the end of the allocated buffer is at least 16 bytes.
    // [last_read_bytes]    = bytes belonging to the last block in the bitstream
    // [alignment_padding] = existing alignment padding (64-bit)
    // [bytes_after]        = subsequent data present after the gap bitstream
    let mut space_after = alignment_padding + bytes_after_components;
    while last_read_bytes + space_after < 16 {
        space_after += 8;
    }
    space_after - bytes_after_components
}

/// Block-by-block decoder over one encoded vector with `u8` values. Built by a view
/// (`iter_raw`) or `nibble4bit::gap_iter`; relies on the module-level safety invariants.
pub struct DotPacking8Iter<'a> {
    pub bulk_blocks: usize,
    pub block_idx: usize,
    pub n: usize,
    pub selectors: &'a [u8],
    pub payload_ptr: *const u8,
    pub val_ptr: *const u8,
}

impl<'a> Iterator for DotPacking8Iter<'a> {
    type Item = (Simd<u32, 8>, Simd<u8, 8>);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.block_idx >= self.bulk_blocks {
            return None;
        }

        let gaps = self.decode_lane();

        let vals = unsafe { self.val_ptr.cast::<Simd<u8, N>>().read_unaligned() };
        self.val_ptr = unsafe { self.val_ptr.add(N) };
        self.block_idx += 1;
        Some((gaps, vals))
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remain = self.bulk_blocks - self.block_idx;
        (remain, Some(remain))
    }
}

impl<'a> ExactSizeIterator for DotPacking8Iter<'a> {}

impl<'a> DotPacking8Iter<'a> {
    #[inline(always)]
    fn extract_block_b(&self) -> usize {
        debug_assert!(self.block_idx / 2 < self.selectors.len());
        // SAFETY: invariant (3) in the module docs.
        let sel_byte = *unsafe { self.selectors.get_unchecked(self.block_idx / 2) };
        (((sel_byte >> ((self.block_idx & 1) << 2)) & 0x0F) as usize) + 1
    }

    #[inline(always)]
    fn decode_lane_with_b(&mut self, b: usize, advance_bytes: usize) -> Simd<u32, 8> {
        // SAFETY: 16 bytes are readable from the start of every block, invariant (2).
        let comps = unsafe { load128_and_broadcast_to_256(self.payload_ptr) };
        let table = unsafe { BLOCK8_TABLE.get_unchecked(b - 1) };
        let shuffled = fast_lane_swizzle(comps, table.shuffle);
        let gaps: Simd<u32, 8> = unsafe { std::mem::transmute(shuffled) };
        self.payload_ptr = unsafe { self.payload_ptr.add(advance_bytes) };
        (gaps >> table.shifts) & table.masks
    }

    #[inline(always)]
    pub fn decode_lane(&mut self) -> Simd<u32, 8> {
        let b = self.extract_block_b();
        self.decode_lane_with_b(b, b)
    }

    #[inline(always)]
    pub fn decode_tail(&mut self) -> (Simd<u32, N>, &'a [u8], usize) {
        let remaining = self.n % N;
        if remaining == 0 {
            return (Simd::splat(0), &[], 0);
        }
        let b = self.extract_block_b();
        let bytes_to_advance = (remaining * b).div_ceil(8);
        let gaps = self.decode_lane_with_b(b, bytes_to_advance);
        let tail_vals = unsafe { std::slice::from_raw_parts(self.val_ptr, remaining) };
        self.val_ptr = unsafe { self.val_ptr.add(remaining) };
        (gaps, tail_vals, remaining)
    }
}

/// Load 8 consecutive `f16` values from `ptr` and widen them to `f32`.
///
/// # Safety
///
/// `ptr` must be valid for reading 8 `f16` values (16 bytes). It does not need to be aligned.
#[inline(always)]
pub unsafe fn f16_ptr_to_f32x8(ptr: *const f16) -> Simd<f32, 8> {
    #[cfg(all(target_arch = "x86_64", target_feature = "f16c"))]
    {
        use std::arch::x86_64::*;
        unsafe {
            let m128 = _mm_loadu_si128(ptr.cast());
            let m256 = _mm256_cvtph_ps(m128);
            std::mem::transmute(m256)
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "f16c")))]
    {
        let val_f32: [f32; 8] = std::array::from_fn(|i| (unsafe { *ptr.add(i) }).to_f32());
        Simd::<f32, 8>::from_array(val_f32)
    }
}

/// Same as [`DotPacking8Iter`] with `f16` values.
pub struct DotPacking8F16Iter<'a> {
    pub bulk_blocks: usize,
    pub block_idx: usize,
    pub n: usize,
    pub selectors: &'a [u8],
    pub payload_ptr: *const u8,
    pub val_ptr: *const f16,
}

impl<'a> Iterator for DotPacking8F16Iter<'a> {
    type Item = (Simd<u32, 8>, Simd<f32, 8>);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.block_idx >= self.bulk_blocks {
            return None;
        }

        let gaps = self.decode_lane();

        let vals = unsafe { f16_ptr_to_f32x8(self.val_ptr) };
        self.val_ptr = unsafe { self.val_ptr.add(8) };
        self.block_idx += 1;
        Some((gaps, vals))
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remain = self.bulk_blocks - self.block_idx;
        (remain, Some(remain))
    }
}

impl<'a> ExactSizeIterator for DotPacking8F16Iter<'a> {}

impl<'a> DotPacking8F16Iter<'a> {
    #[inline(always)]
    fn extract_block_b(&self) -> usize {
        debug_assert!(self.block_idx / 2 < self.selectors.len());
        // SAFETY: invariant (3) in the module docs.
        let sel_byte = *unsafe { self.selectors.get_unchecked(self.block_idx / 2) };
        (((sel_byte >> ((self.block_idx & 1) << 2)) & 0x0F) as usize) + 1
    }

    #[inline(always)]
    fn decode_lane_with_b(&mut self, b: usize, advance_bytes: usize) -> Simd<u32, 8> {
        // SAFETY: 16 bytes are readable from the start of every block, invariant (2).
        let comps = unsafe { load128_and_broadcast_to_256(self.payload_ptr) };
        let table = unsafe { BLOCK8_TABLE.get_unchecked(b - 1) };
        let shuffled = fast_lane_swizzle(comps, table.shuffle);
        let gaps: Simd<u32, 8> = unsafe { std::mem::transmute(shuffled) };
        self.payload_ptr = unsafe { self.payload_ptr.add(advance_bytes) };
        (gaps >> table.shifts) & table.masks
    }

    #[inline(always)]
    pub fn decode_lane(&mut self) -> Simd<u32, 8> {
        let b = self.extract_block_b();
        self.decode_lane_with_b(b, b)
    }

    #[inline(always)]
    pub fn decode_tail(&mut self) -> (Simd<u32, 8>, &'a [f16], usize) {
        let remaining = self.n % 8;
        if remaining == 0 {
            return (Simd::splat(0), &[], 0);
        }
        let b = self.extract_block_b();
        let bytes_to_advance = (remaining * b).div_ceil(8);
        let gaps = self.decode_lane_with_b(b, bytes_to_advance);
        let tail_vals = unsafe { std::slice::from_raw_parts(self.val_ptr, remaining) };
        self.val_ptr = unsafe { self.val_ptr.add(remaining) };
        (gaps, tail_vals, remaining)
    }
}

/// Add `query_value * vbits[i]` to `acc` in every lane `i` where `comps[i] == query_component`.
///
/// # Safety
///
/// The CPU must support AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[inline]
pub unsafe fn match_and_fma_8(
    query_component: u32,
    query_value: f32,
    comps: __m256i,
    vbits: __m256,
    acc: &mut __m256,
) {
    use std::arch::x86_64::*;
    let q32: __m256i = _mm256_set1_epi32(query_component as i32);
    let cmp: __m256i = _mm256_cmpeq_epi32(comps, q32);
    let vbits_masked: __m256 = _mm256_and_ps(vbits, _mm256_castsi256_ps(cmp));
    let scale: __m256 = _mm256_set1_ps(query_value);
    *acc = _mm256_fmadd_ps(vbits_masked, scale, *acc);
}
