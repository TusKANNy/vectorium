use std::simd::{Simd};


use crate::{encoders::dotpacking8::swizzle::fast_lane_swizzle, utils::load128_and_broadcast_to_256};

const N: usize = 8;

#[repr(C, align(32))]
#[derive(Copy, Clone)]
struct Block8Constants {
    shuffle: Simd<u8, 32>, // 32 byte
    shifts: Simd<u32, 8>,  // 32 byte
    masks: Simd<u32, 8>,   // 32 byte
}

static BLOCK8_TABLE: [Block8Constants; 16] = gen_block8_table();

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
    let n_blocks = (gaps.len() + N - 1) / N;
    let mut selectors = vec![0u8; (n_blocks + 1) / 2];
    let mut payloads = Vec::new();
    for (idx, chunk) in gaps.chunks(N).enumerate() {
        let max_val = chunk.iter().cloned().max().unwrap_or(0);
        let b = if max_val == 0 {
            1
        } else {
            (32 - max_val.leading_zeros()) as u8
        };

        let selector_b = b - 1;
        if idx % 2 == 0 {
            selectors[idx / 2] |= selector_b & 0x0F;
        } else {
            selectors[idx / 2] |= (selector_b & 0x0F) << 4;
        }

        let mut acc = 0u128;
        for (i, &g) in chunk.iter().enumerate() {
            acc |= (g as u128) << (i as u8 * b);
        }
        let bytes_to_write = (chunk.len() * (b as usize) + 7) / 8;
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

#[inline(always)]
pub fn compute_safe_simd_padding(
    n_elem: usize,
    selectors: &[u8],
    padding_before: usize,
    trailing_bytes: usize,
) -> usize {
    if n_elem == 0 {
        return padding_before;
    }
    let n_blocks = (n_elem + N - 1) / N;
    let idx = n_blocks - 1;
    let sel = selectors[idx / 2];
    let b = (if idx % 2 == 0 {
        sel & 0x0F
    } else {
        (sel >> 4) & 0x0F
    }) as usize
        + 1;
    let rem = n_elem % N;
    let elems = if rem == 0 { N } else { rem };
    let last_read_bytes = (elems * b + 7) / 8;

    // SIMD Padding Logic:
    // The decoder performs 128-bit (16 bytes) unaligned loads.
    // To prevent Out-of-Bounds (OOB) access, we must ensure that the distance between
    // the start of the last SIMD read and the end of the allocated buffer is at least 16 bytes.
    // [last_read_bytes] = bytes belonging to the last block in the bitstream
    // [padding_before]  = existing alignment padding (64-bit)
    // [trailing_bytes]  = subsequent data present after the gap bitstream
    let mut space_after = padding_before + trailing_bytes;
    while last_read_bytes + space_after < 16 {
        space_after += 8;
    }
    space_after - trailing_bytes
}

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
        let sel_byte = *unsafe { self.selectors.get_unchecked(self.block_idx / 2) };
        (((sel_byte >> ((self.block_idx & 1) << 2)) & 0x0F) as usize) + 1
    }

    #[inline(always)]
    fn decode_lane_with_b(&mut self, b: usize, advance_bytes: usize) -> Simd<u32, 8> {
        let comps = load128_and_broadcast_to_256(self.payload_ptr);
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
        let bytes_to_advance = (remaining * b + 7) / 8;
        let gaps = self.decode_lane_with_b(b, bytes_to_advance);
        let tail_vals = unsafe { std::slice::from_raw_parts(self.val_ptr, remaining) };
        self.val_ptr = unsafe { self.val_ptr.add(remaining) };
        (gaps, tail_vals, remaining)
    }
}
