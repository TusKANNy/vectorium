use std::simd::{Mask, Simd};

use crate::{encoders::dotpacking8::swizzle::swizzle, utils::load128_and_broadcast_to_256};
use std::simd::Select;

const N: usize = 8;

#[repr(C, align(32))]
#[derive(Copy, Clone)]
struct BlockDpConstants {
    shuffle: Simd<u8, 32>,
    shifts: Simd<u32, 8>,
    masks: Simd<u32, 8>,
}

// Ora abbiamo solo 16 (b) * 8 (s) = 128 combinazioni
static BLOCK_DP_TABLE: [BlockDpConstants; 128] = gen_block_dp_table();

const fn gen_block_dp_table() -> [BlockDpConstants; 128] {
    let mut all = [BlockDpConstants {
        shuffle: Simd::from_array([0u8; 32]),
        shifts: Simd::from_array([0u32; 8]),
        masks: Simd::from_array([0u32; 8]),
    }; 128];

    let mut b_idx = 0;
    while b_idx < 16 {
        let b = b_idx + 1;
        let mask = (1u32 << b) - 1;
        let mut s_idx = 0;
        while s_idx < 8 {
            let mut m_arr = [0xFFu8; 32];
            let mut shifts_arr = [0u32; 8];
            let mut masks_arr = [0u32; 8];
            let mut i = 0;
            while i < 8 {
                let pos_bits = s_idx + (i * b);
                let byte_off = (pos_bits / 8) as u8;
                let dest_idx = if i < 4 { i * 4 } else { (i - 4) * 4 + 16 };

                m_arr[dest_idx] = byte_off;
                m_arr[dest_idx + 1] = byte_off + 1;
                m_arr[dest_idx + 2] = byte_off + 2;
                m_arr[dest_idx + 3] = byte_off + 3;

                shifts_arr[i] = (pos_bits % 8) as u32;
                masks_arr[i] = mask;
                i += 1;
            }

            let table_idx = b_idx * 8 + s_idx;
            all[table_idx].shuffle = Simd::from_array(m_arr);
            all[table_idx].shifts = Simd::from_array(shifts_arr);
            all[table_idx].masks = Simd::from_array(masks_arr);
            s_idx += 1;
        }
        b_idx += 1;
    }
    all
}

pub struct DotPackingDpIter<'a, const MAX_BLOCK_LEN: usize> {
    pub n: usize,
    pub processed_count: usize,
    pub sel_idx: usize,
    pub chunk_rem: usize,
    pub b: usize,
    pub global_bit_pos: usize,
    pub overflow_count: usize,
    pub overflow_reg: Simd<u32, N>,
    pub permuted_reg: Simd<u32, N>,
    pub selectors: &'a [u8],
    pub payload_ptr: *const u8,
    pub val_ptr: *const u8,
}

impl<'a, const MAX_BLOCK_LEN: usize> Iterator for DotPackingDpIter<'a, MAX_BLOCK_LEN> {
    type Item = (Simd<u32, N>, Simd<u8, N>);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if MAX_BLOCK_LEN <= 8 {
            self.next_8()
        } else {
            self.next_16()
        }
    }
}

impl<'a, const MAX_BLOCK_LEN: usize> DotPackingDpIter<'a, MAX_BLOCK_LEN> {
    #[inline(always)]
    fn next_8(&mut self) -> Option<(Simd<u32, N>, Simd<u8, N>)> {
        loop {
            if self.processed_count >= self.n {
                return None;
            }
            let sel = *unsafe { self.selectors.get_unchecked(self.sel_idx) };
            self.sel_idx += 1;

            self.b = (sel >> 4) as usize + 1;
            let l = (sel & 0x0F) as usize + 1;

            self.process_chunk(l);
            if self.overflow_count >= N {
                return Some(self.take_full_register());
            }
        }
    }

    #[inline(always)]
    fn next_16(&mut self) -> Option<(Simd<u32, N>, Simd<u8, N>)> {
        if self.chunk_rem > 0 {
            let rem = self.chunk_rem;
            self.chunk_rem = 0;
            self.process_chunk(rem);
            if self.overflow_count >= N {
                return Some(self.take_full_register());
            }
        }

        loop {
            if self.processed_count >= self.n {
                return None;
            }
            let sel = *unsafe { self.selectors.get_unchecked(self.sel_idx) };
            self.sel_idx += 1;

            let b = (sel >> 4) as usize + 1;
            let len = (sel & 0x0F) as usize + 1;
            self.b = b;

            if len > 8 {
                self.process_chunk(8);
                self.chunk_rem = len - 8;
                return Some(self.take_full_register());
            } else {
                self.process_chunk(len);
                if self.overflow_count >= N {
                    return Some(self.take_full_register());
                }
            }
        }
    }

    #[inline(always)]
    pub fn process_chunk(&mut self, to_process: usize) {
        let byte_off = self.global_bit_pos >> 3;
        let bit_off = self.global_bit_pos & 7;
        let ptr = unsafe { self.payload_ptr.add(byte_off) };
        let overflow_entry = unsafe { OVERFLOW_TABLE.get_unchecked(self.overflow_count) };
        let table_idx = (self.b - 1) * 8 + bit_off;
        let table_ptr = unsafe { BLOCK_DP_TABLE.get_unchecked(table_idx) };
        let comps = load128_and_broadcast_to_256(ptr);
        let shuffled_simd_u8 = swizzle(comps, table_ptr.shuffle);
        let shuffled: Simd<u32, N> = unsafe { std::mem::transmute(shuffled_simd_u8) };
        let gaps_zero_aligned = (shuffled >> table_ptr.shifts) & table_ptr.masks;
        let reg = unsafe { vpermd_u32x8(gaps_zero_aligned, overflow_entry.rotate_left) };

        self.global_bit_pos += to_process * self.b;
        let mask = unsafe { Mask::from_simd_unchecked(overflow_entry.blend_mask) };
        self.overflow_reg = mask.select(self.overflow_reg, reg);
        self.overflow_count += to_process;
        self.processed_count += to_process;
        self.permuted_reg = reg;
    }

    #[inline(always)]
    pub fn take_full_register(&mut self) -> (Simd<u32, N>, Simd<u8, N>) {
        let gaps = self.overflow_reg;
        self.overflow_reg = self.permuted_reg;
        self.overflow_count -= N;

        let vals = unsafe { self.val_ptr.cast::<Simd<u8, N>>().read_unaligned() };
        self.val_ptr = unsafe { self.val_ptr.add(8) };

        (gaps, vals)
    }

    #[inline(always)]
    pub fn decode_tail(&self) -> (Simd<u32, N>, &'a [u8], usize) {
        let remaining = self.n % N;
        if remaining == 0 {
            return (Simd::splat(0), &[], 0);
        }
        let remaining = self.overflow_count;
        let vals_slice = unsafe { std::slice::from_raw_parts(self.val_ptr, remaining) };
        (self.overflow_reg, vals_slice, remaining)
    }
}

#[inline]
fn lane_decode_fits_128(bit_off: usize, b: usize) -> bool {
    bit_off + (N * b) <= 128
}

#[inline]
fn block_decode_fits_128(bit_off: usize, b: usize, len: usize) -> bool {
    if !lane_decode_fits_128(bit_off, b) {
        return false;
    }
    if len > N {
        let lane2_bit_off = (bit_off + (N * b)) % 8;
        return lane_decode_fits_128(lane2_bit_off, b);
    }
    true
}

#[inline]
pub fn min_bit_width_16(n: u32) -> usize {
    if n == 0 {
        1
    } else {
        (32 - n.leading_zeros()) as usize
    }
}

pub fn encode_blocks_dp<const MAX_BLOCK_LEN: usize>(gaps: &[u32]) -> (Vec<u8>, Vec<u8>) {
    let plan = plan_blocks_bit_aligned::<MAX_BLOCK_LEN>(gaps);
    let mut selectors = Vec::with_capacity(plan.len());
    let mut gap_bitstream = Vec::new();
    let mut bitbuf = 0u64;
    let mut bits_in_buf = 0usize;

    let mut write_bits = |value: u64, nbits: usize| {
        if nbits == 0 {
            return;
        }
        bitbuf |= (value & ((1u64 << nbits) - 1)) << bits_in_buf;
        bits_in_buf += nbits;
        while bits_in_buf >= 8 {
            gap_bitstream.push((bitbuf & 0xFF) as u8);
            bitbuf >>= 8;
            bits_in_buf -= 8;
        }
    };

    let mut pos = 0;
    for &(len, b) in &plan {
        selectors.push((((b - 1) as u8) << 4) | (len - 1) as u8);
        for i in 0..len {
            write_bits(gaps[pos + i] as u64, b);
        }
        pos += len;
    }

    if bits_in_buf > 0 {
        gap_bitstream.push((bitbuf & 0xFF) as u8);
    }

    (selectors, gap_bitstream)
}

#[inline(always)]
pub fn compute_safe_simd_padding(
    selectors: &[u8],
    padding_needed: usize,
    trailing_bytes: usize,
) -> usize {
    if selectors.is_empty() {
        return padding_needed;
    }
    let sel = *selectors.last().unwrap();
    let b = (sel >> 4) as usize + 1;
    let len = (sel & 0x0F) as usize + 1;

    // In DP, if len > 8, the block is processed in two SIMD reads.
    // The first read (8 elements) is guaranteed safe by the second half of the block.
    // Only the second read (len - 8 elements) can go OOB.
    let last_read_elems = if len > 8 { len - 8 } else { len };
    let last_read_bytes = (last_read_elems * b + 7) / 8;

    // SIMD Padding Logic:
    // The decoder performs 128-bit (16 bytes) unaligned loads.
    // To prevent Out-of-Bounds (OOB) access, we must ensure that the distance between
    // the start of the last SIMD read and the end of the allocated buffer is at least 16 bytes.
    // [last_read_bytes] = bytes belonging to the last SIMD read in the bitstream
    // [padding_needed]  = existing alignment padding (64-bit)
    // [trailing_bytes]  = subsequent data (e.g., values) already present after the bitstream
    let mut space_after = padding_needed + trailing_bytes;
    while last_read_bytes + space_after < 16 {
        space_after += 8;
    }
    space_after - trailing_bytes
}

pub fn plan_blocks_bit_aligned<const MAX_BLOCK_LEN: usize>(gaps: &[u32]) -> Vec<(usize, usize)> {
    let n = gaps.len();
    if n == 0 {
        return Vec::new();
    }

    let bw: Vec<u8> = gaps.iter().map(|&g| min_bit_width_16(g) as u8).collect();

    let mut dp = vec![[usize::MAX; 8]; n + 1];
    let mut choice_l = vec![[0usize; 8]; n + 1];
    let mut choice_b = vec![[0usize; 8]; n + 1];
    let mut parent_s = vec![[0usize; 8]; n + 1];

    dp[0][0] = 0;

    for i in 1..=n {
        let max_l = i.min(MAX_BLOCK_LEN);
        let mut current_max_b = 0;

        for l in 1..=max_l {
            current_max_b = current_max_b.max(bw[i - l] as usize);
            let b = current_max_b;
            for prev_s in 0..8 {
                if dp[i - l][prev_s] == usize::MAX {
                    continue;
                }
                if !block_decode_fits_128(prev_s, b, l) {
                    continue;
                }

                let curr_s = (prev_s + (l * b)) % 8;
                let cost = dp[i - l][prev_s] + 8 + (l * b);

                if cost <= dp[i][curr_s] {
                    dp[i][curr_s] = cost;
                    choice_l[i][curr_s] = l;
                    choice_b[i][curr_s] = b;
                    parent_s[i][curr_s] = prev_s;
                }
            }
        }
    }

    let mut plan = Vec::new();
    let mut curr_i = n;
    let mut curr_s = 0;
    let mut min_cost = usize::MAX;
    for s in 0..8 {
        if dp[n][s] < min_cost {
            min_cost = dp[n][s];
            curr_s = s;
        }
    }
    while curr_i > 0 {
        let l = choice_l[curr_i][curr_s];
        let b = choice_b[curr_i][curr_s];
        let prev_s = parent_s[curr_i][curr_s];
        plan.push((l, b));
        curr_i -= l;
        curr_s = prev_s;
    }
    plan.reverse();
    plan
}

#[inline(always)]
pub fn simd_prefix_sum(mut n: Simd<u32, N>) -> Simd<u32, N> {
    n += n.shift_elements_right::<1>(0);
    n += n.shift_elements_right::<2>(0);
    n += n.shift_elements_right::<4>(0);
    n
}

#[repr(C, align(32))]
#[derive(Copy, Clone)]
struct OverflowConstants {
    rotate_left: Simd<u32, 8>,
    blend_mask: Simd<i32, 8>,
}

static OVERFLOW_TABLE: [OverflowConstants; 8] = gen_overflow_table();

const fn gen_overflow_table() -> [OverflowConstants; 8] {
    let mut table = [OverflowConstants {
        rotate_left: Simd::from_array([0; 8]),
        blend_mask: Simd::from_array([0; 8]),
    }; 8];

    let mut k = 0;
    while k < 8 {
        // PERM_ROTATE_LEFT logic
        let mut rot_arr = [0u32; 8];
        let mut i = 0;
        while i < 8 {
            rot_arr[i] = ((i + (8 - k)) % 8) as u32;
            i += 1;
        }

        // BLEND_MASKS logic
        let mut blend_arr = [0i32; 8];
        let mut j = 0;
        while j < k {
            blend_arr[j] = -1;
            j += 1;
        }

        table[k].rotate_left = Simd::from_array(rot_arr);
        table[k].blend_mask = Simd::from_array(blend_arr);
        k += 1;
    }
    table
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn vpermd_u32x8(v: Simd<u32, 8>, idxs: Simd<u32, 8>) -> Simd<u32, 8> {
    use std::arch::x86_64::_mm256_permutevar8x32_epi32;
    use std::mem::transmute;
    unsafe { transmute(_mm256_permutevar8x32_epi32(transmute(v), transmute(idxs))) }
}

#[cfg(test)]
mod tests {
    use super::plan_blocks_bit_aligned;
    use super::min_bit_width_16;

    #[test]
    fn test_block_dp_plan() {
        let gaps: Vec<u16> = vec![
            7, 7, 3, 7, 7, 15, 15, 15, 3, 3, 16382, 16382, 4095, 16382, 4095,
        ];
        let plan =
            plan_blocks_bit_aligned::<16>(&gaps.iter().map(|&g| g as u32).collect::<Vec<u32>>());
        let expected_plan = vec![(10, 4), (5, 14)];
        assert_eq!(plan, expected_plan);
    }

    #[test]
    fn test_block_dp_plan2() {
        let gaps: Vec<u16> = vec![7, 7, 3, 7, 7, 15, 15, 15, 3, 16382, 7, 7, 3, 7, 31, 7];
        let plan =
            plan_blocks_bit_aligned::<16>(&gaps.iter().map(|&g| g as u32).collect::<Vec<u32>>());
        let expected_plan = vec![(9, 4), (1, 14), (6, 5)];
        let cost_dp = plan.iter().map(|(l, b)| 8 + (l * b)).sum::<usize>();
        let cost_block8 = gaps
            .chunks(8)
            .map(|chunks| {
                let max_b = chunks
                    .iter()
                    .map(|&g| min_bit_width_16(g as u32))
                    .max()
                    .unwrap_or(1) as usize;
                4 + (chunks.len() * max_b)
            })
            .sum::<usize>();
        println!("BlockDp Cost: {}, BPI: {}", cost_dp, cost_dp as f64 / 16.0);
        println!(
            "Block8 Cost: {}, BPI: {}",
            cost_block8,
            cost_block8 as f64 / 16.0
        );
        assert_eq!(plan, expected_plan);
    }
}
