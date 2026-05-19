use dsi_bitstream::prelude::*;

pub fn find_best_k_rice(gaps: &[u32]) -> u8 {
    (0..16)
        .map(|k| {
            let total_len: usize = gaps.iter().map(|&g| len_exp_golomb(g as u64, k)).sum();
            (k, total_len as f64 / gaps.len() as f64)
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(k, _)| k as u8)
        .unwrap_or(0)
}
