use std::collections::HashSet;
use std::hash::Hash;
use std::simd::Simd;

use crate::{ComponentType, Dataset, PlainSparseDataset, ScalarSparseSupportedDistance};

#[inline]
pub fn prefetch_read_slice<T>(data: &[T]) {
    let ptr = data.as_ptr() as *const u8;

    // Cache line size on x86 and most ARM CPUs is 64 bytes.
    // This is only a reasonable heuristic, not a strict guarantee.
    const CACHE_LINE: usize = 64;

    let len = core::mem::size_of_val(data);

    // Looping with pointer arithmetic improves unrolling and avoids bounds checks.
    // Prefetching only 256 bytes per vector is enough and prevents cache pollution.
    let len_to_prefetch = len.min(256);
    let mut i = 0;
    while i < len_to_prefetch {
        unsafe {
            // locality = 1: data will be used soon, but is not extremely hot.
            core::intrinsics::prefetch_read_data::<u8, 1>(ptr.add(i));
        }
        i += CACHE_LINE;
    }
}

#[inline]
pub fn is_strictly_sorted<T: Ord>(slice: &[T]) -> bool {
    slice.windows(2).all(|w| w[0] < w[1])
}

/// Computes the size of the intersection of two unsorted lists of integers.
pub fn intersection<T: Eq + Hash + Clone>(s: &[T], groundtruth: &[T]) -> usize {
    let s_set: HashSet<_> = s.iter().cloned().collect();
    let mut size = 0;
    for v in groundtruth {
        if s_set.contains(v) {
            size += 1;
        }
    }
    size
}

/// Compute a permutation of components ordered by decreasing document frequency: the component
/// that occurs in the most vectors gets the new id 0. Ties are broken by decreasing original id.
///
/// Gap-coded encoders benefit because frequent components end up packed together at low ids, so
/// each vector starts with small gaps and its few rare components come last. On MS MARCO
/// (cocondenser, splade_v3, inference_less) this is within 0.9% of the best ordering measured
/// (reversed graph bisection) and needs only one pass over the data.
///
/// Returns `permutation` with `permutation[original_id] = new_id`.
pub(crate) fn permute_components_by_frequency<C, Item>(
    dim: usize,
    vectors: impl Iterator<Item = Item>,
) -> Box<[usize]>
where
    C: ComponentType,
    Item: AsRef<[C]>,
{
    let mut frequency = vec![0usize; dim];
    for vector in vectors {
        for &component_id in vector.as_ref().iter() {
            let component_idx: usize = component_id.as_();
            frequency[component_idx] += 1;
        }
    }

    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_unstable_by_key(|&c| (std::cmp::Reverse(frequency[c]), std::cmp::Reverse(c)));

    let mut permutation = vec![0usize; dim];
    for (new_id, &original_id) in order.iter().enumerate() {
        permutation[original_id] = new_id;
    }

    permutation.into_boxed_slice()
}

/// Train per-component scalar quantization steps from percentile bounds.
///
/// Returns one step per component of `training_data`'s output dimension: the value `v` of a
/// component `c` encodes to `(v / quants[c]).clamp(0, 255)`. A step of `0.0` marks a component
/// that never occurs in the training data.
///
/// `lower_percentile` must be in `[0.0, 1.0)` and `upper_percentile` in
/// `(lower_percentile, 1.0]`.
///
/// Ported from branch feature/quantization (135e42e, extended in 72c2395), originally by
/// CosimoRulli.
pub fn train_sparse_scalar_quantizer<C, D>(
    training_data: &PlainSparseDataset<C, f32, D>,
    lower_percentile: f32,
    upper_percentile: f32,
) -> Vec<f32>
where
    C: ComponentType,
    D: ScalarSparseSupportedDistance,
{
    assert!(
        (0.0..1.0).contains(&lower_percentile),
        "lower_percentile must be in [0.0, 1.0), got {lower_percentile}"
    );
    assert!(
        (0.0..=1.0).contains(&upper_percentile) && upper_percentile > lower_percentile,
        "upper_percentile must be in (lower_percentile, 1.0], got {upper_percentile}"
    );

    let dim = training_data.output_dim();

    // Fast path: classic min-max quantization. Sorting every per-component bucket
    // is wasteful when we only need the maximum, and on high-dim sparse datasets
    // (e.g. SPLADE on MS MARCO) that sort dominates training time. Single pass,
    // no per-component allocations.
    if lower_percentile == 0.0 && upper_percentile >= 1.0 {
        let mut maxes = vec![0.0f32; dim];
        for doc in training_data.iter() {
            for (&c, &v) in doc.components().iter().zip(doc.values()) {
                let idx: usize = c.as_();
                if v > maxes[idx] {
                    maxes[idx] = v;
                }
            }
        }
        for q in maxes.iter_mut() {
            if *q > 0.0 {
                *q /= 255.0;
            }
        }
        return maxes;
    }

    // Collect per-component values
    let mut per_component: Vec<Vec<f32>> = vec![Vec::new(); dim];
    for doc in training_data.iter() {
        for (&c, &v) in doc.components().iter().zip(doc.values()) {
            let idx: usize = c.as_();
            per_component[idx].push(v);
        }
    }

    let mut quants = vec![0.0f32; dim];

    for i in 0..dim {
        let vals = &mut per_component[i];
        if vals.is_empty() {
            continue;
        }
        vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let min = if lower_percentile <= 0.0 {
            vals[0]
        } else {
            let idx = ((vals.len() as f32) * lower_percentile) as usize;
            let idx = idx.min(vals.len() - 1);
            vals[idx]
        };

        let max = if upper_percentile >= 1.0 {
            *vals.last().unwrap()
        } else {
            let idx = ((vals.len() as f32) * upper_percentile) as usize;
            let idx = idx.min(vals.len() - 1);
            vals[idx]
        };

        // This quantizer stores only a per-component step (no per-component offset),
        // so we fit the step to the robust span [min, max].
        let span = (max - min).max(0.0);
        if span > 0.0 {
            quants[i] = span / 255.0;
        } else if max > 0.0 {
            // Degenerate bucket (or identical percentiles): keep previous behavior.
            quants[i] = max / 255.0;
        }
    }

    quants
}

/// Load 16 bytes from `ptr` and broadcast them into both halves of a 256-bit vector.
///
/// The load is unaligned, so `ptr` only needs 16 readable bytes.
///
/// Ported from branch feature/quantization (7e5f5a0), originally by luca.rizzo.
#[inline(always)]
pub fn load128_and_broadcast_to_256(ptr: *const u8) -> Simd<u8, 32> {
    use std::simd::simd_swizzle;
    const BROADCAST_MASK: [usize; 32] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        12, 13, 14, 15,
    ];
    let raw = unsafe { ptr.cast::<Simd<u8, 16>>().read_unaligned() };
    simd_swizzle!(raw, BROADCAST_MASK)
}

#[cfg(test)]
mod tests {
    use super::{
        intersection, is_strictly_sorted, permute_components_by_frequency,
        train_sparse_scalar_quantizer,
    };
    use crate::core::dataset::DatasetGrowable;
    use crate::core::vector::SparseVectorView;
    use crate::{
        PlainSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer,
        SquaredEuclideanDistance,
    };

    fn build_1d_training_data(
        values: &[f32],
    ) -> PlainSparseDataset<u16, f32, SquaredEuclideanDistance> {
        let q = PlainSparseQuantizer::<u16, f32, SquaredEuclideanDistance>::new(1, 1);
        let mut g = PlainSparseDatasetGrowable::new(q);
        for &v in values {
            g.push(SparseVectorView::new(&[0_u16], &[v]));
        }
        g.into()
    }

    #[test]
    fn lower_percentile_affects_trained_quant_step() {
        let td = build_1d_training_data(&[1.0, 2.0, 3.0, 4.0]);

        let q0 = train_sparse_scalar_quantizer(&td, 0.0, 1.0);
        let q50 = train_sparse_scalar_quantizer(&td, 0.5, 1.0);
        assert!(
            q50[0] < q0[0],
            "expected lower_percentile=0.5 to reduce quant step: q0={} q50={}",
            q0[0],
            q50[0]
        );
    }

    #[test]
    fn permute_components_by_frequency_returns_permutation() {
        let vectors_components = vec![vec![0_u16, 2], vec![1_u16, 3]];

        let perm = permute_components_by_frequency(4, vectors_components.into_iter());
        assert_eq!(perm.len(), 4);

        let mut seen = vec![false; 4];
        for &p in perm.iter() {
            assert!(p < 4);
            assert!(!seen[p], "permutation contains duplicate value");
            seen[p] = true;
        }
        assert!(seen.into_iter().all(|v| v));
    }

    #[test]
    fn permute_components_by_frequency_puts_frequent_components_first() {
        // frequencies: c0 = 1, c1 = 3, c2 = 2, c3 = 0, c4 = 2
        let vectors_components = vec![vec![0_u16, 1, 2], vec![1_u16, 4], vec![1_u16, 2, 4]];

        let perm = permute_components_by_frequency(5, vectors_components.into_iter());

        assert_eq!(perm[1], 0, "most frequent component gets id 0");
        // c2 and c4 tie at frequency 2: decreasing original id breaks the tie.
        assert_eq!(perm[4], 1);
        assert_eq!(perm[2], 2);
        assert_eq!(perm[0], 3);
        assert_eq!(perm[3], 4, "unused component goes last");
    }

    #[test]
    fn permute_components_by_frequency_handles_empty_input() {
        let perm = permute_components_by_frequency::<u16, Vec<u16>>(3, std::iter::empty());
        // all frequencies are 0, so the order is decreasing original id
        assert_eq!(&*perm, &[2, 1, 0]);
    }

    #[test]
    fn is_strictly_sorted_handles_duplicates() {
        assert!(is_strictly_sorted(&[1u32, 2, 3]));
        assert!(!is_strictly_sorted(&[1u32, 1, 2]));
    }

    #[test]
    fn intersection_counts_shared_elements() {
        let a = vec![1i32, 2, 3, 5];
        let b = vec![2i32, 3, 4];
        assert_eq!(intersection(&a, &b), 2);
    }
}
