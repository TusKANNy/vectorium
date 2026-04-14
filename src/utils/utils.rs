use std::collections::HashSet;
use std::hash::Hash;

use crate::{ComponentType, Dataset, PlainSparseDataset, SquaredEuclideanDistance};

use rgb::forward::Doc;

#[inline]
pub fn prefetch_read_slice<T>(data: &[T]) {
    let ptr = data.as_ptr() as *const u8;

    // Cache line size on x86 and most ARM CPUs is 64 bytes.
    // This is only a reasonable heuristic, not a strict guarantee.
    const CACHE_LINE: usize = 64;

    let len = core::mem::size_of_val(data);

    // Looping with pointer arithmetic improves unrolling and avoids bounds checks.
    // Prefetching the first two cache lines only is faster in modern CPUs. TODO: experiment this more.
    let mut i = 0;
    while i < len {
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

/// Compute a permutation of components using recursive graph bisection.
/// Components that often appear together in documents will be grouped close together.
pub fn permute_components_with_bisection<C, Item>(
    dim: usize,
    vectors: impl Iterator<Item = Item>,
) -> Box<[usize]>
where
    C: ComponentType,
    Item: AsRef<[C]>,
{
    // One Doc for each component. RGB's terminology is the opposite of what we need for SparseVectors.
    let mut components = Vec::with_capacity(dim);
    for component_id in 0..dim {
        components.push(Doc {
            terms: Vec::with_capacity(256), // initial estimate for uniq terms in doc
            org_id: component_id as u32,
            gain: 0.0,
            leaf_id: -1,
        });
    }

    let mut doc_count = 0usize;
    for (doc_id, vector) in vectors.enumerate() {
        for &component_id in vector.as_ref().iter() {
            let component_idx: usize = component_id.as_();
            components[component_idx].terms.push(doc_id as u32);
        }
        doc_count = doc_id + 1;
    }

    const ITERATIONS: usize = 20;
    const MIN_PARTITION_SIZE: usize = 16;
    const MAX_DEPTH: usize = 100;
    const PARALLEL_SWITCH: usize = 10;

    rgb::recursive_graph_bisection(
        &mut components,
        doc_count,
        ITERATIONS,
        MIN_PARTITION_SIZE,
        MAX_DEPTH,
        PARALLEL_SWITCH,
        1,
        true,
        1,
    );

    let mut permutation = vec![0usize; components.len()];
    for (new_id, comp) in components.iter().enumerate() {
        permutation[comp.org_id as usize] = new_id;
    }

    permutation.into_boxed_slice()
}

/// Train the quantizer from data. This function is in the utils module as it is used by different quantizers.
///
/// `lower_percentile` controls the lower bound of the quantization range per component.
/// - `0.0` uses the absolute min (classic min–max uniform quantization).
/// - `0.25` uses the 25th percentile as a robust floor for the fitted range,
///   giving finer resolution to the upper 75% of values.
///
/// `upper_percentile` controls the upper bound of the quantization range per component.
/// - `1.0` uses the absolute max.
/// - `0.99` uses the 99th percentile as the upper bound. Values above are clipped to 255.
pub fn train_sparse_scalar_quantizer<C>(
    training_data: &PlainSparseDataset<C, f32, SquaredEuclideanDistance>,
    //training_data: impl Iterator<Item = SparseVectorView<'a, C, f32>>,
    lower_percentile: f32,
    upper_percentile: f32,
) -> Vec<f32>
where
    C: ComponentType,
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

/// Same as `train_sparse_scalar_quantizer` but divides by `num_levels` instead of 255.
/// `num_levels` is typically `(1 << nbits) - 1`.
pub fn train_sparse_scalar_quantizer_with_levels<C>(
    training_data: &PlainSparseDataset<C, f32, SquaredEuclideanDistance>,
    lower_percentile: f32,
    upper_percentile: f32,
    num_levels: f32,
) -> Vec<f32>
where
    C: ComponentType,
{
    assert!(
        (0.0..1.0).contains(&lower_percentile),
        "lower_percentile must be in [0.0, 1.0), got {lower_percentile}"
    );
    assert!(
        (0.0..=1.0).contains(&upper_percentile) && upper_percentile > lower_percentile,
        "upper_percentile must be in (lower_percentile, 1.0], got {upper_percentile}"
    );
    assert!(num_levels > 0.0, "num_levels must be positive, got {num_levels}");

    let dim = training_data.output_dim();

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
                *q /= num_levels;
            }
        }
        return maxes;
    }

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
            quants[i] = span / num_levels;
        } else if max > 0.0 {
            // Degenerate bucket (or identical percentiles): keep previous behavior.
            quants[i] = max / num_levels;
        }
    }

    quants
}

#[cfg(test)]
mod tests {
    use super::{
        intersection, is_strictly_sorted, permute_components_with_bisection,
        train_sparse_scalar_quantizer, train_sparse_scalar_quantizer_with_levels,
    };
    use crate::PlainSparseDatasetGrowable;
    use crate::core::dataset::DatasetGrowable;
    use crate::core::vector::SparseVectorView;
    use crate::encoders::sparse_scalar::PlainSparseQuantizer;
    use crate::{PlainSparseDataset, SquaredEuclideanDistance};

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
    fn permute_components_with_bisection_returns_permutation() {
        let vectors_components = vec![vec![0_u16, 2], vec![1_u16, 3]];

        let perm = permute_components_with_bisection(4, vectors_components.into_iter());
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

        let q0_levels = train_sparse_scalar_quantizer_with_levels(&td, 0.0, 1.0, 15.0);
        let q50_levels = train_sparse_scalar_quantizer_with_levels(&td, 0.5, 1.0, 15.0);
        assert!(
            q50_levels[0] < q0_levels[0],
            "expected lower_percentile=0.5 to reduce quant step (with levels): q0={} q50={}",
            q0_levels[0],
            q50_levels[0]
        );
    }
}
