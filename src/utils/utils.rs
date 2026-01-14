use std::collections::HashSet;
use std::hash::Hash;

use crate::ComponentType;
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
pub(crate) fn permute_components_with_bisection<C, Item>(
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

#[cfg(test)]
mod tests {
    use super::{
        intersection, is_strictly_sorted, permute_components_with_bisection,
    };

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
}
