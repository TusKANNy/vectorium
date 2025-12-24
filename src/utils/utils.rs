use crate::{ComponentType, ValueType, Vector1D};
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

// #[inline]
// pub fn prefetch_read_slice<T>(data: &[T]) {
//     let ptr = data.as_ptr() as *const i8;
//     // Cache line size on x86 is 64 bytes.
//     // The function is written with a pointer because iterating the array seems to prevent loop unrolling, for some reason.
//     // Prefetching the first two cache lines only is faster in modern CPUs. TODO: experiment this more.
//     let len = core::mem::size_of::<T>() * data.len();
//     for i in (0..len).step_by(64) {
//         core::intrinsics::prefetch_read_data::<_, 1>(ptr.wrapping_add(i));
//     }
// }

/// Compute a permutation of components using recursive graph bisection.
/// Components that often appear together in documents will be grouped close together.
pub fn permute_graph_bisection<C, V, InputVector>(
    dim: usize,
    vectors: impl Iterator<Item = InputVector>,
) -> Box<[usize]>
where
    InputVector: Vector1D<ComponentType = C, ValueType = V>,
    C: ComponentType,
    V: ValueType,
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
        for &component_id in vector.components_as_slice().iter() {
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
    use super::permute_graph_bisection;
    use crate::SparseVector1D;

    #[test]
    fn permute_graph_bisection_returns_permutation() {
        let vectors = vec![
            SparseVector1D::new(vec![0_u16, 2], vec![1.0_f32, 2.0]),
            SparseVector1D::new(vec![1_u16, 3], vec![3.0_f32, 4.0]),
        ];

        let perm = permute_graph_bisection(4, vectors.into_iter());
        assert_eq!(perm.len(), 4);

        let mut seen = vec![false; 4];
        for &p in perm.iter() {
            assert!(p < 4);
            assert!(!seen[p], "permutation contains duplicate value");
            seen[p] = true;
        }
        assert!(seen.into_iter().all(|v| v));
    }
}
