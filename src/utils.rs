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
