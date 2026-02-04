/// Simplified distance helpers for Product Quantization.
/// These are placeholders for the SIMD-optimized path and currently fallback to scalar loops.
use crate::core::vector::DenseVectorView;
use crate::distances::SquaredEuclideanDistance;
use crate::{Dataset, PlainDenseDataset};

/// Build the squared Euclidean distance table for a specific subspace.
pub fn compute_distance_table(
    distance_table: &mut [f32],
    query: DenseVectorView<'_, f32>,
    centroids: &PlainDenseDataset<f32, SquaredEuclideanDistance>,
) {
    for (i, slot) in distance_table.iter_mut().enumerate() {
        let centroid = centroids.get(i as crate::VectorId);
        *slot = query
            .values()
            .iter()
            .zip(centroid.values().iter())
            .map(|(&q, &c)| (q - c).algebraic_mul(q - c))
            .fold(0.0f32, |acc, x| acc.algebraic_add(x));
    }
}

/// Build the dot-product table for a specific subspace.
pub fn compute_dot_product_table(
    distance_table: &mut [f32],
    query: DenseVectorView<'_, f32>,
    centroids: &PlainDenseDataset<f32, SquaredEuclideanDistance>,
) {
    for (i, slot) in distance_table.iter_mut().enumerate() {
        let centroid = centroids.get(i as crate::VectorId);
        *slot = query
            .values()
            .iter()
            .zip(centroid.values().iter())
            .map(|(&q, &c)| q.algebraic_mul(c))
            .fold(0.0f32, |acc, x| acc.algebraic_add(x));
    }
}

/// Find the nearest centroid inside a subspace by scanning all `ksub` entries.
pub fn find_nearest_centroid_idx(
    query_sub: DenseVectorView<'_, f32>,
    centroids: &PlainDenseDataset<f32, SquaredEuclideanDistance>,
) -> usize {
    let mut best_idx = 0;
    let mut best_distance = f32::INFINITY;
    for i in 0..centroids.len() {
        let centroid = centroids.get(i as crate::VectorId);
        let dist = query_sub
            .values()
            .iter()
            .zip(centroid.values().iter())
            .map(|(&q, &c)| (q - c).algebraic_mul(q - c))
            .fold(0.0f32, |acc, x| acc.algebraic_add(x));
        if dist < best_distance {
            best_distance = dist;
            best_idx = i;
        }
    }
    best_idx
}
