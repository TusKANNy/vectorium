use std::marker::PhantomData;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::vector_encoder::{
    QueryEvaluator, SparseDataEncoder, SparseVectorEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use crate::utils::is_strictly_sorted;
use crate::{ComponentType, Dataset, PlainSparseDataset, SpaceUsage, SparseVectorView};

/// Number of centroids per dimension (u8 codebook).
const NUM_CENTROIDS: usize = 256;

pub fn greedy_kmeans(values: &[f32], min: f32, max: f32, n_iterations: usize) -> Vec<f32> {
    let mut centroids = Vec::with_capacity(NUM_CENTROIDS);
    let span = (max - min) / (NUM_CENTROIDS as f32);
    for i in 0..NUM_CENTROIDS {
        centroids.push(min + span * i as f32);
    }

    let mut assignments = vec![0_usize; values.len()];

    for _ in 0..n_iterations {
        for (val_index, v) in values.iter().enumerate() {
            let distances = centroids.iter().map(|c| (c - v) * (c - v));
            let mut min_dist = f32::MAX;
            let mut argmin_index = 0_usize;
            for (i, d) in distances.enumerate() {
                if d < min_dist {
                    min_dist = d;
                    argmin_index = i;
                }
            }
            assignments[val_index] = argmin_index;
        }

        for centroid_index in 0..centroids.len() {
            let assigned_vals: Vec<_> = values
                .iter()
                .zip(assignments.iter().copied())
                .filter(|&(_, index)| index == centroid_index)
                .map(|(val, _)| *val)
                .collect();

            if !assigned_vals.is_empty() {
                centroids[centroid_index] =
                    assigned_vals.iter().sum::<f32>() / (assigned_vals.len() as f32);
            }
        }
    }

    centroids
}

/// Centroid-based sparse quantizer.
///
/// Each dimension has its own codebook of 256 centroids. Encoding maps each value
/// to the nearest centroid index (u8), and decoding looks up the centroid value.
///
/// Scoring uses a precomputed lookup table (LUT): for each query component `c`,
/// `lut[c][v] = q[c] * centroids[c][v]` for all 256 codes. Then the dot product
/// is just a sum of table lookups over the document's non-zero entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentroidSparseQuantizer<C, D> {
    dim: usize,
    /// Flat array of centroids: `centroids[c * 256 + v]` is the v-th centroid for dimension c.
    centroids: Box<[f32]>,
    _phantom: PhantomData<(C, D)>,
}

impl<C, D> PartialEq for CentroidSparseQuantizer<C, D> {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
    }
}

impl<C, D> CentroidSparseQuantizer<C, D> {
    /// Create a new quantizer with the given per-dimension centroids.
    ///
    /// `centroids` must have length `dim * 256`. Centroids for dimension `i` are at
    /// `centroids[i * 256 .. (i+1) * 256]` and must be sorted in ascending order.
    pub fn from_centroids(dim: usize, centroids: Box<[f32]>) -> Self {
        assert_eq!(
            centroids.len(),
            dim * NUM_CENTROIDS,
            "centroids length must be dim * 256"
        );
        Self {
            dim,
            centroids,
            _phantom: PhantomData,
        }
    }

    /// Train the quantizer using uniform centroids derived from the data.
    ///
    /// For each dimension, computes min/max and creates 256 evenly spaced centroids.
    /// `lower_percentile` and `upper_percentile` control the range, same as
    /// `UniformSparseQuantizer::train`.
    pub fn train(
        training_data: &PlainSparseDataset<C, f32, SquaredEuclideanDistance>,
        lower_percentile: f32,
        upper_percentile: f32,
        n_iterations: usize,
    ) -> Self
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

        // Collect per-component values
        let mut per_component: Vec<Vec<f32>> = vec![Vec::new(); dim];
        for doc in training_data.iter() {
            for (&c, &v) in doc.components().iter().zip(doc.values()) {
                let idx: usize = c.as_();
                per_component[idx].push(v);
            }
        }

        let mut centroids = vec![0.0f32; dim * NUM_CENTROIDS];

        per_component
            .par_iter_mut()
            .zip(centroids.par_chunks_mut(NUM_CENTROIDS))
            .for_each(|(vals, slot)| {
                if vals.is_empty() {
                    return;
                }
                vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

                // Apply percentile clipping before clustering
                let min = if lower_percentile == 0.0 {
                    *vals.first().unwrap()
                } else {
                    let idx = ((vals.len() as f32) * lower_percentile) as usize;
                    vals[idx.min(vals.len() - 1)]
                };

                let max = if upper_percentile >= 1.0 {
                    *vals.last().unwrap()
                } else {
                    let idx = ((vals.len() as f32) * upper_percentile) as usize;
                    vals[idx.min(vals.len() - 1)]
                };

                for v in vals.iter_mut() {
                    *v = v.clamp(min, max);
                }

                let dim_centroids = greedy_kmeans(vals, min, max, n_iterations);
                slot.copy_from_slice(&dim_centroids);
            });

        Self {
            dim,
            centroids: centroids.into_boxed_slice(),
            _phantom: PhantomData,
        }
    }

    /// Get the centroid table for a given dimension.
    #[inline]
    pub fn centroids_for_dim(&self, dim: usize) -> &[f32] {
        &self.centroids[dim * NUM_CENTROIDS..(dim + 1) * NUM_CENTROIDS]
    }

    /// Find the nearest centroid index for a value in a given dimension.
    #[inline]
    fn quantize(&self, dim: usize, value: f32) -> u8 {
        let table = self.centroids_for_dim(dim);
        // Binary search: find the insertion point, then pick the closer neighbor.
        match table.binary_search_by(|c| c.partial_cmp(&value).unwrap()) {
            Ok(idx) => idx as u8,
            Err(idx) => {
                if idx == 0 {
                    0
                } else if idx >= NUM_CENTROIDS {
                    255
                } else {
                    let lo = table[idx - 1];
                    let hi = table[idx];
                    if (value - lo) <= (hi - value) {
                        (idx - 1) as u8
                    } else {
                        idx as u8
                    }
                }
            }
        }
    }

    /// Dequantize: look up the centroid value.
    #[inline]
    fn dequantize(&self, dim: usize, code: u8) -> f32 {
        self.centroids[dim * NUM_CENTROIDS + code as usize]
    }
}

/// Distance dispatch trait for centroid-quantized sparse vectors.
///
/// For DotProduct:
///   `q · dequant(v) = Σ_{(c,v_int) in doc} lut[c][v_int]`
///   where `lut[c][v] = q[c] * centroids[c][v]`
///
/// For SquaredEuclidean:
///   `||q - dequant(v)||² = ||q||² - 2·Σ dot_lut[c][v_int] + Σ sq_lut[c][v_int]`
///   where `dot_lut[c][v] = q[c] * centroids[c][v]`, `sq_lut[c][v] = centroids[c][v]²`
pub trait CentroidQuantizedSparseSupportedDistance: Distance {
    fn requires_sq_lut() -> bool {
        false
    }

    fn requires_dot_query() -> bool {
        false
    }

    /// Score using a dense LUT (dim < 2^20).
    fn compute_dense_lut<C: ComponentType>(
        dense_lut: &[f32],
        dense_sq_lut: Option<&[f32]>,
        vector: SparseVectorView<'_, C, u8>,
        dot_query: Option<f32>,
    ) -> Self;

    /// Score using sparse merge (dim >= 2^20).
    fn compute_sparse<C: ComponentType>(
        query: &SparseVectorOwned<C, f32>,
        centroids: &[f32],
        vector: SparseVectorView<'_, C, u8>,
        dot_query: Option<f32>,
    ) -> Self;
}

impl CentroidQuantizedSparseSupportedDistance for DotProduct {
    #[inline]
    fn compute_dense_lut<C: ComponentType>(
        dense_lut: &[f32],
        _dense_sq_lut: Option<&[f32]>,
        vector: SparseVectorView<'_, C, u8>,
        _dot_query: Option<f32>,
    ) -> Self {
        let result =
            vector
                .components()
                .iter()
                .zip(vector.values())
                .fold(0.0f32, |acc, (&c, &v)| {
                    let idx: usize = c.as_();
                    acc.algebraic_add(unsafe {
                        *dense_lut.get_unchecked(idx * NUM_CENTROIDS + v as usize)
                    })
                });
        DotProduct::from(result)
    }

    #[inline]
    fn compute_sparse<C: ComponentType>(
        query: &SparseVectorOwned<C, f32>,
        centroids: &[f32],
        vector: SparseVectorView<'_, C, u8>,
        _dot_query: Option<f32>,
    ) -> Self {
        let result = sparse_merge_dot_product_with_centroids(query.as_view(), vector, centroids);
        DotProduct::from(result)
    }
}

fn compute_query_squared_norm(values: &[f32]) -> f32 {
    values
        .iter()
        .fold(0.0f32, |acc, &v| acc.algebraic_add(v.algebraic_mul(v)))
}

impl CentroidQuantizedSparseSupportedDistance for SquaredEuclideanDistance {
    #[inline]
    fn requires_sq_lut() -> bool {
        true
    }

    #[inline]
    fn requires_dot_query() -> bool {
        true
    }

    #[inline]
    fn compute_dense_lut<C: ComponentType>(
        dense_lut: &[f32],
        dense_sq_lut: Option<&[f32]>,
        vector: SparseVectorView<'_, C, u8>,
        dot_query: Option<f32>,
    ) -> Self {
        let dot_query =
            dot_query.expect("SquaredEuclideanDistance requires a precomputed ||q||² value");
        let sq_lut = dense_sq_lut.expect("SquaredEuclideanDistance requires sq_lut");

        let mut dot_qv = 0.0f32;
        let mut v_norm_sq = 0.0f32;
        for (&c, &v) in vector.components().iter().zip(vector.values()) {
            let idx: usize = c.as_();
            let offset = idx * NUM_CENTROIDS + v as usize;
            dot_qv = dot_qv.algebraic_add(unsafe { *dense_lut.get_unchecked(offset) });
            v_norm_sq = v_norm_sq.algebraic_add(unsafe { *sq_lut.get_unchecked(offset) });
        }

        let dist = dot_query
            .algebraic_add(v_norm_sq)
            .algebraic_sub(2.0f32.algebraic_mul(dot_qv));
        SquaredEuclideanDistance::from(dist)
    }

    #[inline]
    fn compute_sparse<C: ComponentType>(
        query: &SparseVectorOwned<C, f32>,
        centroids: &[f32],
        vector: SparseVectorView<'_, C, u8>,
        dot_query: Option<f32>,
    ) -> Self {
        let dot_query =
            dot_query.expect("SquaredEuclideanDistance requires a precomputed ||q||² value");

        let dot_qv = sparse_merge_dot_product_with_centroids(query.as_view(), vector, centroids);

        let v_norm_sq =
            vector
                .components()
                .iter()
                .zip(vector.values())
                .fold(0.0f32, |acc, (&c, &v)| {
                    let idx: usize = c.as_();
                    let v_real = centroids[idx * NUM_CENTROIDS + v as usize];
                    acc.algebraic_add(v_real.algebraic_mul(v_real))
                });

        let dist = dot_query
            .algebraic_add(v_norm_sq)
            .algebraic_sub(2.0f32.algebraic_mul(dot_qv));
        SquaredEuclideanDistance::from(dist)
    }
}

/// Merge-sort style dot product between a sparse f32 query and a sparse u8 vector,
/// dequantizing via centroid lookup.
#[inline]
fn sparse_merge_dot_product_with_centroids<C: ComponentType>(
    query: SparseVectorView<'_, C, f32>,
    vector: SparseVectorView<'_, C, u8>,
    centroids: &[f32],
) -> f32 {
    let q_components = query.components();
    let q_values = query.values();
    let v_components = vector.components();
    let v_values = vector.values();

    let mut qi = 0;
    let mut vi = 0;
    let mut result = 0.0f32;

    while qi < q_components.len() && vi < v_components.len() {
        unsafe {
            let qc: usize = q_components.get_unchecked(qi).as_();
            let vc: usize = v_components.get_unchecked(vi).as_();
            if qc == vc {
                let v_real = centroids[vc * NUM_CENTROIDS + v_values[vi] as usize];
                result = result.algebraic_add(q_values.get_unchecked(qi).algebraic_mul(v_real));
                qi += 1;
                vi += 1;
            } else if qc < vc {
                qi += 1;
            } else {
                vi += 1;
            }
        }
    }

    result
}

impl<C, D> SparseDataEncoder for CentroidSparseQuantizer<C, D>
where
    C: ComponentType,
    D: CentroidQuantizedSparseSupportedDistance,
{
    type InputComponentType = C;
    type InputValueType = f32;
    type OutputComponentType = C;
    type OutputValueType = u8;

    #[inline]
    fn decode_vector<'a>(
        &self,
        encoded: SparseVectorView<'a, Self::OutputComponentType, Self::OutputValueType>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let components = encoded.components().to_vec();
        let values: Vec<f32> = encoded
            .components()
            .iter()
            .zip(encoded.values())
            .map(|(&c, &v)| self.dequantize(c.as_(), v))
            .collect();
        SparseVectorOwned::new(components, values)
    }
}

impl<C, D> SparseVectorEncoder for CentroidSparseQuantizer<C, D>
where
    C: ComponentType,
    D: CentroidQuantizedSparseSupportedDistance,
{
    fn push_encoded<'a, ComponentContainer, ValueContainer>(
        &self,
        input: SparseVectorView<'a, C, f32>,
        components: &mut ComponentContainer,
        values: &mut ValueContainer,
    ) where
        ComponentContainer: Extend<Self::OutputComponentType>,
        ValueContainer: Extend<Self::OutputValueType>,
    {
        components.extend(input.components().iter().cloned());
        values.extend(
            input
                .components()
                .iter()
                .zip(input.values())
                .map(|(&c, &v)| self.quantize(c.as_(), v)),
        );
    }

    fn encode_vector<'a>(
        &self,
        input: Self::InputVector<'a>,
    ) -> SparseVectorOwned<Self::OutputComponentType, Self::OutputValueType> {
        let mut components = Vec::new();
        let mut values = Vec::new();
        self.push_encoded(input, &mut components, &mut values);
        SparseVectorOwned::new(components, values)
    }
}

impl<C, D> VectorEncoder for CentroidSparseQuantizer<C, D>
where
    C: ComponentType,
    D: CentroidQuantizedSparseSupportedDistance,
{
    type Distance = D;
    type InputVector<'a> = SparseVectorView<'a, C, f32>;
    type QueryVector<'q> = SparseVectorView<'q, C, f32>;
    type EncodedVector<'a> = SparseVectorView<'a, C, u8>;

    type Evaluator<'e>
        = CentroidSparseQueryEvaluator<'e, C, D>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        CentroidSparseQueryEvaluator::new(query, self)
    }

    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        CentroidSparseQueryEvaluator::new_from_owned_query(decoded, self)
    }

    #[inline]
    fn output_dim(&self) -> usize {
        self.dim
    }

    #[inline]
    fn input_dim(&self) -> usize {
        self.dim
    }
}

/// Query evaluator using precomputed LUT for centroid-based scoring.
///
/// For the dense case (dim < 2^20), stores a flat LUT of `dim * 256` entries:
/// `dense_lut[c * 256 + v] = q[c] * centroids[c][v]`
///
/// Scoring is a single pass: `Σ_{(c,v) in doc} dense_lut[c * 256 + v]`
#[derive(Debug, Clone)]
pub struct CentroidSparseQueryEvaluator<'e, C, D>
where
    C: ComponentType,
    D: CentroidQuantizedSparseSupportedDistance,
{
    // Dense LUT (dim < 2^20): dense_lut[c * 256 + v] = q[c] * centroids[c][v]
    dense_lut: Option<Vec<f32>>,
    // Dense squared LUT for Euclidean: sq_lut[c * 256 + v] = centroids[c][v]²
    dense_sq_lut: Option<Vec<f32>>,
    // Sparse query fallback (dim >= 2^20)
    sparse_query: Option<SparseVectorOwned<C, f32>>,
    // Precomputed ||q||² for Euclidean
    dot_query: Option<f32>,
    // Borrowed centroids for sparse path
    centroids: &'e [f32],
    _phantom: PhantomData<D>,
}

impl<'e, C, D> CentroidSparseQueryEvaluator<'e, C, D>
where
    C: ComponentType,
    D: CentroidQuantizedSparseSupportedDistance,
{
    pub fn new(
        query: SparseVectorView<'_, C, f32>,
        quantizer: &'e CentroidSparseQuantizer<C, D>,
    ) -> Self {
        let dot_query = if D::requires_dot_query() {
            Some(compute_query_squared_norm(query.values()))
        } else {
            None
        };

        let max_c = query
            .components()
            .iter()
            .map(|c| c.as_())
            .max()
            .unwrap_or(0);

        assert!(
            max_c < quantizer.input_dim(),
            "Query vector component exceeds quantizer input dimension."
        );

        assert_eq!(
            query.components().len(),
            query.values().len(),
            "Query vector components and values length mismatch."
        );

        let small_dim = quantizer.input_dim() < 2_usize.pow(20);

        if small_dim {
            let lut_size = quantizer.dim * NUM_CENTROIDS;
            let mut lut = vec![0.0f32; lut_size];

            // Only fill entries for query components (rest stays 0)
            for (&c, &qv) in query.components().iter().zip(query.values()) {
                let idx: usize = c.as_();
                let base = idx * NUM_CENTROIDS;
                for v in 0..NUM_CENTROIDS {
                    lut[base + v] = qv.algebraic_mul(quantizer.centroids[base + v]);
                }
            }

            // For Euclidean, also precompute sq_lut for all dimensions that appear
            // in any document (we fill all dims since we don't know which docs we'll see)
            let sq_lut = if D::requires_sq_lut() {
                let mut sq = vec![0.0f32; lut_size];
                for i in 0..quantizer.dim {
                    let base = i * NUM_CENTROIDS;
                    for v in 0..NUM_CENTROIDS {
                        let c_val = quantizer.centroids[base + v];
                        sq[base + v] = c_val.algebraic_mul(c_val);
                    }
                }
                Some(sq)
            } else {
                None
            };

            Self {
                dense_lut: Some(lut),
                dense_sq_lut: sq_lut,
                sparse_query: None,
                dot_query,
                centroids: &quantizer.centroids,
                _phantom: PhantomData,
            }
        } else {
            assert!(
                is_strictly_sorted(query.components()),
                "Query components must be sorted in strictly ascending order."
            );

            Self {
                dense_lut: None,
                dense_sq_lut: None,
                sparse_query: Some(SparseVectorOwned::new(
                    query.components().to_vec(),
                    query.values().to_vec(),
                )),
                dot_query,
                centroids: &quantizer.centroids,
                _phantom: PhantomData,
            }
        }
    }

    pub fn new_from_owned_query(
        query: SparseVectorOwned<C, f32>,
        quantizer: &'e CentroidSparseQuantizer<C, D>,
    ) -> Self {
        let dot_query = if D::requires_dot_query() {
            Some(compute_query_squared_norm(query.values()))
        } else {
            None
        };

        let max_c = query
            .components()
            .iter()
            .map(|c| c.as_())
            .max()
            .unwrap_or(0);

        assert!(
            max_c < quantizer.input_dim(),
            "Query vector component exceeds quantizer input dimension."
        );

        let small_dim = quantizer.input_dim() < 2_usize.pow(20);

        if small_dim {
            let lut_size = quantizer.dim * NUM_CENTROIDS;
            let mut lut = vec![0.0f32; lut_size];

            for (&c, &qv) in query.components().iter().zip(query.values()) {
                let idx: usize = c.as_();
                let base = idx * NUM_CENTROIDS;
                for v in 0..NUM_CENTROIDS {
                    lut[base + v] = qv.algebraic_mul(quantizer.centroids[base + v]);
                }
            }

            let sq_lut = if D::requires_sq_lut() {
                let mut sq = vec![0.0f32; lut_size];
                for i in 0..quantizer.dim {
                    let base = i * NUM_CENTROIDS;
                    for v in 0..NUM_CENTROIDS {
                        let c_val = quantizer.centroids[base + v];
                        sq[base + v] = c_val.algebraic_mul(c_val);
                    }
                }
                Some(sq)
            } else {
                None
            };

            Self {
                dense_lut: Some(lut),
                dense_sq_lut: sq_lut,
                sparse_query: None,
                dot_query,
                centroids: &quantizer.centroids,
                _phantom: PhantomData,
            }
        } else {
            assert!(
                is_strictly_sorted(query.components()),
                "Query components must be sorted in strictly ascending order."
            );

            Self {
                dense_lut: None,
                dense_sq_lut: None,
                sparse_query: Some(query),
                dot_query,
                centroids: &quantizer.centroids,
                _phantom: PhantomData,
            }
        }
    }
}

impl<'e, 'v, C, D> QueryEvaluator<SparseVectorView<'v, C, u8>>
    for CentroidSparseQueryEvaluator<'e, C, D>
where
    C: ComponentType,
    D: CentroidQuantizedSparseSupportedDistance,
{
    type Distance = D;

    #[inline]
    fn compute_distance(&self, vector: SparseVectorView<'v, C, u8>) -> D {
        if let Some(lut) = &self.dense_lut {
            D::compute_dense_lut(lut, self.dense_sq_lut.as_deref(), vector, self.dot_query)
        } else {
            D::compute_sparse(
                self.sparse_query.as_ref().unwrap(),
                self.centroids,
                vector,
                self.dot_query,
            )
        }
    }
}

impl<C, D> SpaceUsage for CentroidSparseQuantizer<C, D>
where
    C: ComponentType,
    D: CentroidQuantizedSparseSupportedDistance,
{
    fn space_usage_bytes(&self) -> usize {
        self.dim.space_usage_bytes() + self.centroids.space_usage_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PlainSparseDatasetGrowable;
    use crate::core::dataset::DatasetGrowable;
    use crate::core::vector::SparseVectorView;
    use crate::encoders::sparse_scalar::PlainSparseQuantizer;

    type DotQuantizer = CentroidSparseQuantizer<u16, DotProduct>;

    fn build_training_data(
        dim: usize,
        vectors: &[(&[u16], &[f32])],
    ) -> PlainSparseDataset<u16, f32, SquaredEuclideanDistance> {
        let q = PlainSparseQuantizer::<u16, f32, SquaredEuclideanDistance>::new(dim, dim);
        let mut g = PlainSparseDatasetGrowable::new(q);
        for &(c, v) in vectors {
            g.push(SparseVectorView::new(c, v));
        }
        g.into()
    }

    #[test]
    fn encode_min_gives_zero_max_gives_255() {
        let td = build_training_data(
            3,
            &[
                (&[0, 1, 2], &[0.0, 10.0, -5.0]),
                (&[0, 1, 2], &[1.0, 20.0, 5.0]),
            ],
        );
        let q = DotQuantizer::train(&td, 0.0, 1.0, 10);

        let enc_min = <DotQuantizer as SparseVectorEncoder>::encode_vector(
            &q,
            SparseVectorView::new(&[0_u16, 1, 2], &[0.0_f32, 10.0, -5.0]),
        );
        assert_eq!(enc_min.values(), &[0_u8, 0, 0]);

        let enc_max = <DotQuantizer as SparseVectorEncoder>::encode_vector(
            &q,
            SparseVectorView::new(&[0_u16, 1, 2], &[1.0_f32, 20.0, 5.0]),
        );
        for &v in enc_max.values() {
            assert!(v >= 254, "max value should encode to 254 or 255, got {v}");
        }
    }

    #[test]
    fn decode_reconstructs_within_step() {
        // Provide 255 distinct training values over [0, 10] so ckmeans learns 255 centroids.
        // With that many centroids the max quantization error is ~one inter-centroid gap.
        let n = 255usize;
        let comps: Vec<u16> = vec![0, 1];
        let vecs: Vec<(Vec<u16>, Vec<f32>)> = (0..n)
            .map(|i| {
                let v = i as f32 * 10.0 / (n - 1) as f32;
                (comps.clone(), vec![v, v])
            })
            .collect();
        let refs: Vec<(&[u16], &[f32])> = vecs
            .iter()
            .map(|(c, v)| (c.as_slice(), v.as_slice()))
            .collect();
        let td = build_training_data(2, &refs);
        let q = DotQuantizer::train(&td, 0.0, 1.0, 10);
        let tolerance = 10.0 / (n as f32 - 2.0) + 1e-4;

        let input = SparseVectorView::new(&[0_u16, 1], &[3.7_f32, 8.2]);
        let enc = <DotQuantizer as SparseVectorEncoder>::encode_vector(&q, input);
        let dec = <DotQuantizer as SparseDataEncoder>::decode_vector(
            &q,
            SparseVectorView::new(enc.components(), enc.values()),
        );

        for (&orig, &got) in input.values().iter().zip(dec.values()) {
            assert!(
                (orig - got).abs() <= tolerance,
                "error {} exceeds tolerance {}",
                (orig - got).abs(),
                tolerance
            );
        }
    }

    #[test]
    fn dot_product_matches_dequantized_reference() {
        let td = build_training_data(
            4,
            &[(&[0, 1, 2, 3], &[0.0; 4]), (&[0, 1, 2, 3], &[10.0; 4])],
        );
        let q = DotQuantizer::train(&td, 0.0, 1.0, 10);

        let query = SparseVectorView::new(&[0_u16, 2], &[1.0_f32, 3.0]);
        let doc = SparseVectorView::new(&[0_u16, 1, 2], &[5.0_f32, 7.0, 9.0]);

        let enc = <DotQuantizer as SparseVectorEncoder>::encode_vector(&q, doc);
        let enc_view = SparseVectorView::new(enc.components(), enc.values());

        // Compute via evaluator
        let evaluator = q.query_evaluator(query);
        let got: f32 = evaluator.compute_distance(enc_view).distance();

        // Compute reference: dequantize then manual dot
        let dec = <DotQuantizer as SparseDataEncoder>::decode_vector(&q, enc_view);
        let mut ref_dot = 0.0f32;
        for (&qc, &qv) in [0_u16, 2].iter().zip(&[1.0_f32, 3.0]) {
            for (&dc, &dv) in dec.components().iter().zip(dec.values()) {
                if qc == dc {
                    ref_dot += qv * dv;
                }
            }
        }

        assert!(
            (got - ref_dot).abs() < 1e-4,
            "got {got}, expected {ref_dot}"
        );
    }

    #[test]
    fn dot_product_no_overlap_is_zero() {
        let td = build_training_data(
            4,
            &[(&[0, 1, 2, 3], &[0.0; 4]), (&[0, 1, 2, 3], &[10.0; 4])],
        );
        let q = DotQuantizer::train(&td, 0.0, 1.0, 10);

        let query = SparseVectorView::new(&[0_u16, 1], &[5.0_f32, 5.0]);
        let doc = SparseVectorView::new(&[2_u16, 3], &[5.0_f32, 5.0]);

        let enc = <DotQuantizer as SparseVectorEncoder>::encode_vector(&q, doc);
        let evaluator = q.query_evaluator(query);
        let got: f32 = evaluator
            .compute_distance(SparseVectorView::new(enc.components(), enc.values()))
            .distance();

        assert!(got.abs() < 1e-6, "no overlap should give 0, got {got}");
    }
}
