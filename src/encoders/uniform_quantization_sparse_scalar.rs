use std::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::core::vector_encoder::{
    QueryEvaluator, SparseDataEncoder, SparseVectorEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use crate::utils::is_strictly_sorted;
use crate::{ComponentType, Dataset, PlainSparseDataset, SpaceUsage, SparseVectorView};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniformSparseQuantizer<C, D> {
    dim: usize,
    quants: Box<[f32]>,
    mins: Box<[f32]>,
    _phantom: PhantomData<(C, D)>,
}

impl<C, D> PartialEq for UniformSparseQuantizer<C, D> {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
    }
}

impl<C, D> UniformSparseQuantizer<C, D> {
    #[inline]
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        assert_eq!(
            input_dim, output_dim,
            "UniformSparseQuantizer requires input_dim == output_dim"
        );
        Self {
            dim: input_dim,
            quants: vec![0.0; output_dim].into_boxed_slice(),
            mins: vec![0.0; output_dim].into_boxed_slice(),
            _phantom: PhantomData,
        }
    }

    pub fn mins(&self) -> &[f32] {
        &self.mins
    }

    pub fn quants(&self) -> &[f32] {
        &self.quants
    }

    /// Train the quantizer from data.
    ///
    /// `lower_percentile` controls the lower bound of the quantization range per component.
    /// - `0.0` uses the absolute min (classic min–max uniform quantization).
    /// - `0.25` uses the 25th percentile as the lower bound, giving finer resolution
    ///   to the upper 75% of values. Values below the percentile are clipped to 0.
    pub fn train(
        training_data: &PlainSparseDataset<C, f32, SquaredEuclideanDistance>,
        lower_percentile: f32,
    ) -> Self
    where
        C: ComponentType,
    {
        assert!(
            (0.0..1.0).contains(&lower_percentile),
            "lower_percentile must be in [0.0, 1.0), got {lower_percentile}"
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

        let mut mins = vec![0.0f32; dim];
        let mut quants = vec![0.0f32; dim];

        for i in 0..dim {
            let vals = &mut per_component[i];
            if vals.is_empty() {
                continue;
            }
            vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

            let max = *vals.last().unwrap();
            let min = if lower_percentile == 0.0 {
                *vals.first().unwrap()
            } else {
                let idx = ((vals.len() as f32) * lower_percentile) as usize;
                let idx = idx.min(vals.len() - 1);
                vals[idx]
            };

            mins[i] = min;
            if max > min {
                quants[i] = (max - min) / 255.0;
            }
        }

        Self {
            dim,
            quants: quants.into_boxed_slice(),
            mins: mins.into_boxed_slice(),
            _phantom: PhantomData,
        }
    }
}

/// Distance dispatch trait for uniform-quantized sparse vectors.
///
/// Unlike `ScalarSparseSupportedDistance`, this trait works with pre-transformed query data
/// so that scoring avoids per-element dequantization.
///
/// For DotProduct:
///   `q · dequant(v) = Σ_S correction[c] + transformed[c] * v_int[c]`
///   where `transformed[i] = q[i] * quants[i]`, `correction[i] = q[i] * mins[i]`
///
/// For SquaredEuclidean:
///   `||q - dequant(v)||² = ||q||² - 2·(q · dequant(v)) + ||dequant(v)||²`
pub trait UniformQuantizedSparseSupportedDistance: Distance {
    fn requires_dot_query() -> bool {
        false
    }

    /// Score using a dense pre-transformed query (dim < 2^20).
    fn compute_dense<C: ComponentType>(
        dense_transformed: &[f32],
        dense_correction: &[f32],
        mins: &[f32],
        quants: &[f32],
        vector: SparseVectorView<'_, C, u8>,
        dot_query: Option<f32>,
    ) -> Self;

    /// Score using a sparse query with merge-sort (dim >= 2^20).
    fn compute_sparse<C: ComponentType>(
        query: &SparseVectorOwned<C, f32>,
        mins: &[f32],
        quants: &[f32],
        vector: SparseVectorView<'_, C, u8>,
        dot_query: Option<f32>,
    ) -> Self;
}

fn compute_query_squared_norm(values: &[f32]) -> f32 {
    values
        .iter()
        .fold(0.0f32, |acc, &v| acc.algebraic_add(v.algebraic_mul(v)))
}

impl UniformQuantizedSparseSupportedDistance for DotProduct {
    #[inline]
    fn compute_dense<C: ComponentType>(
        dense_transformed: &[f32],
        _dense_correction: &[f32],
        _mins: &[f32],
        _quants: &[f32],
        vector: SparseVectorView<'_, C, u8>,
        _dot_query: Option<f32>,
    ) -> Self {
        // q · dequant(v) = Σ_S transformed[c] * v_int[c]
        // where transformed[c] = q[c] * quants[c]
        let result =
            vector
                .components()
                .iter()
                .zip(vector.values())
                .fold(0.0f32, |acc, (&c, &v)| {
                    let idx: usize = c.as_();
                    acc.algebraic_add(unsafe {
                        dense_transformed.get_unchecked(idx).algebraic_mul(v as f32)
                    })
                });
        DotProduct::from(result)
    }

    #[inline]
    fn compute_sparse<C: ComponentType>(
        query: &SparseVectorOwned<C, f32>,
        mins: &[f32],
        quants: &[f32],
        vector: SparseVectorView<'_, C, u8>,
        _dot_query: Option<f32>,
    ) -> Self {
        let result = sparse_merge_dot_product_with_dequant(query.as_view(), vector, mins, quants);
        DotProduct::from(result)
    }
}

impl UniformQuantizedSparseSupportedDistance for SquaredEuclideanDistance {
    #[inline]
    fn requires_dot_query() -> bool {
        true
    }

    #[inline]
    fn compute_dense<C: ComponentType>(
        dense_transformed: &[f32],
        dense_correction: &[f32],
        mins: &[f32],
        quants: &[f32],
        vector: SparseVectorView<'_, C, u8>,
        dot_query: Option<f32>,
    ) -> Self {
        let dot_query =
            dot_query.expect("SquaredEuclideanDistance requires a precomputed ||q||² value");

        // Single pass: compute q·dequant(v) and ||dequant(v)||² simultaneously
        let mut dot_qv = 0.0f32;
        let mut v_norm_sq = 0.0f32;
        for (&c, &v) in vector.components().iter().zip(vector.values()) {
            let idx: usize = c.as_();
            let vi = v as f32;
            let v_real = mins[idx].algebraic_add(quants[idx].algebraic_mul(vi));
            dot_qv = dot_qv.algebraic_add(
                dense_correction[idx].algebraic_add(dense_transformed[idx].algebraic_mul(vi)),
            );
            v_norm_sq = v_norm_sq.algebraic_add(v_real.algebraic_mul(v_real));
        }

        // ||q - v||² = ||q||² - 2·(q·v) + ||v||²
        let dist = dot_query
            .algebraic_add(v_norm_sq)
            .algebraic_sub(2.0f32.algebraic_mul(dot_qv));
        SquaredEuclideanDistance::from(dist)
    }

    #[inline]
    fn compute_sparse<C: ComponentType>(
        query: &SparseVectorOwned<C, f32>,
        mins: &[f32],
        quants: &[f32],
        vector: SparseVectorView<'_, C, u8>,
        dot_query: Option<f32>,
    ) -> Self {
        let dot_query =
            dot_query.expect("SquaredEuclideanDistance requires a precomputed ||q||² value");

        let dot_qv = sparse_merge_dot_product_with_dequant(query.as_view(), vector, mins, quants);

        // Compute ||dequant(v)||²
        let v_norm_sq =
            vector
                .components()
                .iter()
                .zip(vector.values())
                .fold(0.0f32, |acc, (&c, &v)| {
                    let idx: usize = c.as_();
                    let v_real = mins[idx].algebraic_add(quants[idx].algebraic_mul(v as f32));
                    acc.algebraic_add(v_real.algebraic_mul(v_real))
                });

        let dist = dot_query
            .algebraic_add(v_norm_sq)
            .algebraic_sub(2.0f32.algebraic_mul(dot_qv));
        SquaredEuclideanDistance::from(dist)
    }
}

/// Merge-sort style dot product between a sparse f32 query and a sparse u8 vector,
/// dequantizing the u8 values on the fly using per-component mins/quants.
#[inline]
fn sparse_merge_dot_product_with_dequant<C: ComponentType>(
    query: SparseVectorView<'_, C, f32>,
    vector: SparseVectorView<'_, C, u8>,
    mins: &[f32],
    quants: &[f32],
) -> f32 {
    let q_components = query.components();
    let q_values = query.values();
    let v_components = vector.components();
    let v_values = vector.values();

    let mut qi = 0;
    let mut vi = 0;
    let mut result = 0.0f32;

    while qi < q_components.len() && vi < v_components.len() {
        let qc: usize = q_components[qi].as_();
        let vc: usize = v_components[vi].as_();
        if qc == vc {
            let v_real = mins[vc].algebraic_add(quants[vc].algebraic_mul(v_values[vi] as f32));
            result = result.algebraic_add(q_values[qi].algebraic_mul(v_real));
            qi += 1;
            vi += 1;
        } else if qc < vc {
            qi += 1;
        } else {
            vi += 1;
        }
    }

    result
}

impl<C, D> SparseDataEncoder for UniformSparseQuantizer<C, D>
where
    C: ComponentType,
    D: UniformQuantizedSparseSupportedDistance,
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
            .map(|(&c, &v)| {
                let idx: usize = c.as_();
                self.mins[idx] + (v as f32) * self.quants[idx]
            })
            .collect();
        SparseVectorOwned::new(components, values)
    }
}

impl<C, D> SparseVectorEncoder for UniformSparseQuantizer<C, D>
where
    C: ComponentType,
    D: UniformQuantizedSparseSupportedDistance,
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
                .map(|(&c, &v)| {
                    let idx: usize = c.as_();
                    let q = self.quants[idx];
                    if q > 0.0 {
                        ((v - self.mins[idx]) / q).clamp(0.0, 255.0) as u8
                    } else {
                        0u8
                    }
                }),
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

impl<C, D> VectorEncoder for UniformSparseQuantizer<C, D>
where
    C: ComponentType,
    D: UniformQuantizedSparseSupportedDistance,
{
    type Distance = D;
    type InputVector<'a> = SparseVectorView<'a, C, f32>;
    type QueryVector<'q> = SparseVectorView<'q, C, f32>;
    type EncodedVector<'a> = SparseVectorView<'a, C, u8>;

    type Evaluator<'e>
        = UniformSparseQueryEvaluator<'e, C, D>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        UniformSparseQueryEvaluator::new(query, self)
    }

    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        UniformSparseQueryEvaluator::new_from_owned_query(decoded, self)
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

/// Query evaluator that pre-transforms the query for efficient scoring against u8 vectors.
///
/// For the dense case (dim < 2^20), stores two precomputed arrays:
/// - `dense_transformed[i] = q[i] * quants[i]`
/// - `dense_correction[i] = q[i] * mins[i]`
///
/// Scoring is then a single pass: `Σ_S correction[c] + transformed[c] * v_int[c]`
#[derive(Debug, Clone)]
pub struct UniformSparseQueryEvaluator<'e, C, D>
where
    C: ComponentType,
    D: UniformQuantizedSparseSupportedDistance,
{
    // Dense pre-transformed query (dim < 2^20)
    dense_transformed: Option<Vec<f32>>,
    dense_correction: Option<Vec<f32>>,
    // Sparse query fallback (dim >= 2^20)
    sparse_query: Option<SparseVectorOwned<C, f32>>,
    // Precomputed ||q||² for Euclidean
    dot_query: Option<f32>,
    // Borrowed from quantizer for Euclidean v_norm_sq and sparse dequantization
    quants: &'e [f32],
    mins: &'e [f32],
    _phantom: PhantomData<D>,
}

impl<'e, C, D> UniformSparseQueryEvaluator<'e, C, D>
where
    C: ComponentType,
    D: UniformQuantizedSparseSupportedDistance,
{
    pub fn new(
        query: SparseVectorView<'_, C, f32>,
        quantizer: &'e UniformSparseQuantizer<C, D>,
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
            let mut transformed = vec![0.0f32; quantizer.dim];
            let mut correction = vec![0.0f32; quantizer.dim];
            for (&c, &v) in query.components().iter().zip(query.values()) {
                let idx: usize = c.as_();
                transformed[idx] = v * quantizer.quants[idx];
                correction[idx] = v * quantizer.mins[idx];
            }

            Self {
                dense_transformed: Some(transformed),
                dense_correction: Some(correction),
                sparse_query: None,
                dot_query,
                quants: &quantizer.quants,
                mins: &quantizer.mins,
                _phantom: PhantomData,
            }
        } else {
            assert!(
                is_strictly_sorted(query.components()),
                "Query components must be sorted in strictly ascending order."
            );

            Self {
                dense_transformed: None,
                dense_correction: None,
                sparse_query: Some(SparseVectorOwned::new(
                    query.components().to_vec(),
                    query.values().to_vec(),
                )),
                dot_query,
                quants: &quantizer.quants,
                mins: &quantizer.mins,
                _phantom: PhantomData,
            }
        }
    }

    pub fn new_from_owned_query(
        query: SparseVectorOwned<C, f32>,
        quantizer: &'e UniformSparseQuantizer<C, D>,
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
            let mut transformed = vec![0.0f32; quantizer.dim];
            let mut correction = vec![0.0f32; quantizer.dim];
            for (&c, &v) in query.components().iter().zip(query.values()) {
                let idx: usize = c.as_();
                transformed[idx] = v * quantizer.quants[idx];
                correction[idx] = v * quantizer.mins[idx];
            }

            Self {
                dense_transformed: Some(transformed),
                dense_correction: Some(correction),
                sparse_query: None,
                dot_query,
                quants: &quantizer.quants,
                mins: &quantizer.mins,
                _phantom: PhantomData,
            }
        } else {
            assert!(
                is_strictly_sorted(query.components()),
                "Query components must be sorted in strictly ascending order."
            );

            Self {
                dense_transformed: None,
                dense_correction: None,
                sparse_query: Some(query),
                dot_query,
                quants: &quantizer.quants,
                mins: &quantizer.mins,
                _phantom: PhantomData,
            }
        }
    }
}

impl<'e, 'v, C, D> QueryEvaluator<SparseVectorView<'v, C, u8>>
    for UniformSparseQueryEvaluator<'e, C, D>
where
    C: ComponentType,
    D: UniformQuantizedSparseSupportedDistance,
{
    type Distance = D;

    #[inline]
    fn compute_distance(&self, vector: SparseVectorView<'v, C, u8>) -> D {
        if let (Some(transformed), Some(correction)) =
            (&self.dense_transformed, &self.dense_correction)
        {
            D::compute_dense(
                transformed,
                correction,
                self.mins,
                self.quants,
                vector,
                self.dot_query,
            )
        } else {
            D::compute_sparse(
                self.sparse_query.as_ref().unwrap(),
                self.mins,
                self.quants,
                vector,
                self.dot_query,
            )
        }
    }
}

impl<C, D> SpaceUsage for UniformSparseQuantizer<C, D>
where
    C: ComponentType,
    D: UniformQuantizedSparseSupportedDistance,
{
    fn space_usage_bytes(&self) -> usize {
        self.dim.space_usage_bytes()
            + self.quants.space_usage_bytes()
            + self.mins.space_usage_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PlainSparseDatasetGrowable;
    use crate::core::dataset::DatasetGrowable;
    use crate::core::vector::SparseVectorView;
    use crate::encoders::sparse_scalar::PlainSparseQuantizer;

    type DotQuantizer = UniformSparseQuantizer<u16, DotProduct>;

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
        let q = DotQuantizer::train(&td, 0.0);

        let enc_min = <DotQuantizer as SparseVectorEncoder>::encode_vector(
            &q,
            SparseVectorView::new(&[0_u16, 1, 2], &[0.0_f32, 10.0, -5.0]),
        );
        assert_eq!(enc_min.values(), &[0_u8, 0, 0]);

        let enc_max = <DotQuantizer as SparseVectorEncoder>::encode_vector(
            &q,
            SparseVectorView::new(&[0_u16, 1, 2], &[1.0_f32, 20.0, 5.0]),
        );
        // May be 254 or 255 due to floating-point rounding in (max-min)/quant
        for &v in enc_max.values() {
            assert!(v >= 254, "max value should encode to 254 or 255, got {v}");
        }
    }

    #[test]
    fn decode_reconstructs_within_step() {
        let td = build_training_data(2, &[(&[0, 1], &[0.0, 0.0]), (&[0, 1], &[10.0, 10.0])]);
        let q = DotQuantizer::train(&td, 0.0);
        let step = 10.0 / 255.0;

        let input = SparseVectorView::new(&[0_u16, 1], &[3.7_f32, 8.2]);
        let enc = <DotQuantizer as SparseVectorEncoder>::encode_vector(&q, input);
        let dec = <DotQuantizer as SparseDataEncoder>::decode_vector(
            &q,
            SparseVectorView::new(enc.components(), enc.values()),
        );

        for (&orig, &got) in input.values().iter().zip(dec.values()) {
            assert!((orig - got).abs() <= step + 1e-5);
        }
    }

    #[test]
    fn dot_product_matches_dequantized_reference() {
        let td = build_training_data(
            4,
            &[(&[0, 1, 2, 3], &[0.0; 4]), (&[0, 1, 2, 3], &[10.0; 4])],
        );
        let q = DotQuantizer::train(&td, 0.0);

        let query = SparseVectorView::new(&[0_u16, 2], &[1.0_f32, 3.0]);
        let doc = SparseVectorView::new(&[0_u16, 1, 2], &[5.0_f32, 7.0, 9.0]);

        let enc = <DotQuantizer as SparseVectorEncoder>::encode_vector(&q, doc);
        let enc_view = SparseVectorView::new(enc.components(), enc.values());

        // Compute via evaluator
        let evaluator = q.query_evaluator(query);
        let got: f32 = evaluator.compute_distance(enc_view).distance();

        // Compute reference: dequantize then manual dot
        let dec = <DotQuantizer as SparseDataEncoder>::decode_vector(&q, enc_view);
        // overlap on components 0 and 2: 1.0*dec[0] + 3.0*dec[2]
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
        let q = DotQuantizer::train(&td, 0.0);

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
