use std::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::core::vector_encoder::{
    QueryEvaluator, SparseDataEncoder, SparseVectorEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use crate::utils::{is_strictly_sorted, train_sparse_scalar_quantizer_with_levels};
use crate::{ComponentType, Dataset, PlainSparseDataset, SpaceUsage, SparseVectorView};

/// Per-component uniform sparse quantizer with a configurable number of bits (1..=8).
///
/// Quantized codes are stored as `u8` regardless of `nbits`; only `max_val = 2^nbits - 1`
/// levels are used. This keeps the implementation simple at the cost of wasting a few bits
/// when `nbits < 8`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableBitUniformSparseQuantizer<C, D> {
    dim: usize,
    nbits: u8,
    max_val: f32,
    quants: Box<[f32]>,
    _phantom: PhantomData<(C, D)>,
}

impl<C, D> PartialEq for VariableBitUniformSparseQuantizer<C, D> {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim && self.nbits == other.nbits
    }
}

impl<C, D> VariableBitUniformSparseQuantizer<C, D> {
    #[inline]
    pub fn new(input_dim: usize, output_dim: usize, nbits: u8) -> Self {
        assert_eq!(
            input_dim, output_dim,
            "VariableBitUniformSparseQuantizer requires input_dim == output_dim"
        );
        assert!(
            (1..=8).contains(&nbits),
            "nbits must be in [1, 8], got {nbits}"
        );
        let max_val = ((1u32 << nbits) - 1) as f32;
        Self {
            dim: input_dim,
            nbits,
            max_val,
            quants: vec![0.0; output_dim].into_boxed_slice(),
            _phantom: PhantomData,
        }
    }

    pub fn nbits(&self) -> u8 {
        self.nbits
    }

    pub fn max_val(&self) -> f32 {
        self.max_val
    }

    pub fn quants(&self) -> &[f32] {
        &self.quants
    }

    pub fn train(
        training_data: &PlainSparseDataset<C, f32, SquaredEuclideanDistance>,
        lower_percentile: f32,
        upper_percentile: f32,
        nbits: u8,
    ) -> Self
    where
        C: ComponentType,
    {
        assert!(
            (1..=8).contains(&nbits),
            "nbits must be in [1, 8], got {nbits}"
        );
        let max_val = ((1u32 << nbits) - 1) as f32;

        let quants = train_sparse_scalar_quantizer_with_levels(
            training_data,
            lower_percentile,
            upper_percentile,
            max_val,
        );

        Self {
            dim: training_data.output_dim(),
            nbits,
            max_val,
            quants: quants.into_boxed_slice(),
            _phantom: PhantomData,
        }
    }
}

/// Distance dispatch trait for variable-bit uniform-quantized sparse vectors.
///
/// Identical to `UniformQuantizedSparseSupportedDistance` but works with the variable-bit
/// quantizer. The scoring math is the same; the only difference is that codes span
/// `[0, max_val]` instead of `[0, 255]`.
pub trait VariableBitUniformQuantizedSparseSupportedDistance: Distance {
    fn requires_dot_query() -> bool {
        false
    }

    fn compute_dense<C: ComponentType>(
        dense_transformed: &[f32],
        quants: &[f32],
        vector: SparseVectorView<'_, C, u8>,
        dot_query: Option<f32>,
    ) -> Self;

    fn compute_sparse<C: ComponentType>(
        query: &SparseVectorOwned<C, f32>,
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

impl VariableBitUniformQuantizedSparseSupportedDistance for DotProduct {
    #[inline]
    fn compute_dense<C: ComponentType>(
        dense_transformed: &[f32],
        _quants: &[f32],
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
                        dense_transformed.get_unchecked(idx).algebraic_mul(v as f32)
                    })
                });
        DotProduct::from(result)
    }

    #[inline]
    fn compute_sparse<C: ComponentType>(
        query: &SparseVectorOwned<C, f32>,
        quants: &[f32],
        vector: SparseVectorView<'_, C, u8>,
        _dot_query: Option<f32>,
    ) -> Self {
        let result = sparse_merge_dot_product_with_dequant(query.as_view(), vector, quants);
        DotProduct::from(result)
    }
}

impl VariableBitUniformQuantizedSparseSupportedDistance for SquaredEuclideanDistance {
    #[inline]
    fn requires_dot_query() -> bool {
        true
    }

    #[inline]
    fn compute_dense<C: ComponentType>(
        dense_transformed: &[f32],
        quants: &[f32],
        vector: SparseVectorView<'_, C, u8>,
        dot_query: Option<f32>,
    ) -> Self {
        let dot_query =
            dot_query.expect("SquaredEuclideanDistance requires a precomputed ||q||² value");

        let mut dot_qv = 0.0f32;
        let mut v_norm_sq = 0.0f32;
        for (&c, &v) in vector.components().iter().zip(vector.values()) {
            let idx: usize = c.as_();
            let vi = v as f32;
            let v_real = quants[idx].algebraic_mul(vi);
            dot_qv = dot_qv.algebraic_add(dense_transformed[idx].algebraic_mul(vi));
            v_norm_sq = v_norm_sq.algebraic_add(v_real.algebraic_mul(v_real));
        }

        let dist = dot_query
            .algebraic_add(v_norm_sq)
            .algebraic_sub(2.0f32.algebraic_mul(dot_qv));
        SquaredEuclideanDistance::from(dist)
    }

    #[inline]
    fn compute_sparse<C: ComponentType>(
        query: &SparseVectorOwned<C, f32>,
        quants: &[f32],
        vector: SparseVectorView<'_, C, u8>,
        dot_query: Option<f32>,
    ) -> Self {
        let dot_query =
            dot_query.expect("SquaredEuclideanDistance requires a precomputed ||q||² value");

        let dot_qv = sparse_merge_dot_product_with_dequant(query.as_view(), vector, quants);

        let v_norm_sq =
            vector
                .components()
                .iter()
                .zip(vector.values())
                .fold(0.0f32, |acc, (&c, &v)| {
                    let idx: usize = c.as_();
                    let v_real = quants[idx].algebraic_mul(v as f32);
                    acc.algebraic_add(v_real.algebraic_mul(v_real))
                });

        let dist = dot_query
            .algebraic_add(v_norm_sq)
            .algebraic_sub(2.0f32.algebraic_mul(dot_qv));
        SquaredEuclideanDistance::from(dist)
    }
}

#[inline]
fn sparse_merge_dot_product_with_dequant<C: ComponentType>(
    query: SparseVectorView<'_, C, f32>,
    vector: SparseVectorView<'_, C, u8>,
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
            let v_real = quants[vc].algebraic_mul(v_values[vi] as f32);
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

impl<C, D> SparseDataEncoder for VariableBitUniformSparseQuantizer<C, D>
where
    C: ComponentType,
    D: VariableBitUniformQuantizedSparseSupportedDistance,
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
                (v as f32) * self.quants[idx]
            })
            .collect();
        SparseVectorOwned::new(components, values)
    }
}

impl<C, D> SparseVectorEncoder for VariableBitUniformSparseQuantizer<C, D>
where
    C: ComponentType,
    D: VariableBitUniformQuantizedSparseSupportedDistance,
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
        let max_val = self.max_val;
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
                        (v / q).clamp(0.0, max_val) as u8
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

impl<C, D> VectorEncoder for VariableBitUniformSparseQuantizer<C, D>
where
    C: ComponentType,
    D: VariableBitUniformQuantizedSparseSupportedDistance,
{
    type Distance = D;
    type InputVector<'a> = SparseVectorView<'a, C, f32>;
    type QueryVector<'q> = SparseVectorView<'q, C, f32>;
    type EncodedVector<'a> = SparseVectorView<'a, C, u8>;

    type Evaluator<'e>
        = VariableBitUniformSparseQueryEvaluator<'e, C, D>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        VariableBitUniformSparseQueryEvaluator::new(query, self)
    }

    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        VariableBitUniformSparseQueryEvaluator::new_from_owned_query(decoded, self)
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

#[derive(Debug, Clone)]
pub struct VariableBitUniformSparseQueryEvaluator<'e, C, D>
where
    C: ComponentType,
    D: VariableBitUniformQuantizedSparseSupportedDistance,
{
    dense_transformed: Option<Vec<f32>>,
    sparse_query: Option<SparseVectorOwned<C, f32>>,
    dot_query: Option<f32>,
    quants: &'e [f32],
    _phantom: PhantomData<D>,
}

impl<'e, C, D> VariableBitUniformSparseQueryEvaluator<'e, C, D>
where
    C: ComponentType,
    D: VariableBitUniformQuantizedSparseSupportedDistance,
{
    pub fn new(
        query: SparseVectorView<'_, C, f32>,
        quantizer: &'e VariableBitUniformSparseQuantizer<C, D>,
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
            for (&c, &v) in query.components().iter().zip(query.values()) {
                let idx: usize = c.as_();
                transformed[idx] = v * quantizer.quants[idx];
            }

            Self {
                dense_transformed: Some(transformed),
                sparse_query: None,
                dot_query,
                quants: &quantizer.quants,
                _phantom: PhantomData,
            }
        } else {
            assert!(
                is_strictly_sorted(query.components()),
                "Query components must be sorted in strictly ascending order."
            );

            Self {
                dense_transformed: None,
                sparse_query: Some(SparseVectorOwned::new(
                    query.components().to_vec(),
                    query.values().to_vec(),
                )),
                dot_query,
                quants: &quantizer.quants,
                _phantom: PhantomData,
            }
        }
    }

    pub fn new_from_owned_query(
        query: SparseVectorOwned<C, f32>,
        quantizer: &'e VariableBitUniformSparseQuantizer<C, D>,
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
            for (&c, &v) in query.components().iter().zip(query.values()) {
                let idx: usize = c.as_();
                transformed[idx] = v * quantizer.quants[idx];
            }

            Self {
                dense_transformed: Some(transformed),
                sparse_query: None,
                dot_query,
                quants: &quantizer.quants,
                _phantom: PhantomData,
            }
        } else {
            assert!(
                is_strictly_sorted(query.components()),
                "Query components must be sorted in strictly ascending order."
            );

            Self {
                dense_transformed: None,
                sparse_query: Some(query),
                dot_query,
                quants: &quantizer.quants,
                _phantom: PhantomData,
            }
        }
    }
}

impl<'e, 'v, C, D> QueryEvaluator<SparseVectorView<'v, C, u8>>
    for VariableBitUniformSparseQueryEvaluator<'e, C, D>
where
    C: ComponentType,
    D: VariableBitUniformQuantizedSparseSupportedDistance,
{
    type Distance = D;

    #[inline]
    fn compute_distance(&self, vector: SparseVectorView<'v, C, u8>) -> D {
        if let Some(transformed) = &self.dense_transformed {
            D::compute_dense(transformed, self.quants, vector, self.dot_query)
        } else {
            D::compute_sparse(
                self.sparse_query.as_ref().unwrap(),
                self.quants,
                vector,
                self.dot_query,
            )
        }
    }
}

impl<C, D> SpaceUsage for VariableBitUniformSparseQuantizer<C, D>
where
    C: ComponentType,
    D: VariableBitUniformQuantizedSparseSupportedDistance,
{
    fn space_usage_bytes(&self) -> usize {
        self.dim.space_usage_bytes()
            + std::mem::size_of::<u8>()
            + std::mem::size_of::<f32>()
            + self.quants.space_usage_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PlainSparseDatasetGrowable;
    use crate::core::dataset::{Dataset, DatasetGrowable, ScoredVector};
    use crate::core::vector::SparseVectorView;
    use crate::datasets::sparse_dataset::SparseDatasetGrowable;
    use crate::encoders::sparse_scalar::PlainSparseQuantizer;

    type DotQuantizer = VariableBitUniformSparseQuantizer<u16, DotProduct>;
    type EuclidQuantizer = VariableBitUniformSparseQuantizer<u16, SquaredEuclideanDistance>;

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

    fn build_quantized_dot_dataset(
        training_data: &PlainSparseDataset<u16, f32, SquaredEuclideanDistance>,
        docs: &[(&[u16], &[f32])],
        nbits: u8,
    ) -> crate::SparseDataset<VariableBitUniformSparseQuantizer<u16, DotProduct>> {
        let quantizer = DotQuantizer::train(training_data, 0.0, 1.0, nbits);
        let mut growable = SparseDatasetGrowable::new(quantizer);
        for &(c, v) in docs {
            growable.push(SparseVectorView::new(c, v));
        }
        growable.into()
    }

    #[test]
    fn encode_zero_gives_zero_max_gives_max_val() {
        for nbits in 1..=8u8 {
            let max_val = ((1u32 << nbits) - 1) as u8;
            let td = build_training_data(
                3,
                &[
                    (&[0, 1, 2], &[0.0, 0.0, 0.0]),
                    (&[0, 1, 2], &[1.0, 20.0, 5.0]),
                ],
            );
            let q = DotQuantizer::train(&td, 0.0, 1.0, nbits);

            // Zero values should encode to 0
            let enc_zero = <DotQuantizer as SparseVectorEncoder>::encode_vector(
                &q,
                SparseVectorView::new(&[0_u16, 1, 2], &[0.0_f32, 0.0, 0.0]),
            );
            assert_eq!(enc_zero.values(), &[0_u8, 0, 0]);

            // Negative values should be clamped to 0
            let enc_neg = <DotQuantizer as SparseVectorEncoder>::encode_vector(
                &q,
                SparseVectorView::new(&[0_u16, 1, 2], &[-1.0_f32, -5.0, -3.0]),
            );
            assert_eq!(enc_neg.values(), &[0_u8, 0, 0]);

            // Max values should encode to max_val (or max_val-1 due to FP rounding)
            let enc_max = <DotQuantizer as SparseVectorEncoder>::encode_vector(
                &q,
                SparseVectorView::new(&[0_u16, 1, 2], &[1.0_f32, 20.0, 5.0]),
            );
            for &v in enc_max.values() {
                assert!(
                    v >= max_val.saturating_sub(1),
                    "nbits={nbits}: max value should encode to {max_val} or {}, got {v}",
                    max_val.saturating_sub(1)
                );
            }
        }
    }

    #[test]
    fn encode_values_never_exceed_max_val() {
        for nbits in 1..=8u8 {
            let max_val = ((1u32 << nbits) - 1) as u8;
            let td = build_training_data(
                3,
                &[
                    (&[0, 1, 2], &[0.0, 0.0, 0.0]),
                    (&[0, 1, 2], &[1.0, 20.0, 5.0]),
                ],
            );
            let q = DotQuantizer::train(&td, 0.0, 1.0, nbits);

            // Values above the training max should be clamped to max_val
            let enc = <DotQuantizer as SparseVectorEncoder>::encode_vector(
                &q,
                SparseVectorView::new(&[0_u16, 1, 2], &[100.0_f32, 200.0, 500.0]),
            );
            for &v in enc.values() {
                assert!(
                    v <= max_val,
                    "nbits={nbits}: encoded value {v} exceeds max_val {max_val}"
                );
            }
        }
    }

    #[test]
    fn decode_reconstructs_within_step() {
        for nbits in 1..=8u8 {
            let max_val = ((1u32 << nbits) - 1) as f32;
            let td =
                build_training_data(2, &[(&[0, 1], &[0.0, 0.0]), (&[0, 1], &[10.0, 10.0])]);
            let q = DotQuantizer::train(&td, 0.0, 1.0, nbits);
            let step = 10.0 / max_val;

            let input = SparseVectorView::new(&[0_u16, 1], &[3.7_f32, 8.2]);
            let enc = <DotQuantizer as SparseVectorEncoder>::encode_vector(&q, input);
            let dec = <DotQuantizer as SparseDataEncoder>::decode_vector(
                &q,
                SparseVectorView::new(enc.components(), enc.values()),
            );

            for (&orig, &got) in input.values().iter().zip(dec.values()) {
                assert!(
                    (orig - got).abs() <= step + 1e-5,
                    "nbits={nbits}: orig={orig}, got={got}, step={step}"
                );
            }
        }
    }

    #[test]
    fn decode_preserves_components() {
        for nbits in [1, 4, 8] {
            let td = build_training_data(
                5,
                &[(&[0, 1, 2, 3, 4], &[0.0; 5]), (&[0, 1, 2, 3, 4], &[10.0; 5])],
            );
            let q = DotQuantizer::train(&td, 0.0, 1.0, nbits);

            let input = SparseVectorView::new(&[1_u16, 3], &[5.0_f32, 7.0]);
            let enc = <DotQuantizer as SparseVectorEncoder>::encode_vector(&q, input);
            let dec = <DotQuantizer as SparseDataEncoder>::decode_vector(
                &q,
                SparseVectorView::new(enc.components(), enc.values()),
            );

            assert_eq!(
                dec.components(),
                &[1_u16, 3],
                "nbits={nbits}: components should be preserved"
            );
        }
    }

    #[test]
    fn dot_product_matches_dequantized_reference() {
        for nbits in 1..=8u8 {
            let td = build_training_data(
                4,
                &[(&[0, 1, 2, 3], &[0.0; 4]), (&[0, 1, 2, 3], &[10.0; 4])],
            );
            let q = DotQuantizer::train(&td, 0.0, 1.0, nbits);

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
                "nbits={nbits}: got {got}, expected {ref_dot}"
            );
        }
    }

    #[test]
    fn dot_product_no_overlap_is_zero() {
        for nbits in 1..=8u8 {
            let td = build_training_data(
                4,
                &[(&[0, 1, 2, 3], &[0.0; 4]), (&[0, 1, 2, 3], &[10.0; 4])],
            );
            let q = DotQuantizer::train(&td, 0.0, 1.0, nbits);

            let query = SparseVectorView::new(&[0_u16, 1], &[5.0_f32, 5.0]);
            let doc = SparseVectorView::new(&[2_u16, 3], &[5.0_f32, 5.0]);

            let enc = <DotQuantizer as SparseVectorEncoder>::encode_vector(&q, doc);
            let evaluator = q.query_evaluator(query);
            let got: f32 = evaluator
                .compute_distance(SparseVectorView::new(enc.components(), enc.values()))
                .distance();

            assert!(
                got.abs() < 1e-6,
                "nbits={nbits}: no overlap should give 0, got {got}"
            );
        }
    }

    #[test]
    fn dot_product_approximation_improves_with_more_bits() {
        let td = build_training_data(
            4,
            &[(&[0, 1, 2, 3], &[0.0; 4]), (&[0, 1, 2, 3], &[10.0; 4])],
        );

        let query = SparseVectorView::new(&[0_u16, 1, 2, 3], &[1.0_f32, 2.0, 3.0, 4.0]);
        let doc = SparseVectorView::new(&[0_u16, 1, 2, 3], &[3.5_f32, 6.1, 2.8, 7.3]);

        // True dot product
        let true_dot: f32 = query
            .values()
            .iter()
            .zip(doc.values())
            .map(|(&q, &d)| q * d)
            .sum();

        let mut prev_error = f32::MAX;
        for nbits in 1..=8u8 {
            let q = DotQuantizer::train(&td, 0.0, 1.0, nbits);
            let enc = <DotQuantizer as SparseVectorEncoder>::encode_vector(&q, doc);
            let enc_view = SparseVectorView::new(enc.components(), enc.values());

            let evaluator = q.query_evaluator(query);
            let got: f32 = evaluator.compute_distance(enc_view).distance();
            let error = (got - true_dot).abs();

            assert!(
                error <= prev_error + 1e-5,
                "nbits={nbits}: error {error} should not exceed previous {prev_error}"
            );
            prev_error = error;
        }
    }

    #[test]
    fn euclidean_distance_matches_dequantized_reference() {
        for nbits in [2, 4, 8] {
            let td = build_training_data(
                4,
                &[(&[0, 1, 2, 3], &[0.0; 4]), (&[0, 1, 2, 3], &[10.0; 4])],
            );
            let q = EuclidQuantizer::train(&td, 0.0, 1.0, nbits);

            let query = SparseVectorView::new(&[0_u16, 1, 2], &[1.0_f32, 5.0, 3.0]);
            let doc = SparseVectorView::new(&[0_u16, 2, 3], &[4.0_f32, 8.0, 2.0]);

            let enc = <EuclidQuantizer as SparseVectorEncoder>::encode_vector(&q, doc);
            let enc_view = SparseVectorView::new(enc.components(), enc.values());

            let evaluator = q.query_evaluator(query);
            let got: f32 = evaluator.compute_distance(enc_view).distance();

            // Reference: dequantize, then compute ||q - dequant(v)||^2
            // treating missing components as 0
            let dec = <EuclidQuantizer as SparseDataEncoder>::decode_vector(&q, enc_view);
            let dim = 4;
            let mut q_dense = vec![0.0f32; dim];
            for (&c, &v) in query.components().iter().zip(query.values()) {
                q_dense[c as usize] = v;
            }
            let mut d_dense = vec![0.0f32; dim];
            for (&c, &v) in dec.components().iter().zip(dec.values()) {
                d_dense[c as usize] = v;
            }
            let ref_dist: f32 = q_dense
                .iter()
                .zip(d_dense.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();

            assert!(
                (got - ref_dist).abs() < 1e-3,
                "nbits={nbits}: euclidean got {got}, expected {ref_dist}"
            );
        }
    }

    #[test]
    fn search_returns_correct_top_k_ranking() {
        // Build a small dataset, quantize, and verify search ranking matches brute-force
        let docs: Vec<(&[u16], &[f32])> = vec![
            (&[0, 1, 2], &[1.0, 0.0, 0.0]),
            (&[0, 1, 2], &[0.0, 1.0, 0.0]),
            (&[0, 1, 2], &[0.0, 0.0, 1.0]),
            (&[0, 1, 2], &[5.0, 5.0, 5.0]),
            (&[0, 1, 2], &[3.0, 1.0, 0.5]),
        ];
        let td = build_training_data(3, &docs);

        for nbits in [1, 4, 8] {
            let dataset = build_quantized_dot_dataset(&td, &docs, nbits);
            let query = SparseVectorView::new(&[0_u16, 1, 2], &[1.0_f32, 1.0, 1.0]);
            let results: Vec<ScoredVector<DotProduct>> = dataset.search(query, 3);

            assert_eq!(
                results.len(),
                3,
                "nbits={nbits}: search should return k=3 results"
            );
            // doc 3 ([5,5,5]) should always be top-1 for query [1,1,1]
            assert_eq!(
                results[0].vector, 3,
                "nbits={nbits}: doc 3 should be top-1, got doc {}",
                results[0].vector
            );
            // Scores should be in non-increasing order (DotProduct: larger is better)
            for i in 1..results.len() {
                assert!(
                    results[i - 1].distance.distance() >= results[i].distance.distance(),
                    "nbits={nbits}: results not sorted by distance: {} < {}",
                    results[i - 1].distance.distance(),
                    results[i].distance.distance()
                );
            }
        }
    }

    #[test]
    fn nbits_8_matches_original_quantizer() {
        use crate::encoders::uniform_quantization_sparse_scalar::UniformSparseQuantizer;

        let td = build_training_data(
            4,
            &[(&[0, 1, 2, 3], &[0.0; 4]), (&[0, 1, 2, 3], &[10.0; 4])],
        );
        let q8 = DotQuantizer::train(&td, 0.0, 1.0, 8);
        let q_orig = UniformSparseQuantizer::<u16, DotProduct>::train(&td, 0.0, 1.0);

        // quants should be identical
        assert_eq!(q8.quants().len(), q_orig.quants().len());
        for (a, b) in q8.quants().iter().zip(q_orig.quants()) {
            assert!((a - b).abs() < 1e-10, "quants differ: {a} vs {b}");
        }

        // Encoded values should be identical
        let doc = SparseVectorView::new(&[0_u16, 1, 2], &[5.0_f32, 7.0, 9.0]);
        let enc8 = <DotQuantizer as SparseVectorEncoder>::encode_vector(&q8, doc);
        let enc_orig = <UniformSparseQuantizer<u16, DotProduct> as SparseVectorEncoder>::encode_vector(&q_orig, doc);
        assert_eq!(enc8.values(), enc_orig.values());

        // Dot products should be identical
        let query = SparseVectorView::new(&[0_u16, 2], &[1.0_f32, 3.0]);
        let eval8 = q8.query_evaluator(query);
        let eval_orig = q_orig.query_evaluator(query);
        let dot8: f32 = eval8
            .compute_distance(SparseVectorView::new(enc8.components(), enc8.values()))
            .distance();
        let dot_orig: f32 = eval_orig
            .compute_distance(SparseVectorView::new(
                enc_orig.components(),
                enc_orig.values(),
            ))
            .distance();
        assert!(
            (dot8 - dot_orig).abs() < 1e-6,
            "nbits=8 dot {dot8} differs from original {dot_orig}"
        );
    }

    #[test]
    fn fewer_bits_produces_coarser_quantization() {
        let td = build_training_data(2, &[(&[0, 1], &[0.0, 0.0]), (&[0, 1], &[10.0, 10.0])]);

        // With fewer bits, the quantization step is larger
        for nbits in 1..=8u8 {
            let q = DotQuantizer::train(&td, 0.0, 1.0, nbits);
            let max_val = ((1u32 << nbits) - 1) as f32;
            let expected_quant = 10.0 / max_val;
            for &qv in q.quants() {
                if qv > 0.0 {
                    assert!(
                        (qv - expected_quant).abs() < 1e-6,
                        "nbits={nbits}: quant={qv}, expected={expected_quant}"
                    );
                }
            }
        }
    }
}
