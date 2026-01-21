//! Scalar dense quantizers that simply rescale floats between numeric types.
//! These modules back the dense datasets and keep distances in `f32` for accurate queries.
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::core::distances::{
    Distance, DotProduct, SquaredEuclideanDistance, dot_product_dense, dot_product_dense_unchecked,
    squared_euclidean_distance_dense_unchecked,
};
use crate::core::vector::DenseVectorView;
use crate::core::vector_encoder::{
    DenseVectorEncoder, DenseVectorOwned, QueryEvaluator, VectorEncoder,
};
use crate::{Float, FromF32, SpaceUsage, ValueType};

/// Marker trait for distance types supported by scalar dense quantizers.
/// Provides the computation method specific to dense vectors.
pub trait ScalarDenseSupportedDistance: Distance {
    /// Compute a numeric distance between two dense float views.
    /// Implementations usually call the same distance kernel the dataset uses.
    fn compute_dense<Q: ValueType, V: ValueType>(
        dot_query: f32,
        query: DenseVectorView<'_, Q>,
        vector: DenseVectorView<'_, V>,
    ) -> Self;
}

impl ScalarDenseSupportedDistance for SquaredEuclideanDistance {
    fn compute_dense<Q: ValueType, V: ValueType>(
        dot_query: f32,
        query: DenseVectorView<'_, Q>,
        vector: DenseVectorView<'_, V>,
    ) -> Self {
        unsafe { squared_euclidean_distance_dense_unchecked(dot_query, query, vector) }
    }
}

impl ScalarDenseSupportedDistance for DotProduct {
    fn compute_dense<Q: ValueType, V: ValueType>(
        _dot_query: f32,
        query: DenseVectorView<'_, Q>,
        vector: DenseVectorView<'_, V>,
    ) -> Self {
        unsafe { dot_product_dense_unchecked(query, vector) }
    }
}

/// A scalar quantizer that casts each component from `In` to `Out`.
/// The quantizer stores `d` to keep the input/output dimensionality in sync.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarDenseQuantizer<In, Out, D> {
    d: usize,
    _phantom: PhantomData<(In, Out, D)>,
}

pub type ScalarDenseQuantizerSame<V, D> = ScalarDenseQuantizer<V, V, D>;
pub type PlainDenseQuantizer<V, D> = ScalarDenseQuantizer<V, V, D>;
pub type PlainDenseQuantizerSquaredEuclidean<V> = PlainDenseQuantizer<V, SquaredEuclideanDistance>;
pub type PlainDenseQuantizerDotProduct<V> = PlainDenseQuantizer<V, DotProduct>;
pub type ScalarDenseQuantizerSquaredEuclidean<V> =
    ScalarDenseQuantizer<V, V, SquaredEuclideanDistance>;
pub type ScalarDenseQuantizerDotProduct<V> = ScalarDenseQuantizer<V, V, DotProduct>;

impl<In, Out, D> ScalarDenseQuantizer<In, Out, D> {
    /// Build a quantizer that keeps the input and output dimensionality equal.
    #[inline]
    pub fn new(d: usize) -> Self {
        Self {
            d,
            _phantom: PhantomData,
        }
    }
}

impl<In, Out, D> DenseVectorEncoder for ScalarDenseQuantizer<In, Out, D>
where
    In: ValueType + Float,
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
{
    type InputValueType = In;
    type OutputValueType = Out;

    /// Decode any encoded vector into `DenseVectorOwned<f32>` by converting each element to `f32`.
    #[inline]
    fn decode_vector<'a>(
        &self,
        encoded: DenseVectorView<'a, Self::OutputValueType>,
    ) -> DenseVectorOwned<f32> {
        let values = encoded
            .values()
            .iter()
            .map(|&v| v.to_f32().expect("Failed to convert value to f32"))
            .collect();
        DenseVectorOwned::new(values)
    }

    /// Encode the input view into an existing buffer to avoid temporary allocations.
    #[inline]
    fn push_encoded<'a, OutputContainer>(
        &self,
        input: DenseVectorView<'a, Self::InputValueType>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<Self::OutputValueType>,
    {
        output.extend(input.values().iter().map(|&in_val| {
            let f32_val = in_val.to_f32().unwrap();
            Out::from_f32_saturating(f32_val)
        }));
    }
}

/// Evaluator that computes distances between a frozen query (owned `f32` vector) and encoded vectors.
#[derive(Debug, Clone)]
pub struct ScalarDenseQueryEvaluator<'e, In, Out, D>
where
    In: ValueType,
    Out: ValueType + FromF32,
    D: ScalarDenseSupportedDistance,
{
    encoder: &'e ScalarDenseQuantizer<In, Out, D>,
    query: DenseVectorOwned<f32>,
    dot_query: f32,
}

impl<'e, In, Out, D> ScalarDenseQueryEvaluator<'e, In, Out, D>
where
    In: ValueType,
    Out: ValueType + FromF32,
    D: ScalarDenseSupportedDistance,
{
    #[inline]
    pub fn new(
        encoder: &'e ScalarDenseQuantizer<In, Out, D>,
        query: DenseVectorOwned<f32>,
    ) -> Self {
        let dot_query = dot_product_dense(query.as_view(), query.as_view()).distance();
        Self {
            encoder,
            query,
            dot_query,
        }
    }
}

impl<'e, 'v, In, Out, D> QueryEvaluator<DenseVectorView<'v, Out>>
    for ScalarDenseQueryEvaluator<'e, In, Out, D>
where
    In: ValueType,
    Out: ValueType + FromF32,
    D: ScalarDenseSupportedDistance,
{
    type Distance = D;

    #[inline]
    fn compute_distance(&self, vector: DenseVectorView<'v, Out>) -> D {
        let _ = self.encoder;
        D::compute_dense(self.dot_query, self.query.as_view(), vector)
    }
}

impl<In, Out, D> VectorEncoder for ScalarDenseQuantizer<In, Out, D>
where
    In: ValueType + Float,
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
{
    type Distance = D;
    type InputVector<'a> = DenseVectorView<'a, In>;
    type QueryVector<'q> = DenseVectorView<'q, f32>;
    type EncodedVector<'a> = DenseVectorView<'a, Out>;

    type Evaluator<'e>
        = ScalarDenseQueryEvaluator<'e, In, Out, D>
    where
        Self: 'e;

    /// Build an evaluator from a dense `f32` query.
    #[inline]
    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        assert_eq!(
            query.len(),
            self.input_dim(),
            "Query vector length exceeds quantizer input dimension."
        );
        ScalarDenseQueryEvaluator::new(self, query.to_owned())
    }

    /// Build an evaluator from an encoded vector stored in the dataset.
    #[inline]
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as DenseVectorEncoder>::decode_vector(self, vector);
        ScalarDenseQueryEvaluator::new(self, decoded)
    }

    fn input_dim(&self) -> usize {
        self.d
    }

    fn output_dim(&self) -> usize {
        self.d
    }
}

impl<In, Out, D> SpaceUsage for ScalarDenseQuantizer<In, Out, D>
where
    In: ValueType + Float,
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
{
    fn space_usage_bytes(&self) -> usize {
        self.d.space_usage_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vector::DenseVectorView;
    use crate::distances::DotProduct;

    #[test]
    fn scalar_dense_quantizer_encodes_and_decodes_values() {
        type Quant = ScalarDenseQuantizer<f32, f32, DotProduct>;
        let quant = Quant::new(2);

        let input = DenseVectorView::new(&[0.5f32, 1.5]);
        let mut buffer = Vec::new();
        quant.push_encoded(input, &mut buffer);
        assert_eq!(buffer, vec![0.5f32, 1.5]);

        let encoded = quant.encode_vector(input);
        assert_eq!(encoded.values(), &[0.5f32, 1.5]);

        let decoded = quant.decode_vector(encoded.as_view());
        assert_eq!(decoded.values(), &[0.5f32, 1.5]);
    }

    #[test]
    fn scalar_dense_query_and_vector_evaluator_use_dot_product() {
        type Quant = ScalarDenseQuantizer<f32, f32, DotProduct>;
        let quant = Quant::new(2);
        let query = DenseVectorView::new(&[1.0f32, 2.0]);

        let evaluator = quant.query_evaluator(query);
        let vector = DenseVectorView::new(&[3.0f32, 4.0]);
        assert_eq!(evaluator.compute_distance(vector), DotProduct::from(11.0));

        let encoded = quant.encode_vector(DenseVectorView::new(&[1.0f32, 1.0]));
        let vector_eval = quant.vector_evaluator(encoded.as_view());
        assert_eq!(
            vector_eval.compute_distance(DenseVectorView::new(&[2.0f32, 0.0])),
            DotProduct::from(2.0)
        );
    }
}
