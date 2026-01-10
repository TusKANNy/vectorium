use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::core::distances::{
    Distance, DotProduct, SquaredEuclideanDistance, dot_product_dense_unchecked,
    squared_euclidean_distance_dense_unchecked,
};
use crate::core::vector::DenseVectorView;
use crate::core::vector_encoder::{DenseVectorEncoder, QueryEvaluator, VectorEncoder};
use crate::{Float, FromF32, SpaceUsage, ValueType};

/// Marker trait for distance types supported by scalar dense quantizers.
/// Provides the computation method specific to dense vectors.
pub trait ScalarDenseSupportedDistance: Distance {
    /// Compute distance between two dense float vectors
    fn compute_dense<Q: ValueType, V: ValueType>(
        query: DenseVectorView<'_, Q>,
        vector: DenseVectorView<'_, V>,
    ) -> Self;
}

impl ScalarDenseSupportedDistance for SquaredEuclideanDistance {
    fn compute_dense<Q: ValueType, V: ValueType>(
        query: DenseVectorView<'_, Q>,
        vector: DenseVectorView<'_, V>,
    ) -> Self {
        unsafe { squared_euclidean_distance_dense_unchecked(query, vector) }
    }
}

impl ScalarDenseSupportedDistance for DotProduct {
    fn compute_dense<Q: ValueType, V: ValueType>(
        query: DenseVectorView<'_, Q>,
        vector: DenseVectorView<'_, V>,
    ) -> Self {
        unsafe { dot_product_dense_unchecked(query, vector) }
    }
}

/// A scalar dense quantizer that converts values from one float type to another.
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

/// Query evaluator for ScalarDenseQuantizer.
#[derive(Debug, Clone, Copy)]
pub struct ScalarDenseQueryEvaluator<'e, 'q, In, Out, D, V>
where
    In: ValueType,
    Out: ValueType + FromF32,
    D: ScalarDenseSupportedDistance,
    V: ValueType,
{
    encoder: &'e ScalarDenseQuantizer<In, Out, D>,
    query: DenseVectorView<'q, V>,
}

impl<'e, 'q, In, Out, D, V> ScalarDenseQueryEvaluator<'e, 'q, In, Out, D, V>
where
    In: ValueType,
    Out: ValueType + FromF32,
    D: ScalarDenseSupportedDistance,
    V: ValueType,
{
    #[inline]
    pub fn new(
        encoder: &'e ScalarDenseQuantizer<In, Out, D>,
        query: DenseVectorView<'q, V>,
    ) -> Self {
        Self { encoder, query }
    }
}

impl<'e, 'q, 'v, In, Out, D, V> QueryEvaluator<DenseVectorView<'v, Out>>
    for ScalarDenseQueryEvaluator<'e, 'q, In, Out, D, V>
where
    In: ValueType,
    Out: ValueType + FromF32,
    D: ScalarDenseSupportedDistance,
    V: ValueType,
{
    type Distance = D;

    #[inline]
    fn compute_distance(&mut self, vector: DenseVectorView<'v, Out>) -> D {
        let _ = self.encoder;
        D::compute_dense(self.query, vector)
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
    type QueryVector<'q, V>
        = DenseVectorView<'q, V>
    where
        V: ValueType;
    type EncodedVector<'a> = DenseVectorView<'a, Out>;

    type Evaluator<'e, 'q, V>
        = ScalarDenseQueryEvaluator<'e, 'q, In, Out, D, V>
    where
        V: ValueType,
        Self: 'e;

    #[inline]
    fn query_evaluator<'e, 'q, V>(
        &'e self,
        query: Self::QueryVector<'q, V>,
    ) -> Self::Evaluator<'e, 'q, V>
    where
        V: ValueType,
    {
        assert_eq!(
            query.len(),
            self.input_dim(),
            "Query vector length exceeds quantizer input dimension."
        );
        ScalarDenseQueryEvaluator::new(self, query)
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
