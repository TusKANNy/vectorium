use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::core::distances::{
    Distance, DotProduct, SquaredEuclideanDistance, dot_product_dense_unchecked,
    squared_euclidean_distance_dense_unchecked,
};
use crate::core::vector_encoder::{DenseVectorEncoder, QueryEvaluator, VectorEncoder};
use crate::core::vector1d::DenseVector1DView;
use crate::{Float, FromF32, SpaceUsage, ValueType};

/// Marker trait for distance types supported by scalar dense quantizers.
/// Provides the computation method specific to dense vectors.
pub trait ScalarDenseSupportedDistance: Distance {
    /// Compute distance between two dense float vectors
    fn compute_dense<Q: ValueType + Float, V: ValueType + Float>(
        query: DenseVector1DView<'_, Q>,
        vector: DenseVector1DView<'_, V>,
    ) -> Self;
}

impl ScalarDenseSupportedDistance for SquaredEuclideanDistance {
    fn compute_dense<Q: ValueType + Float, V: ValueType + Float>(
        query: DenseVector1DView<'_, Q>,
        vector: DenseVector1DView<'_, V>,
    ) -> Self {
        assert_eq!(
            query.len(),
            vector.len(),
            "Dense vectors must have the same length"
        );
        unsafe { squared_euclidean_distance_dense_unchecked(query, vector) }
    }
}

impl ScalarDenseSupportedDistance for DotProduct {
    fn compute_dense<Q: ValueType + Float, V: ValueType + Float>(
        query: DenseVector1DView<'_, Q>,
        vector: DenseVector1DView<'_, V>,
    ) -> Self {
        let q_len = query.len();
        let v_len = vector.len();
        assert_eq!(q_len, v_len, "Dense vectors must have the same length");
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
        input: DenseVector1DView<'a, Self::InputValueType>,
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
#[derive(Debug, Clone)]
pub struct ScalarDenseQueryEvaluator<'a, Out, D, V>
where
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
    V: ValueType,
{
    query: Vec<Out>,
    _phantom: PhantomData<&'a (D, V)>,
}

impl<'a, Out, D, V> ScalarDenseQueryEvaluator<'a, Out, D, V>
where
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
    V: ValueType,
{
    pub fn new(query: DenseVector1DView<'a, V>) -> Self {
        let query_converted: Vec<Out> = query
            .values()
            .iter()
            .map(|&v| {
                let f = v.to_f32().expect("Failed to convert value to f32");
                Out::from_f32_saturating(f)
            })
            .collect();

        Self {
            query: query_converted,
            _phantom: PhantomData,
        }
    }
}

impl<'a, 'v, Out, D, V> QueryEvaluator<DenseVector1DView<'v, Out>>
    for ScalarDenseQueryEvaluator<'a, Out, D, V>
where
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
    V: ValueType,
{
    type Distance = D;

    #[inline]
    fn compute_distance(&mut self, vector: DenseVector1DView<'v, Out>) -> D {
        D::compute_dense(DenseVector1DView::new(&self.query), vector)
    }
}

impl<In, Out, D> VectorEncoder for ScalarDenseQuantizer<In, Out, D>
where
    In: ValueType + Float,
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
{
    type Distance = D;
    type InputVector<'a> = DenseVector1DView<'a, In>;
    type QueryVector<'a, V>
        = DenseVector1DView<'a, V>
    where
        V: ValueType;
    type EncodedVector<'a> = DenseVector1DView<'a, Out>;

    type Evaluator<'a, V>
        = ScalarDenseQueryEvaluator<'a, Out, D, V>
    where
        V: ValueType,
        Self: 'a;

    #[inline]
    fn query_evaluator<'a, V>(&'a self, query: Self::QueryVector<'a, V>) -> Self::Evaluator<'a, V>
    where
        V: ValueType,
    {
        assert_eq!(
            query.len(),
            self.input_dim(),
            "Query vector length exceeds quantizer input dimension."
        );
        ScalarDenseQueryEvaluator::new(query)
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
