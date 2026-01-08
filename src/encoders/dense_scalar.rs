use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::core::distances::{
    Distance, DotProduct, SquaredEuclideanDistance, dot_product_dense_unchecked,
    squared_euclidean_distance_dense_unchecked,
};
use crate::core::vector_encoder::{
    DenseVector1DOwned, DenseVectorEncoder, QueryEvaluator, VectorEncoder,
};
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
        let q_len = query.len();
        let v_len = vector.len();
        assert_eq!(q_len, v_len, "Dense vectors must have the same length");
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

    fn encode_vector<'a>(
        &self,
        input: DenseVector1DView<'a, Self::InputValueType>,
    ) -> DenseVector1DOwned<Self::OutputValueType> {
        let values: Vec<Out> = input
            .values()
            .iter()
            .map(|&in_val| {
                let f32_val = in_val.to_f32().unwrap();
                Out::from_f32_saturating(f32_val)
            })
            .collect();
        DenseVector1DOwned::new(values)
    }

    fn create_view<'a>(&self, data: &'a [Self::OutputValueType]) -> Self::EncodedVector<'a> {
        DenseVector1DView::new(data)
    }
}

/// Query evaluator for ScalarDenseQuantizer.
#[derive(Debug, Clone)]
pub struct ScalarDenseQueryEvaluator<Out, D>
where
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
{
    query: Vec<f32>,
    _phantom: PhantomData<(Out, D)>,
}

impl<Out, D> ScalarDenseQueryEvaluator<Out, D>
where
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
{
    pub fn new(query: DenseVector1DView<'_, f32>) -> Self {
        Self {
            query: query.values().to_vec(),
            _phantom: PhantomData,
        }
    }
}

impl<Out, D> QueryEvaluator for ScalarDenseQueryEvaluator<Out, D>
where
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
{
    type Distance = D;
    type EncodedVector<'a> = DenseVector1DView<'a, Out>;

    #[inline]
    fn compute_distance(&mut self, vector: DenseVector1DView<'_, Out>) -> D {
        let query_view = DenseVector1DView::new(&self.query);
        D::compute_dense(query_view, vector)
    }
}

impl<In, Out, D> VectorEncoder for ScalarDenseQuantizer<In, Out, D>
where
    In: ValueType + Float,
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
{
    type Distance = D;
    type QueryVector<'a> = DenseVector1DView<'a, f32>;
    type EncodedVector<'a> = DenseVector1DView<'a, Out>;

    type Evaluator<'a>
        = ScalarDenseQueryEvaluator<Out, D>
    where
        Self: 'a;

    #[inline]
    fn query_evaluator<'a>(&'a self, query: Self::QueryVector<'a>) -> Self::Evaluator<'a> {
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
