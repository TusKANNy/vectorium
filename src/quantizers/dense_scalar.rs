use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::distances::{
    Distance, DotProduct, EuclideanDistance, dot_product_dense, euclidean_distance_dense,
};
use crate::num_marker::DenseComponent;
use crate::quantizers::{DenseQuantizer, Quantizer, QueryEvaluator};
use crate::{DenseVector1D, Float, FromF32, SpaceUsage, ValueType, Vector1D};

/// Marker trait for distance types supported by scalar dense quantizers.
/// Provides the computation method specific to dense vectors.
pub trait ScalarDenseSupportedDistance: Distance {
    /// Compute distance between two dense float vectors
    fn compute_dense<Q: ValueType + Float, V: ValueType + Float>(
        query: DenseVector1D<Q, &[Q]>,
        vector: DenseVector1D<V, &[V]>,
    ) -> Self;
}

impl ScalarDenseSupportedDistance for EuclideanDistance {
    fn compute_dense<Q: ValueType + Float, V: ValueType + Float>(
        query: DenseVector1D<Q, &[Q]>,
        vector: DenseVector1D<V, &[V]>,
    ) -> Self {
        euclidean_distance_dense(query, vector)
    }
}

impl ScalarDenseSupportedDistance for DotProduct {
    fn compute_dense<Q: ValueType + Float, V: ValueType + Float>(
        query: DenseVector1D<Q, &[Q]>,
        vector: DenseVector1D<V, &[V]>,
    ) -> Self {
        dot_product_dense(query, vector)
    }
}

/// A scalar dense quantizer that converts values from one float type to another.
///
/// - `In`: Input value type (to be encoded, must be Float)
/// - `Out`: Output value type (quantized, must be Float)
/// - `D`: Distance type (must implement ScalarDenseSupportedDistance)
///
/// Query value type is always f32.
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarDenseQuantizer<In, Out, D> {
    d: usize,
    _phantom: PhantomData<(In, Out, D)>,
}

/// Scalar quantizer where input and output types are the same.
pub type ScalarDenseQuantizerSame<V, D> = ScalarDenseQuantizer<V, V, D>;

/// Plain quantizer where input and output types are the same.
pub type PlainDenseQuantizer<V, D> = ScalarDenseQuantizer<V, V, D>;

/// Convenience aliases for common configurations
pub type PlainDenseQuantizerEuclidean<V> = PlainDenseQuantizer<V, EuclideanDistance>;
pub type PlainDenseQuantizerDotProduct<V> = PlainDenseQuantizer<V, DotProduct>;
pub type ScalarDenseQuantizerEuclidean<V> = ScalarDenseQuantizer<V, V, EuclideanDistance>;
pub type ScalarDenseQuantizerDotProduct<V> = ScalarDenseQuantizer<V, V, DotProduct>;

impl<In, Out, D> ScalarDenseQuantizer<In, Out, D> {
    pub fn new(d: usize) -> Self {
        Self {
            d,
            _phantom: PhantomData,
        }
    }
}

impl<In, Out, D> DenseQuantizer for ScalarDenseQuantizer<In, Out, D>
where
    In: ValueType + Float,
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
{
    fn extend_with_encode<InputVector, ValueContainer>(
        &self,
        input_vector: InputVector,
        values: &mut ValueContainer,
    ) where
        InputVector:
            Vector1D<ValueType = Self::InputValueType, ComponentType = Self::InputComponentType>,
        ValueContainer: Extend<Self::OutputValueType>,
    {
        let input = input_vector.values_as_slice();
        values.extend(input.iter().map(|&in_val| {
            let f32_val = in_val.to_f32().unwrap();
            Out::from_f32_saturating(f32_val)
        }));
    }
}

impl<In, Out, D> Quantizer for ScalarDenseQuantizer<In, Out, D>
where
    In: ValueType + Float,
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
{
    type Distance = D;

    type QueryValueType = f32;
    type QueryComponentType = DenseComponent;
    type InputValueType = In;
    type InputComponentType = DenseComponent;
    type OutputValueType = Out;
    type OutputComponentType = DenseComponent;

    type Evaluator = ScalarDenseQueryEvaluator<Out, D>;

    fn get_query_evaluator<QueryVector>(&self, query: QueryVector) -> Self::Evaluator
    where
        QueryVector:
            Vector1D<ValueType = Self::QueryValueType, ComponentType = Self::QueryComponentType>,
    {
        ScalarDenseQueryEvaluator::from_query(query)
    }

    #[inline]
    fn m(&self) -> usize {
        self.d
    }

    #[inline]
    fn dim(&self) -> usize {
        self.d
    }
}

/// Query evaluator for ScalarDenseQuantizer.
/// Stores the query (as f32) and computes distances against encoded vectors.
pub struct ScalarDenseQueryEvaluator<Out, D>
where
    Out: ValueType + Float,
    D: ScalarDenseSupportedDistance,
{
    query: Vec<f32>,
    _phantom: PhantomData<(Out, D)>,
}

impl<Out, D> ScalarDenseQueryEvaluator<Out, D>
where
    Out: ValueType + Float,
    D: ScalarDenseSupportedDistance,
{
    pub fn from_query<QueryVector>(query: QueryVector) -> Self
    where
        QueryVector: Vector1D<ValueType = f32, ComponentType = DenseComponent>,
    {
        Self {
            query: query.values_as_slice().to_vec(),
            _phantom: PhantomData,
        }
    }
}

impl<In, Out, D> SpaceUsage for ScalarDenseQuantizer<In, Out, D>
where
    In: ValueType + Float,
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
{
    fn space_usage_byte(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

impl<In, Out, D> QueryEvaluator<ScalarDenseQuantizer<In, Out, D>>
    for ScalarDenseQueryEvaluator<Out, D>
where
    In: ValueType + Float,
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
{
    fn new<QueryVector>(query: QueryVector) -> Self
    where
        QueryVector: Vector1D<
                ValueType = <ScalarDenseQuantizer<In, Out, D> as Quantizer>::QueryValueType,
                ComponentType = <ScalarDenseQuantizer<In, Out, D> as Quantizer>::QueryComponentType,
            >,
    {
        Self::from_query(query)
    }

    fn compute_distance<EncodedVector>(&self, vector: EncodedVector) -> <ScalarDenseQuantizer<In, Out, D> as Quantizer>::Distance
    where
        EncodedVector: Vector1D<
                ValueType = <ScalarDenseQuantizer<In, Out, D> as Quantizer>::OutputValueType,
                ComponentType = <ScalarDenseQuantizer<In, Out, D> as Quantizer>::OutputComponentType,
            >,
    {
        D::compute_dense(
            DenseVector1D::new(self.query.as_slice()),
            DenseVector1D::new(vector.values_as_slice()),
        )
    }
}
