use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::distances::{
    Distance, DotProduct, EuclideanDistance, dot_product_dense, euclidean_distance_dense,
};
use crate::quantizers::{Quantizer, QueryEvaluator};
use crate::{DenseComponent, DenseVector1D, FromF32, MutableVector1D, ValueType, Vector1D};

/// Marker trait for distance types supported by plain dense quantizers.
/// Provides the computation method specific to dense vectors.
pub trait PlainDenseSupportedDistance: Distance {
    /// Compute distance between two dense f32 vectors
    fn compute_dense(query: DenseVector1D<f32, &[f32]>, vector: DenseVector1D<f32, &[f32]>)
    -> Self;
}

impl PlainDenseSupportedDistance for EuclideanDistance {
    fn compute_dense(
        query: DenseVector1D<f32, &[f32]>,
        vector: DenseVector1D<f32, &[f32]>,
    ) -> Self {
        euclidean_distance_dense(query, vector)
    }
}

impl PlainDenseSupportedDistance for DotProduct {
    fn compute_dense(
        query: DenseVector1D<f32, &[f32]>,
        vector: DenseVector1D<f32, &[f32]>,
    ) -> Self {
        dot_product_dense(query, vector)
    }
}

/// A scalar dense quantizer that converts values from one float type to another.
///
/// - `In`: Input value type (to be encoded)
/// - `Out`: Output value type (quantized)
/// - `D`: Distance type (must implement PlainDenseSupportedDistance)
///
/// Query value type is always f32.
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarDenseQuantizer<In, Out, D> {
    d: usize,
    _phantom: PhantomData<(In, Out, D)>,
}

/// Plain quantizer where input and output types are the same.
pub type PlainDenseQuantizer<V, D> = ScalarDenseQuantizer<V, V, D>;

/// Convenience aliases for common configurations
pub type PlainDenseQuantizerEuclidean<V> = PlainDenseQuantizer<V, EuclideanDistance>;
pub type PlainDenseQuantizerDotProduct<V> = PlainDenseQuantizer<V, DotProduct>;

impl<In, Out, D> ScalarDenseQuantizer<In, Out, D> {
    pub fn new(d: usize) -> Self {
        Self {
            d,
            _phantom: PhantomData,
        }
    }
}

impl<In, Out, D> Quantizer for ScalarDenseQuantizer<In, Out, D>
where
    In: ValueType,
    Out: ValueType + FromF32,
    D: PlainDenseSupportedDistance,
{
    type Distance = D;

    type QueryValueType = f32;
    type QueryComponentType = DenseComponent;
    type InputValueType = In;
    type InputComponentType = DenseComponent;
    type OutputValueType = Out;
    type OutputComponentType = DenseComponent;

    type Evaluator = ScalarDenseQueryEvaluator<Out, D>;

    fn encode<InputVector, OutputVector>(
        &self,
        input_vector: InputVector,
        output_vector: &mut OutputVector,
    ) where
        InputVector:
            Vector1D<ValueType = Self::InputValueType, ComponentType = Self::InputComponentType>,
        OutputVector: MutableVector1D<
                ValueType = Self::OutputValueType,
                ComponentType = Self::OutputComponentType,
            >,
    {
        let input = input_vector.values_as_slice();
        let output = output_vector.values_as_mut_slice();

        for (out_val, in_val) in output.iter_mut().zip(input.iter()) {
            let f32_val = in_val.to_f32().unwrap();
            *out_val = Out::from_f32_saturating(f32_val);
        }
    }

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
    Out: ValueType,
    D: PlainDenseSupportedDistance,
{
    query: Vec<f32>,
    _phantom: PhantomData<(Out, D)>,
}

impl<Out, D> ScalarDenseQueryEvaluator<Out, D>
where
    Out: ValueType,
    D: PlainDenseSupportedDistance,
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

impl<In, Out, D> QueryEvaluator<ScalarDenseQuantizer<In, Out, D>>
    for ScalarDenseQueryEvaluator<Out, D>
where
    In: ValueType,
    Out: ValueType + FromF32,
    D: PlainDenseSupportedDistance,
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
        // Convert encoded vector to f32
        let encoded = vector.values_as_slice();
        let vec_f32: Vec<f32> = encoded.iter().map(|v| v.to_f32().unwrap()).collect();
        D::compute_dense(
            DenseVector1D::new(self.query.as_slice()),
            DenseVector1D::new(vec_f32.as_slice()),
        )
    }
}
