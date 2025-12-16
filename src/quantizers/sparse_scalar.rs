use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::distances::{Distance, DotProduct, dot_product_dense_sparse};
use crate::quantizers::{Quantizer, QueryEvaluator, SparseQuantizer};
use crate::{
    ComponentType, DenseVector1D, Float, FromF32, MutableVector1D, SparseVector1D, ValueType,
    Vector1D,
};

/// Marker trait for distance types supported by scalar sparse quantizers.
/// Provides the computation method specific to sparse vectors.
pub trait ScalarSparseSupportedDistance: Distance {
    /// Compute distance between a dense query (f32) and a sparse encoded vector
    fn compute_sparse<C: ComponentType, V: ValueType + Float>(
        query_dense: &DenseVector1D<f32, &[f32]>,
        vector_sparse: &SparseVector1D<C, V, &[C], &[V]>,
    ) -> Self;
}

// For now, only DotProduct is implemented.
// EuclideanDistance for sparse vectors would require custom implementation.
impl ScalarSparseSupportedDistance for DotProduct {
    fn compute_sparse<C: ComponentType, V: ValueType + Float>(
        query_dense: &DenseVector1D<f32, &[f32]>,
        vector_sparse: &SparseVector1D<C, V, &[C], &[V]>,
    ) -> Self {
        dot_product_dense_sparse(query_dense, vector_sparse)
    }
}

/// A scalar sparse quantizer that converts values from one float type to another.
/// Component type (C) is the same for input and output.
///
/// - `C`: Component type (input = output)
/// - `InValue`: Input value type (to be encoded, must be ValueType + Float)
/// - `OutValue`: Output value type (quantized, must be ValueType + Float + FromF32)
/// - `D`: Distance type (must implement ScalarSparseSupportedDistance)
///
/// Query value type is always f32.
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarSparseQuantizer<C, InValue, OutValue, D> {
    _phantom: PhantomData<(C, InValue, OutValue, D)>,
}

/// Scalar quantizer where input and output value types are the same.
pub type ScalarSparseQuantizerSame<C, V, D> = ScalarSparseQuantizer<C, V, V, D>;

/// Plain quantizer where input and output value types are the same.
pub type PlainSparseQuantizer<C, V, D> = ScalarSparseQuantizer<C, V, V, D>;

/// Convenience aliases for common configurations (currently only DotProduct is supported)
pub type PlainSparseQuantizerDotProduct<C, V> = PlainSparseQuantizer<C, V, DotProduct>;
pub type ScalarSparseQuantizerDotProduct<C, V> = ScalarSparseQuantizer<C, V, V, DotProduct>;

impl<C, InValue, OutValue, D> ScalarSparseQuantizer<C, InValue, OutValue, D> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<C, InValue, OutValue, D> SparseQuantizer for ScalarSparseQuantizer<C, InValue, OutValue, D>
where
    C: ComponentType,
    InValue: ValueType + Float,
    OutValue: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
{
}

impl<C, InValue, OutValue, D> Quantizer for ScalarSparseQuantizer<C, InValue, OutValue, D>
where
    C: ComponentType,
    InValue: ValueType + Float,
    OutValue: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
{
    type Distance = D;

    type QueryValueType = f32;
    type QueryComponentType = C;
    type InputValueType = InValue;
    type InputComponentType = C;
    type OutputValueType = OutValue;
    type OutputComponentType = C;

    type Evaluator = ScalarSparseQueryEvaluator<C, OutValue, D>;

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
            *out_val = OutValue::from_f32_saturating(f32_val);
        }
    }

    fn get_query_evaluator<QueryVector>(&self, query: QueryVector) -> Self::Evaluator
    where
        QueryVector:
            Vector1D<ValueType = Self::QueryValueType, ComponentType = Self::QueryComponentType>,
    {
        ScalarSparseQueryEvaluator::from_query(query)
    }

    #[inline]
    fn m(&self) -> usize {
        0 // Sparse vectors don't have a fixed dimensionality per vector
    }

    #[inline]
    fn dim(&self) -> usize {
        0 // Sparse quantizer doesn't know global dimensionality
    }
}

/// Query evaluator for ScalarSparseQuantizer.
/// Stores the query (as f32) components and values, and computes distances against encoded vectors.
pub struct ScalarSparseQueryEvaluator<C, OutValue, D>
where
    C: ComponentType,
    OutValue: ValueType + Float,
    D: ScalarSparseSupportedDistance,
{
    query_components: Vec<C>,
    query_values: Vec<f32>,
    _phantom: PhantomData<(OutValue, D)>,
}

impl<C, OutValue, D> ScalarSparseQueryEvaluator<C, OutValue, D>
where
    C: ComponentType,
    OutValue: ValueType + Float,
    D: ScalarSparseSupportedDistance,
{
    pub fn from_query<QueryVector>(query: QueryVector) -> Self
    where
        QueryVector: Vector1D<ValueType = f32, ComponentType = C>,
    {
        let components = query.components_as_slice().to_vec();
        let values = query.values_as_slice().to_vec();
        Self {
            query_components: components,
            query_values: values,
            _phantom: PhantomData,
        }
    }
}

impl<C, InValue, OutValue, D> QueryEvaluator<ScalarSparseQuantizer<C, InValue, OutValue, D>>
    for ScalarSparseQueryEvaluator<C, OutValue, D>
where
    C: ComponentType,
    InValue: ValueType + Float,
    OutValue: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
{
    fn new<QueryVector>(query: QueryVector) -> Self
    where
        QueryVector: Vector1D<
                ValueType = <ScalarSparseQuantizer<C, InValue, OutValue, D> as Quantizer>::QueryValueType,
                ComponentType = <ScalarSparseQuantizer<C, InValue, OutValue, D> as Quantizer>::QueryComponentType,
            >,
    {
        Self::from_query(query)
    }

    fn compute_distance<EncodedVector>(
        &self,
        vector: EncodedVector,
    ) -> <ScalarSparseQuantizer<C, InValue, OutValue, D> as Quantizer>::Distance
    where
        EncodedVector: Vector1D<
                ValueType = <ScalarSparseQuantizer<C, InValue, OutValue, D> as Quantizer>::OutputValueType,
                ComponentType = <ScalarSparseQuantizer<C, InValue, OutValue, D> as Quantizer>::OutputComponentType,
            >,
    {
        // Create dense query wrapper and sparse vector wrapper
        let query_dense = DenseVector1D::new(self.query_values.as_slice());
        let vector_sparse =
            SparseVector1D::new(vector.components_as_slice(), vector.values_as_slice());
        D::compute_sparse(&query_dense, &vector_sparse)
    }
}
