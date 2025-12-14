use crate::ComponentType as ComponentTypeTrait;
use crate::ValueType as ValueTypeTrait;
use crate::{MutableVector1D, Vector1D, distances::Distance};

pub mod dense_scalar;

/// A query evaluator computes distances between a query and encoded vectors.
pub trait QueryEvaluator<Q: Quantizer>: Sized {
    fn new<QueryVector>(query: QueryVector) -> Self
    where
        QueryVector: Vector1D<ValueType = Q::QueryValueType, ComponentType = Q::QueryComponentType>;

    fn compute_distance<EncodedVector>(&self, vector: EncodedVector) -> Q::Distance
    where
        EncodedVector:
            Vector1D<ValueType = Q::OutputValueType, ComponentType = Q::OutputComponentType>;
}

pub trait Quantizer: Sized {
    type Distance: Distance;

    type QueryValueType: ValueTypeTrait;
    type QueryComponentType: ComponentTypeTrait;
    type InputValueType: ValueTypeTrait;
    type InputComponentType: ComponentTypeTrait;
    type OutputValueType: ValueTypeTrait;
    type OutputComponentType: ComponentTypeTrait;

    /// The query evaluator type for this quantizer and distance
    type Evaluator: QueryEvaluator<Self>;

    /// Encode input vectors into quantized output vectors
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
            >;

    /// Get a query evaluator for the given distance type
    fn get_query_evaluator<QueryVector>(&self, query: QueryVector) -> Self::Evaluator
    where
        QueryVector:
            Vector1D<ValueType = Self::QueryValueType, ComponentType = Self::QueryComponentType>;

    /// Number of components in the quantized vector
    fn m(&self) -> usize;

    /// Dimensionality of the original vector space
    fn dim(&self) -> usize;
}
