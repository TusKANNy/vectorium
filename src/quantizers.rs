use crate::ComponentType as ComponentTypeTrait;
use crate::ValueType as ValueTypeTrait;
use crate::num_marker::DenseComponent;
use crate::{Vector1D, distances::Distance};

pub mod dense_scalar;
pub mod sparse_scalar;

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

    // type InputStorage: AsRef<[Self::InputValueType]>;
    // type QueryStorage: AsRef<[Self::QueryValueType]>;

    // type InputVectorType: Vector1D<ValueType = Self::InputValueType, ComponentType = Self::InputComponentType>;
    // type QueryVectorType: Vector1D<ValueType = Self::QueryValueType, ComponentType = Self::QueryComponentType>;

    /// The query evaluator type for this quantizer and distance
    type Evaluator: QueryEvaluator<Self>;

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

pub trait DenseQuantizer:
    Quantizer<
        QueryComponentType = DenseComponent,
        InputComponentType = DenseComponent,
        OutputComponentType = DenseComponent,
        // InputVectorType = DenseVector1D<Self::InputValueType, Self::InputStorage>,
        // QueryVectorType = DenseVector1D<Self::QueryValueType, Self::QueryStorage>,
    >
{
    /// Encode input vectors into quantized output vectors
    fn extend_with_encode<InputVector, ValueContainer>(
        &self,
        input_vector: InputVector,
        values: &mut ValueContainer,
    ) where
        InputVector: Vector1D<
            ValueType = Self::InputValueType,
            ComponentType = Self::InputComponentType,
        >,
        ValueContainer: Extend<Self::OutputValueType>;
}

pub trait SparseQuantizer: Quantizer {
    /// Encode input vectors into quantized output vectors
    fn extend_with_encode<InputVector, ValueContainer, ComponentContainer>(
        &self,
        input_vector: InputVector,
        components: &mut ComponentContainer,
        values: &mut ValueContainer,
    ) where
        InputVector:
            Vector1D<ValueType = Self::InputValueType, ComponentType = Self::InputComponentType>,
        ValueContainer: Extend<Self::OutputValueType>,
        ComponentContainer: Extend<Self::OutputComponentType>;
}
