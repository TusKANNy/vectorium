use crate::ComponentType as ComponentTypeTrait;
use crate::ValueType as ValueTypeTrait;
use crate::{MutableVector1D, Vector1D, distances::Distance};

pub mod plain;

/// A query evaluator should be minimal. The new method takes the quantizer because it may want to initialize some structure, e.g., distance table, densify a sparse query, and so on.
/// TODO: add a method to compute more distances at once for better performance (e.g., SIMD).
pub trait QueryEvaluator {
    type QuantizerType: Quantizer;
    type DistanceType: Distance;

    fn new<QueryVector>(query: QueryVector) -> Self
    where
        QueryVector: Vector1D<
                ValueType = <Self::QuantizerType as Quantizer>::QueryValueType,
                ComponentType = <Self::QuantizerType as Quantizer>::QueryComponentType,
            >;

    fn compute_distance<EncodedVector>(&self, vector: EncodedVector) -> Self::DistanceType
    where
        EncodedVector: Vector1D<
                ValueType = <Self::QuantizerType as Quantizer>::OutputValueType,
                ComponentType = <Self::QuantizerType as Quantizer>::OutputComponentType,
            >;

    // fn compute_distance(
    //     &self,
    //     dataset: &<Self::Q as Quantizer>::DatasetType,
    //     index: usize,
    // ) -> Self::DistanceType;

    // #[inline]
    // fn compute_distances(
    //     &self,
    //     dataset: &<Self::Q as Quantizer>::DatasetType,
    //     indexes: impl IntoIterator<Item = usize>,
    // ) -> impl Iterator<Item = f32> {
    //     indexes
    //         .into_iter()
    //         .map(|index| self.compute_distance(dataset, index))
    // }

    // #[inline]
    // fn compute_four_distances(
    //     &self,
    //     dataset: &<Self::Q as Quantizer>::DatasetType,
    //     indexes: impl IntoIterator<Item = usize>,
    // ) -> impl Iterator<Item = f32> {
    //     indexes
    //         .into_iter()
    //         .map(|index| self.compute_distance(dataset, index))
    // }

    // fn topk_retrieval<I, H>(&self, distances: I, heap: &mut H) -> Vec<(f32, usize)>
    // where
    //     I: Iterator<Item = f32>,
    //     H: OnlineTopKSelector;
}

pub trait Quantizer: Sized {
    type QueryValueType: ValueTypeTrait;
    type QueryComponentType: ComponentTypeTrait;
    type InputValueType: ValueTypeTrait;
    type InputComponentType: ComponentTypeTrait;
    type OutputValueType: ValueTypeTrait;
    type OutputComponentType: ComponentTypeTrait;
    type Evaluator: QueryEvaluator<QuantizerType = Self>;

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

    fn get_query_evaluator<QueryVector>(&self, query: QueryVector) -> Self::Evaluator
    where
        QueryVector:
            Vector1D<ValueType = Self::QueryValueType, ComponentType = Self::QueryComponentType>;

    /// Number of components in the quantized vector
    fn m(&self) -> usize;

    /// Dimensionality of the original vector space
    fn dim(&self) -> usize;

    // fn distance(&self) -> Self::Evaluator::DistanceType;

    // fn get_space_usage_bytes(&self) -> usize;
}
