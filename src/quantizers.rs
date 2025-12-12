use crate::{Vector1D, distances::Distance};

pub mod plain;

/// A query evaluator should be minimal. The new method takes the quantizer because it may want to initialize some structure, e.g., distance table, densify a sparse query, and so on.
/// TODO: add a method to compute more distances at once for better performance (e.g., SIMD).
pub trait QueryEvaluator {
    type QuantizerType: Quantizer;
    type DistanceType: Distance;

    fn new(query: <Self::QuantizerType as Quantizer>::QueryType) -> Self;

    fn compute_distance(
        &self,
        vector: <Self::QuantizerType as Quantizer>::OutputVector1D,
    ) -> Self::DistanceType;

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
    type QueryType: Vector1D;
    type InputVector1D: Vector1D;
    type OutputVector1D: Vector1D;
    type Evaluator: QueryEvaluator;

    /// Encode input vectors into quantized output vectors
    fn encode(&self, input_vectors: Self::InputVector1D, output_vectors: &mut Self::OutputVector1D);

    fn get_query_evaluator(&self, query: Self::QueryType) -> Self::Evaluator;

    /// Number of components in the quantized vector
    fn m(&self) -> usize;

    /// Dimensionality of the original vector space
    fn dim(&self) -> usize;

    // fn distance(&self) -> Self::Evaluator::DistanceType;

    // fn get_space_usage_bytes(&self) -> usize;
}
