pub use crate::core::vector1d::{
    DenseVector1DOwned, DenseVector1DView, PackedVectorOwned, PackedVectorView,
    SparseVector1DOwned, SparseVector1DView, Vector1DViewTrait,
};
use crate::{ComponentType, SpaceUsage, ValueType};

/// A query evaluator computes distances between a query and encoded vectors.
pub trait QueryEvaluator {
    type Distance: Copy + Ord;
    type EncodedVector<'a>: Vector1DViewTrait;

    fn compute_distance(&mut self, vector: Self::EncodedVector<'_>) -> Self::Distance;
}

/// Core encoder trait.
pub trait VectorEncoder: Send + Sync + SpaceUsage {
    type Distance: Copy + Ord;

    type QueryVector<'a>: Vector1DViewTrait
    where
        Self: 'a;

    type EncodedVector<'a>: Vector1DViewTrait
    where
        Self: 'a;

    type Evaluator<'a>: QueryEvaluator<Distance = Self::Distance, EncodedVector<'a> = Self::EncodedVector<'a>>
    where
        Self: 'a;

    fn query_evaluator<'a>(&'a self, query: Self::QueryVector<'a>) -> Self::Evaluator<'a>;

    fn input_dim(&self) -> usize;
    fn output_dim(&self) -> usize;
}

pub trait DenseVectorEncoder: VectorEncoder {
    type InputValueType: ValueType;
    type OutputValueType: ValueType;

    fn encode_vector<'a>(
        &self,
        input: DenseVector1DView<'a, Self::InputValueType>,
    ) -> DenseVector1DOwned<Self::OutputValueType>;

    /// Create a view from raw slice data.
    fn create_view<'a>(&self, data: &'a [Self::OutputValueType]) -> Self::EncodedVector<'a>;
}

pub trait SparseVectorEncoder: VectorEncoder {
    type InputComponentType: ComponentType;
    type InputValueType: ValueType;
    type OutputComponentType: ComponentType;
    type OutputValueType: ValueType;

    fn encode_vector<'a>(
        &self,
        input: SparseVector1DView<'a, Self::InputComponentType, Self::InputValueType>,
    ) -> SparseVector1DOwned<Self::OutputComponentType, Self::OutputValueType>;

    /// Create a view from raw component and value slices.
    fn create_view<'a>(
        &self,
        components: &'a [Self::OutputComponentType],
        values: &'a [Self::OutputValueType],
    ) -> Self::EncodedVector<'a>;
}

pub trait PackedVectorEncoder: VectorEncoder {
    type InputComponentType: ComponentType;
    type InputValueType: ValueType;
    type PackedValueType: ValueType + SpaceUsage;

    fn encode_vector<'a>(
        &self,
        input: SparseVector1DView<'a, Self::InputComponentType, Self::InputValueType>,
    ) -> PackedVectorOwned<Self::PackedValueType>;

    /// Create a view from raw packed data.
    fn create_view<'a>(&self, data: &'a [Self::PackedValueType]) -> Self::EncodedVector<'a>;
}

/// Helper trait to query directly from encoded reference if supported?
/// (Optional, derived from old design, maybe remove if unused)
pub trait QueryFromEncoded: VectorEncoder {
    // This seems problematic with GATs and View markers.
    // Leaving it empty or just removing it if not used.
    // It was used in SparseScalar impl but implementation was questionable.
    fn query_from_encoded<'a, V>(&self, encoded: &'a V) -> Self::QueryVector<'a>
    where
        V: ?Sized; // Simplified or removed.
}
