pub use crate::core::vector1d::{
    DenseVector1DOwned, DenseVector1DView, PackedVectorOwned, PackedVectorView,
    SparseVector1DOwned, SparseVector1DView, Vector1DViewTrait,
};
use crate::{ComponentType, SpaceUsage, ValueType};

/// A query evaluator computes distances between a query and encoded vectors.
pub trait QueryEvaluator<V: Vector1DViewTrait> {
    type Distance: Copy + Ord;

    fn compute_distance(&mut self, vector: V) -> Self::Distance;
}

/// Core encoder trait.
pub trait VectorEncoder: Send + Sync + SpaceUsage {
    type Distance: Copy + Ord;

    type InputVector<'a>: Vector1DViewTrait;

    type QueryVector<'a>: Vector1DViewTrait
    where
        Self: 'a;

    type EncodedVector<'a>: Vector1DViewTrait;

    type Evaluator<'a>: for<'b> QueryEvaluator<Self::EncodedVector<'b>, Distance = Self::Distance>
    where
        Self: 'a;

    fn query_evaluator<'a>(&'a self, query: Self::QueryVector<'a>) -> Self::Evaluator<'a>;

    fn input_dim(&self) -> usize;
    fn output_dim(&self) -> usize;
}

pub trait DenseVectorEncoder:
    for<'a> VectorEncoder<
        InputVector<'a> = DenseVector1DView<'a, Self::InputValueType>,
        EncodedVector<'a> = DenseVector1DView<'a, Self::OutputValueType>,
    >
{
    type InputValueType: ValueType;
    type OutputValueType: ValueType;

    fn push_encoded<'a, OutputContainer>(
        &self,
        input: Self::InputVector<'a>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<Self::OutputValueType>;

    fn encode_vector<'a>(
        &self,
        input: Self::InputVector<'a>,
    ) -> DenseVector1DOwned<Self::OutputValueType> {
        let mut values = Vec::new();
        self.push_encoded(input, &mut values);
        DenseVector1DOwned::new(values)
    }
}

pub trait SparseVectorEncoder:
    for<'a> VectorEncoder<
        InputVector<'a> = SparseVector1DView<'a, Self::InputComponentType, Self::InputValueType>,
        EncodedVector<'a> = SparseVector1DView<
            'a,
            Self::OutputComponentType,
            Self::OutputValueType,
        >,
    >
{
    type InputComponentType: ComponentType;
    type InputValueType: ValueType;
    type OutputComponentType: ComponentType;
    type OutputValueType: ValueType;

    fn push_encoded<'a, ComponentContainer, ValueContainer>(
        &self,
        input: Self::InputVector<'a>,
        components: &mut ComponentContainer,
        values: &mut ValueContainer,
    ) where
        ComponentContainer: Extend<Self::OutputComponentType>,
        ValueContainer: Extend<Self::OutputValueType>;

    fn encode_vector<'a>(
        &self,
        input: Self::InputVector<'a>,
    ) -> SparseVector1DOwned<Self::OutputComponentType, Self::OutputValueType> {
        let mut components = Vec::new();
        let mut values = Vec::new();
        self.push_encoded(input, &mut components, &mut values);
        SparseVector1DOwned::new(components, values)
    }
}

pub trait PackedVectorEncoder:
    for<'a> VectorEncoder<
        InputVector<'a> = SparseVector1DView<'a, Self::InputComponentType, Self::InputValueType>,
        EncodedVector<'a> = PackedVectorView<'a, Self::PackedValueType>,
    >
{
    type InputComponentType: ComponentType;
    type InputValueType: ValueType;
    type PackedValueType: ValueType + SpaceUsage;

    fn push_encoded<'a, OutputContainer>(
        &self,
        input: Self::InputVector<'a>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<Self::PackedValueType>;

    fn encode_vector<'a>(
        &self,
        input: Self::InputVector<'a>,
    ) -> PackedVectorOwned<Self::PackedValueType> {
        let mut data = Vec::new();
        self.push_encoded(input, &mut data);
        PackedVectorOwned::new(data)
    }
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
