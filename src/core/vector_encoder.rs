pub use crate::core::vector::{
    DenseVectorOwned, DenseVectorView, PackedVectorOwned, PackedVectorView,
    SparseVectorOwned, SparseVectorView, VectorView,
};
use crate::{ComponentType, SpaceUsage, ValueType};

/// A query evaluator computes distances between a query and encoded vectors.
pub trait QueryEvaluator<V: VectorView> {
    type Distance: Copy + Ord;

    fn compute_distance(&mut self, vector: V) -> Self::Distance;
}

/// Core encoder trait.
pub trait VectorEncoder: Send + Sync + SpaceUsage {
    type Distance: Copy + Ord;

    type InputVector<'a>: VectorView;

    type QueryVector<'a, V>: VectorView
    where
        V: ValueType,
        Self: 'a;

    type EncodedVector<'a>: VectorView;

    type Evaluator<'a, V>: for<'b> QueryEvaluator<Self::EncodedVector<'b>, Distance = Self::Distance>
    where
        V: ValueType,
        Self: 'a;

    fn query_evaluator<'a, V>(&'a self, query: Self::QueryVector<'a, V>) -> Self::Evaluator<'a, V>
    where
        V: ValueType;

    fn input_dim(&self) -> usize;
    fn output_dim(&self) -> usize;
}

pub trait DenseVectorEncoder:
    for<'a> VectorEncoder<
        InputVector<'a> = DenseVectorView<'a, Self::InputValueType>,
        EncodedVector<'a> = DenseVectorView<'a, Self::OutputValueType>,
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
    ) -> DenseVectorOwned<Self::OutputValueType> {
        let mut values = Vec::new();
        self.push_encoded(input, &mut values);
        DenseVectorOwned::new(values)
    }
}

pub trait SparseVectorEncoder:
    for<'a> VectorEncoder<
        InputVector<'a> = SparseVectorView<'a, Self::InputComponentType, Self::InputValueType>,
        EncodedVector<'a> = SparseVectorView<
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
    ) -> SparseVectorOwned<Self::OutputComponentType, Self::OutputValueType> {
        let mut components = Vec::new();
        let mut values = Vec::new();
        self.push_encoded(input, &mut components, &mut values);
        SparseVectorOwned::new(components, values)
    }
}

pub trait PackedSparseVectorEncoder:
    for<'a> VectorEncoder<
        InputVector<'a> = SparseVectorView<'a, Self::InputComponentType, Self::InputValueType>,
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
}
