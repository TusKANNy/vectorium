pub use crate::core::vector::{
    DenseVectorOwned, DenseVectorView, PackedVectorOwned, PackedVectorView, PlainVectorView,
    SparseVectorOwned, SparseVectorView, VectorView,
};
use crate::{ComponentType, SpaceUsage, ValueType};

/// A query evaluator computes distances between a query and encoded vectors.
pub trait QueryEvaluator<V: VectorView> {
    type Distance: Copy + Ord;

    fn compute_distance(&self, vector: V) -> Self::Distance;
}

/// Core encoder trait.
pub trait VectorEncoder: Send + Sync + SpaceUsage {
    type Distance: Copy + Ord;

    type InputVector<'a>: VectorView;

    type QueryVector<'q>: PlainVectorView<f32>;

    type EncodedVector<'a>: VectorView;

    /// Evaluator created from a query.
    ///
    /// This evaluator is intended to be created once and then used to compute many distances.
    /// Implementations are allowed to do expensive preprocessing here (e.g., densification,
    /// lookup tables) as long as the resulting evaluator is cheap to use.
    ///
    /// IMPORTANT: the evaluator must not borrow from the query vector.
    type Evaluator<'e>: for<'b> QueryEvaluator<Self::EncodedVector<'b>, Distance = Self::Distance>
    where
        Self: 'e;

    /// Create an evaluator from a *plain* query vector.
    ///
    /// This is designed for search-time queries provided by the user.
    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e>;

    /// Create an evaluator from a dataset-internal vector.
    ///
    /// This is designed for index-build-time scenarios where we need to compute distances
    /// between vectors stored in the dataset. Implementations typically:
    ///
    /// 1) materialize/decompress the encoded vector into a plain `...Owned<..., f32>`;
    /// 2) build and return an evaluator from that plain representation.
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e>;

    fn input_dim(&self) -> usize;
    fn output_dim(&self) -> usize;
}

pub trait DenseVectorEncoder:
    for<'a> VectorEncoder<
        InputVector<'a> = DenseVectorView<'a, Self::InputValueType>,
        QueryVector<'a> = DenseVectorView<'a, f32>,
        EncodedVector<'a> = DenseVectorView<'a, Self::OutputValueType>,
    >
{
    type InputValueType: ValueType;
    type OutputValueType: ValueType;

    /// Decode an encoded vector to a plain dense `f32` vector.
    ///
    /// Intended for index build and other internal workflows.
    fn decode_vector<'a>(&self, encoded: Self::EncodedVector<'a>) -> DenseVectorOwned<f32>;

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

/// A minimal shared trait capturing the sparse input/query contract, component/value types, and decode capability.
/// This lets code reason about sparse datasets without caring whether the encoder produces component/value pairs
/// or a packed blob; it also keeps trait bounds lean by avoiding the heavier `SparseVectorEncoder`/`PackedSparseVectorEncoder`
/// when only the shared behavior is needed.
pub trait SparseDataEncoder
where
    for<'a> Self: VectorEncoder<
            InputVector<'a> = SparseVectorView<'a, Self::InputComponentType, Self::InputValueType>,
            QueryVector<'a> = SparseVectorView<'a, Self::InputComponentType, f32>,
        >,
{
    type InputComponentType: ComponentType;
    type InputValueType: ValueType;
    type OutputComponentType: ComponentType;
    type OutputValueType: ValueType;

    /// Decode an encoded vector to a plain sparse `f32` vector in the *input component* space.
    ///
    /// Intended for index build and other internal workflows.
    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32>;
}

pub trait SparseVectorEncoder: SparseDataEncoder
where
    for<'a> Self: VectorEncoder<
        EncodedVector<'a> = SparseVectorView<'a, Self::OutputComponentType, Self::OutputValueType>,
    >,
{
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

pub trait PackedSparseVectorEncoder: SparseDataEncoder
where
    for<'a> Self: VectorEncoder<EncodedVector<'a> = PackedVectorView<'a, Self::PackedDataType>>,
{
    type PackedDataType: ValueType + SpaceUsage;

    fn push_encoded<'a, OutputContainer>(
        &self,
        input: Self::InputVector<'a>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<Self::PackedDataType>;

    fn encode_vector<'a>(
        &self,
        input: Self::InputVector<'a>,
    ) -> PackedVectorOwned<Self::PackedDataType> {
        let mut data = Vec::new();
        self.push_encoded(input, &mut data);
        PackedVectorOwned::new(data)
    }
}
