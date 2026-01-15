pub use crate::core::vector::{
    DenseVectorOwned, DenseVectorView, PackedVectorOwned, PackedVectorView, PlainVectorView,
    SparseVectorOwned, SparseVectorView, VectorView,
};
use crate::{ComponentType, SpaceUsage, ValueType};

/// A query evaluator computes distances between a query and vectors.
///
/// We usually create an evaluator from a query vector once and then use it to compute
/// distances to many dataset vectors.
pub trait QueryEvaluator<V: VectorView> {
    type Distance: Copy + Ord;

    /// Calculates the distance between the evaluator’s query and `vector`.
    ///
    /// Implementations must be fast and should not mutate `vector`.
    fn compute_distance(&self, vector: V) -> Self::Distance;
}

/// The encoder abstraction used by datasets.
///
/// Each encoder maps user-supplied inputs into an encoded representation and exposes
/// the types necessary to build evaluators and query decoders.
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

/// Encoder contract for dense vectors.
///
/// The encoder consumes plain dense inputs (`DenseVectorView`) and produces dense
/// outputs with the encoder-specific `OutputValueType`. Queries are always
/// dense `f32` vectors and the encoded representation is also dense.
///
/// Implementations must support decoding encoded vectors back to `f32` and
/// provide `push_encoded`/`encode_vector` functions to perform the encoding.
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

    /// Encode `input` into the provided `output` container.
    ///
    /// This lets callers either store directly within the dataset container or reuse buffers to amortize allocations.
    fn push_encoded<'a, OutputContainer>(
        &self,
        input: Self::InputVector<'a>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<Self::OutputValueType>;

    /// Convenience helper that builds a `DenseVectorOwned` from `input`.
    ///
    /// The default implementation reuses `push_encoded` to append into a temporary
    /// `Vec`, so callers can drop this method if they already expose `push_encoded`.
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

/// Encoder contract for producing sparse component/value pairs.
///
/// Extends `SparseDataEncoder` by requiring the encoded representation be exposed
/// as sparse component/value slices so callers can build datasets that store explicit
/// offsets/components/values. Implementations are requested to expose `push_encoded`
/// helpers for component/value containers, while the blanket `encode_vector`
/// builds an owned sparse vector.
pub trait SparseVectorEncoder: SparseDataEncoder
where
    for<'a> Self: VectorEncoder<
        EncodedVector<'a> = SparseVectorView<'a, Self::OutputComponentType, Self::OutputValueType>,
    >,
{
    /// Encode `input` directly into dataset-owned component/value containers
    /// to avoid intermediate allocations.
    fn push_encoded<'a, ComponentContainer, ValueContainer>(
        &self,
        input: Self::InputVector<'a>,
        components: &mut ComponentContainer,
        values: &mut ValueContainer,
    ) where
        ComponentContainer: Extend<Self::OutputComponentType>,
        ValueContainer: Extend<Self::OutputValueType>;

    /// Convenience helper that builds an owned sparse vector by replaying `push_encoded`.
    ///
    /// Callers who already manage buffers can skip this method to avoid the additional allocation.
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

/// Encoder contract for handling packed sparse representations.
///
/// Implementors produce `PackedVectorView` outputs while still exposing the sparse
/// metadata captured by `SparseDataEncoder`, letting callers reason about sparsity
/// without committing to explicit component/value buffers.
pub trait PackedSparseVectorEncoder: SparseDataEncoder
where
    for<'a> Self: VectorEncoder<EncodedVector<'a> = PackedVectorView<'a, Self::PackedDataType>>,
{
    type PackedDataType: ValueType + SpaceUsage;

    /// Encode `input` directly into an existing packed data container so datasets can
    /// avoid intermediate allocations.
    fn push_encoded<'a, OutputContainer>(
        &self,
        input: Self::InputVector<'a>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<Self::PackedDataType>;

    /// Convenience helper that builds an owned packed vector by replaying `push_encoded`.
    ///
    /// Implementations expecting to reuse buffers can rely solely on `push_encoded`.
    fn encode_vector<'a>(
        &self,
        input: Self::InputVector<'a>,
    ) -> PackedVectorOwned<Self::PackedDataType> {
        let mut data = Vec::new();
        self.push_encoded(input, &mut data);
        PackedVectorOwned::new(data)
    }
}
