use crate::ComponentType as ComponentTypeTrait;
use crate::ValueType as ValueTypeTrait;
use crate::num_marker::DenseComponent;
use crate::{DenseVector1D, SparseVector1D, Vector1D, distances::Distance};

pub mod dense_scalar;
pub mod sparse_scalar;

/// Marker trait for types that are valid query vectors for a given quantizer `Q`.
///
/// This allows `Quantizer::get_query_evaluator` to remain on the base trait while
/// still enforcing that dense/sparse quantizers only accept the corresponding
/// concrete vector representations.
pub trait QueryVectorFor<Q: Quantizer>:
    Vector1D<ValueType = Q::QueryValueType, ComponentType = Q::QueryComponentType>
{
}

impl<Q, AV> QueryVectorFor<Q> for DenseVector1D<Q::QueryValueType, AV>
where
    Q: DenseQuantizer,
    AV: AsRef<[Q::QueryValueType]>,
{
}

impl<Q, AC, AV> QueryVectorFor<Q>
    for SparseVector1D<Q::QueryComponentType, Q::QueryValueType, AC, AV>
where
    Q: SparseQuantizer,
    AC: AsRef<[Q::QueryComponentType]>,
    AV: AsRef<[Q::QueryValueType]>,
{
}

impl<'a, Q, T> QueryVectorFor<Q> for &'a T
where
    Q: Quantizer,
    T: QueryVectorFor<Q> + ?Sized,
{
}

/// A query evaluator computes distances between a query and encoded vectors.
pub trait QueryEvaluator<Q: Quantizer>: Sized {
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

    /// TODO: Do we need these associated types for input and query vector types as a shorthand?

    // type InputVectorType: Vector1D<ValueType = Self::InputValueType, ComponentType = Self::InputComponentType>;
    // type QueryVectorType: Vector1D<ValueType = Self::QueryValueType, ComponentType = Self::QueryComponentType>;

    /// The query evaluator type for this quantizer and distance.
    ///
    /// The evaluator may borrow the query vector, hence it is lifetime-parameterized.
    type Evaluator<'a>: QueryEvaluator<Self>
    where
        Self: 'a;

    // /// Create a new quantizer for input vectors of the given dimensionality `input_dim` and number of components in the quantized vector `output_dim`.
    fn new(input_dim: usize, output_dim: usize) -> Self;

    /// TODO: do we need fn train(data: Option<Dataset<Q>>) -> Self; ?

    /// Get a query evaluator for the given distance type
    fn get_query_evaluator<'a, QueryVector>(&'a self, query: &'a QueryVector) -> Self::Evaluator<'a>
    where
        QueryVector: QueryVectorFor<Self> + ?Sized;

    /// Dimensionality of the encoded vector space.
    ///
    /// For dense encoders, this is typically the number of values produced per vector.
    fn output_dim(&self) -> usize;

    /// Dimensionality of the original (input) vector space.
    ///
    /// For sparse vectors, this is the maximum possible component index + 1.
    fn input_dim(&self) -> usize;
}

pub trait DenseQuantizer:
    Quantizer<
        QueryComponentType = DenseComponent,
        InputComponentType = DenseComponent,
        OutputComponentType = DenseComponent,   
        // TODO: Do we to specify these types for input and query vector types to force?
        // InputVectorType = DenseVector1D<Self::InputValueType, Self::InputStorage>,
        // QueryVectorType = DenseVector1D<Self::QueryValueType, Self::QueryStorage>,
    >
{
    /// Encode input vectors into quantized output vectors
    fn extend_with_encode<ValueContainer>(
        &self,
        input_vector: DenseVector1D<
            Self::InputValueType,
            impl AsRef<[Self::InputValueType]>>,
        values: &mut ValueContainer,
    ) where  
        ValueContainer: Extend<Self::OutputValueType>;
}

pub trait SparseQuantizer: Quantizer {
    /// Encode input vectors into quantized output vectors
    fn extend_with_encode< ValueContainer, ComponentContainer>(
        &self,
        input_vector: SparseVector1D<
            Self::InputComponentType,
            Self::InputValueType,
            impl AsRef<[Self::InputComponentType]>,
            impl AsRef<[Self::InputValueType]>>,
        components: &mut ComponentContainer,
        values: &mut ValueContainer,
    ) where
        ValueContainer: Extend<Self::OutputValueType>,
        ComponentContainer: Extend<Self::OutputComponentType>;
}
