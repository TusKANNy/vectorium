use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::core::sealed;
use crate::distances::{
    Distance, DotProduct, SquaredEuclideanDistance, dot_product_dense_unchecked,
    squared_euclidean_distance_dense_unchecked,
};
use crate::numeric_markers::DenseComponent;
use crate::{DenseVector1D, Float, FromF32, SpaceUsage, ValueType, Vector1D};
use crate::{DenseVectorEncoder, QueryEvaluator, QueryFromEncoded, VectorEncoder};

/// Marker trait for distance types supported by scalar dense quantizers.
/// Provides the computation method specific to dense vectors.
pub trait ScalarDenseSupportedDistance: Distance {
    /// Compute distance between two dense float vectors
    fn compute_dense<Q: ValueType + Float, V: ValueType + Float>(
        query: DenseVector1D<Q, &[Q]>,
        vector: DenseVector1D<V, &[V]>,
    ) -> Self;
}

impl ScalarDenseSupportedDistance for SquaredEuclideanDistance {
    fn compute_dense<Q: ValueType + Float, V: ValueType + Float>(
        query: DenseVector1D<Q, &[Q]>,
        vector: DenseVector1D<V, &[V]>,
    ) -> Self {
        let q_len = query.len();
        let v_len = vector.len();
        assert_eq!(q_len, v_len, "Dense vectors must have the same length");
        // SAFETY: We just validated that lengths are equal.
        unsafe { squared_euclidean_distance_dense_unchecked(query, vector) }
    }
}

impl ScalarDenseSupportedDistance for DotProduct {
    fn compute_dense<Q: ValueType + Float, V: ValueType + Float>(
        query: DenseVector1D<Q, &[Q]>,
        vector: DenseVector1D<V, &[V]>,
    ) -> Self {
        let q_len = query.len();
        let v_len = vector.len();
        assert_eq!(q_len, v_len, "Dense vectors must have the same length");
        // SAFETY: We just validated that lengths are equal.
        unsafe { dot_product_dense_unchecked(query, vector) }
    }
}

/// A scalar dense quantizer that converts values from one float type to another.
///
/// - `In`: Input value type (to be encoded, must be Float)
/// - `Out`: Output value type (quantized, must be Float)
/// - `D`: Distance type (must implement ScalarDenseSupportedDistance)
///
/// Query value type is always f32.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarDenseQuantizer<In, Out, D> {
    d: usize,
    _phantom: PhantomData<(In, Out, D)>,
}

impl<In, Out, D> sealed::Sealed for ScalarDenseQuantizer<In, Out, D> {}

/// Scalar quantizer where input and output types are the same.
pub type ScalarDenseQuantizerSame<V, D> = ScalarDenseQuantizer<V, V, D>;

/// Plain quantizer where input and output types are the same.
pub type PlainDenseQuantizer<V, D> = ScalarDenseQuantizer<V, V, D>;

/// Convenience aliases for common configurations
pub type PlainDenseQuantizerSquaredEuclidean<V> = PlainDenseQuantizer<V, SquaredEuclideanDistance>;
pub type PlainDenseQuantizerDotProduct<V> = PlainDenseQuantizer<V, DotProduct>;
pub type ScalarDenseQuantizerSquaredEuclidean<V> =
    ScalarDenseQuantizer<V, V, SquaredEuclideanDistance>;
pub type ScalarDenseQuantizerDotProduct<V> = ScalarDenseQuantizer<V, V, DotProduct>;

impl<In, Out, D> ScalarDenseQuantizer<In, Out, D> {
    pub fn new(d: usize) -> Self {
        Self {
            d,
            _phantom: PhantomData,
        }
    }
}

impl<In, Out, D> DenseVectorEncoder for ScalarDenseQuantizer<In, Out, D>
where
    In: ValueType + Float,
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
{
    fn extend_with_encode<ValueContainer>(
        &self,
        input_vector: DenseVector1D<Self::InputValueType, impl AsRef<[Self::InputValueType]>>,
        values: &mut ValueContainer,
    ) where
        ValueContainer: Extend<Self::OutputValueType>,
    {
        let input = input_vector.values_as_slice();
        values.extend(input.iter().map(|&in_val| {
            // xxxxx_TODO: avoid to_f32 conversion, use direct cast if possible
            let f32_val = in_val.to_f32().unwrap();
            Out::from_f32_saturating(f32_val)
        }));
    }

    #[inline]
    fn encoded_from_slice<'a>(&self, values: &'a [Out]) -> Self::EncodedVectorType<'a>
    where
        Self::EncodedVectorType<'a>: Vector1D<Component = DenseComponent, Value = Out>,
    {
        DenseVector1D::new(values)
    }
}

impl<In, Out, D> VectorEncoder for ScalarDenseQuantizer<In, Out, D>
where
    In: ValueType + Float,
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
{
    type Distance = D;

    type QueryValueType = f32;
    type QueryComponentType = DenseComponent;
    type InputValueType = In;
    type InputComponentType = DenseComponent;
    type InputVectorType<'a> = DenseVector1D<In, &'a [In]>
    where
        Self: 'a;

    type EncodedVectorType<'a> = DenseVector1D<Out, &'a [Out]>
    ;

    type QueryVectorType<'a> = DenseVector1D<f32, &'a [f32]>
    where
        Self: 'a;

    type OutputValueType = Out;
    type OutputComponentType = DenseComponent;

    type Evaluator<'a>
        = ScalarDenseQueryEvaluator<Out, D>
    where
        Self: 'a;

    #[inline]
    fn new(input_dim: usize, output_dim: usize) -> Self {
        assert_eq!(
            input_dim, output_dim,
            "ScalarDenseQuantizer requires input_dim == output_dim"
        );
        Self::new(output_dim)
    }

    #[inline]
    fn query_evaluator<'a>(
        &'a self,
        query: &'a Self::QueryVectorType<'a>,
    ) -> Self::Evaluator<'a> {
        assert_eq!(
            query.len(),
            self.input_dim(),
            "Query vector length exceeds quantizer input dimension."
        );
        ScalarDenseQueryEvaluator::from_query(query)
    }

    #[inline]
    fn output_dim(&self) -> usize {
        self.d
    }

    #[inline]
    fn input_dim(&self) -> usize {
        self.d
    }
}

impl<In, D> QueryFromEncoded for ScalarDenseQuantizer<In, f32, D>
where
    In: ValueType + Float,
    D: ScalarDenseSupportedDistance,
{
    fn query_from_encoded<'a, V>(
        &self,
        encoded: &'a V,
    ) -> Self::QueryVectorType<'a>
    where
        V: Vector1D<Component = DenseComponent, Value = f32> + ?Sized,
        D: 'a,
    {
        DenseVector1D::new(encoded.values_as_slice())
    }
}

/// Query evaluator for ScalarDenseQuantizer.
/// Stores the query (as f32) and computes distances against encoded vectors.
#[derive(Debug, Clone)]
pub struct ScalarDenseQueryEvaluator<Out, D>
where
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
{
    query: Vec<f32>,
    _phantom: PhantomData<(Out, D)>,
}

impl<Out, D> ScalarDenseQueryEvaluator<Out, D>
where
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
{
    pub fn from_query<QueryVector>(query: QueryVector) -> Self
    where
        QueryVector: Vector1D<Value = f32, Component = DenseComponent>,
    {
        Self {
            query: query.values_as_slice().to_vec(),
            _phantom: PhantomData,
        }
    }
}

impl<In, Out, D> SpaceUsage for ScalarDenseQuantizer<In, Out, D>
where
    In: ValueType + Float,
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
{
    fn space_usage_bytes(&self) -> usize {
        self.d.space_usage_bytes()
    }
}

impl<'a, Out, D> QueryEvaluator<DenseVector1D<Out, &'a [Out]>, D>
    for ScalarDenseQueryEvaluator<Out, D>
where
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
{
    #[inline]
    fn compute_distance(&self, vector: DenseVector1D<Out, &'a [Out]>) -> D {
        D::compute_dense(
            DenseVector1D::new(self.query.as_slice()),
            DenseVector1D::new(vector.values_as_slice()),
        )
    }
}
