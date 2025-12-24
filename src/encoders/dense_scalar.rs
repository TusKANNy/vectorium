use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::distances::{
    Distance, DotProduct, EuclideanDistance, dot_product_dense, euclidean_distance_dense,
};
use crate::numeric_markers::DenseComponent;
use crate::{DenseQuantizer, QueryEvaluator, QueryVectorFor, VectorEncoder};
use crate::{DenseVector1D, Float, FromF32, SpaceUsage, ValueType, Vector1D};

/// Marker trait for distance types supported by scalar dense quantizers.
/// Provides the computation method specific to dense vectors.
pub trait ScalarDenseSupportedDistance: Distance {
    /// Compute distance between two dense float vectors
    fn compute_dense<Q: ValueType + Float, V: ValueType + Float>(
        query: DenseVector1D<Q, &[Q]>,
        vector: DenseVector1D<V, &[V]>,
    ) -> Self;
}

impl ScalarDenseSupportedDistance for EuclideanDistance {
    fn compute_dense<Q: ValueType + Float, V: ValueType + Float>(
        query: DenseVector1D<Q, &[Q]>,
        vector: DenseVector1D<V, &[V]>,
    ) -> Self {
        let q_len = query.len();
        let v_len = vector.len();
        assert_eq!(q_len, v_len, "Dense vectors must have the same length");
        unsafe { euclidean_distance_dense(query, vector) }
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
        unsafe { dot_product_dense(query, vector) }
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

/// Scalar quantizer where input and output types are the same.
pub type ScalarDenseQuantizerSame<V, D> = ScalarDenseQuantizer<V, V, D>;

/// Plain quantizer where input and output types are the same.
pub type PlainDenseQuantizer<V, D> = ScalarDenseQuantizer<V, V, D>;

/// Convenience aliases for common configurations
pub type PlainDenseQuantizerEuclidean<V> = PlainDenseQuantizer<V, EuclideanDistance>;
pub type PlainDenseQuantizerDotProduct<V> = PlainDenseQuantizer<V, DotProduct>;
pub type ScalarDenseQuantizerEuclidean<V> = ScalarDenseQuantizer<V, V, EuclideanDistance>;
pub type ScalarDenseQuantizerDotProduct<V> = ScalarDenseQuantizer<V, V, DotProduct>;

impl<In, Out, D> ScalarDenseQuantizer<In, Out, D> {
    pub fn new(d: usize) -> Self {
        Self {
            d,
            _phantom: PhantomData,
        }
    }
}

impl<In, Out, D> DenseQuantizer for ScalarDenseQuantizer<In, Out, D>
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
            let f32_val = in_val.to_f32().unwrap();
            Out::from_f32_saturating(f32_val)
        }));
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
    type OutputValueType = Out;
    type OutputComponentType = DenseComponent;

    type Evaluator<'a> = ScalarDenseQueryEvaluator<Out, D> where Self: 'a;
    type EncodedVector<'a> = DenseVector1D<Out, &'a [Out]>;

    #[inline]
    fn new(input_dim: usize, output_dim: usize) -> Self {
        assert_eq!(
            input_dim, output_dim,
            "ScalarDenseQuantizer requires input_dim == output_dim"
        );
        Self::new(output_dim)
    }

    #[inline]
    fn get_query_evaluator<'a, QueryVector>(&'a self, query: &'a QueryVector) -> Self::Evaluator<'a>
    where
        QueryVector: QueryVectorFor<Self> + ?Sized,
    {
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

/// Query evaluator for ScalarDenseQuantizer.
/// Stores the query (as f32) and computes distances against encoded vectors.
pub struct ScalarDenseQueryEvaluator<Out, D>
where
    Out: ValueType + Float,
    D: ScalarDenseSupportedDistance,
{
    query: Vec<f32>,
    _phantom: PhantomData<(Out, D)>,
}

impl<Out, D> ScalarDenseQueryEvaluator<Out, D>
where
    Out: ValueType + Float,
    D: ScalarDenseSupportedDistance,
{
    pub fn from_query<QueryVector>(query: QueryVector) -> Self
    where
        QueryVector: Vector1D<ValueType = f32, ComponentType = DenseComponent>,
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
    fn space_usage_byte(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

impl<In, Out, D> QueryEvaluator<ScalarDenseQuantizer<In, Out, D>>
    for ScalarDenseQueryEvaluator<Out, D>
where
    In: ValueType + Float,
    Out: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
{
    fn compute_distance(
        &self,
        vector: <ScalarDenseQuantizer<In, Out, D> as VectorEncoder>::EncodedVector<'_>,
    ) -> <ScalarDenseQuantizer<In, Out, D> as VectorEncoder>::Distance {
        D::compute_dense(
            DenseVector1D::new(self.query.as_slice()),
            DenseVector1D::new(vector.values_as_slice()),
        )
    }
}
