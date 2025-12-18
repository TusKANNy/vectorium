use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::distances::{
    Distance, DotProduct, dot_product_dense_sparse, dot_product_sparse_with_merge,
};
use crate::quantizers::{Quantizer, QueryEvaluator, QueryVectorFor, SparseQuantizer};
use crate::{
    ComponentType, DenseVector1D, Float, FromF32, SpaceUsage, SparseVector1D, ValueType, Vector1D,
};

/// Marker trait for distance types supported by scalar sparse quantizers.
/// Provides the computation method specific to sparse vectors.
pub trait ScalarSparseSupportedDistance: Distance {
    /// Compute distance between a dense query (f32) and a sparse encoded vector
    fn compute_sparse<C: ComponentType, V: ValueType + Float>(
        dense_query: &Option<DenseVector1D<f32, impl AsRef<[f32]>>>,
        query: &SparseVector1D<C, f32, impl AsRef<[C]>, impl AsRef<[f32]>>,
        vector_sparse: &SparseVector1D<C, V, impl AsRef<[C]>, impl AsRef<[V]>>,
    ) -> Self;
}

// For now, only DotProduct is implemented.
// EuclideanDistance for sparse vectors would require custom implementation.
impl ScalarSparseSupportedDistance for DotProduct {
    #[inline]
    fn compute_sparse<C: ComponentType, V: ValueType + Float>(
        dense_query: &Option<DenseVector1D<f32, impl AsRef<[f32]>>>,
        query: &SparseVector1D<C, f32, impl AsRef<[C]>, impl AsRef<[f32]>>,
        vector_sparse: &SparseVector1D<C, V, impl AsRef<[C]>, impl AsRef<[V]>>,
    ) -> Self {
        if dense_query.is_none() {
            unsafe { dot_product_sparse_with_merge(query, vector_sparse) }
        } else {
            unsafe { dot_product_dense_sparse(dense_query.as_ref().unwrap(), vector_sparse) }
        }
    }
}

/// A scalar sparse quantizer that converts values from one float type to another.
/// Component type (C) is the same for input and output.
///
/// - `C`: Component type (input = output)
/// - `InValue`: Input value type (to be encoded, must be ValueType + Float)
/// - `OutValue`: Output value type (quantized, must be ValueType + Float + FromF32)
/// - `D`: Distance type (must implement ScalarSparseSupportedDistance)
///
/// Query value type is always f32.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarSparseQuantizer<C, InValue, OutValue, D> {
    dim: usize,
    _phantom: PhantomData<(C, InValue, OutValue, D)>,
}

/// Scalar quantizer where input and output value types are the same.
pub type ScalarSparseQuantizerSame<C, V, D> = ScalarSparseQuantizer<C, V, V, D>;

/// Plain quantizer where input and output value types are the same.
pub type PlainSparseQuantizer<C, V, D> = ScalarSparseQuantizer<C, V, V, D>;

/// Convenience aliases for common configurations (currently only DotProduct is supported)
pub type PlainSparseQuantizerDotProduct<C, V> = PlainSparseQuantizer<C, V, DotProduct>;
pub type ScalarSparseQuantizerDotProduct<C, V> = ScalarSparseQuantizer<C, V, V, DotProduct>;

impl<C, InValue, OutValue, D> SparseQuantizer for ScalarSparseQuantizer<C, InValue, OutValue, D>
where
    C: ComponentType,
    InValue: ValueType + Float,
    OutValue: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
{
    fn extend_with_encode<ValueContainer, ComponentContainer>(
        &self,
        input_vector: SparseVector1D<
            Self::InputComponentType,
            Self::InputValueType,
            impl AsRef<[Self::InputComponentType]>,
            impl AsRef<[Self::InputValueType]>,
        >,
        components: &mut ComponentContainer,
        values: &mut ValueContainer,
    ) where
        ValueContainer: Extend<Self::OutputValueType>,
        ComponentContainer: Extend<Self::OutputComponentType>,
    {
        let input_components = input_vector.components_as_slice();
        let input_values = input_vector.values_as_slice();

        components.extend(input_components.iter().copied());
        values.extend(input_values.iter().map(|in_val| {
            let f32_val = in_val.to_f32().unwrap();
            OutValue::from_f32_saturating(f32_val)
        }));
    }
}

impl<C, InValue, OutValue, D> Quantizer for ScalarSparseQuantizer<C, InValue, OutValue, D>
where
    C: ComponentType,
    InValue: ValueType + Float,
    OutValue: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
{
    type Distance = D;

    type QueryValueType = f32;
    type QueryComponentType = C;
    type InputValueType = InValue;
    type InputComponentType = C;
    type OutputValueType = OutValue;
    type OutputComponentType = C;

    type Evaluator = ScalarSparseQueryEvaluator<C, OutValue, D, Vec<C>, Vec<f32>>;

    #[inline]
    fn new(input_dim: usize, output_dim: usize) -> Self {
        assert_eq!(
            input_dim, output_dim,
            "ScalarSparseQuantizer requires input_dim == output_dim"
        );

        Self {
            dim: input_dim,
            _phantom: PhantomData,
        }
    }

    fn get_query_evaluator<QueryVector>(&self, query: QueryVector) -> Self::Evaluator
    where
        QueryVector: QueryVectorFor<Self>,
    {
        <Self::Evaluator as QueryEvaluator<Self>>::new(query, self)
    }

    #[inline]
    fn output_dim(&self) -> usize {
        self.dim
    }

    #[inline]
    fn input_dim(&self) -> usize {
        self.dim
    }
}

/// Query evaluator for ScalarSparseQuantizer.
/// Stores the query (as f32) components and values, and computes distances against encoded vectors.
pub struct ScalarSparseQueryEvaluator<C, OutValue, D, AC, AV>
where
    C: ComponentType,
    OutValue: ValueType + Float,
    D: ScalarSparseSupportedDistance,
    AC: AsRef<[C]>,
    AV: AsRef<[f32]>,
{
    dense_query: Option<DenseVector1D<f32, Vec<f32>>>,
    query: SparseVector1D<C, f32, AC, AV>,
    _phantom: PhantomData<(OutValue, D)>,
}

impl<C, InValue, OutValue, D> QueryEvaluator<ScalarSparseQuantizer<C, InValue, OutValue, D>>
    for ScalarSparseQueryEvaluator<C, OutValue, D, Vec<C>, Vec<f32>>
where
    C: ComponentType,
    InValue: ValueType + Float,
    OutValue: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
{
    fn new<QueryVector>(
        query: QueryVector,
        quantizer: &ScalarSparseQuantizer<C, InValue, OutValue, D>,
    ) -> Self
    where
        QueryVector: Vector1D<
                ValueType = <ScalarSparseQuantizer<C, InValue, OutValue, D> as Quantizer>::QueryValueType,
                ComponentType = <ScalarSparseQuantizer<C, InValue, OutValue, D> as Quantizer>::QueryComponentType,
            >,
    {
        let dense_query = if quantizer.input_dim() < 2_usize.pow(20) {
            // For small dimensions, create a dense representation
            let mut dense_query = vec![0.0; quantizer.input_dim()];
            let components = query.components_as_slice();
            let values = query.values_as_slice();
            for (&i, &v) in components.iter().zip(values) {
                dense_query[i.as_()] = v;
            }
            Some(DenseVector1D::new(dense_query))
        } else {
            // For large dimensions, keep sparse representation
            None
        };

        let query_components = query.components_as_slice().to_vec();
        let query_values = query.values_as_slice().to_vec();
        Self {
            dense_query,
            query: SparseVector1D::new(query_components, query_values),
            _phantom: PhantomData,
        }
    }

    #[inline]
    fn compute_distance<EncodedVector>(
        &self,
        vector: EncodedVector,
    ) -> <ScalarSparseQuantizer<C, InValue, OutValue, D> as Quantizer>::Distance
    where
        EncodedVector: Vector1D<
                ValueType = <ScalarSparseQuantizer<C, InValue, OutValue, D> as Quantizer>::OutputValueType,
                ComponentType = <ScalarSparseQuantizer<C, InValue, OutValue, D> as Quantizer>::OutputComponentType,
            >,
    {
        let vector_sparse =
            SparseVector1D::new(vector.components_as_slice(), vector.values_as_slice());
        D::compute_sparse(&self.dense_query, &self.query, &vector_sparse)
    }
}

/// SpaceUsage implementation for ScalarSparseQuantizer
/// Since ScalarSparseQuantizer only contains PhantomData, it has zero size
impl<C, InValue, OutValue, D> SpaceUsage for ScalarSparseQuantizer<C, InValue, OutValue, D>
where
    C: ComponentType,
    InValue: ValueType,
    OutValue: ValueType,
    D: ScalarSparseSupportedDistance,
{
    fn space_usage_byte(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}
