use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::core::sealed;
use crate::distances::{
    Distance, DotProduct, dot_product_dense_sparse_unchecked,
    dot_product_sparse_with_merge_unchecked,
};
use crate::utils::is_strictly_sorted;
use crate::{
    ComponentType, DenseVector1D, Float, FromF32, SpaceUsage, SparseVector1D, ValueType, Vector1D,
};
use crate::{QueryEvaluator, QueryFromEncoded, SparseVectorEncoder, VectorEncoder};

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
// SquaredEuclideanDistance for sparse vectors would require custom implementation.
impl ScalarSparseSupportedDistance for DotProduct {
    #[inline]
    fn compute_sparse<C: ComponentType, V: ValueType + Float>(
        dense_query: &Option<DenseVector1D<f32, impl AsRef<[f32]>>>,
        query: &SparseVector1D<C, f32, impl AsRef<[C]>, impl AsRef<[f32]>>,
        vector_sparse: &SparseVector1D<C, V, impl AsRef<[C]>, impl AsRef<[V]>>,
    ) -> Self {
        if dense_query.is_none() {
            // SAFETY: Query evaluator construction ensures query components are sorted.
            // Dataset vectors are assumed to be correctly formed.
            unsafe { dot_product_sparse_with_merge_unchecked(query, vector_sparse) }
        } else {
            // SAFETY: dense_query is built with sufficient dimension in query evaluator.
            // Dataset vector components are assumed to be within bounds.
            unsafe {
                dot_product_dense_sparse_unchecked(dense_query.as_ref().unwrap(), vector_sparse)
            }
        }
    }
}

/// A scalar sparse quantizer that converts values from one float type to another.
/// Component type (C) is the same for input and output.
///
/// - `C`: Component type (input = output)
/// - `InValue`: Input value type (to be encoded, must be ValueType + Float)
/// - `OutValue`: Output value type (quantized, must be ValueType + Float)
/// - `D`: Distance type (must implement ScalarSparseSupportedDistance)
///
/// Query value type is always f32.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarSparseQuantizer<C, InValue, OutValue, D> {
    dim: usize,
    _phantom: PhantomData<(C, InValue, OutValue, D)>,
}

impl<C, InValue, OutValue, D> sealed::Sealed for ScalarSparseQuantizer<C, InValue, OutValue, D> {}

/// Scalar quantizer where input and output value types are the same.
pub type ScalarSparseQuantizerSame<C, V, D> = ScalarSparseQuantizer<C, V, V, D>;

/// Plain quantizer where input and output value types are the same.
pub type PlainSparseQuantizer<C, V, D> = ScalarSparseQuantizer<C, V, V, D>;

/// Convenience aliases for common configurations (currently only DotProduct is supported)
pub type PlainSparseQuantizerDotProduct<C, V> = PlainSparseQuantizer<C, V, DotProduct>;
pub type ScalarSparseQuantizerDotProduct<C, V> = ScalarSparseQuantizer<C, V, V, DotProduct>;

impl<C, InValue, OutValue, D> SparseVectorEncoder
    for ScalarSparseQuantizer<C, InValue, OutValue, D>
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

    #[inline]
    fn encoded_from_slices<'a>(
        &self,
        components: &'a [C],
        values: &'a [OutValue],
    ) -> Self::EncodedVectorType<'a>
    where
        Self::EncodedVectorType<'a>: Vector1D<Component = C, Value = OutValue>,
    {
        SparseVector1D::new(components, values)
    }
}

impl<C, InValue, D> QueryFromEncoded for ScalarSparseQuantizer<C, InValue, f32, D>
where
    C: ComponentType,
    InValue: ValueType + Float,
    D: ScalarSparseSupportedDistance,
    D: 'static,
{
    fn query_from_encoded<'a, V>(
        &self,
        encoded: &'a V,
    ) -> Self::QueryVectorType<'a>
    where
        V: Vector1D<Component = C, Value = f32> + ?Sized,
    {
        SparseVector1D::new(encoded.components_as_slice(), encoded.values_as_slice())
    }
}

impl<C, InValue, OutValue, D> VectorEncoder for ScalarSparseQuantizer<C, InValue, OutValue, D>
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
    type InputVectorType<'a> = SparseVector1D<C, InValue, &'a [C], &'a [InValue]>
    where
        Self: 'a;

    type EncodedVectorType<'a> = SparseVector1D<C, OutValue, &'a [C], &'a [OutValue]>
    ;

    type QueryVectorType<'a> = SparseVector1D<C, f32, &'a [C], &'a [f32]>
    where
        Self: 'a;

    type OutputValueType = OutValue;
    type OutputComponentType = C;

    type Evaluator<'a>
        = ScalarSparseQueryEvaluator<'a, C, OutValue, D>
    where
        Self: 'a;

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

    fn query_evaluator<'a>(
        &'a self,
        query: Self::QueryVectorType<'a>,
    ) -> Self::Evaluator<'a> {
        ScalarSparseQueryEvaluator::new(query, self)
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
#[derive(Debug, Clone)]
pub struct ScalarSparseQueryEvaluator<'a, C, OutValue, D>
where
    C: ComponentType,
    OutValue: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
{
    dense_query: Option<DenseVector1D<f32, Vec<f32>>>,
    query: SparseVector1D<C, f32, &'a [C], &'a [f32]>,
    _phantom: PhantomData<(OutValue, D)>,
}

impl<'a, C, OutValue, D> ScalarSparseQueryEvaluator<'a, C, OutValue, D>
where
    C: ComponentType,
    OutValue: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
{
    pub fn new<InValue>(
        query: SparseVector1D<C, f32, &'a [C], &'a [f32]>,
        quantizer: &ScalarSparseQuantizer<C, InValue, OutValue, D>,
    ) -> Self
    where
        InValue: ValueType + Float,
    {
        let max_c = query
            .components_as_slice()
            .iter()
            .map(|c| c.as_())
            .max()
            .unwrap_or(0);

        assert!(
            max_c < quantizer.input_dim(),
            "Query vector component exceeds quantizer input dimension."
        );

        assert_eq!(
            query.components_as_slice().len(),
            query.values_as_slice().len(),
            "Query vector components and values length mismatch."
        );

        let dense_query = if quantizer.input_dim() < 2_usize.pow(20) {
            // For small dimensions, create a dense representation
            let mut dense_query = vec![0.0; quantizer.input_dim()];
            for (&i, &v) in query
                .components_as_slice()
                .iter()
                .zip(query.values_as_slice().iter())
            {
                dense_query[i.as_()] = v;
            }
            Some(DenseVector1D::new(dense_query))
        } else {
            // For large dimensions, keep sparse representation and do merge based dot product computation.
            assert!(
                is_strictly_sorted(query.components_as_slice()),
                "Query components must be sorted in strictly ascending order."
            );
            None
        };

        Self {
            dense_query,
            query,
            _phantom: PhantomData,
        }
    }
}

impl<'a, C, OutValue, D> QueryEvaluator<SparseVector1D<C, OutValue, &'a [C], &'a [OutValue]>, D>
    for ScalarSparseQueryEvaluator<'a, C, OutValue, D>
where
    C: ComponentType,
    OutValue: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
{
    #[inline]
    fn compute_distance(&self, vector: SparseVector1D<C, OutValue, &'a [C], &'a [OutValue]>) -> D {
        D::compute_sparse(&self.dense_query, &self.query, &vector)
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
    fn space_usage_bytes(&self) -> usize {
        self.dim.space_usage_bytes()
    }
}
