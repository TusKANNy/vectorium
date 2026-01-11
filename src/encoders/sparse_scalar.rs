use serde::{Deserialize, Serialize};
use std::any::TypeId;
use std::marker::PhantomData;

use crate::core::vector::SparseVectorView;
use crate::core::vector_encoder::{
    DenseVectorOwned, QueryEvaluator, SparseVectorEncoder, VectorEncoder,
};
use crate::distances::{
    Distance, DotProduct, dot_product_dense_sparse_unchecked,
    dot_product_sparse_with_merge_unchecked,
};
use crate::utils::is_strictly_sorted;
use crate::{ComponentType, Float, FromF32, SpaceUsage, ValueType};

/// Marker trait for distance types supported by scalar sparse quantizers.
/// Provides the computation method specific to sparse vectors.
pub trait ScalarSparseSupportedDistance: Distance {
    /// Compute distance between a dense query and a sparse encoded vector
    fn compute_sparse<C: ComponentType, Q: ValueType, V: ValueType + Float>(
        dense_query: &Option<DenseVectorOwned<Q>>,
        query: SparseVectorView<'_, C, Q>,
        vector_sparse: SparseVectorView<'_, C, V>,
    ) -> Self;
}

// For now, only DotProduct is implemented.
// SquaredEuclideanDistance for sparse vectors would require custom implementation.
impl ScalarSparseSupportedDistance for DotProduct {
    #[inline]
    fn compute_sparse<C: ComponentType, Q: ValueType, V: ValueType + Float>(
        dense_query: &Option<DenseVectorOwned<Q>>,
        query: SparseVectorView<'_, C, Q>,
        vector_sparse: SparseVectorView<'_, C, V>,
    ) -> Self {
        // If query is dense (for small dimensions), use dense-sparse dot product
        if let Some(dense_q) = dense_query {
            // Need a view of dense_q
            let dense_view = dense_q.as_view();

            unsafe { dot_product_dense_sparse_unchecked(dense_view, vector_sparse) }
        } else {
            // Otherwise use sparse-sparse dot product (merge sort style)
            // Assumes both are strictly sorted
            unsafe { dot_product_sparse_with_merge_unchecked(query, vector_sparse) }
        }
    }
}

/// Scalar Quantizer for Sparse Vectors.
///
/// Quantizes values component-wise using a conversion function (e.g. `f32` -> `u8`).
/// Does not compress indices/components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarSparseQuantizer<C, InValue, OutValue, D> {
    dim: usize,
    _phantom: PhantomData<(C, InValue, OutValue, D)>,
}

pub type PlainSparseQuantizer<C, V, D> = ScalarSparseQuantizer<C, V, V, D>;

/// Convenience aliases for common configurations (currently only DotProduct is supported)
pub type PlainSparseQuantizerDotProduct<C, V> = PlainSparseQuantizer<C, V, DotProduct>;
pub type ScalarSparseQuantizerDotProduct<C, V> = ScalarSparseQuantizer<C, V, V, DotProduct>;

impl<C, InValue, OutValue, D> ScalarSparseQuantizer<C, InValue, OutValue, D> {
    #[inline]
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        assert_eq!(
            input_dim, output_dim,
            "ScalarSparseQuantizer requires input_dim == output_dim"
        );

        Self {
            dim: input_dim,
            _phantom: PhantomData,
        }
    }
}

impl<C, InValue, OutValue, D> SparseVectorEncoder for ScalarSparseQuantizer<C, InValue, OutValue, D>
where
    C: ComponentType,
    InValue: ValueType + Float,
    OutValue: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
{
    type InputComponentType = C;
    type InputValueType = InValue;
    type OutputComponentType = C;
    type OutputValueType = OutValue;

    fn push_encoded<'a, ComponentContainer, ValueContainer>(
        &self,
        input: SparseVectorView<'a, C, InValue>,
        components: &mut ComponentContainer,
        values: &mut ValueContainer,
    ) where
        ComponentContainer: Extend<Self::OutputComponentType>,
        ValueContainer: Extend<Self::OutputValueType>,
    {
        components.extend(input.components().iter().cloned());
        values.extend(input.values().iter().map(|&in_val| {
            let f32_val = in_val.to_f32().unwrap();
            OutValue::from_f32_saturating(f32_val)
        }));
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
    type InputVector<'a> = SparseVectorView<'a, C, InValue>;
    type QueryVector<'q, V>
        = SparseVectorView<'q, C, V>
    where
        V: ValueType;
    type EncodedVector<'a> = SparseVectorView<'a, C, OutValue>;

    type Evaluator<'e, 'q, V>
        = ScalarSparseQueryEvaluator<'e, 'q, C, OutValue, D, V>
    where
        V: ValueType,
        Self: 'e;

    fn query_evaluator<'e, 'q, V>(
        &'e self,
        query: Self::QueryVector<'q, V>,
    ) -> Self::Evaluator<'e, 'q, V>
    where
        V: ValueType,
    {
        ScalarSparseQueryEvaluator::new::<InValue>(query, self)
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
#[derive(Debug, Clone)]
pub struct ScalarSparseQueryEvaluator<'e, 'q, C, OutValue, D, V>
where
    C: ComponentType,
    OutValue: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
    V: ValueType,
{
    dense_query: Option<DenseVectorOwned<f32>>,
    query_values: Option<QueryValues<'q>>,
    query_components: Option<&'q [C]>,
    _phantom: PhantomData<(&'e (), OutValue, D, V)>,
}

#[derive(Debug, Clone)]
enum QueryValues<'q> {
    Borrowed(&'q [f32]),
    Owned(Vec<f32>),
}

impl<'q> QueryValues<'q> {
    #[inline]
    fn as_slice(&self) -> &[f32] {
        match self {
            Self::Borrowed(slice) => slice,
            Self::Owned(values) => values,
        }
    }
}

impl<'e, 'q, C, OutValue, D, V> ScalarSparseQueryEvaluator<'e, 'q, C, OutValue, D, V>
where
    C: ComponentType,
    OutValue: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
    V: ValueType,
{
    #[inline]
    fn values_as_f32_slice(values: &'q [V]) -> Option<&'q [f32]> {
        if TypeId::of::<V>() != TypeId::of::<f32>() {
            return None;
        }

        // SAFETY: TypeId check ensures `V == f32`, so the slice layout matches.
        Some(unsafe { &*(values as *const [V] as *const [f32]) })
    }

    pub fn new<InValue>(
        query: SparseVectorView<'q, C, V>,
        quantizer: &'e ScalarSparseQuantizer<C, InValue, OutValue, D>,
    ) -> Self
    where
        InValue: ValueType + Float,
    {
        let max_c = query
            .components()
            .iter()
            .map(|c| c.as_())
            .max()
            .unwrap_or(0);

        assert!(
            max_c < quantizer.input_dim(),
            "Query vector component exceeds quantizer input dimension."
        );

        assert_eq!(
            query.components().len(),
            query.values().len(),
            "Query vector components and values length mismatch."
        );

        let small_dim = quantizer.input_dim() < 2_usize.pow(20);

        // Build a dense query only for small dimensionalities.
        // For large dimensionalities we keep a sparse representation and use merge-based computation.
        let (dense_query, query_values, query_components) = if small_dim {
            let values_f32: QueryValues<'q> = if let Some(values) = Self::values_as_f32_slice(query.values()) {
                QueryValues::Borrowed(values)
            } else {
                QueryValues::Owned(
                    query
                        .values()
                        .iter()
                        .map(|v| v.to_f32().expect("Failed to convert value to f32"))
                        .collect(),
                )
            };

            let mut dense_query_vec = vec![0.0f32; quantizer.input_dim()];
            for (&i, &v) in query.components().iter().zip(values_f32.as_slice().iter()) {
                dense_query_vec[i.as_()] = v;
            }

            (Some(DenseVectorOwned::new(dense_query_vec)), None, None)
        } else {
            assert!(
                is_strictly_sorted(query.components()),
                "Query components must be sorted in strictly ascending order."
            );

            let query_values = if let Some(values) = Self::values_as_f32_slice(query.values()) {
                QueryValues::Borrowed(values)
            } else {
                QueryValues::Owned(
                    query
                        .values()
                        .iter()
                        .map(|v| v.to_f32().expect("Failed to convert value to f32"))
                        .collect(),
                )
            };

            (None, Some(query_values), Some(query.components()))
        };

        Self {
            dense_query,
            query_values,
            query_components,
            _phantom: PhantomData,
        }
    }
}

impl<'e, 'q, 'v, C, OutValue, D, V> QueryEvaluator<SparseVectorView<'v, C, OutValue>>
    for ScalarSparseQueryEvaluator<'e, 'q, C, OutValue, D, V>
where
    C: ComponentType,
    OutValue: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
    V: ValueType,
{
    type Distance = D;

    #[inline]
    fn compute_distance(&self, vector: SparseVectorView<'v, C, OutValue>) -> D {
        debug_assert!(self.dense_query.is_some() || self.query_values.is_some());

        // If we have a dense query, the distance implementation should prefer it.
        // Still, we must pass a well-formed sparse view for the API.
        let (query_components, query_values): (&[C], &[f32]) =
            match (&self.query_components, &self.query_values) {
                (Some(components), Some(values)) => (*components, values.as_slice()),
                _ => (&[] as &[C], &[] as &[f32]),
            };

        let query_view = SparseVectorView::new(query_components, query_values);
        D::compute_sparse(&self.dense_query, query_view, vector)
    }
}

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
