use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::core::vector::SparseVectorView;
use crate::core::vector_encoder::{
    DenseVectorOwned, QueryEvaluator, SparseDataEncoder, SparseVectorEncoder, SparseVectorOwned,
    VectorEncoder,
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

impl<C, InValue, OutValue, D> SparseDataEncoder for ScalarSparseQuantizer<C, InValue, OutValue, D>
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

    #[inline]
    fn decode_vector<'a>(
        &self,
        encoded: SparseVectorView<'a, Self::OutputComponentType, Self::OutputValueType>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let components = encoded.components().to_vec();
        let values = encoded
            .values()
            .iter()
            .map(|&v| v.to_f32().expect("Failed to convert value to f32"))
            .collect();
        SparseVectorOwned::new(components, values)
    }
}

impl<C, InValue, OutValue, D> SparseVectorEncoder for ScalarSparseQuantizer<C, InValue, OutValue, D>
where
    C: ComponentType,
    InValue: ValueType + Float,
    OutValue: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
{
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
    type QueryVector<'q> = SparseVectorView<'q, C, f32>;
    type EncodedVector<'a> = SparseVectorView<'a, C, OutValue>;

    type Evaluator<'e>
        = ScalarSparseQueryEvaluator<'e, C, OutValue, D>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        ScalarSparseQueryEvaluator::new(query, self)
    }

    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        ScalarSparseQueryEvaluator::new_from_owned_query(decoded, self)
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
pub struct ScalarSparseQueryEvaluator<'e, C, OutValue, D>
where
    C: ComponentType,
    OutValue: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
{
    dense_query: Option<DenseVectorOwned<f32>>,
    sparse_query: Option<SparseVectorOwned<C, f32>>,
    _phantom: PhantomData<(&'e (), OutValue, D)>,
}

impl<'e, C, OutValue, D> ScalarSparseQueryEvaluator<'e, C, OutValue, D>
where
    C: ComponentType,
    OutValue: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
{
    #[inline]
    pub fn new_from_owned_query<InValue>(
        query: SparseVectorOwned<C, f32>,
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

        let small_dim = quantizer.input_dim() < 2_usize.pow(20);

        if small_dim {
            let mut dense_query_vec = vec![0.0f32; quantizer.input_dim()];
            for (&i, &v) in query.components().iter().zip(query.values().iter()) {
                dense_query_vec[i.as_()] = v;
            }

            Self {
                dense_query: Some(DenseVectorOwned::new(dense_query_vec)),
                sparse_query: None,
                _phantom: PhantomData,
            }
        } else {
            assert!(
                is_strictly_sorted(query.components()),
                "Query components must be sorted in strictly ascending order."
            );

            Self {
                dense_query: None,
                sparse_query: Some(query),
                _phantom: PhantomData,
            }
        }
    }

    pub fn new<InValue>(
        query: SparseVectorView<'_, C, f32>,
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
        let (dense_query, sparse_query) = if small_dim {
            let values_f32 = query.values();

            let mut dense_query_vec = vec![0.0f32; quantizer.input_dim()];
            for (&i, &v) in query.components().iter().zip(values_f32.iter()) {
                dense_query_vec[i.as_()] = v;
            }

            (Some(DenseVectorOwned::new(dense_query_vec)), None)
        } else {
            assert!(
                is_strictly_sorted(query.components()),
                "Query components must be sorted in strictly ascending order."
            );

            (
                None,
                Some(SparseVectorOwned::new(
                    query.components().to_vec(),
                    query.values().to_vec(),
                )),
            )
        };

        Self {
            dense_query,
            sparse_query,
            _phantom: PhantomData,
        }
    }
}

impl<'e, 'v, C, OutValue, D> QueryEvaluator<SparseVectorView<'v, C, OutValue>>
    for ScalarSparseQueryEvaluator<'e, C, OutValue, D>
where
    C: ComponentType,
    OutValue: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
{
    type Distance = D;

    #[inline]
    fn compute_distance(&self, vector: SparseVectorView<'v, C, OutValue>) -> D {
        debug_assert!(self.dense_query.is_some() || self.sparse_query.is_some());

        // If we have a dense query, the distance implementation should prefer it.
        // Still, we must pass a well-formed sparse view for the API.
        let (query_components, query_values): (&[C], &[f32]) = match &self.sparse_query {
            Some(sparse) => (sparse.components(), sparse.values()),
            None => (&[] as &[C], &[] as &[f32]),
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
