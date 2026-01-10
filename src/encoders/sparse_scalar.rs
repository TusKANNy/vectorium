use serde::{Deserialize, Serialize};
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
    query_values: Vec<f32>,
    query_components: &'q [C],
    _phantom: PhantomData<(&'e (), OutValue, D, V)>,
}

impl<'e, 'q, C, OutValue, D, V> ScalarSparseQueryEvaluator<'e, 'q, C, OutValue, D, V>
where
    C: ComponentType,
    OutValue: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
    V: ValueType,
{
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

        let dense_query = if quantizer.input_dim() < 2_usize.pow(20) {
            // For small dimensions, create a dense representation
            let mut dense_query_vec = vec![0.0f32; quantizer.input_dim()];
            for (&i, &v) in query.components().iter().zip(query.values().iter()) {
                dense_query_vec[i.as_()] = v.to_f32().expect("Failed to convert value to f32");
            }
            Some(DenseVectorOwned::new(dense_query_vec))
        } else {
            assert!(
                is_strictly_sorted(query.components()),
                "Query components must be sorted in strictly ascending order."
            );
            None
        };

        let query_values: Vec<f32> = query
            .values()
            .iter()
            .map(|v| v.to_f32().expect("Failed to convert value to f32"))
            .collect();

        Self {
            dense_query,
            query_values,
            query_components: query.components(),
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
    fn compute_distance(&mut self, vector: SparseVectorView<'v, C, OutValue>) -> D {
        let query_view = SparseVectorView::new(self.query_components, &self.query_values);
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
