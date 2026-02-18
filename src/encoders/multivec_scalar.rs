//! MaxSim encoder for multivector late-interaction scoring.
//!
//! Each document and query is a variable-length collection of dense token vectors
//! stored flat as `token_dim * num_tokens` values. The encoder computes the MaxSim
//! distance: for each query token, find the maximum dot product with any document
//! token, then sum across all query tokens.
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::core::distances::{Distance, DotProduct, dot_product_dense_unchecked};
use crate::core::vector::DenseVectorView;
use crate::core::vector_encoder::{
    DenseVectorOwned, MultiVecEncoder, QueryEvaluator, VectorEncoder,
};
use crate::{Float, FromF32, SpaceUsage, ValueType};

/// A MaxSim encoder parameterized by input/output value types.
///
/// `token_dim` is the dimensionality of each individual token vector.
/// Input multivectors are flat `DenseVectorView`s of length `token_dim * num_tokens`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarMultiVecQuantizer<In, Out> {
    token_dim: usize,
    _phantom: PhantomData<(In, Out)>,
}

pub type PlainMultiVecQuantizer<V> = ScalarMultiVecQuantizer<V, V>;

impl<In, Out> ScalarMultiVecQuantizer<In, Out> {
    /// Build a MaxSim encoder for token vectors of the given dimensionality.
    #[inline]
    pub fn new(token_dim: usize) -> Self {
        Self {
            token_dim,
            _phantom: PhantomData,
        }
    }

    /// Dimensionality of each individual token vector.
    #[inline]
    pub fn token_dim(&self) -> usize {
        self.token_dim
    }
}

/// Evaluator that computes MaxSim between a frozen query multivector and document multivectors.
#[derive(Debug, Clone)]
pub struct ScalarMultiVecQueryEvaluator<'e, In, Out> {
    encoder: &'e ScalarMultiVecQuantizer<In, Out>,
    query: DenseVectorOwned<f32>,
}

impl<'e, In, Out> ScalarMultiVecQueryEvaluator<'e, In, Out> {
    #[inline]
    pub fn new(
        encoder: &'e ScalarMultiVecQuantizer<In, Out>,
        query: DenseVectorOwned<f32>,
    ) -> Self {
        Self { encoder, query }
    }
}

impl<'e, 'v, In, Out> QueryEvaluator<DenseVectorView<'v, Out>>
    for ScalarMultiVecQueryEvaluator<'e, In, Out>
where
    In: ValueType,
    Out: ValueType + FromF32,
{
    type Distance = DotProduct;

    /// Compute MaxSim: sum over query tokens of max(dot(q_i, d_j) for all doc tokens j).
    #[inline]
    fn compute_distance(&self, vector: DenseVectorView<'v, Out>) -> DotProduct {
        let token_dim = self.encoder.token_dim;

        let total: f32 = self
            .query
            .values()
            .chunks_exact(token_dim)
            .map(|q_token| {
                let q_view = DenseVectorView::new(q_token);
                vector
                    .values()
                    .chunks_exact(token_dim)
                    .map(|d_token| {
                        let d_view = DenseVectorView::new(d_token);
                        unsafe { dot_product_dense_unchecked(q_view, d_view) }.distance()
                    })
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .sum();

        DotProduct::from(total)
    }
}

impl<In, Out> VectorEncoder for ScalarMultiVecQuantizer<In, Out>
where
    In: ValueType + Float,
    Out: ValueType + Float + FromF32,
{
    type Distance = DotProduct;
    type InputVector<'a> = DenseVectorView<'a, In>;
    type QueryVector<'q> = DenseVectorView<'q, f32>;
    type EncodedVector<'a> = DenseVectorView<'a, Out>;

    type Evaluator<'e>
        = ScalarMultiVecQueryEvaluator<'e, In, Out>
    where
        Self: 'e;

    #[inline]
    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        assert!(
            query.len() % self.token_dim == 0,
            "Query length must be a multiple of token_dim ({}), got {}",
            self.token_dim,
            query.len()
        );
        ScalarMultiVecQueryEvaluator::new(self, query.to_owned())
    }

    #[inline]
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded: Vec<f32> = vector
            .values()
            .iter()
            .map(|&v| v.to_f32().expect("Failed to convert value to f32"))
            .collect();
        ScalarMultiVecQueryEvaluator::new(self, DenseVectorOwned::new(decoded))
    }

    fn input_dim(&self) -> usize {
        self.token_dim
    }

    fn output_dim(&self) -> usize {
        self.token_dim
    }
}

impl<In, Out> MultiVecEncoder for ScalarMultiVecQuantizer<In, Out>
where
    In: ValueType + Float,
    Out: ValueType + Float + FromF32,
{
    type InputValueType = In;
    type OutputValueType = Out;

    #[inline]
    fn push_encoded<'a, OutputContainer>(
        &self,
        input: DenseVectorView<'a, In>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<Out>,
    {
        output.extend(
            input
                .values()
                .iter()
                .map(|&v| Out::from_f32_saturating(v.to_f32().expect("value to f32"))),
        );
    }
}

impl<In, Out> SpaceUsage for ScalarMultiVecQuantizer<In, Out>
where
    In: ValueType + Float,
    Out: ValueType + Float + FromF32,
{
    fn space_usage_bytes(&self) -> usize {
        self.token_dim.space_usage_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vector::DenseVectorView;

    #[test]
    fn maxsim_single_token_equals_dot_product() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(3);

        let query = DenseVectorView::new(&[1.0f32, 2.0, 3.0]);
        let evaluator = encoder.query_evaluator(query);

        let doc = DenseVectorView::new(&[4.0f32, 5.0, 6.0]);
        // dot = 1*4 + 2*5 + 3*6 = 32
        assert_eq!(evaluator.compute_distance(doc), DotProduct::from(32.0));
    }

    #[test]
    fn maxsim_multi_query_single_doc() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(2);

        // 2 query tokens: [1, 0] and [0, 1]
        let query = DenseVectorView::new(&[1.0f32, 0.0, 0.0, 1.0]);
        let evaluator = encoder.query_evaluator(query);

        // 1 doc token: [3, 4]
        let doc = DenseVectorView::new(&[3.0f32, 4.0]);
        // q0 max = dot([1,0], [3,4]) = 3
        // q1 max = dot([0,1], [3,4]) = 4
        // MaxSim = 3 + 4 = 7
        assert_eq!(evaluator.compute_distance(doc), DotProduct::from(7.0));
    }

    #[test]
    fn maxsim_multi_query_multi_doc() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(2);

        // 2 query tokens: [1, 0] and [0, 1]
        let query = DenseVectorView::new(&[1.0f32, 0.0, 0.0, 1.0]);
        let evaluator = encoder.query_evaluator(query);

        // 2 doc tokens: [3, 0] and [0, 5]
        let doc = DenseVectorView::new(&[3.0f32, 0.0, 0.0, 5.0]);
        // q0: max(dot([1,0],[3,0]), dot([1,0],[0,5])) = max(3, 0) = 3
        // q1: max(dot([0,1],[3,0]), dot([0,1],[0,5])) = max(0, 5) = 5
        // MaxSim = 3 + 5 = 8
        assert_eq!(evaluator.compute_distance(doc), DotProduct::from(8.0));
    }

    #[test]
    fn maxsim_vector_evaluator_works() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(2);

        // Use a stored doc vector as query via vector_evaluator
        let doc_as_query = DenseVectorView::new(&[1.0f32, 0.0]);
        let evaluator = encoder.vector_evaluator(doc_as_query);

        let doc = DenseVectorView::new(&[2.0f32, 3.0]);
        assert_eq!(evaluator.compute_distance(doc), DotProduct::from(2.0));
    }

    #[test]
    #[should_panic(expected = "Query length must be a multiple of token_dim")]
    fn maxsim_panics_on_misaligned_query() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(3);
        let query = DenseVectorView::new(&[1.0f32, 2.0]); // length 2, not divisible by 3
        encoder.query_evaluator(query);
    }
}
