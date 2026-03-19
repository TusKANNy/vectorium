//! MaxSim encoder for multivector late-interaction scoring.
//!
//! Each document and query is a variable-length collection of dense token vectors
//! represented as a [`DenseMultiVectorView`] with an explicit `dim` and `num_vecs`.
//! The encoder computes the MaxSim distance: for each query token, find the maximum
//! dot product with any document token, then sum across all query tokens.
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::core::distances::{DotProduct, maxsim};
use crate::core::vector::{DenseMultiVectorOwned, DenseMultiVectorView};
use crate::core::vector_encoder::{MultiVecEncoder, QueryEvaluator, VectorEncoder};
use crate::{Float, FromF32, SpaceUsage, ValueType};

/// A MaxSim encoder parameterized by input/output value types.
///
/// `token_dim` is the dimensionality of each individual token vector.
/// Input and encoded multivectors are [`DenseMultiVectorView`]s whose `dim` must equal
/// `token_dim`.
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
    query: DenseMultiVectorOwned<f32>,
}

impl<'e, In, Out> ScalarMultiVecQueryEvaluator<'e, In, Out> {
    #[inline]
    pub fn new(
        encoder: &'e ScalarMultiVecQuantizer<In, Out>,
        query: DenseMultiVectorOwned<f32>,
    ) -> Self {
        Self { encoder, query }
    }
}

impl<'e, 'v, In, Out> QueryEvaluator<DenseMultiVectorView<'v, Out>>
    for ScalarMultiVecQueryEvaluator<'e, In, Out>
where
    In: ValueType,
    Out: ValueType + FromF32,
{
    type Distance = DotProduct;

    /// Compute MaxSim: sum over query tokens of max(dot(q_i, d_j) for all doc tokens j).
    #[inline]
    fn compute_distance(&self, vector: DenseMultiVectorView<'v, Out>) -> DotProduct {
        let _ = self.encoder;
        let mut d_buf = vec![0.0f32; self.query.dim()];
        let mut max_scores = vec![f32::NEG_INFINITY; self.query.num_vecs()];
        DotProduct::from(maxsim(
            self.query.as_view(),
            vector,
            &mut d_buf,
            &mut max_scores,
        ))
    }
}

impl<In, Out> VectorEncoder for ScalarMultiVecQuantizer<In, Out>
where
    In: ValueType + Float,
    Out: ValueType + Float + FromF32,
{
    type Distance = DotProduct;
    type InputVector<'a> = DenseMultiVectorView<'a, In>;
    type QueryVector<'q> = DenseMultiVectorView<'q, f32>;
    type EncodedVector<'a> = DenseMultiVectorView<'a, Out>;

    type Evaluator<'e>
        = ScalarMultiVecQueryEvaluator<'e, In, Out>
    where
        Self: 'e;

    #[inline]
    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        assert_eq!(
            query.dim(),
            self.token_dim,
            "Query dim ({}) must match token_dim ({})",
            query.dim(),
            self.token_dim
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
        ScalarMultiVecQueryEvaluator::new(self, DenseMultiVectorOwned::new(decoded, self.token_dim))
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
        input: DenseMultiVectorView<'a, In>,
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

    /// Decode any encoded multivector into `DenseMultiVectorOwned<f32>` by converting each element to `f32`.
    #[inline]
    fn decode_vector<'a>(
        &self,
        encoded: DenseMultiVectorView<'a, Self::OutputValueType>,
    ) -> DenseMultiVectorOwned<f32> {
        let values = encoded
            .values()
            .iter()
            .map(|&v| v.to_f32().expect("Failed to convert value to f32"))
            .collect();
        DenseMultiVectorOwned::new(values, encoded.dim())
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
    use crate::core::vector::DenseMultiVectorView;

    #[test]
    fn maxsim_single_token_equals_dot_product() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(3);

        // 1 query token of dim 3
        let query = DenseMultiVectorView::new(&[1.0f32, 2.0, 3.0], 3);
        let evaluator = encoder.query_evaluator(query);

        // 1 doc token of dim 3: dot = 1*4 + 2*5 + 3*6 = 32
        let doc = DenseMultiVectorView::new(&[4.0f32, 5.0, 6.0], 3);
        assert_eq!(evaluator.compute_distance(doc), DotProduct::from(32.0));
    }

    #[test]
    fn maxsim_multi_query_single_doc() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(2);

        // 2 query tokens: [1, 0] and [0, 1]
        let query = DenseMultiVectorView::new(&[1.0f32, 0.0, 0.0, 1.0], 2);
        let evaluator = encoder.query_evaluator(query);

        // 1 doc token: [3, 4]
        // q0 max = dot([1,0], [3,4]) = 3
        // q1 max = dot([0,1], [3,4]) = 4  →  MaxSim = 7
        let doc = DenseMultiVectorView::new(&[3.0f32, 4.0], 2);
        assert_eq!(evaluator.compute_distance(doc), DotProduct::from(7.0));
    }

    #[test]
    fn maxsim_multi_query_multi_doc() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(2);

        // 2 query tokens: [1, 0] and [0, 1]
        let query = DenseMultiVectorView::new(&[1.0f32, 0.0, 0.0, 1.0], 2);
        let evaluator = encoder.query_evaluator(query);

        // 2 doc tokens: [3, 0] and [0, 5]
        // q0: max(3, 0) = 3 ;  q1: max(0, 5) = 5  →  MaxSim = 8
        let doc = DenseMultiVectorView::new(&[3.0f32, 0.0, 0.0, 5.0], 2);
        assert_eq!(evaluator.compute_distance(doc), DotProduct::from(8.0));
    }

    #[test]
    fn maxsim_vector_evaluator_works() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(2);

        // Use a stored doc vector as query via vector_evaluator
        let doc_as_query = DenseMultiVectorView::new(&[1.0f32, 0.0], 2);
        let evaluator = encoder.vector_evaluator(doc_as_query);

        let doc = DenseMultiVectorView::new(&[2.0f32, 3.0], 2);
        assert_eq!(evaluator.compute_distance(doc), DotProduct::from(2.0));
    }

    #[test]
    #[should_panic(expected = "Query dim")]
    fn maxsim_panics_on_mismatched_dim() {
        let encoder = PlainMultiVecQuantizer::<f32>::new(3);
        // dim=2 doesn't match token_dim=3
        let query = DenseMultiVectorView::new(&[1.0f32, 2.0], 2);
        encoder.query_evaluator(query);
    }

    #[test]
    #[should_panic]
    fn multivec_view_panics_on_misaligned_slice() {
        // 5 values is not divisible by dim=3
        DenseMultiVectorView::<f32>::new(&[1.0, 2.0, 3.0, 4.0, 5.0], 3);
    }
}
