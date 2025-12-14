use crate::quantizers::{Quantizer, QueryEvaluator};
use crate::{Distance, SpaceUsage, Vector1D};

use std::collections::BinaryHeap;

pub mod dense_dataset;

#[derive(Debug, PartialOrd, Eq, Ord, PartialEq, Copy, Clone)]
pub struct Result<D: Distance> {
    pub distance: D,
    pub id: usize,
}

pub trait Dataset<Q>: SpaceUsage
where
    Q: Quantizer,
{
    fn quantizer(&self) -> &Q;

    fn shape(&self) -> (usize, usize);

    fn dim(&self) -> usize;

    fn len(&self) -> usize;

    // fn get_space_usage_bytes(&self) -> usize;

    #[inline]
    #[must_use]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn nnz(&self) -> usize;

    // fn data<'a>(&'a self) -> Self::Vector1DType<'a>;

    fn get(
        &self,
        index: usize,
    ) -> impl Vector1D<ComponentType = Q::OutputComponentType, ValueType = Q::OutputValueType>;

    // fn compute_distance_by_id(&self, idx1: usize, idx2: usize) -> impl Distance;

    fn iter(
        &self,
    ) -> impl Iterator<
        Item = impl Vector1D<ComponentType = Q::OutputComponentType, ValueType = Q::OutputValueType>,
    >;

    #[inline]
    fn search(
        &self,
        query: impl Vector1D<ComponentType = Q::QueryComponentType, ValueType = Q::QueryValueType>,
        k: usize,
    ) -> Vec<Result<<Q as Quantizer>::Distance>> {
        let evaluator = self.quantizer().get_query_evaluator(query);

        if k == 0 {
            return Vec::new();
        }

        // Use a min-heap (via Reverse) to track top-k: root is the worst candidate
        let mut heap: BinaryHeap<Result<<Q as Quantizer>::Distance>> = BinaryHeap::with_capacity(k);

        for (id, vector) in self.iter().enumerate() {
            let distance = evaluator.compute_distance(vector);
            let result = Result { distance, id };

            if heap.len() < k {
                heap.push(result);
            } else if result < *heap.peek().unwrap() {
                heap.pop();
                heap.push(result);
            }
        }

        // Convert min-heap to sorted vec
        let mut results: Vec<_> = heap.into_vec().into_iter().map(|r| r).collect();
        results.sort();
        results
    }
}

pub trait GrowableDataset<Q>: Dataset<Q>
where
    Q: Quantizer,
{
    fn new(quantizer: Q, d: usize) -> Self;
    fn push(
        &mut self,
        vec: impl Vector1D<ComponentType = Q::InputComponentType, ValueType = Q::InputValueType>,
    );
}
