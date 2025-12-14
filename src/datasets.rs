use crate::quantizers::{Quantizer, QueryEvaluator};
use crate::{Distance, SpaceUsage, Vector1D};

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

        let mut results: Vec<Result<<Q as Quantizer>::Distance>> = self
            .iter()
            .enumerate()
            .map(|(id, vector)| Result {
                distance: evaluator.compute_distance(vector),
                id,
            })
            .collect();

        results.sort();
        results.truncate(k);

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
