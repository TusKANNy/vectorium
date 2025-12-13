use crate::quantizers::Quantizer;
use crate::{SpaceUsage, Vector1D};

pub mod dense_dataset;

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

    // fn search<'a, H: OnlineTopKSelector>(
    //     &self,
    //     query: <Q::Evaluator<'a> as QueryEvaluator<'a>>::QueryType>,
    //     heap: &mut H,
    // ) -> Vec<(f32, usize)>
    // where
    //     Q::InputItem: Float + EuclideanDistance<Q::InputItem> + DotProduct<Q::InputItem>;
}

pub trait GrowableDataset<Q: Quantizer>: Dataset<Q> {
    fn new(quantizer: Q, d: usize) -> Self;
    fn push(
        &mut self,
        vec: impl Vector1D<ComponentType = Q::InputComponentType, ValueType = Q::InputValueType>,
    );
}
