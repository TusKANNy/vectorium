use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::Dataset;
use crate::VectorId;
use crate::core::dataset::ScoredVector;
use crate::core::index::Index;
use crate::core::vector_encoder::{QueryEvaluator, VectorEncoder};

/// Brute-force index that wraps a dataset and scores every vector exhaustively.
///
/// Useful for small collections or as a correctness baseline for approximate indexes.
#[derive(Serialize, Deserialize)]
pub struct FlatIndex<D: Dataset> {
    dataset: D,
}

impl<D: Dataset> FlatIndex<D> {
    pub fn dataset(&self) -> &D {
        &self.dataset
    }

    pub fn into_dataset(self) -> D {
        self.dataset
    }

    /// Returns the single nearest neighbor, or `None` if the dataset is empty.
    pub fn search_nearest<'q>(
        &self,
        query: <D::Encoder as VectorEncoder>::QueryVector<'q>,
    ) -> Option<ScoredVector<<D::Encoder as VectorEncoder>::Distance>> {
        let evaluator = self.dataset.encoder().query_evaluator(query);
        self.dataset
            .iter()
            .enumerate()
            .map(|(i, vector)| ScoredVector {
                distance: evaluator.compute_distance(vector),
                vector: i as VectorId,
            })
            .min_by_key(|s| s.distance)
    }
}

impl<D: Dataset> From<D> for FlatIndex<D> {
    fn from(dataset: D) -> Self {
        FlatIndex { dataset }
    }
}

impl<D: Dataset> crate::core::index::IndexStats for FlatIndex<D> {
    fn n_elements(&self) -> usize {
        self.dataset.len()
    }

    fn dim(&self) -> usize {
        self.dataset.input_dim()
    }
}

impl<D: Dataset> FlatIndex<D> {
    /// Space usage in bytes (a flat index adds nothing beyond its dataset).
    pub fn print_space_usage_bytes(&self) {
        println!("FlatIndex: no overhead beyond the underlying dataset");
    }
}

impl<D: Dataset> Index for FlatIndex<D>
where
    <D::Encoder as VectorEncoder>::Distance: crate::distances::Distance,
{
    type Query<'q> = <D::Encoder as VectorEncoder>::QueryVector<'q>;
    type Distance = <D::Encoder as VectorEncoder>::Distance;
    type SearchParams = ();

    fn search<'q>(
        &self,
        query: Self::Query<'q>,
        k: usize,
        _: &(),
    ) -> Vec<ScoredVector<Self::Distance>> {
        if k == 0 {
            return Vec::new();
        }

        let evaluator = self.dataset.encoder().query_evaluator(query);

        self.dataset
            .iter()
            .enumerate()
            .map(|(i, vector)| ScoredVector {
                distance: evaluator.compute_distance(vector),
                vector: i as VectorId,
            })
            .k_smallest(k)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vector::DenseVectorView;
    use crate::datasets::dense_dataset::DenseDatasetGrowable;
    use crate::distances::DotProduct;
    use crate::{DatasetGrowable, DenseDataset, PlainDenseQuantizer};

    #[test]
    fn flat_index_search_matches_dataset_order() {
        type Enc = PlainDenseQuantizer<f32, DotProduct>;
        let mut growable = DenseDatasetGrowable::new(Enc::new(2));
        growable.push(DenseVectorView::new(&[1.0f32, 0.5]));
        growable.push(DenseVectorView::new(&[0.0f32, 1.0]));
        growable.push(DenseVectorView::new(&[2.0f32, 1.0]));

        let dataset: DenseDataset<Enc> = growable.into();
        let index = FlatIndex::from(dataset);

        let query = DenseVectorView::new(&[1.5f32, 1.0]);
        let results = index.search(query, 2, &());

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].vector, 2);
        assert_eq!(results[1].vector, 0);
    }

    #[test]
    fn flat_index_search_zero_k_returns_empty() {
        type Enc = PlainDenseQuantizer<f32, DotProduct>;
        let mut growable = DenseDatasetGrowable::new(Enc::new(2));
        growable.push(DenseVectorView::new(&[1.0f32, 0.0]));
        let dataset: DenseDataset<Enc> = growable.into();
        let index = FlatIndex::from(dataset);

        assert!(
            index
                .search(DenseVectorView::new(&[1.0f32, 0.0]), 0, &())
                .is_empty()
        );
    }
}
