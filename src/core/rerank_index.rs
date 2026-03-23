//! An implementation of a two-stage retrieval index.
//!
//! ## Architecture
//!
//! The two-stage approach combines a first-stage index, that usually employs lighter vector
//! representations for fast candidates retrieval, with a usually heavier/slower but higher-quality
//! representation for candidates re-scoring and final selection.
//!
//! 1. **First Stage**: Uses an ANN index (e.g., HNSW from kANNolo) to retrieve k_candidates
//!    from a potentially lower-precision or quantized representation.
//!
//! 2. **Second Stage**: Reranks the retrieved candidates using a higher-quality dataset
//!    (e.g., full-precision vectors or multi-vector representations) to compute exact scores
//!    and return the top k_final results.

use super::dataset::ScoredVector;
use super::distances::Distance;
use super::index::Index;
use super::vector_encoder::VectorEncoder;
use crate::Dataset;
use crate::QueryEvaluator;
use crate::VectorId;
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::fmt;
use std::marker::PhantomData;

/// A reranking index that combines a first-stage index for candidate retrieval
/// with a higher-quality dataset for reranking.
///
/// # Type Parameters
/// - `FirstStageIndex`: The index type (e.g., HNSW from kANNolo) that implements Index
/// - `FirstStageDataset`: The dataset type used by the first-stage index
/// - `RerankDataset`: The dataset type used for reranking (e.g., MultiVectorDataset, DenseDataset)
///
/// # Constraints
/// Both the first-stage index and rerank dataset must contain the same number of items
/// with the same vector IDs (same semantic content, different representations).
#[derive(Serialize, Deserialize)]
pub struct RerankIndex<FirstStageIndex, FirstStageDataset, RerankDataset>
where
    FirstStageIndex: Index<FirstStageDataset>,
    FirstStageDataset: Dataset,
    RerankDataset: Dataset,
{
    first_stage_index: FirstStageIndex,
    rerank_dataset: RerankDataset,
    _phantom_first_stage: PhantomData<FirstStageDataset>,
}

impl<FirstStageIndex, FirstStageDataset, RerankDataset>
    RerankIndex<FirstStageIndex, FirstStageDataset, RerankDataset>
where
    FirstStageIndex: Index<FirstStageDataset>,
    FirstStageDataset: Dataset,
    RerankDataset: Dataset,
{
    /// Creates a new rerank index from a first-stage index and a rerank dataset.
    ///
    /// # Arguments
    /// - `first_stage_index`: The first-stage index for candidate retrieval
    /// - `rerank_dataset`: The dataset for reranking candidates
    ///
    /// # Note
    /// The caller should ensure that both the first-stage index and rerank dataset
    /// have the same number of items with matching vector IDs.
    pub fn new(first_stage_index: FirstStageIndex, rerank_dataset: RerankDataset) -> Self {
        Self {
            first_stage_index,
            rerank_dataset,
            _phantom_first_stage: PhantomData,
        }
    }

    /// Performs a complete two-stage search: candidate retrieval followed by reranking.
    ///
    /// This is the main search method that:
    /// 1. Retrieves k_candidates from the first-stage index
    /// 2. Reranks all candidates using the rerank dataset  
    /// 3. Returns the top k_final results along with timing information
    ///
    /// # Arguments
    /// - `first_stage_query`: Query vector for the first-stage index
    /// - `rerank_query`: Query vector for reranking
    /// - `k_candidates`: Number of candidates to retrieve from first-stage index
    /// - `k_final`: Number of final results to return after reranking
    /// - `first_stage_search_params`: Search parameters for the first-stage index
    /// - `alpha`: Optional Candidates Pruning (CP) threshold parameter (None for no pruning)
    /// - `beta`: Optional early-exit (EE) parameter for reranking (None for no early exit)
    ///
    /// # Returns
    /// - results: Vector of scored vectors, sorted by distance using the Distance type's Ord implementation
    pub fn search<'q>(
        &'q self,
        first_stage_query: <FirstStageDataset::Encoder as VectorEncoder>::QueryVector<'q>,
        rerank_query: <RerankDataset::Encoder as VectorEncoder>::QueryVector<'q>,
        k_candidates: usize,
        k_final: usize,
        first_stage_search_params: &FirstStageIndex::SearchParams,
        alpha: Option<f32>,
        beta: Option<usize>,
    ) -> Vec<ScoredVector<<RerankDataset::Encoder as VectorEncoder>::Distance>>
    where
        <FirstStageDataset::Encoder as VectorEncoder>::Distance: Distance,
        <RerankDataset::Encoder as VectorEncoder>::Distance: Distance,
    {
        // Stage 1: Get candidates from first stage index using its own search method
        let first_stage_results = self.first_stage_index.search(
            first_stage_query,
            k_candidates,
            first_stage_search_params,
        );

        // Extract candidate IDs from first-stage search results
        let mut candidates: Vec<VectorId> = first_stage_results
            .iter()
            .map(|result| result.vector)
            .collect();

        // Apply alpha-based filtering if provided (based on first-stage distance scores)
        if let Some(alpha_val) = alpha
            && !first_stage_results.is_empty()
            && k_final <= first_stage_results.len()
        {
            let threshold =
                first_stage_results[k_final - 1].distance.distance() * (1.0 - alpha_val);
            candidates.retain(|&id| {
                if let Some(scored_vec) = first_stage_results.iter().find(|r| r.vector == id) {
                    scored_vec.distance.distance() >= threshold
                } else {
                    false
                }
            });
        }

        // Stage 2: Rerank candidates using the rerank dataset
        if let Some(beta_val) = beta {
            self.rerank_candidates_with_early_exit(rerank_query, &candidates, k_final, beta_val)
        } else {
            self.rerank_candidates(rerank_query, &candidates, k_final)
        }
    }

    /// Reranks a given list of candidates using the rerank dataset.
    ///
    /// This method takes candidate IDs (typically from a graph index search) and
    /// reranks them using the high-quality rerank dataset.
    ///
    /// # Arguments
    /// - `rerank_query`: Query vector for reranking
    /// - `candidates`: List of vector IDs from initial search
    /// - `k_final`: Number of final results to return
    ///
    /// # Returns
    /// Vector of scored vectors, sorted by distance using the Distance type's Ord implementation
    fn rerank_candidates<'q>(
        &'q self,
        rerank_query: <RerankDataset::Encoder as VectorEncoder>::QueryVector<'q>,
        candidates: &[VectorId],
        k_final: usize,
    ) -> Vec<ScoredVector<<RerankDataset::Encoder as VectorEncoder>::Distance>>
    where
        <RerankDataset::Encoder as VectorEncoder>::Distance: Distance,
    {
        // Create query evaluator for the rerank dataset
        let encoder = self.rerank_dataset.encoder();
        let query_evaluator = encoder.query_evaluator(rerank_query);

        // Rerank candidates by computing exact distances
        let mut reranked: Vec<ScoredVector<<RerankDataset::Encoder as VectorEncoder>::Distance>> =
            candidates
                .iter()
                .map(|vector_id| {
                    let encoded_vector = self.rerank_dataset.get(*vector_id);
                    let distance = query_evaluator.compute_distance(encoded_vector);
                    ScoredVector {
                        distance,
                        vector: *vector_id,
                    }
                })
                .collect();

        // Sort by distance using the Distance type's Ord implementation and take top k_final
        reranked.sort_by_key(|b| b.distance);
        reranked.truncate(k_final);

        reranked
    }

    fn rerank_candidates_with_early_exit<'q>(
        &'q self,
        rerank_query: <RerankDataset::Encoder as VectorEncoder>::QueryVector<'q>,
        candidates: &[VectorId],
        k_final: usize,
        beta: usize,
    ) -> Vec<ScoredVector<<RerankDataset::Encoder as VectorEncoder>::Distance>>
    where
        <RerankDataset::Encoder as VectorEncoder>::Distance: Distance,
    {
        // Ensure there are enough candidates
        if candidates.len() < k_final {
            return Vec::new();
        }

        let encoder = self.rerank_dataset.encoder();
        let query_evaluator = encoder.query_evaluator(rerank_query);

        // Rerank first k_final candidates
        let first_reranked: Vec<ScoredVector<<RerankDataset::Encoder as VectorEncoder>::Distance>> =
            candidates[..k_final]
                .iter()
                .map(|vector_id| {
                    let encoded_vector = self.rerank_dataset.get(*vector_id);
                    let distance = query_evaluator.compute_distance(encoded_vector);
                    ScoredVector {
                        distance,
                        vector: *vector_id,
                    }
                })
                .collect();

        // Create a max heap to keep track of the top k_final candidates
        let mut heap: BinaryHeap<
            ScoredVector<<RerankDataset::Encoder as VectorEncoder>::Distance>,
        > = BinaryHeap::from(first_reranked);

        let mut n_stalls = 0;
        for vector_id in &candidates[k_final..] {
            let encoded_vector = self.rerank_dataset.get(*vector_id);
            let distance = query_evaluator.compute_distance(encoded_vector);
            let candidate = ScoredVector {
                distance,
                vector: *vector_id,
            };

            if let Some(mut worst) = heap.peek_mut() {
                if candidate.distance < worst.distance {
                    *worst = candidate;
                    n_stalls = 0;
                } else {
                    n_stalls += 1;
                    if n_stalls >= beta {
                        break;
                    }
                }
            }
        }

        // Extract results from heap
        heap.into_sorted_vec()
    }

    /// Returns a reference to the underlying first-stage index.
    pub fn first_stage_index(&self) -> &FirstStageIndex {
        &self.first_stage_index
    }

    /// Returns a reference to the rerank dataset.
    pub fn rerank_dataset(&self) -> &RerankDataset {
        &self.rerank_dataset
    }

    /// Returns the number of items in the index.
    pub fn len(&self) -> usize {
        self.rerank_dataset.len()
    }

    /// Returns true if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.rerank_dataset.is_empty()
    }
}

impl<FirstStageIndex, FirstStageDataset, RerankDataset> fmt::Debug
    for RerankIndex<FirstStageIndex, FirstStageDataset, RerankDataset>
where
    FirstStageIndex: Index<FirstStageDataset> + fmt::Debug,
    FirstStageDataset: Dataset + fmt::Debug,
    RerankDataset: Dataset + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RerankIndex")
            .field("first_stage_index", &self.first_stage_index)
            .field("rerank_dataset", &self.rerank_dataset)
            .finish()
    }
}
