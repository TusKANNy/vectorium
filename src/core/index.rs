use crate::Dataset;
use crate::core::dataset::ScoredVector;
use crate::core::vector_encoder::VectorEncoder;

/// Generic trait for vector indices that can be built and searched.
pub trait Index<D>
where
    D: Dataset,
{
    /// Type for parameters used during index building.
    type BuildParams;

    /// Type for parameters used during search.
    type SearchParams;

    /// Returns the number of vectors in the index.
    fn n_vectors(&self) -> usize;

    /// Returns the dimensionality of the vectors in the index.
    fn dim(&self) -> usize;

    /// Prints the space usage of the index in bytes,
    /// including the dataset and the index structure.
    fn print_space_usage_bytes(&self);

    /// Builds an index from an already-encoded dataset.
    fn build_index(dataset: D, build_params: &Self::BuildParams) -> Self;

    /// Searches the index with the given query vector.
    fn search<'q>(
        &'q self,
        query: <D::Encoder as VectorEncoder>::QueryVector<'q>,
        k: usize,
        search_params: &Self::SearchParams,
    ) -> Vec<ScoredVector<<D::Encoder as VectorEncoder>::Distance>>;
}
