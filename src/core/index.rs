use crate::core::dataset::ScoredVector;
use crate::distances::Distance;

/// Generic trait for a vector index: searched with a query to return the
/// `k` nearest results.
///
/// This trait says nothing about how an index is built, building is
/// inherently type-directed (different indexes need different inputs and
/// configuration), so each index type exposes its own build entry point
/// (an inherent method or a free function) instead of a common trait method.
///
/// The query ([`Query`](Index::Query)) and the result distance
/// ([`Distance`](Index::Distance)) are the index's own associated types:
/// `search` receives the raw query and handles it however it wants.
pub trait Index {
    /// Raw query accepted by [`search`](Index::search). The query lifetime `'q`
    /// is independent of the index, so it imposes no `Self: 'q` bound.
    type Query<'q>;

    /// Distance/score type carried by each result.
    type Distance: Distance;

    /// Parameters for [`search`](Index::search).
    type SearchParams;

    /// Search the index with `query`, returning up to `k` scored results.
    fn search<'q>(
        &self,
        query: Self::Query<'q>,
        k: usize,
        search_params: &Self::SearchParams,
    ) -> Vec<ScoredVector<Self::Distance>>;
}

/// Optional metadata for indexes that store a fixed collection of vectors.
///
/// Kept separate from [`Index`] because we may have indexes where these
/// values are not known or not meaningful.
pub trait IndexStats {
    /// Number of indexed elements.
    ///
    /// For multivector datasets each element may contain several token vectors;
    /// this returns the element count (e.g. documents), not the token count.
    fn n_elements(&self) -> usize;

    /// Dimensionality of the indexed vectors.
    fn dim(&self) -> usize;
}
