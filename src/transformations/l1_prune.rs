//! L1-mass pruning: split a sparse vector into the components that carry a% of its
//! $\ell_1$ mass and the exact complement.
//!
//! # What this is for
//!
//! A learned sparse representation spreads its mass very unevenly and a small minority of
//! the components of a document carry almost all of its $\ell_1$ norm. Keeping only those
//! components gives a cheaper collection to quickly build indexes and gather candidates,
//! and the components left are kept for reranking.
//!
//! # The rule
//!
//! For a vector `x` and a percentage `a`:
//!
//! 1. Order the components by **descending `|value|`**, breaking ties on **minimum component
//!    id**.
//! 2. Accumulate `|value|` in that order and keep components until the running sum
//!    **surpasses** `a`% of `sum(|x_i|)`. The component that crosses the threshold is kept.
//! 3. The residual is exactly the complement, so `kept.len() + residual.len() == x.len()` and
//!    no component appears in both.

use crate::core::vector::{SparseVectorOwned, SparseVectorView};
use crate::encoders::sparse_scalar::ScalarSparseSupportedDistance;
use crate::{ComponentType, Dataset, DatasetGrowable, Float, FromF32, SpaceUsage, ValueType};
use crate::{PlainSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer};

/// Scratch buffers reused across vectors, so a whole-dataset split does not allocate per vector.
#[derive(Default)]
struct SplitScratch {
    /// Positions into the input vector, ordered by descending `|value|` then ascending component.
    order: Vec<u32>,
    /// Whether the position at this index was kept.
    keep: Vec<bool>,
}

/// Number of leading entries of `order` that carry the top `percentage`% of the L1 mass.
///
/// Fills `scratch.order` with the positions of `values`, ordered by descending `|value|` and
/// then by ascending component id.
fn select_kept<V: ValueType>(values: &[V], percentage: f32, scratch: &mut SplitScratch) -> usize {
    let n = values.len();
    scratch.order.clear();
    scratch.order.extend(0..n as u32);

    // `total_cmp` rather than `partial_cmp`: it is a total order, so the sort is well defined
    // even if a stored value is NaN, and the result does not depend on the sort's stability.
    let magnitude = |i: u32| values[i as usize].to_f32().unwrap_or(0.0).abs();
    scratch.order.sort_unstable_by(|&i, &j| {
        magnitude(j)
            .total_cmp(&magnitude(i))
            .then_with(|| i.cmp(&j))
    });

    let l1: f32 = scratch
        .order
        .iter()
        .map(|&i| magnitude(i))
        .fold(0.0, |a, b| a + b);

    // `percentage` of zero mass is zero, and nothing can surpass it: keep the whole vector
    // rather than emit an all-empty first stage whose graph has nothing to search.
    if l1 <= 0.0 {
        return n;
    }

    let target = l1 * (percentage / 100.0);
    let mut acc = 0.0f32;
    for (rank, &i) in scratch.order.iter().enumerate() {
        acc += magnitude(i);
        // Surpasses, not reaches: the crossing component is kept.
        if acc > target {
            return rank + 1;
        }
    }
    n
}

/// Splits one sparse vector into its L1-pruned part and the exact complement.
///
/// The kept part holds the components carrying the top `percentage`% of the vector's L1 mass
/// (the exact rule is in the [module documentation](self)), and the residual holds the rest.
/// Both are in strictly ascending component order. The kept part is never empty unless the
/// input is; the residual can be empty.
///
/// # Panics
///
/// Panics unless `0.0 <= percentage <= 100.0`.
///
/// # Examples
///
/// ```
/// use vectorium::transformations::l1_prune::l1_prune_split_vector;
/// use vectorium::SparseVectorView;
///
/// // L1 mass is 10.0; 50% of it is 5.0. Descending |value| is 4.0, 3.0, 2.0, 1.0, and the
/// // running sum surpasses 5.0 at the second component, so two components are kept.
/// let v = SparseVectorView::new(&[0_u32, 1, 2, 3], &[1.0_f32, 4.0, 2.0, 3.0]);
/// let (kept, residual) = l1_prune_split_vector(v, 50.0);
/// assert_eq!(kept.components(), &[1, 3]);
/// assert_eq!(residual.components(), &[0, 2]);
/// ```
pub fn l1_prune_split_vector<C: ComponentType, V: ValueType>(
    vector: SparseVectorView<'_, C, V>,
    percentage: f32,
) -> (SparseVectorOwned<C, V>, SparseVectorOwned<C, V>) {
    let mut scratch = SplitScratch::default();
    let mut kept_c = Vec::new();
    let mut kept_v = Vec::new();
    let mut res_c = Vec::new();
    let mut res_v = Vec::new();
    split_into(
        vector,
        percentage,
        &mut scratch,
        (&mut kept_c, &mut kept_v),
        (&mut res_c, &mut res_v),
    );
    (
        SparseVectorOwned::new(kept_c, kept_v),
        SparseVectorOwned::new(res_c, res_v),
    )
}

/// Splits `vector` into the two component/value buffer pairs, which are cleared first.
fn split_into<C: ComponentType, V: ValueType>(
    vector: SparseVectorView<'_, C, V>,
    percentage: f32,
    scratch: &mut SplitScratch,
    kept: (&mut Vec<C>, &mut Vec<V>),
    residual: (&mut Vec<C>, &mut Vec<V>),
) {
    assert!(
        (0.0..=100.0).contains(&percentage),
        "percentage must be in [0, 100], got {percentage}"
    );

    let components = vector.components();
    let values = vector.values();
    let (kept_c, kept_v) = kept;
    let (res_c, res_v) = residual;
    kept_c.clear();
    kept_v.clear();
    res_c.clear();
    res_v.clear();

    let n_kept = select_kept(values, percentage, scratch);

    scratch.keep.clear();
    scratch.keep.resize(values.len(), false);
    for &i in &scratch.order[..n_kept] {
        scratch.keep[i as usize] = true;
    }

    // Emit in the input's own order, which is ascending by component, so both outputs satisfy
    // the strictly-ascending invariant a sparse dataset requires on push.
    for (pos, (&c, &v)) in components.iter().zip(values.iter()).enumerate() {
        if scratch.keep[pos] {
            kept_c.push(c);
            kept_v.push(v);
        } else {
            res_c.push(c);
            res_v.push(v);
        }
    }
}

/// Splits a whole sparse dataset into its L1-pruned part and the exact residual.
///
/// Both returned datasets have the same length and the same vector ids as the input, and both
/// keep the input's dimensionality, so a query valid against the input is valid against either
/// of them. Per vector, `pruned.get(i)` and `residual.get(i)` partition `dataset.get(i)`, hence
/// `pruned.nnz() + residual.nnz() == dataset.nnz()`.
///
/// # Panics
///
/// Panics unless `0.0 <= percentage <= 100.0`.
///
/// # Examples
///
/// ```
/// use vectorium::transformations::l1_prune::l1_prune_split;
/// use vectorium::{Dataset, DatasetGrowable, DotProduct};
/// use vectorium::{PlainSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView};
///
/// let quantizer = PlainSparseQuantizer::<u32, f32, DotProduct>::new(8, 8);
/// let mut growable = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
/// growable.push(SparseVectorView::new(&[0, 1, 2, 3], &[1.0, 4.0, 2.0, 3.0]));
/// growable.push(SparseVectorView::new(&[5], &[7.0]));
/// let dataset: PlainSparseDataset<u32, f32, DotProduct> = growable.into();
///
/// let (pruned, residual) = l1_prune_split(&dataset, 50.0);
/// assert_eq!(pruned.nnz() + residual.nnz(), dataset.nnz());
/// // The 1-nnz vector keeps its only component and leaves an empty residual.
/// assert_eq!(pruned.get(1).components(), &[5]);
/// assert!(residual.get(1).components().is_empty());
/// # let _ = ();
/// ```
pub fn l1_prune_split<C, V, D>(
    dataset: &PlainSparseDataset<C, V, D>,
    percentage: f32,
) -> (PlainSparseDataset<C, V, D>, PlainSparseDataset<C, V, D>)
where
    C: ComponentType,
    V: ValueType + Float + FromF32 + SpaceUsage,
    D: ScalarSparseSupportedDistance,
{
    assert!(
        (0.0..=100.0).contains(&percentage),
        "percentage must be in [0, 100], got {percentage}"
    );

    let dim = dataset.input_dim();
    let mut pruned: PlainSparseDatasetGrowable<C, V, D> =
        DatasetGrowable::with_capacity(PlainSparseQuantizer::new(dim, dim), dataset.len());
    let mut residual: PlainSparseDatasetGrowable<C, V, D> =
        DatasetGrowable::with_capacity(PlainSparseQuantizer::new(dim, dim), dataset.len());

    let mut scratch = SplitScratch::default();
    let (mut kept_c, mut kept_v) = (Vec::new(), Vec::new());
    let (mut res_c, mut res_v) = (Vec::new(), Vec::new());

    for vector in dataset.iter() {
        split_into(
            vector,
            percentage,
            &mut scratch,
            (&mut kept_c, &mut kept_v),
            (&mut res_c, &mut res_v),
        );
        pruned.push(SparseVectorView::new(&kept_c, &kept_v));
        residual.push(SparseVectorView::new(&res_c, &res_v));
    }

    (pruned.into(), residual.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distances::DotProduct;
    use crate::{Distance, PlainSparseDatasetGrowable, QueryEvaluator, VectorEncoder, VectorId};
    use half::f16;

    fn dataset_of<V: ValueType + Float + FromF32 + SpaceUsage>(
        dim: usize,
        vectors: &[(Vec<u32>, Vec<V>)],
    ) -> PlainSparseDataset<u32, V, DotProduct> {
        let mut growable: PlainSparseDatasetGrowable<u32, V, DotProduct> =
            DatasetGrowable::new(PlainSparseQuantizer::new(dim, dim));
        for (c, v) in vectors {
            growable.push(SparseVectorView::new(c, v));
        }
        growable.into()
    }

    #[test]
    fn crossing_component_is_kept_and_the_split_is_a_partition() {
        // |values| 4, 3, 2, 1, L1 = 10. At a = 50 the target is 5.0: 4 does not surpass it,
        // 4 + 3 = 7 does, so exactly two components are kept.
        let v = SparseVectorView::new(&[0_u32, 1, 2, 3], &[1.0_f32, 4.0, 2.0, 3.0]);
        let (kept, residual) = l1_prune_split_vector(v, 50.0);
        assert_eq!(kept.components(), &[1, 3]);
        assert_eq!(kept.values(), &[4.0, 3.0]);
        assert_eq!(residual.components(), &[0, 2]);
        assert_eq!(residual.values(), &[1.0, 2.0]);
        assert_eq!(kept.len() + residual.len(), 4);
    }

    #[test]
    fn exactly_reaching_the_target_does_not_stop_the_accumulation() {
        // L1 = 10, a = 40, target = 4.0. The largest component is 4.0, which *reaches* but does
        // not *surpass* the target, so a second component is kept. This is the boundary the
        // rule's wording turns on.
        let v = SparseVectorView::new(&[0_u32, 1, 2, 3], &[1.0_f32, 4.0, 2.0, 3.0]);
        let (kept, _) = l1_prune_split_vector(v, 40.0);
        assert_eq!(kept.components(), &[1, 3]);
    }

    #[test]
    fn ties_break_on_minimum_component_id() {
        // Four equal magnitudes and a target that admits two of them. The rule must take the
        // two smallest component ids, and must do so whatever order the sort happens to visit.
        let v = SparseVectorView::new(&[9_u32, 11, 40, 77], &[2.0_f32, 2.0, 2.0, 2.0]);
        let (kept, residual) = l1_prune_split_vector(v, 30.0);
        assert_eq!(kept.components(), &[9, 11]);
        assert_eq!(residual.components(), &[40, 77]);

        // Same multiset of values presented in a different component order: the kept set is
        // still the two smallest ids present.
        let v = SparseVectorView::new(&[1_u32, 2, 3, 4], &[2.0_f32, 2.0, 2.0, 2.0]);
        let (kept, _) = l1_prune_split_vector(v, 30.0);
        assert_eq!(kept.components(), &[1, 2]);
    }

    #[test]
    fn ties_break_deterministically_across_repeated_calls() {
        let v = SparseVectorView::new(&[3_u32, 5, 8, 13, 21], &[1.0_f32; 5]);
        let first = l1_prune_split_vector(v, 55.0).0.components().to_vec();
        for _ in 0..16 {
            assert_eq!(l1_prune_split_vector(v, 55.0).0.components(), &first[..]);
        }
    }

    #[test]
    fn magnitude_not_signed_value_decides_what_is_kept() {
        // The most negative component is the most influential one. Ordering on the raw value
        // would prune it first and keep the near-zero components instead.
        let v = SparseVectorView::new(&[0_u32, 1, 2], &[-9.0_f32, 0.5, 0.5]);
        let (kept, residual) = l1_prune_split_vector(v, 50.0);
        assert_eq!(kept.components(), &[0]);
        assert_eq!(kept.values(), &[-9.0]);
        assert_eq!(residual.components(), &[1, 2]);
    }

    #[test]
    fn a_one_component_vector_always_leaves_an_empty_residual() {
        for a in [0.0_f32, 10.0, 30.0, 50.0, 90.0, 99.0] {
            let v = SparseVectorView::new(&[7_u32], &[3.5_f32]);
            let (kept, residual) = l1_prune_split_vector(v, a);
            assert_eq!(kept.components(), &[7], "a = {a}");
            assert!(residual.components().is_empty(), "a = {a}");
            assert!(residual.values().is_empty(), "a = {a}");
        }
    }

    #[test]
    fn a_dominant_component_leaves_a_single_component_first_stage() {
        // One component holding more than a% of the mass on its own. The first stage is then a
        // 1-nnz vector -- the cheapest thing the graph walk can score -- while the residual
        // holds everything else. It is *not* empty here: only a vector that had one component
        // to begin with leaves nothing behind, which is the case the test above pins.
        let v = SparseVectorView::new(&[0_u32, 1, 2], &[100.0_f32, 0.01, 0.01]);
        let (kept, residual) = l1_prune_split_vector(v, 90.0);
        assert_eq!(kept.components(), &[0]);
        assert_eq!(residual.components(), &[1, 2]);
    }

    #[test]
    fn a_equal_to_100_keeps_everything_and_empties_every_residual() {
        // The other way an empty residual arises, and the one that does not depend on the
        // collection containing 1-nnz vectors: nothing can surpass 100% of the mass, so every
        // component is kept and every residual is empty. The rerank path must score all of
        // them as 0.0.
        let v = SparseVectorView::new(&[0_u32, 1, 2, 3], &[1.0_f32, 4.0, 2.0, 3.0]);
        let (kept, residual) = l1_prune_split_vector(v, 100.0);
        assert_eq!(kept.components(), &[0, 1, 2, 3]);
        assert!(residual.components().is_empty());
    }

    #[test]
    fn an_empty_residual_is_storable_and_scores_zero() {
        // The property the two-stage rerank path depends on: an empty residual is a legal
        // dataset member and contributes 0.0 to the summed score, rather than panicking.
        let dataset = dataset_of::<f32>(
            8,
            &[
                (vec![0, 1, 2], vec![100.0, 0.01, 0.01]),
                (vec![5], vec![7.0]),
            ],
        );
        let (pruned, residual) = l1_prune_split(&dataset, 90.0);

        // Vector 0 keeps a residual; vector 1 had a single component and so has none.
        assert_eq!(residual.get(0 as VectorId).components(), &[1, 2]);
        assert!(residual.get(1 as VectorId).components().is_empty());

        let query_c = [0_u32, 1, 2, 5];
        let query_v = [1.0_f32, 1.0, 1.0, 1.0];
        let query = SparseVectorView::new(&query_c, &query_v);

        let evaluator = residual.encoder().query_evaluator(query, &());
        let score = evaluator
            .compute_distance(residual.get(1 as VectorId))
            .distance();
        assert_eq!(score, 0.0, "an empty residual must score 0, not panic");

        let _ = &pruned;
    }

    #[test]
    fn the_two_stage_scores_sum_to_the_full_score() {
        // The property that makes the rescoring exact rather than estimated.
        let vectors: Vec<(Vec<u32>, Vec<f32>)> = vec![
            (vec![0, 1, 2, 3, 4], vec![1.0, 4.0, 2.0, 3.0, 0.5]),
            (vec![1, 3, 5], vec![9.0, 0.2, 0.3]),
            (vec![2], vec![6.0]),
            (vec![0, 2, 4, 6], vec![1.0, 1.0, 1.0, 1.0]),
        ];
        let dataset = dataset_of::<f32>(8, &vectors);

        let query_c = [0_u32, 1, 2, 3, 4, 5, 6, 7];
        let query_v = [0.5_f32, 1.5, 2.5, 0.25, 1.0, 3.0, 0.75, 9.0];
        let query = SparseVectorView::new(&query_c, &query_v);

        for a in [10.0_f32, 30.0, 50.0, 70.0, 90.0, 99.0] {
            let (pruned, residual) = l1_prune_split(&dataset, a);
            assert_eq!(pruned.nnz() + residual.nnz(), dataset.nnz(), "a = {a}");

            let full_eval = dataset.encoder().query_evaluator(query, &());
            let pruned_eval = pruned.encoder().query_evaluator(query, &());
            let residual_eval = residual.encoder().query_evaluator(query, &());

            for id in 0..dataset.len() as VectorId {
                let full = full_eval.compute_distance(dataset.get(id)).distance();
                let first = pruned_eval.compute_distance(pruned.get(id)).distance();
                let rest = residual_eval.compute_distance(residual.get(id)).distance();
                assert!(
                    (full - (first + rest)).abs() <= 1e-5 * full.abs().max(1.0),
                    "a = {a}, id = {id}: {full} != {first} + {rest}"
                );
            }
        }
    }

    #[test]
    fn the_split_partitions_every_vector_of_a_dataset() {
        let vectors: Vec<(Vec<u32>, Vec<f32>)> = vec![
            (vec![0, 1, 2, 3, 4], vec![1.0, 4.0, 2.0, 3.0, 0.5]),
            (vec![1, 3, 5], vec![9.0, 0.2, 0.3]),
            (vec![2], vec![6.0]),
        ];
        let dataset = dataset_of::<f32>(8, &vectors);
        let (pruned, residual) = l1_prune_split(&dataset, 60.0);

        assert_eq!(pruned.len(), dataset.len());
        assert_eq!(residual.len(), dataset.len());
        assert_eq!(pruned.input_dim(), dataset.input_dim());
        assert_eq!(residual.input_dim(), dataset.input_dim());

        for id in 0..dataset.len() as VectorId {
            let mut merged: Vec<(u32, f32)> = pruned
                .get(id)
                .iter()
                .chain(residual.get(id).iter())
                .collect();
            merged.sort_by_key(|&(c, _)| c);
            let original: Vec<(u32, f32)> = dataset.get(id).iter().collect();
            assert_eq!(merged, original, "id = {id}");
        }
    }

    #[test]
    fn works_at_the_stored_precision_the_campaign_uses() {
        // f16 is the value type both stages are stored at, and the ordering must be taken on
        // the stored values rather than on some wider staging type.
        let vectors: Vec<(Vec<u32>, Vec<f16>)> = vec![(
            vec![0, 1, 2, 3],
            vec![
                f16::from_f32(1.0),
                f16::from_f32(4.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
            ],
        )];
        let dataset = dataset_of::<f16>(8, &vectors);
        let (pruned, residual) = l1_prune_split(&dataset, 50.0);
        assert_eq!(pruned.get(0 as VectorId).components(), &[1, 3]);
        assert_eq!(residual.get(0 as VectorId).components(), &[0, 2]);
    }

    #[test]
    fn a_zero_vector_keeps_everything_rather_than_emitting_nothing() {
        let v = SparseVectorView::new(&[0_u32, 1], &[0.0_f32, 0.0]);
        let (kept, residual) = l1_prune_split_vector(v, 50.0);
        assert_eq!(kept.components(), &[0, 1]);
        assert!(residual.components().is_empty());
    }

    #[test]
    #[should_panic(expected = "percentage must be in [0, 100]")]
    fn a_percentage_above_100_is_rejected() {
        let v = SparseVectorView::new(&[0_u32], &[1.0_f32]);
        let _ = l1_prune_split_vector(v, 101.0);
    }
}
