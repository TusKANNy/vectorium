use half::f16;
use serde::{Deserialize, Serialize};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::core::data_block::{BLOCK_SIZE, DataBlock};
#[cfg(target_arch = "x86_64")]
use crate::core::data_block::{hsum256_ps, match_and_fma_16};
use crate::core::sealed;
use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    DenseVectorOwned, PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder,
    SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::{Distance, SpaceUsage};

/// Encoder that packs sparse vectors into cache-line-aligned [`DataBlock`]s.
///
/// - Input:  `SparseVectorView<u16, f16>` (components must be sorted ascending)
/// - Query:  `SparseVectorView<u16, f32>`
/// - Output: `PackedVectorView<DataBlock>` (sequence of 64-byte blocks)
///
/// The evaluator uses a runtime threshold ([`SPARSE_QUERY_THRESHOLD`]) to decide
/// the dot-product strategy: short queries (nnz < threshold) use the v1
/// block-skipping SIMD algorithm; longer queries are densified into a `Vec<f32>`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlockedSparseEncoder {
    dim: usize,
}

impl sealed::Sealed for BlockedSparseEncoder {}

impl BlockedSparseEncoder {
    pub fn new(dim: usize) -> Self {
        assert!(
            dim <= u16::MAX as usize + 1,
            "Dimension {} exceeds u16 range",
            dim
        );
        Self { dim }
    }
}

impl SpaceUsage for BlockedSparseEncoder {
    fn space_usage_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

impl VectorEncoder for BlockedSparseEncoder {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u16, f16>;
    type QueryVector<'q> = SparseVectorView<'q, u16, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, DataBlock>;

    type Evaluator<'e>
        = BlockedSparseQueryEvaluator
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        BlockedSparseQueryEvaluator::new(query, self)
    }

    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        BlockedSparseQueryEvaluator::new(decoded.as_view(), self)
    }

    #[inline]
    fn input_dim(&self) -> usize {
        self.dim
    }

    #[inline]
    fn output_dim(&self) -> usize {
        self.dim
    }
}

impl SparseDataEncoder for BlockedSparseEncoder {
    type InputComponentType = u16;
    type InputValueType = f16;
    type OutputComponentType = u16;
    type OutputValueType = f16;

    fn decode_vector<'a>(
        &self,
        encoded: PackedVectorView<'a, DataBlock>,
    ) -> SparseVectorOwned<u16, f32> {
        let mut components = Vec::new();
        let mut values = Vec::new();

        for block in encoded.data() {
            for i in 0..BLOCK_SIZE {
                if block.values[i] == f16::ZERO {
                    break; // Hit padding
                }
                components.push(block.components[i]);
                values.push(block.values[i].to_f32());
            }
        }

        SparseVectorOwned::new(components, values)
    }
}

impl PackedSparseVectorEncoder for BlockedSparseEncoder {
    type PackedDataType = DataBlock;

    fn push_encoded<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, u16, f16>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<DataBlock>,
    {
        let components = input.components();
        let values = input.values();
        let nnz = components.len();

        debug_assert!(
            components.windows(2).all(|w| w[0] < w[1]),
            "Components must be sorted in strictly ascending order"
        );

        let num_blocks = nnz.div_ceil(BLOCK_SIZE);

        for block_idx in 0..num_blocks {
            let start = block_idx * BLOCK_SIZE;
            let end = std::cmp::min(start + BLOCK_SIZE, nnz);

            let mut block = DataBlock::default();

            // Fill with actual data.
            for (i, idx) in (start..end).enumerate() {
                block.components[i] = components[idx];
                block.values[i] = values[idx];
            }

            // Pad remaining slots: component = last real component, value = ZERO (already default).
            let real_count = end - start;
            if real_count < BLOCK_SIZE {
                let last_comp = components[end - 1];
                for i in real_count..BLOCK_SIZE {
                    block.components[i] = last_comp;
                }
            }

            output.extend(std::iter::once(block));
        }
    }
}

/// If the query has fewer than this many non-zero components, the evaluator
/// keeps it sparse and uses the v1 block-skipping SIMD algorithm. Otherwise
/// it densifies the query and uses gather-based dot product.
pub const SPARSE_QUERY_THRESHOLD: usize = 33;

/// Evaluator for [`BlockedSparseEncoder`].
///
/// At construction time the query is either densified (when `nnz >= SPARSE_QUERY_THRESHOLD`)
/// or kept sparse (when `nnz < SPARSE_QUERY_THRESHOLD`). The `compute_distance`
/// method dispatches to the appropriate path.
#[derive(Debug, Clone)]
pub struct BlockedSparseQueryEvaluator {
    /// Dense path: populated when query nnz >= SPARSE_QUERY_THRESHOLD.
    dense_query: Option<DenseVectorOwned<f32>>,
    /// Sparse v1 path: populated when query nnz < SPARSE_QUERY_THRESHOLD.
    sparse_query: Option<SparseVectorOwned<u16, f32>>,
}

impl BlockedSparseQueryEvaluator {
    #[inline]
    pub fn new(query: SparseVectorView<'_, u16, f32>, encoder: &BlockedSparseEncoder) -> Self {
        for (c, _) in query.iter() {
            assert!(
                (c as usize) < encoder.dim,
                "Query component {} exceeds dimension {}",
                c,
                encoder.dim
            );
        }

        if query.components().len() >= SPARSE_QUERY_THRESHOLD {
            // Dense path: densify the query.
            let mut dense_query = vec![0.0f32; encoder.dim];
            for (c, v) in query.iter() {
                dense_query[c as usize] = v;
            }
            Self {
                dense_query: Some(DenseVectorOwned::new(dense_query)),
                sparse_query: None,
            }
        } else {
            // Sparse v1 path: keep query as sorted component/value arrays.
            debug_assert!(
                query.components().windows(2).all(|w| w[0] < w[1]),
                "Query components must be sorted in strictly ascending order"
            );
            Self {
                dense_query: None,
                sparse_query: Some(SparseVectorOwned::new(
                    query.components().to_vec(),
                    query.values().to_vec(),
                )),
            }
        }
    }

    /// Dense gather-based dot product (existing algorithm).
    #[inline]
    fn compute_distance_dense(
        dense_query: crate::core::vector::DenseVectorView<'_, f32>,
        vector: PackedVectorView<'_, DataBlock>,
    ) -> DotProduct {
        let query = dense_query.values();
        let mut result = 0.0f32;
        for block in vector.data() {
            result = result.algebraic_add(block.dot_product_dense_query(query).distance());
        }
        DotProduct::from(result)
    }

    /// V1 block-skipping SIMD dot product.
    ///
    /// Iterates over sorted query components and skips blocks whose
    /// `last_component()` is smaller than the current query component.
    /// For matching blocks, uses AVX2 SIMD to find and accumulate values.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "f16c")]
    #[target_feature(enable = "fma")]
    #[inline]
    unsafe fn compute_distance_v1(
        query: SparseVectorView<'_, u16, f32>,
        vector: PackedVectorView<'_, DataBlock>,
    ) -> DotProduct {
        if query.is_empty() {
            return DotProduct::from(0.0f32);
        }

        let blocks = vector.data();
        let mut cur_q = 0;
        let mut sum: __m256 = _mm256_setzero_ps();
        let query_components = query.components();
        let query_values = query.values();

        unsafe {
            'block_loop: for block in blocks {
                if query_components[cur_q] <= block.last_component() {
                    let (comps, vbits) = block.load_in_registers();
                    while query_components[cur_q] <= block.last_component() {
                        match_and_fma_16(
                            query_components[cur_q],
                            query_values[cur_q],
                            comps,
                            vbits,
                            &mut sum,
                        );
                        cur_q += 1;
                        if cur_q >= query_components.len() {
                            break 'block_loop;
                        }
                    }
                }
            }

            DotProduct::from(hsum256_ps(sum))
        }
    }
}

impl<'v> QueryEvaluator<PackedVectorView<'v, DataBlock>> for BlockedSparseQueryEvaluator {
    type Distance = DotProduct;

    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, DataBlock>) -> DotProduct {
        if let Some(dense_query) = &self.dense_query {
            Self::compute_distance_dense(dense_query.as_view(), vector)
        } else {
            let query = self.sparse_query.as_ref().unwrap().as_view();
            #[cfg(target_arch = "x86_64")]
            {
                // SAFETY: we build with target-cpu=native which guarantees AVX2/F16C/FMA.
                unsafe { Self::compute_distance_v1(query, vector) }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                let _ = query;
                // Fallback: densify on the fly (shouldn't happen in practice).
                panic!("v1 sparse path requires x86_64 with AVX2/F16C/FMA");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dataset, DatasetGrowable, PackedSparseDatasetGrowable};

    #[test]
    fn blocked_sparse_basic_push_and_get() {
        let encoder = BlockedSparseEncoder::new(100);
        let mut dataset = PackedSparseDatasetGrowable::<BlockedSparseEncoder>::new(encoder);

        let comps = [1u16, 10, 50];
        let vals = [f16::from_f32(1.5), f16::from_f32(2.0), f16::from_f32(0.5)];
        dataset.push(SparseVectorView::new(&comps, &vals));

        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset.nnz(), 3);

        // Get the vector back.
        let view = dataset.get(0);
        assert_eq!(view.data().len(), 1); // 3 elements fit in 1 block
    }

    #[test]
    fn blocked_sparse_dot_product() {
        let encoder = BlockedSparseEncoder::new(100);
        let mut dataset = PackedSparseDatasetGrowable::<BlockedSparseEncoder>::new(encoder);

        // Document: [(1, 1.0), (10, 2.0)]
        let comps = [1u16, 10];
        let vals = [f16::from_f32(1.0), f16::from_f32(2.0)];
        dataset.push(SparseVectorView::new(&comps, &vals));

        // Query: [(1, 3.0), (10, 4.0)]
        let query = SparseVectorView::new(&[1u16, 10], &[3.0f32, 4.0]);
        let evaluator = dataset.encoder().query_evaluator(query);
        let dist = evaluator.compute_distance(dataset.get(0));

        // Expected: 1.0 * 3.0 + 2.0 * 4.0 = 11.0
        // (with f16 quantization error)
        let expected = f16::from_f32(1.0).to_f32() * 3.0 + f16::from_f32(2.0).to_f32() * 4.0;
        assert!(
            (dist.distance() - expected).abs() < 0.01,
            "Expected {expected}, got {}",
            dist.distance()
        );
    }

    #[test]
    fn blocked_sparse_multiple_blocks() {
        let encoder = BlockedSparseEncoder::new(1000);
        let mut dataset = PackedSparseDatasetGrowable::<BlockedSparseEncoder>::new(encoder);

        // Create a vector with >16 non-zeros (needs 2+ blocks).
        let comps: Vec<u16> = (0..20).collect();
        let vals: Vec<f16> = (1..=20).map(|i| f16::from_f32(i as f32 * 0.1)).collect();
        dataset.push(SparseVectorView::new(&comps, &vals));

        let view = dataset.get(0);
        assert_eq!(view.data().len(), 2); // ceil(20/16) = 2 blocks
    }

    #[test]
    fn blocked_sparse_roundtrip_growable_immutable() {
        use crate::PackedSparseDataset;

        let encoder = BlockedSparseEncoder::new(50);
        let mut growable = PackedSparseDatasetGrowable::<BlockedSparseEncoder>::new(encoder);

        growable.push(SparseVectorView::new(
            &[5u16, 10],
            &[f16::from_f32(1.0), f16::from_f32(2.0)],
        ));

        let frozen: PackedSparseDataset<BlockedSparseEncoder> = growable.into();
        assert_eq!(frozen.len(), 1);

        let mut growable_again: PackedSparseDatasetGrowable<BlockedSparseEncoder> = frozen.into();
        growable_again.push(SparseVectorView::new(&[3u16], &[f16::from_f32(0.5)]));
        assert_eq!(growable_again.len(), 2);
    }

    #[test]
    fn convert_from_plain_sparse_dataset() {
        use crate::{
            BlockedSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer,
            VectorEncoder as _,
        };

        let quantizer = PlainSparseQuantizer::<u16, f16, DotProduct>::new(100, 100);
        let mut plain = PlainSparseDatasetGrowable::new(quantizer);

        plain.push(SparseVectorView::new(
            &[1u16, 10, 50],
            &[f16::from_f32(1.5), f16::from_f32(2.0), f16::from_f32(0.5)],
        ));
        plain.push(SparseVectorView::new(
            &[5u16, 20],
            &[f16::from_f32(3.0), f16::from_f32(1.0)],
        ));

        let frozen: crate::PlainSparseDataset<u16, f16, DotProduct> = plain.into();
        let blocked: BlockedSparseDataset = frozen.into();

        assert_eq!(blocked.len(), 2);
        assert_eq!(blocked.nnz(), 5);

        // Verify dot product matches.
        let query = SparseVectorView::new(&[1u16, 10], &[2.0f32, 3.0]);
        let evaluator = blocked.encoder().query_evaluator(query);
        let dist = evaluator.compute_distance(blocked.get(0));

        let expected = f16::from_f32(1.5).to_f32() * 2.0 + f16::from_f32(2.0).to_f32() * 3.0;
        assert!(
            (dist.distance() - expected).abs() < 0.01,
            "Expected {expected}, got {}",
            dist.distance()
        );
    }

    // -----------------------------------------------------------------------
    // V1 sparse-query path tests
    // -----------------------------------------------------------------------

    /// Helper: force the dense path by creating an evaluator with nnz >= SPARSE_QUERY_THRESHOLD.
    fn make_dense_evaluator(
        query_comps: &[u16],
        query_vals: &[f32],
        dim: usize,
    ) -> BlockedSparseQueryEvaluator {
        assert!(query_comps.len() >= SPARSE_QUERY_THRESHOLD);
        let encoder = BlockedSparseEncoder::new(dim);
        let query = SparseVectorView::new(query_comps, query_vals);
        let eval = BlockedSparseQueryEvaluator::new(query, &encoder);
        assert!(eval.dense_query.is_some(), "Expected dense path");
        eval
    }

    /// Helper: force the v1 path by creating an evaluator with nnz < SPARSE_QUERY_THRESHOLD.
    fn make_v1_evaluator(
        query_comps: &[u16],
        query_vals: &[f32],
        dim: usize,
    ) -> BlockedSparseQueryEvaluator {
        assert!(query_comps.len() < SPARSE_QUERY_THRESHOLD);
        let encoder = BlockedSparseEncoder::new(dim);
        let query = SparseVectorView::new(query_comps, query_vals);
        let eval = BlockedSparseQueryEvaluator::new(query, &encoder);
        assert!(eval.sparse_query.is_some(), "Expected v1 path");
        eval
    }

    #[test]
    fn threshold_selects_dense_path() {
        let dim = 1000;
        let comps: Vec<u16> = (0..SPARSE_QUERY_THRESHOLD as u16).collect();
        let vals: Vec<f32> = vec![1.0; SPARSE_QUERY_THRESHOLD];
        let encoder = BlockedSparseEncoder::new(dim);
        let query = SparseVectorView::new(&comps, &vals);
        let eval = BlockedSparseQueryEvaluator::new(query, &encoder);
        assert!(eval.dense_query.is_some());
        assert!(eval.sparse_query.is_none());
    }

    #[test]
    fn threshold_selects_v1_path() {
        let dim = 1000;
        let comps: Vec<u16> = (0..(SPARSE_QUERY_THRESHOLD as u16 - 1)).collect();
        let vals: Vec<f32> = vec![1.0; SPARSE_QUERY_THRESHOLD - 1];
        let encoder = BlockedSparseEncoder::new(dim);
        let query = SparseVectorView::new(&comps, &vals);
        let eval = BlockedSparseQueryEvaluator::new(query, &encoder);
        assert!(eval.sparse_query.is_some());
        assert!(eval.dense_query.is_none());
    }

    #[test]
    fn v1_dot_product_basic() {
        let dim = 100;
        let encoder = BlockedSparseEncoder::new(dim);
        let mut dataset = PackedSparseDatasetGrowable::<BlockedSparseEncoder>::new(encoder);

        // Document: [(1, 1.0), (10, 2.0)]
        let doc_comps = [1u16, 10];
        let doc_vals = [f16::from_f32(1.0), f16::from_f32(2.0)];
        dataset.push(SparseVectorView::new(&doc_comps, &doc_vals));

        // Query with nnz < SPARSE_QUERY_THRESHOLD → v1 path
        let query_comps = [1u16, 10];
        let query_vals = [3.0f32, 4.0];
        let eval = make_v1_evaluator(&query_comps, &query_vals, dim);
        let dist = eval.compute_distance(dataset.get(0));

        let expected = f16::from_f32(1.0).to_f32() * 3.0 + f16::from_f32(2.0).to_f32() * 4.0;
        assert!(
            (dist.distance() - expected).abs() < 0.01,
            "Expected {expected}, got {}",
            dist.distance()
        );
    }

    #[test]
    fn v1_dot_product_multiple_blocks() {
        let dim = 1000;
        let encoder = BlockedSparseEncoder::new(dim);
        let mut dataset = PackedSparseDatasetGrowable::<BlockedSparseEncoder>::new(encoder);

        // Document with 20 nnz → 2 blocks
        let doc_comps: Vec<u16> = (0..20).collect();
        let doc_vals: Vec<f16> = (1..=20).map(|i| f16::from_f32(i as f32 * 0.1)).collect();
        dataset.push(SparseVectorView::new(&doc_comps, &doc_vals));

        // Query overlaps components in both blocks (comp 5 in block 0, comp 18 in block 1)
        let query_comps = [5u16, 18];
        let query_vals = [2.0f32, 3.0];
        let eval = make_v1_evaluator(&query_comps, &query_vals, dim);
        let dist = eval.compute_distance(dataset.get(0));

        let expected =
            f16::from_f32(6.0 * 0.1).to_f32() * 2.0 + f16::from_f32(19.0 * 0.1).to_f32() * 3.0;
        assert!(
            (dist.distance() - expected).abs() < 0.05,
            "Expected {expected}, got {}",
            dist.distance()
        );
    }

    #[test]
    fn v1_empty_query() {
        let dim = 100;
        let encoder = BlockedSparseEncoder::new(dim);
        let mut dataset = PackedSparseDatasetGrowable::<BlockedSparseEncoder>::new(encoder);

        let doc_comps = [1u16, 10];
        let doc_vals = [f16::from_f32(1.0), f16::from_f32(2.0)];
        dataset.push(SparseVectorView::new(&doc_comps, &doc_vals));

        let query_comps: [u16; 0] = [];
        let query_vals: [f32; 0] = [];
        let eval = make_v1_evaluator(&query_comps, &query_vals, dim);
        let dist = eval.compute_distance(dataset.get(0));

        assert_eq!(dist.distance(), 0.0);
    }

    #[test]
    fn v1_query_beyond_all_blocks() {
        let dim = 1000;
        let encoder = BlockedSparseEncoder::new(dim);
        let mut dataset = PackedSparseDatasetGrowable::<BlockedSparseEncoder>::new(encoder);

        // Document components: [0..3], all in block 0 with last_component = 2
        let doc_comps = [0u16, 1, 2];
        let doc_vals = [f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];
        dataset.push(SparseVectorView::new(&doc_comps, &doc_vals));

        // Query components all > last_component of any block → no matches
        let query_comps = [500u16, 600];
        let query_vals = [1.0f32, 2.0];
        let eval = make_v1_evaluator(&query_comps, &query_vals, dim);
        let dist = eval.compute_distance(dataset.get(0));

        assert_eq!(dist.distance(), 0.0);
    }

    #[test]
    fn v1_disjoint_components() {
        let dim = 100;
        let encoder = BlockedSparseEncoder::new(dim);
        let mut dataset = PackedSparseDatasetGrowable::<BlockedSparseEncoder>::new(encoder);

        // Doc has even components, query has odd → no overlap
        let doc_comps = [0u16, 2, 4, 6];
        let doc_vals = [
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        dataset.push(SparseVectorView::new(&doc_comps, &doc_vals));

        let query_comps = [1u16, 3, 5];
        let query_vals = [1.0f32, 1.0, 1.0];
        let eval = make_v1_evaluator(&query_comps, &query_vals, dim);
        let dist = eval.compute_distance(dataset.get(0));

        assert_eq!(dist.distance(), 0.0);
    }

    #[test]
    fn v1_matches_dense_path() {
        use rand::rngs::StdRng;
        use rand::seq::SliceRandom;
        use rand::{Rng, SeedableRng};

        let dim = 1 << 15; // 32768
        let doc_nnz = 128;
        let query_nnz = 20; // below threshold → v1

        let mut rng = StdRng::seed_from_u64(42);

        // Generate random document
        let mut doc_indices: Vec<u16> = (0..dim as u16).collect();
        doc_indices.shuffle(&mut rng);
        doc_indices.truncate(doc_nnz);
        doc_indices.sort_unstable();
        let doc_vals: Vec<f16> = (0..doc_nnz)
            .map(|_| f16::from_f32(rng.gen_range(0.01..1.0)))
            .collect();

        // Generate random query
        let mut query_indices: Vec<u16> = (0..dim as u16).collect();
        query_indices.shuffle(&mut rng);
        query_indices.truncate(query_nnz);
        query_indices.sort_unstable();
        let query_vals: Vec<f32> = (0..query_nnz).map(|_| rng.gen_range(0.01..1.0)).collect();

        let encoder = BlockedSparseEncoder::new(dim);
        let mut dataset = PackedSparseDatasetGrowable::<BlockedSparseEncoder>::new(encoder);
        dataset.push(SparseVectorView::new(&doc_indices, &doc_vals));

        // V1 path (nnz < threshold)
        let v1_eval = make_v1_evaluator(&query_indices, &query_vals, dim);
        let v1_dist = v1_eval.compute_distance(dataset.get(0));

        // Dense path: pad query to reach threshold
        let mut padded_comps = query_indices.clone();
        let mut padded_vals = query_vals.clone();
        // Add components that don't overlap with the document (use high indices)
        let mut extra_comp = (dim as u16) - 1;
        while padded_comps.len() < SPARSE_QUERY_THRESHOLD {
            if !padded_comps.contains(&extra_comp) {
                padded_comps.push(extra_comp);
                padded_vals.push(0.0); // zero value → no contribution
            }
            extra_comp -= 1;
        }
        padded_comps.sort_unstable();
        // Re-sort values to match sorted components
        let mut pairs: Vec<(u16, f32)> = padded_comps
            .iter()
            .copied()
            .zip(padded_vals.iter().copied())
            .collect();
        pairs.sort_by_key(|p| p.0);
        let sorted_comps: Vec<u16> = pairs.iter().map(|p| p.0).collect();
        let sorted_vals: Vec<f32> = pairs.iter().map(|p| p.1).collect();

        let dense_eval = make_dense_evaluator(&sorted_comps, &sorted_vals, dim);
        let dense_dist = dense_eval.compute_distance(dataset.get(0));

        assert!(
            (v1_dist.distance() - dense_dist.distance()).abs() < 0.01,
            "v1={} dense={}",
            v1_dist.distance(),
            dense_dist.distance()
        );
    }

    #[test]
    fn v1_fuzz_random() {
        use rand::rngs::StdRng;
        use rand::seq::SliceRandom;
        use rand::{Rng, SeedableRng};

        let dim = 1000;
        let mut rng = StdRng::seed_from_u64(0xBEEF_CAFE);

        for _ in 0..100 {
            let doc_nnz = rng.gen_range(1..100);
            let query_nnz = rng.gen_range(1..SPARSE_QUERY_THRESHOLD);

            let mut doc_indices: Vec<u16> = (0..dim as u16).collect();
            doc_indices.shuffle(&mut rng);
            doc_indices.truncate(doc_nnz);
            doc_indices.sort_unstable();
            let doc_vals: Vec<f16> = (0..doc_nnz)
                .map(|_| f16::from_f32(rng.gen_range(0.01..5.0)))
                .collect();

            let mut query_indices: Vec<u16> = (0..dim as u16).collect();
            query_indices.shuffle(&mut rng);
            query_indices.truncate(query_nnz);
            query_indices.sort_unstable();
            let query_vals: Vec<f32> = (0..query_nnz).map(|_| rng.gen_range(0.01..5.0)).collect();

            let encoder = BlockedSparseEncoder::new(dim);
            let mut dataset = PackedSparseDatasetGrowable::<BlockedSparseEncoder>::new(encoder);
            dataset.push(SparseVectorView::new(&doc_indices, &doc_vals));

            // V1 evaluator
            let v1_eval = make_v1_evaluator(&query_indices, &query_vals, dim);
            let v1_dist = v1_eval.compute_distance(dataset.get(0));

            // Reference: compute manually with merge
            let mut expected = 0.0f32;
            let mut qi = 0;
            let mut di = 0;
            while qi < query_indices.len() && di < doc_indices.len() {
                match query_indices[qi].cmp(&doc_indices[di]) {
                    std::cmp::Ordering::Equal => {
                        expected += query_vals[qi] * doc_vals[di].to_f32();
                        qi += 1;
                        di += 1;
                    }
                    std::cmp::Ordering::Less => qi += 1,
                    std::cmp::Ordering::Greater => di += 1,
                }
            }

            assert!(
                (v1_dist.distance() - expected).abs() < 0.1 * (1.0 + expected.abs()),
                "Mismatch: v1={} expected={} (doc_nnz={doc_nnz}, query_nnz={query_nnz})",
                v1_dist.distance(),
                expected
            );
        }
    }
}
