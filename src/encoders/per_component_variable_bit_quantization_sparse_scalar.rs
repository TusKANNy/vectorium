use std::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::core::vector_encoder::{
    QueryEvaluator, SparseDataEncoder, SparseVectorEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::encoders::variable_bit_uniform_quantization_sparse_scalar::VariableBitUniformQuantizedSparseSupportedDistance;
use crate::utils::is_strictly_sorted;
use crate::{ComponentType, Dataset, PlainSparseDataset, SpaceUsage, SparseVectorView};

/// Per-component uniform sparse quantizer with a **per-component** bit width.
///
/// Each vocabulary dimension `c` stores codes using `nbits_per_component[c]` bits
/// (1..=8). All codes are stored as `u8` regardless of the bit width — only the
/// `max_val = 2^nbits - 1` ceiling differs per component, applied at encode time.
/// This is the right shape for posting-list-uniform variable-bit experiments
/// (constraint C1): bit width varies across tokens, but is uniform within a single
/// posting list.
///
/// We do **not** care about runtime efficiency in this iteration; the storage
/// layout is the same uniform `u8` buffer used by the global-bit-width sibling
/// encoder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerComponentVariableBitUniformSparseQuantizer<C, D> {
    dim: usize,
    nbits_per_component: Box<[u8]>,
    max_val_per_component: Box<[f32]>,
    quants: Box<[f32]>,
    _phantom: PhantomData<(C, D)>,
}

impl<C, D> PartialEq for PerComponentVariableBitUniformSparseQuantizer<C, D> {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim && self.nbits_per_component == other.nbits_per_component
    }
}

impl<C, D> PerComponentVariableBitUniformSparseQuantizer<C, D> {
    #[inline]
    pub fn new(input_dim: usize, output_dim: usize, nbits_per_component: &[u8]) -> Self {
        assert_eq!(
            input_dim, output_dim,
            "PerComponentVariableBitUniformSparseQuantizer requires input_dim == output_dim"
        );
        assert_eq!(
            nbits_per_component.len(),
            input_dim,
            "nbits_per_component length ({}) must equal input_dim ({input_dim})",
            nbits_per_component.len()
        );
        for (c, &nb) in nbits_per_component.iter().enumerate() {
            assert!(
                (1..=8).contains(&nb),
                "nbits_per_component[{c}] must be in [1, 8], got {nb}"
            );
        }

        let max_val_per_component: Vec<f32> = nbits_per_component
            .iter()
            .map(|&nb| ((1u32 << nb) - 1) as f32)
            .collect();

        Self {
            dim: input_dim,
            nbits_per_component: nbits_per_component.to_vec().into_boxed_slice(),
            max_val_per_component: max_val_per_component.into_boxed_slice(),
            quants: vec![0.0; output_dim].into_boxed_slice(),
            _phantom: PhantomData,
        }
    }

    pub fn nbits_per_component(&self) -> &[u8] {
        &self.nbits_per_component
    }

    pub fn max_val_per_component(&self) -> &[f32] {
        &self.max_val_per_component
    }

    pub fn quants(&self) -> &[f32] {
        &self.quants
    }

    /// Train the per-component step `quants[c] = max_c / max_val_per_component[c]`,
    /// where `max_c` is the maximum positive value observed for component `c` in
    /// the training data. Components not seen in the data, or whose maximum is
    /// non-positive, get `quants[c] = 0` (encode-time clamp returns 0).
    ///
    /// This is the per-component analogue of the `[0.0, 1.0]` percentile fast path
    /// in `train_sparse_scalar_quantizer_with_levels`. We do not support arbitrary
    /// percentiles here — the toy experiment doesn't need them.
    pub fn train(
        training_data: &PlainSparseDataset<C, f32, crate::distances::SquaredEuclideanDistance>,
        nbits_per_component: &[u8],
    ) -> Self
    where
        C: ComponentType,
    {
        let dim = training_data.output_dim();
        assert_eq!(
            nbits_per_component.len(),
            dim,
            "nbits_per_component length ({}) must equal training_data.output_dim() ({dim})",
            nbits_per_component.len()
        );
        for (c, &nb) in nbits_per_component.iter().enumerate() {
            assert!(
                (1..=8).contains(&nb),
                "nbits_per_component[{c}] must be in [1, 8], got {nb}"
            );
        }

        let max_val_per_component: Vec<f32> = nbits_per_component
            .iter()
            .map(|&nb| ((1u32 << nb) - 1) as f32)
            .collect();

        // Per-component max scan (single pass). Mirrors the fast-path branch of
        // `train_sparse_scalar_quantizer_with_levels` but uses per-component levels.
        let mut maxes = vec![0.0f32; dim];
        for doc in training_data.iter() {
            for (&c, &v) in doc.components().iter().zip(doc.values()) {
                let idx: usize = c.as_();
                if v > maxes[idx] {
                    maxes[idx] = v;
                }
            }
        }

        let mut quants = vec![0.0f32; dim];
        for c in 0..dim {
            let mv = max_val_per_component[c];
            if maxes[c] > 0.0 && mv > 0.0 {
                quants[c] = maxes[c] / mv;
            }
        }

        Self {
            dim,
            nbits_per_component: nbits_per_component.to_vec().into_boxed_slice(),
            max_val_per_component: max_val_per_component.into_boxed_slice(),
            quants: quants.into_boxed_slice(),
            _phantom: PhantomData,
        }
    }
}

fn compute_query_squared_norm(values: &[f32]) -> f32 {
    values
        .iter()
        .fold(0.0f32, |acc, &v| acc.algebraic_add(v.algebraic_mul(v)))
}

impl<C, D> SparseDataEncoder for PerComponentVariableBitUniformSparseQuantizer<C, D>
where
    C: ComponentType,
    D: VariableBitUniformQuantizedSparseSupportedDistance,
{
    type InputComponentType = C;
    type InputValueType = f32;
    type OutputComponentType = C;
    type OutputValueType = u8;

    #[inline]
    fn decode_vector<'a>(
        &self,
        encoded: SparseVectorView<'a, Self::OutputComponentType, Self::OutputValueType>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let components = encoded.components().to_vec();
        let values: Vec<f32> = encoded
            .components()
            .iter()
            .zip(encoded.values())
            .map(|(&c, &v)| {
                let idx: usize = c.as_();
                (v as f32) * self.quants[idx]
            })
            .collect();
        SparseVectorOwned::new(components, values)
    }
}

impl<C, D> SparseVectorEncoder for PerComponentVariableBitUniformSparseQuantizer<C, D>
where
    C: ComponentType,
    D: VariableBitUniformQuantizedSparseSupportedDistance,
{
    fn push_encoded<'a, ComponentContainer, ValueContainer>(
        &self,
        input: SparseVectorView<'a, C, f32>,
        components: &mut ComponentContainer,
        values: &mut ValueContainer,
    ) where
        ComponentContainer: Extend<Self::OutputComponentType>,
        ValueContainer: Extend<Self::OutputValueType>,
    {
        components.extend(input.components().iter().cloned());
        values.extend(
            input
                .components()
                .iter()
                .zip(input.values())
                .map(|(&c, &v)| {
                    let idx: usize = c.as_();
                    let q = self.quants[idx];
                    let max_val = self.max_val_per_component[idx];
                    if q > 0.0 {
                        (v / q).clamp(0.0, max_val) as u8
                    } else {
                        0u8
                    }
                }),
        );
    }

    fn encode_vector<'a>(
        &self,
        input: Self::InputVector<'a>,
    ) -> SparseVectorOwned<Self::OutputComponentType, Self::OutputValueType> {
        let mut components = Vec::new();
        let mut values = Vec::new();
        self.push_encoded(input, &mut components, &mut values);
        SparseVectorOwned::new(components, values)
    }
}

impl<C, D> VectorEncoder for PerComponentVariableBitUniformSparseQuantizer<C, D>
where
    C: ComponentType,
    D: VariableBitUniformQuantizedSparseSupportedDistance,
{
    type Distance = D;
    type InputVector<'a> = SparseVectorView<'a, C, f32>;
    type QueryVector<'q> = SparseVectorView<'q, C, f32>;
    type EncodedVector<'a> = SparseVectorView<'a, C, u8>;

    type Evaluator<'e>
        = PerComponentVariableBitUniformSparseQueryEvaluator<'e, C, D>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        PerComponentVariableBitUniformSparseQueryEvaluator::new(query, self)
    }

    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        PerComponentVariableBitUniformSparseQueryEvaluator::new_from_owned_query(decoded, self)
    }

    #[inline]
    fn output_dim(&self) -> usize {
        self.dim
    }

    #[inline]
    fn input_dim(&self) -> usize {
        self.dim
    }
}

#[derive(Debug, Clone)]
pub struct PerComponentVariableBitUniformSparseQueryEvaluator<'e, C, D>
where
    C: ComponentType,
    D: VariableBitUniformQuantizedSparseSupportedDistance,
{
    dense_transformed: Option<Vec<f32>>,
    sparse_query: Option<SparseVectorOwned<C, f32>>,
    dot_query: Option<f32>,
    quants: &'e [f32],
    _phantom: PhantomData<D>,
}

impl<'e, C, D> PerComponentVariableBitUniformSparseQueryEvaluator<'e, C, D>
where
    C: ComponentType,
    D: VariableBitUniformQuantizedSparseSupportedDistance,
{
    pub fn new(
        query: SparseVectorView<'_, C, f32>,
        quantizer: &'e PerComponentVariableBitUniformSparseQuantizer<C, D>,
    ) -> Self {
        let dot_query = if D::requires_dot_query() {
            Some(compute_query_squared_norm(query.values()))
        } else {
            None
        };

        let max_c = query
            .components()
            .iter()
            .map(|c| c.as_())
            .max()
            .unwrap_or(0);

        assert!(
            max_c < quantizer.input_dim(),
            "Query vector component exceeds quantizer input dimension."
        );

        assert_eq!(
            query.components().len(),
            query.values().len(),
            "Query vector components and values length mismatch."
        );

        let small_dim = quantizer.input_dim() < 2_usize.pow(20);

        if small_dim {
            let mut transformed = vec![0.0f32; quantizer.dim];
            for (&c, &v) in query.components().iter().zip(query.values()) {
                let idx: usize = c.as_();
                transformed[idx] = v * quantizer.quants[idx];
            }

            Self {
                dense_transformed: Some(transformed),
                sparse_query: None,
                dot_query,
                quants: &quantizer.quants,
                _phantom: PhantomData,
            }
        } else {
            assert!(
                is_strictly_sorted(query.components()),
                "Query components must be sorted in strictly ascending order."
            );

            Self {
                dense_transformed: None,
                sparse_query: Some(SparseVectorOwned::new(
                    query.components().to_vec(),
                    query.values().to_vec(),
                )),
                dot_query,
                quants: &quantizer.quants,
                _phantom: PhantomData,
            }
        }
    }

    pub fn new_from_owned_query(
        query: SparseVectorOwned<C, f32>,
        quantizer: &'e PerComponentVariableBitUniformSparseQuantizer<C, D>,
    ) -> Self {
        let dot_query = if D::requires_dot_query() {
            Some(compute_query_squared_norm(query.values()))
        } else {
            None
        };

        let max_c = query
            .components()
            .iter()
            .map(|c| c.as_())
            .max()
            .unwrap_or(0);

        assert!(
            max_c < quantizer.input_dim(),
            "Query vector component exceeds quantizer input dimension."
        );

        let small_dim = quantizer.input_dim() < 2_usize.pow(20);

        if small_dim {
            let mut transformed = vec![0.0f32; quantizer.dim];
            for (&c, &v) in query.components().iter().zip(query.values()) {
                let idx: usize = c.as_();
                transformed[idx] = v * quantizer.quants[idx];
            }

            Self {
                dense_transformed: Some(transformed),
                sparse_query: None,
                dot_query,
                quants: &quantizer.quants,
                _phantom: PhantomData,
            }
        } else {
            assert!(
                is_strictly_sorted(query.components()),
                "Query components must be sorted in strictly ascending order."
            );

            Self {
                dense_transformed: None,
                sparse_query: Some(query),
                dot_query,
                quants: &quantizer.quants,
                _phantom: PhantomData,
            }
        }
    }
}

impl<'e, 'v, C, D> QueryEvaluator<SparseVectorView<'v, C, u8>>
    for PerComponentVariableBitUniformSparseQueryEvaluator<'e, C, D>
where
    C: ComponentType,
    D: VariableBitUniformQuantizedSparseSupportedDistance,
{
    type Distance = D;

    #[inline]
    fn compute_distance(&self, vector: SparseVectorView<'v, C, u8>) -> D {
        if let Some(transformed) = &self.dense_transformed {
            D::compute_dense(transformed, self.quants, vector, self.dot_query)
        } else {
            D::compute_sparse(
                self.sparse_query.as_ref().unwrap(),
                self.quants,
                vector,
                self.dot_query,
            )
        }
    }
}

impl<C, D> SpaceUsage for PerComponentVariableBitUniformSparseQuantizer<C, D>
where
    C: ComponentType,
    D: VariableBitUniformQuantizedSparseSupportedDistance,
{
    fn space_usage_bytes(&self) -> usize {
        self.dim.space_usage_bytes()
            + self.nbits_per_component.space_usage_bytes()
            + self.max_val_per_component.space_usage_bytes()
            + self.quants.space_usage_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PlainSparseDatasetGrowable;
    use crate::core::dataset::DatasetGrowable;
    use crate::core::vector::SparseVectorView;
    use crate::distances::{Distance, DotProduct, SquaredEuclideanDistance};
    use crate::encoders::sparse_scalar::PlainSparseQuantizer;

    type DotQuantizer = PerComponentVariableBitUniformSparseQuantizer<u16, DotProduct>;

    fn build_training_data(
        dim: usize,
        vectors: &[(&[u16], &[f32])],
    ) -> PlainSparseDataset<u16, f32, SquaredEuclideanDistance> {
        let q = PlainSparseQuantizer::<u16, f32, SquaredEuclideanDistance>::new(dim, dim);
        let mut g = PlainSparseDatasetGrowable::new(q);
        for &(c, v) in vectors {
            g.push(SparseVectorView::new(c, v));
        }
        g.into()
    }

    #[test]
    fn uniform_bits_match_global_quantizer() {
        // When every component shares the same nbits, this encoder must match
        // the global VariableBitUniformSparseQuantizer at the same bit width.
        use crate::encoders::variable_bit_uniform_quantization_sparse_scalar::VariableBitUniformSparseQuantizer;

        let td = build_training_data(
            4,
            &[(&[0, 1, 2, 3], &[0.0; 4]), (&[0, 1, 2, 3], &[10.0; 4])],
        );

        for nbits in [1, 4, 8] {
            let nbits_vec = vec![nbits; 4];
            let q_pc = DotQuantizer::train(&td, &nbits_vec);
            let q_global = VariableBitUniformSparseQuantizer::<u16, DotProduct>::train(
                &td, 0.0, 1.0, nbits,
            );

            assert_eq!(q_pc.quants().len(), q_global.quants().len());
            for (a, b) in q_pc.quants().iter().zip(q_global.quants()) {
                assert!(
                    (a - b).abs() < 1e-10,
                    "nbits={nbits}: quants differ {a} vs {b}"
                );
            }

            let doc = SparseVectorView::new(&[0_u16, 1, 2, 3], &[5.0_f32, 7.0, 9.0, 1.0]);
            let enc_pc =
                <DotQuantizer as SparseVectorEncoder>::encode_vector(&q_pc, doc);
            let enc_global = <VariableBitUniformSparseQuantizer<u16, DotProduct> as SparseVectorEncoder>::encode_vector(
                &q_global, doc,
            );
            assert_eq!(
                enc_pc.values(),
                enc_global.values(),
                "nbits={nbits}: encoded values differ"
            );
        }
    }

    #[test]
    fn mixed_bits_clamp_per_component() {
        // Component 0 has nbits=4 (max_val=15), component 1 has nbits=8 (max_val=255).
        // After training on a max-1.0 distribution, encoding 1.0 should hit the
        // respective max_val per component.
        let td = build_training_data(
            2,
            &[(&[0, 1], &[0.0, 0.0]), (&[0, 1], &[1.0, 1.0])],
        );
        let q = DotQuantizer::train(&td, &[4, 8]);

        let enc = <DotQuantizer as SparseVectorEncoder>::encode_vector(
            &q,
            SparseVectorView::new(&[0_u16, 1], &[1.0_f32, 1.0]),
        );
        // Both should be near their max_val (allow off-by-one due to FP rounding).
        assert!(
            enc.values()[0] >= 14,
            "component 0 (4-bit) should encode 1.0 near max_val=15, got {}",
            enc.values()[0]
        );
        assert!(
            enc.values()[1] >= 254,
            "component 1 (8-bit) should encode 1.0 near max_val=255, got {}",
            enc.values()[1]
        );
    }

    #[test]
    fn mixed_bits_negative_clamps_to_zero() {
        let td = build_training_data(
            2,
            &[(&[0, 1], &[0.0, 0.0]), (&[0, 1], &[1.0, 1.0])],
        );
        let q = DotQuantizer::train(&td, &[4, 8]);

        let enc = <DotQuantizer as SparseVectorEncoder>::encode_vector(
            &q,
            SparseVectorView::new(&[0_u16, 1], &[-2.0_f32, -3.0]),
        );
        assert_eq!(enc.values(), &[0u8, 0]);
    }

    #[test]
    fn opposite_bit_maps_yield_different_search_scores() {
        // End-to-end check: build a tiny SparseDataset, run search, verify that
        // swapping which component is 4-bit vs 8-bit changes the dot products.
        // This is the smoking-gun test for the bench-binary regression where
        // [4, 8] and [8, 4] gave identical Recall@10.
        use crate::core::dataset::{Dataset, DatasetGrowable, ScoredVector};
        use crate::SparseDataset;
        use crate::datasets::sparse_dataset::SparseDatasetGrowable;

        // Component 0 has a small max (~1.0) — 4-bit has step ~0.067, 8-bit has step ~0.004.
        // Component 1 has a large max (~100.0) — 4-bit has step ~6.67, 8-bit has step ~0.39.
        // The reconstruction error magnitudes are very asymmetric, so swapping
        // bit allocations *must* change the dot product.
        let td = build_training_data(
            2,
            &[
                (&[0, 1], &[0.0, 0.0]),
                (&[0, 1], &[1.0, 100.0]),
            ],
        );

        let docs: Vec<(&[u16], &[f32])> = vec![
            (&[0_u16, 1], &[0.7_f32, 73.4]),
            (&[0_u16, 1], &[0.3_f32, 51.1]),
            (&[0_u16, 1], &[0.9_f32, 12.6]),
        ];

        let q_4_8 = DotQuantizer::train(&td, &[4, 8]);
        let q_8_4 = DotQuantizer::train(&td, &[8, 4]);

        let build = |q: PerComponentVariableBitUniformSparseQuantizer<u16, DotProduct>|
                    -> SparseDataset<PerComponentVariableBitUniformSparseQuantizer<u16, DotProduct>>
        {
            let mut g = SparseDatasetGrowable::new(q);
            for &(c, v) in &docs {
                g.push(SparseVectorView::new(c, v));
            }
            g.into()
        };

        let ds_4_8 = build(q_4_8);
        let ds_8_4 = build(q_8_4);

        let query = SparseVectorView::new(&[0_u16, 1], &[0.5_f32, 0.5]);

        let r_4_8: Vec<ScoredVector<DotProduct>> = ds_4_8.search(query, 3);
        let r_8_4: Vec<ScoredVector<DotProduct>> = ds_8_4.search(query, 3);

        // The two configurations *must* produce at least one differing score —
        // otherwise the per-component bit map is being silently ignored.
        let mut any_diff = false;
        for i in 0..3 {
            let s48 = r_4_8[i].distance.distance();
            let s84 = r_8_4[i].distance.distance();
            if (s48 - s84).abs() > 1e-6 {
                any_diff = true;
            }
        }
        assert!(
            any_diff,
            "[4,8] and [8,4] gave identical search scores — per-component bit map ignored. \
             4_8 scores: {:?} | 8_4 scores: {:?}",
            r_4_8.iter().map(|s| s.distance.distance()).collect::<Vec<_>>(),
            r_8_4.iter().map(|s| s.distance.distance()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn decode_reconstructs_within_per_component_step() {
        let td = build_training_data(
            2,
            &[(&[0, 1], &[0.0, 0.0]), (&[0, 1], &[10.0, 10.0])],
        );
        let q = DotQuantizer::train(&td, &[4, 8]);

        let input = SparseVectorView::new(&[0_u16, 1], &[3.7_f32, 8.2]);
        let enc = <DotQuantizer as SparseVectorEncoder>::encode_vector(&q, input);
        let dec = <DotQuantizer as SparseDataEncoder>::decode_vector(
            &q,
            SparseVectorView::new(enc.components(), enc.values()),
        );

        let step_4bit = 10.0 / 15.0;
        let step_8bit = 10.0 / 255.0;
        assert!(
            (input.values()[0] - dec.values()[0]).abs() <= step_4bit + 1e-5,
            "4-bit: orig={}, got={}, step={step_4bit}",
            input.values()[0],
            dec.values()[0]
        );
        assert!(
            (input.values()[1] - dec.values()[1]).abs() <= step_8bit + 1e-5,
            "8-bit: orig={}, got={}, step={step_8bit}",
            input.values()[1],
            dec.values()[1]
        );
    }
}
