use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::simd::num::SimdFloat;
use std::simd::{Simd, StdFloat};

use crate::core::sealed;
use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::encoders::dotpacking8::mapping::ComponentMapping;
use crate::encoders::dotpacking8::common::{
    GapDecoder, compute_safe_simd_padding, encode_blocks, fast_lane_gather, simd_prefix_sum,
};
use crate::encoders::dotpacking8::values::{U8Reader, ValueReader};
use crate::encoders::dotpacking8::encoder::DotPacking8View;
use crate::utils::train_sparse_scalar_quantizer;
use crate::{PlainSparseDataset, ScalarSparseSupportedDistance, SpaceUsage};

use rusty_perm::PermApply as _;
use rusty_perm::PermFromSorting as _;

const N: usize = 8;

/// Bridged block-8 encoder for sparse vectors with `u32` components and scalar
/// quantized `u8` values.
///
/// Mirrors `OptimisticDotVByteScalarU8Encoder` (on `feature/quantization`) but emits the
/// block-8 packed format from [`crate::encoders::dotpacking8`] instead of the VByte format.
/// The block-8 encoding stores per-block gap bit-widths in a 4-bit selector field, so
/// individual gaps must fit in 16 bits (`b <= 16`). To support `u32` components, this encoder
/// injects synthetic zero-valued "bridge" coordinates whenever a gap (including the first,
/// relative to 0) would exceed `u16::MAX`, keeping the encoding on the fast SIMD path.
///
/// # Invariant: code 0 marks a bridge
///
/// A real coordinate never quantizes to code 0: every input value is assumed to be at least
/// its component's quantization step (`max / 255` over the training data), so its code is in
/// `1..=255`. Code 0 is therefore reserved for bridges, which score zero and are removed by
/// `decode_vector` by their code alone. This holds for learned sparse embeddings with
/// strictly positive values whose components all occur in the training data (checked on
/// SPLARE: no entry quantizes to 0); it is checked by a `debug_assert!` in `push_encoded`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BridgedDotPacking8U32ScalarU8Encoder {
    dim: usize,
    quants: Box<[f32]>,
    mapping: Option<ComponentMapping<u32>>,
}

impl sealed::Sealed for BridgedDotPacking8U32ScalarU8Encoder {}

impl BridgedDotPacking8U32ScalarU8Encoder {
    #[inline]
    pub fn new(input_dim: usize) -> Self {
        Self {
            dim: input_dim,
            quants: vec![0.0_f32; input_dim].into_boxed_slice(),
            mapping: None,
        }
    }

    #[inline]
    pub fn component_mapping(&self) -> Option<&[u32]> {
        self.mapping.as_ref().map(|m| m.forward())
    }

    #[inline]
    pub fn inverse_component_mapping(&self) -> Option<&[u32]> {
        self.mapping.as_ref().map(|m| m.inverse())
    }


    /// Reorder components by decreasing frequency in `training_data`, which shortens gaps,
    /// then fit the per-component `u8` steps. The dataset's distance does not matter.
    pub fn train<D>(&mut self, training_data: &PlainSparseDataset<u32, f32, D>)
    where
        D: ScalarSparseSupportedDistance,
    {
        self.mapping = Some(ComponentMapping::by_decreasing_frequency(
            self.input_dim(),
            training_data,
        ));

        self.quants = train_sparse_scalar_quantizer(training_data, 0.0, 1.0).into_boxed_slice();
    }

    /// Inserts synthetic coordinates ("bridges") with code 0 so that every gap, including the
    /// first one (relative to 0), is at most `u16::MAX`. Each bridge sits `u16::MAX` after the
    /// previous coordinate. [`Self::drop_bridges`] recognises them by code 0 alone, which is
    /// reserved for bridges (see the type-level invariant).
    /// Components must be sorted ascending; real coordinates are preserved as-is.
    fn inject_zero_bridges(components: &[u32], values: &[u8]) -> (Vec<u32>, Vec<u8>) {
        assert_eq!(
            components.len(),
            values.len(),
            "components and values length mismatch"
        );

        if components.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let mut bridged_components = Vec::with_capacity(components.len());
        let mut bridged_values = Vec::with_capacity(values.len());

        // The first gap is relative to 0 (see `push_encoded`), so it is bridged as well.
        let mut last = 0u32;
        for (&component, &value) in components.iter().zip(values.iter()) {
            while component - last > u16::MAX as u32 {
                last += u16::MAX as u32;
                bridged_components.push(last);
                bridged_values.push(0);
            }
            bridged_components.push(component);
            bridged_values.push(value);
            last = component;
        }

        (bridged_components, bridged_values)
    }

    /// Removes the bridges inserted by [`Self::inject_zero_bridges`] from decoded components:
    /// the entries with code 0, which real coordinates never have (see the type-level
    /// invariant).
    fn drop_bridges(components: &mut Vec<u32>, values: &mut Vec<u8>) {
        let mut kept = 0;
        for i in 0..components.len() {
            if values[i] != 0 {
                components[kept] = components[i];
                values[kept] = values[i];
                kept += 1;
            }
        }
        components.truncate(kept);
        values.truncate(kept);
    }
}

impl SparseDataEncoder for BridgedDotPacking8U32ScalarU8Encoder {
    type InputComponentType = u32;
    type InputValueType = f32;
    type OutputComponentType = u32;
    type OutputValueType = u8;

    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let view = unsafe { DotPacking8View::from_unchecked_slice(encoded.data()) };
        let n = view.n;

        let mut decoded_comp: Vec<u32> = Vec::with_capacity(n);
        let mut values_raw: Vec<u8> = Vec::with_capacity(n);

        // SAFETY: the view comes from `push_encoded` output, invariants (1)-(3) in `common.rs`.
        let mut gaps = unsafe { GapDecoder::new(view.n, view.selectors, view.payloads) };
        let mut values = U8Reader::new(view.values);
        let mut last_component = 0u32;
        while let Some(block_gaps) = gaps.next_block() {
            let abs = (simd_prefix_sum(block_gaps) + Simd::splat(last_component)).to_array();
            let codes = values.next_block_codes();
            for i in 0..N {
                decoded_comp.push(abs[i]);
                values_raw.push(codes[i]);
            }
            last_component = abs[N - 1];
        }
        let (tail_gaps, rem) = gaps.tail();
        let abs = (simd_prefix_sum(tail_gaps) + Simd::splat(last_component)).to_array();
        let codes = values.tail(rem);
        for i in 0..rem {
            decoded_comp.push(abs[i]);
            values_raw.push(codes[i]);
        }

        Self::drop_bridges(&mut decoded_comp, &mut values_raw);

        if let Some(mapping) = &self.mapping {
            let inverse = mapping.inverse();

            for c in decoded_comp.iter_mut() {
                *c = inverse[*c as usize];
            }

            // Decode AFTER inverse mapping so each raw byte uses its original
            // component's scalar quantization step. Permute raw bytes along with
            // components to keep alignment.
            let permutation = rusty_perm::PermD::from_sort(decoded_comp.as_slice());
            permutation.apply(values_raw.as_mut_slice()).unwrap();
            permutation.apply(decoded_comp.as_mut_slice()).unwrap();

            let values: Vec<f32> = decoded_comp
                .iter()
                .zip(values_raw.iter())
                .map(|(&c, &v)| v as f32 * self.quants[c as usize])
                .collect();

            SparseVectorOwned::new(decoded_comp, values)
        } else {
            let values: Vec<f32> = decoded_comp
                .iter()
                .zip(values_raw.iter())
                .map(|(&c, &v)| v as f32 * self.quants[c as usize])
                .collect();
            SparseVectorOwned::new(decoded_comp, values)
        }
    }
}

impl PackedSparseVectorEncoder for BridgedDotPacking8U32ScalarU8Encoder {
    type PackedDataType = u64;

    fn push_encoded<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, Self::InputComponentType, Self::InputValueType>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<Self::PackedDataType>,
    {
        // Quantize values using each component's ORIGINAL-index scale.
        let mut q_values: Vec<u8> = input
            .components()
            .iter()
            .zip(input.values())
            .map(|(&c, &v)| {
                let q = self.quants[c as usize];
                if q > 0.0 {
                    (v / q).clamp(0.0, 255.0) as u8
                } else {
                    0u8
                }
            })
            .collect();
        debug_assert!(
            q_values.iter().all(|&code| code != 0),
            "a real coordinate quantized to code 0, which is reserved for bridges: every value \
             must be at least its component's quantization step"
        );

        let mut q_components: Vec<u32> = if let Some(mapping) = self.component_mapping() {
            input
                .components()
                .iter()
                .map(|&c| mapping[c as usize])
                .collect()
        } else {
            input.components().to_vec()
        };

        if self.mapping.is_some() {
            let permutation = rusty_perm::PermD::from_sort(q_components.as_slice());
            permutation.apply(q_values.as_mut_slice()).unwrap();
            permutation.apply(q_components.as_mut_slice()).unwrap();
        }

        let (bridged_components, bridged_values) =
            Self::inject_zero_bridges(&q_components, &q_values);

        let n = bridged_components.len();
        if n == 0 {
            return;
        }
        assert!(
            n < u16::MAX as usize,
            "BridgedDotPacking8U32ScalarU8Encoder only supports vectors (after bridging) shorter than 65535, got {n}"
        );

        let mut gaps: Vec<u32> = Vec::with_capacity(n);
        let mut last = 0u32;
        for &comp in &bridged_components {
            let g = comp - last;
            assert!(
                g <= u16::MAX as u32,
                "gap {g} exceeds u16::MAX; bridge insertion failed"
            );
            gaps.push(g);
            last = comp;
        }

        let (selectors, gap_payloads) = encode_blocks(&gaps);
        let mut payload = Vec::new();
        payload.extend_from_slice(&(n as u16).to_le_bytes());
        payload.extend_from_slice(&selectors);
        payload.extend_from_slice(&gap_payloads);

        let current_len = payload.len();
        let padding_needed =
            compute_safe_simd_padding(n, &selectors, (8 - (current_len + n) % 8) % 8, n);
        payload.resize(payload.len() + padding_needed, 0);
        payload.extend_from_slice(bridged_values.as_slice());

        let data = payload
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&chunk| u64::from_le_bytes(chunk));
        output.extend(data);
    }
}

impl VectorEncoder for BridgedDotPacking8U32ScalarU8Encoder {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u32, f32>;
    type QueryVector<'q> = SparseVectorView<'q, u32, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;

    type Evaluator<'e>
        = BridgedDotPacking8U32ScalarU8QueryEvaluator<'e>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        BridgedDotPacking8U32ScalarU8QueryEvaluator::new(query, self)
    }

    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        BridgedDotPacking8U32ScalarU8QueryEvaluator::new(decoded.as_view(), self)
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
pub struct BridgedDotPacking8U32ScalarU8QueryEvaluator<'e> {
    dense_query_transformed: Vec<f32>,
    _phantom: PhantomData<&'e ()>,
}

impl<'e> BridgedDotPacking8U32ScalarU8QueryEvaluator<'e> {
    pub fn new(
        query: SparseVectorView<'_, u32, f32>,
        encoder: &'e BridgedDotPacking8U32ScalarU8Encoder,
    ) -> Self {
        assert_eq!(
            query.components().len(),
            query.values().len(),
            "Query vector components and values length mismatch."
        );

        let component_mapping = encoder.component_mapping();
        let mut transformed = vec![0.0f32; encoder.input_dim()];
        for (&c, &v) in query.components().iter().zip(query.values()) {
            let idx = c as usize;
            assert!(
                idx < encoder.input_dim(),
                "Query vector component exceeds encoder input dimension."
            );
            let mapped: usize = match component_mapping {
                Some(mapping) => mapping[idx] as usize,
                None => idx,
            };
            transformed[mapped] = v * encoder.quants[idx];
        }

        Self {
            dense_query_transformed: transformed,
            _phantom: PhantomData,
        }
    }
}

impl<'e, 'v> QueryEvaluator<PackedVectorView<'v, u64>>
    for BridgedDotPacking8U32ScalarU8QueryEvaluator<'e>
{
    type Distance = DotProduct;

    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> Self::Distance {
        let view = unsafe { DotPacking8View::from_unchecked_slice(vector.data()) };
        // SAFETY: the view comes from `push_encoded` output, invariants (1)-(3) in `common.rs`.
        let mut gaps = unsafe { GapDecoder::new(view.n, view.selectors, view.payloads) };
        let mut values = U8Reader::new(view.values);
        let mut acc = Simd::<f32, N>::splat(0.0);
        let mut query_ptr = self.dense_query_transformed.as_slice();
        while let Some(block_gaps) = gaps.next_block() {
            let components = simd_prefix_sum(block_gaps);
            // SAFETY: every relative component is < query_ptr.len(), invariant (5) in `common.rs`.
            let query_values = unsafe { fast_lane_gather(query_ptr, components) };
            acc = query_values.mul_add(values.next_block(), acc);
            let last_component = components[N - 1];
            debug_assert!(last_component as usize <= query_ptr.len());
            // SAFETY: `last_component <= query_ptr.len()` by invariant (5) in `common.rs`.
            query_ptr = unsafe { query_ptr.split_at_unchecked(last_component as usize).1 };
        }
        let mut total = acc.reduce_sum();
        let (tail_gaps, rem) = gaps.tail();
        if rem > 0 {
            let components = simd_prefix_sum(tail_gaps).to_array();
            total = values.tail_dot(total, query_ptr, &components, rem);
        }
        DotProduct(total)
    }
}

impl SpaceUsage for BridgedDotPacking8U32ScalarU8Encoder {
    fn space_usage_bytes(&self) -> usize {
        let mapping_size = self.mapping.as_ref().map_or(0, |m| m.space_usage_bytes());
        let quants_size = self.quants.len() * std::mem::size_of::<f32>();
        mapping_size + quants_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SquaredEuclideanDistance;
    use crate::PlainSparseDatasetGrowable;
    use crate::core::dataset::DatasetGrowable;
    use crate::core::vector::{PackedVectorView, SparseVectorView};

    fn build_training_data(
        dim: usize,
        vectors: &[(&[u32], &[f32])],
    ) -> PlainSparseDataset<u32, f32, SquaredEuclideanDistance> {
        let encoder = crate::encoders::sparse_scalar::PlainSparseQuantizer::<
            u32,
            f32,
            SquaredEuclideanDistance,
        >::new(dim, dim);
        let mut growable = PlainSparseDatasetGrowable::new(encoder);
        for &(components, values) in vectors {
            growable.push(SparseVectorView::new(components, values));
        }
        growable.into()
    }

    #[test]
    fn new_creates_encoder() {
        let encoder = BridgedDotPacking8U32ScalarU8Encoder::new(100);
        assert_eq!(encoder.input_dim(), 100);
        assert_eq!(encoder.output_dim(), 100);
        assert!(encoder.component_mapping().is_none());
        assert!(encoder.inverse_component_mapping().is_none());
    }

    #[test]
    fn training_sets_component_mapping_and_quants() {
        let mut encoder = BridgedDotPacking8U32ScalarU8Encoder::new(5);
        let td = build_training_data(
            5,
            &[
                (
                    &[0_u32, 1, 2, 3, 4],
                    &[255.0_f32, 255.0, 255.0, 255.0, 255.0],
                ),
                (&[0_u32, 2], &[128.0_f32, 64.0]),
            ],
        );
        encoder.train(&td);

        let mapping = encoder.component_mapping().unwrap();
        let inverse = encoder.inverse_component_mapping().unwrap();
        assert_eq!(mapping.len(), 5);
        assert_eq!(inverse.len(), 5);
        for &v in mapping {
            assert!(v < 5);
        }
        for &v in inverse {
            assert!(v < 5);
        }
        assert_eq!(encoder.quants.len(), 5);
    }

    #[test]
    fn inject_zero_bridges_no_gap() {
        let components = vec![0u32, 1, 2, 3];
        let values = vec![10u8, 20, 30, 40];
        let (bc, bv) =
            BridgedDotPacking8U32ScalarU8Encoder::inject_zero_bridges(&components, &values);
        assert_eq!(bc, components);
        assert_eq!(bv, values);
    }

    #[test]
    fn inject_zero_bridges_large_gap() {
        let gap = u16::MAX as u32 + 100;
        let components = vec![0u32, gap];
        let values = vec![10u8, 20];
        let (bc, bv) =
            BridgedDotPacking8U32ScalarU8Encoder::inject_zero_bridges(&components, &values);
        assert!(bc.len() > 2);
        assert_eq!(*bc.last().unwrap(), gap);
        assert_eq!(*bv.last().unwrap(), 20);
        for &v in &bv[1..bv.len() - 1] {
            assert_eq!(v, 0);
        }
        for window in bc.windows(2) {
            assert!(window[1] - window[0] <= u16::MAX as u32);
        }
    }

    #[test]
    fn encode_decode_preserves_structure() {
        let mut encoder = BridgedDotPacking8U32ScalarU8Encoder::new(10);
        let td = build_training_data(
            10,
            &[
                (&[0_u32, 1, 2, 3], &[255.0_f32, 255.0, 255.0, 255.0]),
                (&[0_u32, 5], &[128.0_f32, 64.0]),
            ],
        );
        encoder.train(&td);

        let values = [1.0_f32, 2.0_f32];
        let input = SparseVectorView::new(&[0_u32, 5], &values);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);
        assert!(!buffer.is_empty());

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &[0_u32, 5]);
        assert_eq!(decoded.values().len(), 2);
        for (&decoded_v, &original_v) in decoded.values().iter().zip(values.iter()) {
            assert!(
                (decoded_v - original_v).abs() <= 1.01,
                "decoded {decoded_v} too far from original {original_v}"
            );
        }
    }

    /// dim=4, quants trained to 1.0 → encoded u8 == value, decoded == value.
    /// dot = 100*2 + 150*3 + 200*1 = 850.
    #[test]
    fn encode_decode_dot_product_exact() {
        let mut encoder = BridgedDotPacking8U32ScalarU8Encoder::new(4);
        let td = build_training_data(4, &[(&[0_u32, 1, 2, 3], &[255.0_f32, 255.0, 255.0, 255.0])]);
        encoder.train(&td);
        assert!(
            encoder.quants.iter().all(|&q| (q - 1.0).abs() < 1e-6),
            "expected quants == 1.0 for all components"
        );

        let vec_values = [100.0_f32, 150.0, 200.0];
        let vec_input = SparseVectorView::new(&[0_u32, 1, 2], &vec_values);

        let mut buffer = Vec::new();
        encoder.push_encoded(vec_input, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &[0_u32, 1, 2]);
        for (&decoded_v, &original_v) in decoded.values().iter().zip(vec_values.iter()) {
            assert!(
                (decoded_v - original_v).abs() < 1e-4,
                "decoded {decoded_v} != original {original_v}"
            );
        }

        let query_values = [2.0_f32, 3.0, 1.0];
        let query = SparseVectorView::new(&[0_u32, 1, 2], &query_values);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
        assert!(
            (dist.0 - 850.0).abs() < 1e-2,
            "expected dot product 850.0, got {}",
            dist.0
        );
    }

    /// Exercise the bridging path: a `(0, u16::MAX + 100)` pair forces at least one
    /// synthetic zero bridge. The dot product must still match the analytical value
    /// because bridges contribute zero by construction.
    #[test]
    fn dot_product_with_bridged_large_gap() {
        let dim = (u16::MAX as usize) + 200;
        let mut encoder = BridgedDotPacking8U32ScalarU8Encoder::new(dim);
        let gap = u16::MAX as u32 + 100;
        let comps = [0_u32, gap];
        let td = build_training_data(dim, &[(&comps, &[255.0_f32, 255.0])]);
        encoder.train(&td);

        let values = [100.0_f32, 200.0];
        let input = SparseVectorView::new(&comps, &values);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        // Analytical: quants for both trained components are 1.0, so dot is exact.
        // dot = 100*2 + 200*5 = 200 + 1000 = 1200.
        let query_values = [2.0_f32, 5.0];
        let query = SparseVectorView::new(&comps, &query_values);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
        assert!(
            (dist.0 - 1200.0).abs() < 1e-2,
            "expected dot product 1200.0, got {}",
            dist.0
        );
    }

    /// An encoder with no component mapping and a unit step for every component, so
    /// encoded codes equal the (integer) input values.
    fn unit_step_encoder(dim: usize) -> BridgedDotPacking8U32ScalarU8Encoder {
        let mut encoder = BridgedDotPacking8U32ScalarU8Encoder::new(dim);
        encoder.quants = vec![1.0_f32; dim].into_boxed_slice();
        encoder
    }

    #[test]
    fn inject_zero_bridges_first_component_beyond_u16() {
        // The first gap is relative to 0, so it must be bridged as well.
        let components = vec![u16::MAX as u32 * 2 + 10];
        let (bc, bv) =
            BridgedDotPacking8U32ScalarU8Encoder::inject_zero_bridges(&components, &[7u8]);
        assert_eq!(bc, vec![u16::MAX as u32, 2 * u16::MAX as u32, components[0]]);
        assert_eq!(bv, vec![0, 0, 7]);
    }

    /// A vector whose smallest component exceeds `u16::MAX` must encode, decode to exactly
    /// its own components (no bridges), and score correctly.
    #[test]
    fn first_component_beyond_u16_roundtrip_and_dot() {
        let dim = 200_000;
        let encoder = unit_step_encoder(dim);
        let comps = [70_000_u32, 70_010, 140_000, 199_999];
        let values = [10.0_f32, 20.0, 30.0, 40.0];
        let mut buffer = Vec::new();
        encoder.push_encoded(SparseVectorView::new(&comps, &values), &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &comps);
        assert_eq!(decoded.values(), &values);

        let query_values = [1.0_f32, 2.0, 3.0, 4.0];
        let evaluator = encoder.query_evaluator(SparseVectorView::new(&comps, &query_values));
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
        assert!((dist.0 - 300.0).abs() < 1e-3, "expected 300, got {}", dist.0);
    }

    /// Bridges are an encoding artifact: decoding returns only the input components, also
    /// across a bulk block (> 8 entries after bridging) and with several bridges in a row.
    #[test]
    fn decode_drops_bridges() {
        let dim = 400_000;
        let encoder = unit_step_encoder(dim);
        let comps: Vec<u32> = vec![0, 3, 200_000, 200_001, 200_002, 200_003, 200_004, 399_999];
        let values: Vec<f32> = (1..=comps.len()).map(|i| i as f32).collect();
        let mut buffer = Vec::new();
        encoder.push_encoded(SparseVectorView::new(&comps, &values), &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), comps.as_slice());
        assert_eq!(decoded.values(), values.as_slice());
    }

    /// A genuine component exactly `u16::MAX` after the previous one needs no bridge and must
    /// be kept by the decoder.
    #[test]
    fn decode_keeps_real_component_at_max_gap() {
        let dim = 200_000;
        let encoder = unit_step_encoder(dim);
        let comps = [5_u32, 5 + u16::MAX as u32];
        let values = [3.0_f32, 4.0];
        let mut buffer = Vec::new();
        encoder.push_encoded(SparseVectorView::new(&comps, &values), &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &comps);
        assert_eq!(decoded.values(), &values);
    }

    /// Vector length not a multiple of 8 → exercises `decode_tail`.
    #[test]
    fn bulk_and_tail_blocks() {
        let dim = 256;
        let mut encoder = BridgedDotPacking8U32ScalarU8Encoder::new(dim);
        let comps: Vec<u32> = (0_u32..10).map(|i| i * 5).collect();
        let train_values: Vec<f32> = (0..10).map(|_| 255.0).collect();
        let td = build_training_data(dim, &[(&comps, &train_values)]);
        encoder.train(&td);

        let values: Vec<f32> = (0..10).map(|i| 10.0 + i as f32 * 7.0).collect();
        let input = SparseVectorView::new(&comps, &values);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let query_values: Vec<f32> = (0..10).map(|i| 1.0 + i as f32 * 0.5).collect();
        let query = SparseVectorView::new(&comps, &query_values);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        // Reference: dot product over the (quantized) values.
        let mut expected = 0.0_f32;
        for (i, &c) in comps.iter().enumerate() {
            let q = encoder.quants[c as usize];
            let encoded_u8 = if q > 0.0 {
                (values[i] / q).clamp(0.0, 255.0) as u8
            } else {
                0u8
            };
            let decoded_v = encoded_u8 as f32 * q;
            expected += decoded_v * query_values[i];
        }
        assert!(
            (dist.0 - expected).abs() < 1e-2,
            "dot {} != expected {}",
            dist.0,
            expected
        );
    }
}
