use bytemuck::cast_slice;
use half::f16;
use std::simd::{Simd, StdFloat, prelude::*};
use rusty_perm::{PermApply as _, PermFromSorting as _};

use crate::{PlainSparseDataset, ScalarSparseSupportedDistance, SpaceUsage};
use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::encoders::dotpacking8::mapping::ComponentMapping;
use crate::encoders::dotpacking8::common::{
    simd_prefix_sum, fast_lane_gather, encode_blocks, compute_safe_simd_padding, DotPacking8F16Iter
};

#[derive(Clone)]
pub struct DotPacking8U16F16View<'a> {
    pub n: usize,
    pub selectors: &'a [u8],
    pub payloads: &'a [u8],
    pub values: &'a [f16],
}

impl<'a> DotPacking8U16F16View<'a> {
    /// Split one encoded vector into its header, selectors, gap payload and values.
    ///
    /// # Safety
    ///
    /// `slice` must be exactly one vector produced by this encoder's `push_encoded`. Decoders
    /// built from the view read the gap payload with unchecked 16-byte loads whose bounds rely
    /// on the padding `push_encoded` inserts before the values.
    pub unsafe fn from_unchecked_slice(slice: &'a [u64]) -> Self {
        let bytes = cast_slice::<u64, u8>(slice);
        let n = bytemuck::from_bytes::<u16>(&bytes[0..2]).to_le() as usize;

        let n_blocks_total = n.div_ceil(8);
        let selectors_size = n_blocks_total.div_ceil(2);
        let selectors_end = 2 + selectors_size;
        let values_start_idx = bytes.len() - 2 * n; // 2 bytes per f16

        Self {
            n,
            selectors: &bytes[2..selectors_end],
            payloads: &bytes[selectors_end..values_start_idx],
            values: cast_slice::<u8, f16>(&bytes[values_start_idx..]),
        }
    }

    pub fn iter_raw(&self) -> DotPacking8F16Iter<'a> {
        DotPacking8F16Iter {
            n: self.n,
            selectors: self.selectors,
            bulk_blocks: self.n / 8,
            block_idx: 0,
            payload_ptr: self.payloads.as_ptr(),
            val_ptr: self.values.as_ptr(),
        }
    }
}


#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DotPacking8U16F16Encoder {
    pub dim: usize,
    pub mapping: Option<ComponentMapping<u16>>,
}

impl DotPacking8U16F16Encoder {
    pub fn new(input_dim: usize) -> Self {
        Self {
            dim: input_dim,
            mapping: None,
        }
    }

    #[inline]
    pub fn component_mapping(&self) -> Option<&[u16]> {
        self.mapping.as_ref().map(|m| m.forward())
    }


    #[inline]
    pub fn inverse_component_mapping(&self) -> Option<&[u16]> {
        self.mapping.as_ref().map(|m| m.inverse())
    }

    /// Reorder components by decreasing frequency in `training_data`, which shortens gaps.
    /// The dataset's distance does not matter.
    pub fn train<D>(&mut self, training_data: &PlainSparseDataset<u16, f32, D>)
    where
        D: ScalarSparseSupportedDistance,
    {
        self.mapping = Some(ComponentMapping::by_decreasing_frequency(
            self.input_dim(),
            training_data,
        ));
    }
}

impl SpaceUsage for DotPacking8U16F16Encoder {
    fn space_usage_bytes(&self) -> usize {
        let mapping_size = self.mapping.as_ref().map_or(0, |m| m.space_usage_bytes());
        mapping_size
    }
}

impl VectorEncoder for DotPacking8U16F16Encoder {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u16, f32>;
    type QueryVector<'q> = SparseVectorView<'q, u16, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;
    type Evaluator<'e> = DotPacking8U16F16QueryEvaluator<'e>;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        DotPacking8U16F16QueryEvaluator::new(query, self)
    }

    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        DotPacking8U16F16QueryEvaluator::new(decoded.as_view(), self)
    }

    fn output_dim(&self) -> usize {
        self.dim
    }

    fn input_dim(&self) -> usize {
        self.dim
    }
}

impl SparseDataEncoder for DotPacking8U16F16Encoder {
    type InputComponentType = u16;
    type InputValueType = f32;
    type OutputComponentType = u16;
    type OutputValueType = f16;

    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let view = unsafe { DotPacking8U16F16View::from_unchecked_slice(encoded.data()) };
        let n = view.n;

        let mut decoded_comp = Vec::with_capacity(n);
        let mut values_raw = Vec::with_capacity(n);

        let mut iter = view.iter_raw();
        let mut last_component = 0u32;

        for (gaps, vals) in &mut iter {
            let components = simd_prefix_sum(gaps);
            let absolute_components = components + Simd::splat(last_component);
            let absolute_arr = absolute_components.to_array();
            let vals_arr = vals.to_array();
            for i in 0..8 {
                decoded_comp.push(absolute_arr[i] as u16);
                values_raw.push(vals_arr[i]);
            }
            last_component = absolute_arr[8 - 1];
        }

        let (gaps, tail_values, remaining) = iter.decode_tail();
        let components = simd_prefix_sum(gaps);
        let absolute_components = components + Simd::splat(last_component);
        let absolute_arr = absolute_components.to_array();
        for i in 0..remaining {
            decoded_comp.push(absolute_arr[i] as u16);
            values_raw.push(tail_values[i].to_f32());
        }

        if let Some(mapping) = &self.mapping {
            let inverse = mapping.inverse();

            for c in decoded_comp.iter_mut() {
                *c = inverse[*c as usize];
            }

            let permutation = rusty_perm::PermD::from_sort(decoded_comp.as_slice());
            permutation.apply(values_raw.as_mut_slice()).unwrap();
            permutation.apply(decoded_comp.as_mut_slice()).unwrap();

            SparseVectorOwned::new(decoded_comp, values_raw)
        } else {
            SparseVectorOwned::new(decoded_comp, values_raw)
        }
    }
}

impl PackedSparseVectorEncoder for DotPacking8U16F16Encoder {
    type PackedDataType = u64;

    fn push_encoded<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, u16, f32>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<u64>,
    {
        let mut q_values: Vec<f16> = input
            .values()
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();

        let mut q_components: Vec<u16> = if let Some(mapping) = self.component_mapping() {
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

        let n = q_components.len();
        if n == 0 {
            return;
        }

        let mut gaps = Vec::with_capacity(n);
        let mut last = 0u32;
        for &comp in &q_components {
            gaps.push(comp as u32 - last);
            last = comp as u32;
        }

        let (selectors, gap_payloads) = encode_blocks(&gaps);
        let mut payload = Vec::new();
        payload.extend_from_slice(&(n as u16).to_le_bytes());
        payload.extend_from_slice(&selectors);
        payload.extend_from_slice(&gap_payloads);

        let current_len = payload.len();
        // alignment_padding ensures that the total payload length (metadata + selectors + gap payloads + padding + values)
        // is a multiple of 8 bytes (so it can be chunked into u64) and that the f16 values are aligned to the end of the [u64] slice.
        let alignment_padding = (8 - (current_len + 2 * n) % 8) % 8;
        let padding_needed =
            compute_safe_simd_padding(n, &selectors, alignment_padding, 2 * n);

        payload.resize(payload.len() + padding_needed, 0);
        payload.extend_from_slice(cast_slice::<f16, u8>(q_values.as_slice()));

        let data = payload
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&chunk| u64::from_le_bytes(chunk));

        output.extend(data);
    }
}

pub struct DotPacking8U16F16QueryEvaluator<'a> {
    dense_query: Vec<f32>,
    _encoder: &'a DotPacking8U16F16Encoder,
}

impl<'a> DotPacking8U16F16QueryEvaluator<'a> {
    pub fn new(query: SparseVectorView<'_, u16, f32>, encoder: &'a DotPacking8U16F16Encoder) -> Self {
        let mut dense_query = vec![0.0f32; encoder.dim];
        for (&c, &v) in query.components().iter().zip(query.values().iter()) {
            let mapped_c = if let Some(mapping) = encoder.component_mapping() {
                mapping[c as usize]
            } else {
                c
            };
            dense_query[mapped_c as usize] = v;
        }
        Self {
            dense_query,
            _encoder: encoder,
        }
    }
}

impl<'a, 'v> QueryEvaluator<PackedVectorView<'v, u64>> for DotPacking8U16F16QueryEvaluator<'a> {
    type Distance = DotProduct;

    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> Self::Distance {
        let view = unsafe { DotPacking8U16F16View::from_unchecked_slice(vector.data()) };
        let mut acc = Simd::<f32, 8>::splat(0.0);
        let mut query_ptr = self.dense_query.as_slice();

        let mut iter = view.iter_raw();
        for (gaps, values) in &mut iter {
            let components = simd_prefix_sum(gaps);
            // SAFETY: every relative component is < query_ptr.len(), invariant (5) in `common.rs`.
            let query_values = unsafe { fast_lane_gather(query_ptr, components) };

            acc = query_values.mul_add(values, acc);
            let last_component = components[7];
            debug_assert!(last_component as usize <= query_ptr.len());
            // SAFETY: `last_component <= query_ptr.len()` by invariant (5) in `common.rs`.
            query_ptr = unsafe { query_ptr.split_at_unchecked(last_component as usize).1 };
        }
        let mut total_unscaled = acc.reduce_sum();

        let (gaps, tail_values, remaining) = iter.decode_tail();
        if remaining > 0 {
            let components = simd_prefix_sum(gaps);
            let components_arr = components.to_array();
            for i in 0..remaining {
                total_unscaled += query_ptr[components_arr[i] as usize] * tail_values[i].to_f32();
            }
        }
        DotProduct(total_unscaled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::distances::Distance;
    use crate::core::vector::{PackedVectorView, SparseVectorView};
    use crate::vector_encoder::SparseDataEncoder;
    use crate::{
        DatasetGrowable, PlainSparseDatasetGrowable, PlainSparseQuantizer, QueryEvaluator, VectorEncoder,
    };

    fn calculate_expected_distance(
        vector: &SparseVectorView<u16, f32>,
        query: &SparseVectorView<u16, f32>,
    ) -> f32 {
        let mut expected = 0.0f32;
        let mut vec_iter = vector.iter().peekable();

        for (comp, val) in query.iter() {
            while let Some(&(vec_comp, _)) = vec_iter.peek() {
                if vec_comp < comp {
                    vec_iter.next();
                } else {
                    break;
                }
            }

            if let Some(&(vec_comp, vec_val)) = vec_iter.peek() {
                if vec_comp == comp {
                    expected += f16::from_f32(vec_val).to_f32() * val;
                    vec_iter.next();
                }
            }
        }
        expected
    }

    #[test]
    fn compute_distance_only_bulk() {
        let encoder = DotPacking8U16F16Encoder::new(100);

        let binding = [1.0, 3.0, 2.0, 3.5, 1.5, 2.0, 1.0, 2.0];
        let input = SparseVectorView::new(&[2, 4, 8, 24, 36, 48, 53, 90], &binding);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let query = SparseVectorView::new(
            &[2, 4, 6, 8, 24, 70, 90],
            &[0.5, 1.5, 2.5, 1.0, 2.0, 1.0, 2.0],
        );
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let expected = calculate_expected_distance(&input, &query);
        assert!((dist.distance() - expected).abs() < 1e-5);
    }

    #[test]
    fn compute_distance_bulk_and_tail() {
        let encoder = DotPacking8U16F16Encoder::new(100);

        let binding = [1.0, 3.0, 2.0, 3.5, 1.5, 2.0, 1.0, 2.0, 3.0, 2.5];
        let input = SparseVectorView::new(&[2, 4, 8, 24, 28, 36, 48, 53, 70, 90], &binding);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let query = SparseVectorView::new(
            &[2, 4, 8, 24, 28, 70, 90],
            &[0.5, 1.5, 2.5, 1.0, 2.0, 1.0, 2.0],
        );
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let expected = calculate_expected_distance(&input, &query);
        assert!((dist.distance() - expected).abs() < 1e-5);
    }

    fn verify_gaps(gaps: &[u32]) {
        let num_vals = gaps.len();
        let mut components = Vec::new();
        let mut curr = 0u32;
        for &g in gaps {
            curr += g;
            components.push(curr as u16);
        }

        let encoder = DotPacking8U16F16Encoder::new(u16::MAX as usize + 1);

        let values: Vec<_> = (0..num_vals).map(|i| 1.0 + (i % 7) as f32 / 10.0).collect();
        let input = SparseVectorView::new(&components, &values);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let query_vals: Vec<_> = (0..num_vals).map(|i| 0.1 + (i % 5) as f32 / 10.0).collect();
        let query = SparseVectorView::new(&components, &query_vals);

        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let expected = calculate_expected_distance(&input, &query);

        assert!(
            (dist.distance() - expected).abs() < 1e-3,
            "Failed: dist={}, expected={}, gaps_len={}",
            dist.distance(),
            expected,
            num_vals
        );
    }

    #[test]
    fn test_plain_multiple_blocks() {
        let gaps = vec![1u32; 32];
        verify_gaps(&gaps);
    }

    #[test]
    fn test_plain_small_gaps() {
        let gaps = vec![1u32; 1];
        verify_gaps(&gaps);
    }

    #[test]
    fn test_plain_with_tail() {
        let gaps = vec![1u32; 11];
        verify_gaps(&gaps);
    }

    #[test]
    fn test_plain_large_gaps() {
        let gaps = vec![1000u32, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000];
        verify_gaps(&gaps);
    }

    #[test]
    fn test_plain_u16_gaps() {
        let gaps = vec![60000, 1, 1, 2, 3, 200, 450, 2000, 3];
        verify_gaps(&gaps);
    }

    fn same_when_quantized(
        before: &SparseVectorView<u16, f32>,
        after: &SparseVectorView<u16, f32>,
    ) {
        assert_eq!(before.components(), after.components());
        for (v1, v2) in before.values().iter().zip(after.values().iter()) {
            let q1 = f16::from_f32(*v1).to_f32();
            let q2 = f16::from_f32(*v2).to_f32();
            assert_eq!(q1, q2);
        }
    }

    #[test]
    fn block8_decode_roundtrip() {
        let mut encoder = DotPacking8U16F16Encoder::new(100);

        let binding = [1.0, 3.0, 2.0, 3.5];
        let input_components = [0, 4, 8, 24];
        let input0 = SparseVectorView::new(&input_components, &binding);
        let binding1 = [3.0, 2.0, 3.5];
        let input_components1 = [4, 8, 24];
        let input1 = SparseVectorView::new(&input_components1, &binding1);

        let quantizer = PlainSparseQuantizer::<u16, f32, crate::SquaredEuclideanDistance>::new(100, 100);
        let mut growable = PlainSparseDatasetGrowable::new(quantizer);

        growable.push(input0.clone());
        growable.push(input1.clone());

        let training_data = growable.into();
        encoder.train(&training_data);

        let mut buffer0 = Vec::new();
        encoder.push_encoded(input0, &mut buffer0);

        let mut buffer1 = Vec::new();
        encoder.push_encoded(input1, &mut buffer1);

        let decoded0 = encoder.decode_vector(PackedVectorView::new(&buffer0));
        let decoded1 = encoder.decode_vector(PackedVectorView::new(&buffer1));

        same_when_quantized(&input0, &decoded0.as_view());
        same_when_quantized(&input1, &decoded1.as_view());
    }
}
