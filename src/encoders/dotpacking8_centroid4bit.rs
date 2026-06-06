//! `dotpacking8` component compression blended with 4-bit per-component **centroid** value
//! compression.
//!
//! Non-clustered counterpart of [`crate::CDotPacking8Centroid4BitEncoder`]: it keeps the
//! component codec of [`crate::DotPacking8ScalarU8Encoder`] (bisection permutation + one
//! variable-bit gap-coded stream over all components) but stores the per-component
//! **values** as 4-bit nibbles whose codes index a per-component **codebook** of 16
//! centroids (`dequant(c, code) = centroids[c·16 + code]`).
//!
//! Scoring a query value `q` against a doc code computes `q · centroids[c·16 + code]`; the
//! evaluator precomputes a dense LUT `lut[c·16 + v] = q · centroids[c·16 + v]` and the hot
//! kernel reduces to a 2-D gather `lut[comp·16 + code]` + add. The codebook quantizer and
//! the nibble layout are shared with the `cdotpacking8` fork via
//! [`crate::encoders::dotpacking8::nibble4bit`]; only the single-stream component skeleton
//! lives here.
//!
//! ## Per-vector byte layout
//! ```text
//! [n u16][selectors][gap payloads...][pad][values: nibble-packed]  (nibble_bytes(n) bytes)
//! ```

use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::encoders::dotpacking8::common::{compute_safe_simd_padding, encode_blocks, simd_prefix_sum};
use crate::encoders::dotpacking8::nibble4bit::{
    gap_iter, nibble_bytes, pack_nibbles, Centroid4BitQuantizer, NibbleReader, DEFAULT_KMEANS_ITERS,
    DEFAULT_LOWER_PCT, DEFAULT_UPPER_PCT, NUM_CENTROIDS,
};
use crate::{Dataset, PlainSparseDataset, SpaceUsage, SquaredEuclideanDistance};
use bytemuck::{cast_slice, from_bytes};
use rusty_perm::{PermApply, PermFromSorting};
use std::borrow::Cow;
use std::simd::prelude::*;

const N: usize = 8;

// ---------------------------------------------------------------------------
// View (single nibble-packed value stream).
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct DotPack8Centroid4BitView<'a> {
    pub n: usize,
    pub selectors: &'a [u8],
    pub payloads: &'a [u8],
    pub values: &'a [u8],
}

impl<'a> DotPack8Centroid4BitView<'a> {
    pub unsafe fn from_unchecked_slice(slice: &'a [u64]) -> Self {
        let bytes = cast_slice::<u64, u8>(slice);
        let n = from_bytes::<u16>(&bytes[0..2]).to_le() as usize;

        let n_blocks_total = (n + N - 1) / N;
        let selectors_size = (n_blocks_total + 1) / 2;
        let selectors_end = 2 + selectors_size;
        let values_start = bytes.len() - nibble_bytes(n);

        Self {
            n,
            selectors: &bytes[2..selectors_end],
            payloads: &bytes[selectors_end..values_start],
            values: &bytes[values_start..],
        }
    }
}

// ---------------------------------------------------------------------------
// Encoder.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DotPacking8Centroid4BitEncoder {
    pub dim: usize,
    pub quantizer: Centroid4BitQuantizer,
    pub component_mapping: Option<Box<[u16]>>,
    pub inverse_component_mapping: Option<Box<[u16]>>,
}

impl DotPacking8Centroid4BitEncoder {
    pub fn new(input_dim: usize) -> Self {
        Self {
            dim: input_dim,
            quantizer: Centroid4BitQuantizer::new(
                input_dim,
                vec![0.0f32; input_dim * NUM_CENTROIDS].into_boxed_slice(),
            ),
            component_mapping: None,
            inverse_component_mapping: None,
        }
    }

    /// Train the component permutation (bisection) and the per-component 4-bit codebook,
    /// using the default centroid k-means knobs ([`DEFAULT_LOWER_PCT`], [`DEFAULT_UPPER_PCT`],
    /// [`DEFAULT_KMEANS_ITERS`]). To tune the codebook fit, call [`Self::train_with_params`].
    pub fn train(&mut self, training_data: &PlainSparseDataset<u16, f32, SquaredEuclideanDistance>) {
        self.train_with_params(
            training_data,
            DEFAULT_LOWER_PCT,
            DEFAULT_UPPER_PCT,
            DEFAULT_KMEANS_ITERS,
        );
    }

    /// Training entry point with explicit centroid k-means knobs — the real API for tuning.
    pub fn train_with_params(
        &mut self,
        training_data: &PlainSparseDataset<u16, f32, SquaredEuclideanDistance>,
        lower_percentile: f32,
        upper_percentile: f32,
        n_iterations: usize,
    ) {
        const SAMPLE_RATE: usize = 20;
        let sample_size = if training_data.len() / SAMPLE_RATE < 50_000 {
            training_data.len()
        } else {
            training_data.len() / SAMPLE_RATE
        };
        self.train_components(training_data.iter().take(sample_size));
        self.quantizer = Centroid4BitQuantizer::train(
            training_data,
            lower_percentile,
            upper_percentile,
            n_iterations,
        );
    }

    pub fn train_components<'a, V>(
        &mut self,
        training_data: impl Iterator<Item = SparseVectorView<'a, u16, V>>,
    ) where
        V: crate::ValueType,
    {
        let components_iter = training_data.map(|v| v.components());
        let permutation =
            crate::utils::permute_components_with_bisection(self.input_dim(), components_iter);
        let component_mapping: Vec<u16> = permutation.iter().map(|i| *i as u16).collect();
        let inverse = Self::compute_inverse_mapping(&component_mapping);
        self.component_mapping = Some(component_mapping.into_boxed_slice());
        self.inverse_component_mapping = Some(inverse.into_boxed_slice());
    }

    fn compute_inverse_mapping(component_mapping: &[u16]) -> Vec<u16> {
        let dim = component_mapping.len();
        let mut inverse = vec![0u16; dim];
        for (old, &new) in component_mapping.iter().enumerate() {
            inverse[new as usize] = old as u16;
        }
        inverse
    }

    #[inline]
    pub fn component_mapping(&self) -> Option<&[u16]> {
        self.component_mapping.as_deref()
    }

    #[inline]
    pub fn inverse_component_mapping(&self) -> Option<&[u16]> {
        self.inverse_component_mapping.as_deref()
    }
}

// ---------------------------------------------------------------------------
// Query evaluator.
// ---------------------------------------------------------------------------

pub struct DotPacking8Centroid4BitQueryEvaluator<'a> {
    /// Dense LUT `dense_lut[mapped_c·16 + v] = q · centroids[orig_c·16 + v]`. Rows for
    /// components absent from the query stay 0, so their gathers contribute 0.
    dense_lut: Vec<f32>,
    _encoder: &'a DotPacking8Centroid4BitEncoder,
}

impl<'a> DotPacking8Centroid4BitQueryEvaluator<'a> {
    pub fn new(
        query: SparseVectorView<'_, u16, f32>,
        encoder: &'a DotPacking8Centroid4BitEncoder,
    ) -> Self {
        let mut dense_lut = vec![0.0f32; encoder.dim * NUM_CENTROIDS];
        let centroids = encoder.quantizer.centroids();
        for (&c, &qv) in query.components().iter().zip(query.values().iter()) {
            let mapped_c = if let Some(mapping) = &encoder.component_mapping {
                mapping[c as usize]
            } else {
                c
            };
            let qbase = mapped_c as usize * NUM_CENTROIDS;
            let cbase = c as usize * NUM_CENTROIDS;
            for v in 0..NUM_CENTROIDS {
                dense_lut[qbase + v] = qv * centroids[cbase + v];
            }
        }
        Self {
            dense_lut,
            _encoder: encoder,
        }
    }
}

impl<'a, 'v> QueryEvaluator<PackedVectorView<'v, u64>>
    for DotPacking8Centroid4BitQueryEvaluator<'a>
{
    type Distance = DotProduct;

    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> DotProduct {
        let view = unsafe { DotPack8Centroid4BitView::from_unchecked_slice(vector.data()) };
        let n = view.n;

        let mut acc = Simd::<f32, N>::splat(0.0);
        let lut = self.dense_lut.as_slice();
        let stride = Simd::<usize, N>::splat(NUM_CENTROIDS);
        let mut total = 0.0f32;

        let mut it = gap_iter(n, view.selectors, view.payloads.as_ptr());
        let mut nib = NibbleReader::new(view.values);
        let mut last = 0u32;

        for _ in 0..(n / N) {
            let gaps = it.decode_lane();
            it.block_idx += 1;
            let comps = simd_prefix_sum(gaps) + Simd::splat(last);
            last = comps[N - 1];
            let codes = nib.next_block_codes();
            let idx = comps.cast::<usize>() * stride + codes.cast::<usize>();
            let g = unsafe {
                Simd::gather_select_unchecked(lut, Mask::splat(true), idx, Simd::splat(0.0))
            };
            acc += g;
        }

        let rem = n % N;
        if rem > 0 {
            let gaps = it.decode_lane();
            let comps = (simd_prefix_sum(gaps) + Simd::splat(last)).to_array();
            let codes = nib.tail_codes(rem);
            for i in 0..rem {
                total += lut[comps[i] as usize * NUM_CENTROIDS + codes[i] as usize];
            }
        }

        total += acc.reduce_sum();
        DotProduct(total)
    }
}

// ---------------------------------------------------------------------------
// Trait impls.
// ---------------------------------------------------------------------------

impl VectorEncoder for DotPacking8Centroid4BitEncoder {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u16, f32>;
    type QueryVector<'q> = SparseVectorView<'q, u16, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;
    type Evaluator<'e>
        = DotPacking8Centroid4BitQueryEvaluator<'e>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        DotPacking8Centroid4BitQueryEvaluator::new(query, self)
    }
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        DotPacking8Centroid4BitQueryEvaluator::new(decoded.as_view(), self)
    }
    fn output_dim(&self) -> usize {
        self.dim
    }
    fn input_dim(&self) -> usize {
        self.dim
    }
}

impl SparseDataEncoder for DotPacking8Centroid4BitEncoder {
    type InputComponentType = u16;
    type InputValueType = f32;
    type OutputComponentType = u16;
    type OutputValueType = u8;

    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let view = unsafe { DotPack8Centroid4BitView::from_unchecked_slice(encoded.data()) };
        let n = view.n;

        let mut decoded_comp = Vec::with_capacity(n);
        let mut values_raw = Vec::with_capacity(n);

        let mut it = gap_iter(n, view.selectors, view.payloads.as_ptr());
        let mut nib = NibbleReader::new(view.values);
        let mut last_component = 0u32;

        for _ in 0..(n / N) {
            let gaps = it.decode_lane();
            it.block_idx += 1;
            let components = simd_prefix_sum(gaps);
            let absolute_components = components + Simd::splat(last_component);
            let absolute_arr = absolute_components.to_array();
            let vals = nib.next_block_u8();
            for i in 0..N {
                decoded_comp.push(absolute_arr[i] as u16);
                values_raw.push(vals[i]);
            }
            last_component = absolute_arr[N - 1];
        }

        let rem = n % N;
        if rem > 0 {
            let gaps = it.decode_lane();
            let absolute_arr = (simd_prefix_sum(gaps) + Simd::splat(last_component)).to_array();
            let codes = nib.tail_codes(rem);
            for i in 0..rem {
                decoded_comp.push(absolute_arr[i] as u16);
                values_raw.push(codes[i]);
            }
        }

        if let Some(component_mapping) = &self.component_mapping {
            let inverse: Cow<'_, [u16]> = match &self.inverse_component_mapping {
                Some(inverse) => Cow::Borrowed(inverse),
                None => Cow::Owned(Self::compute_inverse_mapping(component_mapping)),
            };
            for c in decoded_comp.iter_mut() {
                *c = inverse[*c as usize];
            }

            let permutation = rusty_perm::PermD::from_sort(decoded_comp.as_slice());
            permutation.apply(values_raw.as_mut_slice()).unwrap();
            permutation.apply(decoded_comp.as_mut_slice()).unwrap();

            let values: Vec<f32> = decoded_comp
                .iter()
                .zip(values_raw.iter())
                .map(|(&c, &v)| self.quantizer.decode_value(c, v))
                .collect();
            SparseVectorOwned::new(decoded_comp, values)
        } else {
            let values: Vec<f32> = decoded_comp
                .iter()
                .zip(values_raw.iter())
                .map(|(&c, &v)| self.quantizer.decode_value(c, v))
                .collect();
            SparseVectorOwned::new(decoded_comp, values)
        }
    }
}

impl PackedSparseVectorEncoder for DotPacking8Centroid4BitEncoder {
    type PackedDataType = u64;

    fn push_encoded<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, u16, f32>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<u64>,
    {
        let mut q_values: Vec<u8> = input
            .components()
            .iter()
            .zip(input.values())
            .map(|(&c, &v)| self.quantizer.encode_value(c, v))
            .collect();

        let mut q_components: Vec<u16> = if let Some(mapping) = &self.component_mapping {
            input
                .components()
                .iter()
                .map(|&c| mapping[c as usize])
                .collect()
        } else {
            input.components().to_vec()
        };

        if self.component_mapping.is_some() {
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

        let nb = nibble_bytes(n);
        let current_len = payload.len();
        let padding_needed =
            compute_safe_simd_padding(n, &selectors, (8 - (current_len + nb) % 8) % 8, nb);

        for _ in 0..padding_needed {
            payload.push(0);
        }
        pack_nibbles(&mut payload, &q_values);

        let data = payload
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()));

        output.extend(data);
    }
}

impl SpaceUsage for DotPacking8Centroid4BitEncoder {
    fn space_usage_bytes(&self) -> usize {
        let mapping_size = self
            .component_mapping
            .as_ref()
            .map_or(0, |m| m.len() * std::mem::size_of::<u16>());
        let inverse_mapping_size = self
            .inverse_component_mapping
            .as_ref()
            .map_or(0, |m| m.len() * std::mem::size_of::<u16>());
        mapping_size + inverse_mapping_size + self.quantizer.space_usage_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::distances::Distance;
    use crate::core::vector::{PackedVectorView, SparseVectorView};
    use crate::encoders::sparse_scalar::PlainSparseQuantizer;
    use crate::vector_encoder::SparseDataEncoder;
    use crate::{DatasetGrowable, PlainSparseDatasetGrowable, QueryEvaluator, VectorEncoder};

    /// Push enough vectors per component (values spread across ~[0.2, 4.0]) that per-component
    /// k-means yields a full 16-centroid codebook, so test inputs map to a spread of codes 0..15.
    fn build_training_data(
        dim: usize,
        components: &[u16],
    ) -> PlainSparseDataset<u16, f32, SquaredEuclideanDistance> {
        let quantizer = PlainSparseQuantizer::<u16, f32, SquaredEuclideanDistance>::new(dim, dim);
        let mut growable = PlainSparseDatasetGrowable::new(quantizer);
        for k in 0..32usize {
            let vals: Vec<f32> = components
                .iter()
                .enumerate()
                .map(|(idx, _)| 0.2 + ((idx + k) % 20) as f32 * 0.2)
                .collect();
            growable.push(SparseVectorView::new(components, &vals));
        }
        growable.into()
    }

    fn quantized_values(
        encoder: &DotPacking8Centroid4BitEncoder,
        components: &[u16],
        values: &[f32],
    ) -> Vec<f32> {
        components
            .iter()
            .zip(values.iter())
            .map(|(&c, &v)| {
                let encoded = encoder.quantizer.encode_value(c, v);
                encoder.quantizer.decode_value(c, encoded)
            })
            .collect()
    }

    fn dot_with_query(
        components: &[u16],
        values: &[f32],
        query_components: &[u16],
        query_values: &[f32],
    ) -> f32 {
        let mut sum = 0.0f32;
        let (mut i, mut j) = (0usize, 0usize);
        while i < components.len() && j < query_components.len() {
            match components[i].cmp(&query_components[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    sum += values[i] * query_values[j];
                    i += 1;
                    j += 1;
                }
            }
        }
        sum
    }

    fn assert_parity(
        encoder: &DotPacking8Centroid4BitEncoder,
        comps: &[u16],
        in_vals: &[f32],
        query_comps: &[u16],
        query_vals: &[f32],
    ) {
        let input = SparseVectorView::new(comps, in_vals);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let query = SparseVectorView::new(query_comps, query_vals);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let q_values = quantized_values(encoder, comps, in_vals);
        let expected = dot_with_query(comps, &q_values, query_comps, query_vals);

        let tol = 1e-4 * expected.abs().max(1.0);
        assert!(
            (dist.distance() - expected).abs() <= tol,
            "score {} != expected {} (n={})",
            dist.distance(),
            expected,
            comps.len()
        );
    }

    const NS: [usize; 12] = [1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 23, 40];

    #[test]
    fn parity_various_n() {
        const DIM: usize = 4000;
        let all: Vec<u16> = (0u16..3000).collect();
        let training = build_training_data(DIM, &all);
        let mut enc = DotPacking8Centroid4BitEncoder::new(DIM);
        enc.train_with_params(&training, 0.0, 1.0, 10);

        for &n in &NS {
            let comps: Vec<u16> = (0..n as u16).map(|i| i * 3).collect();
            let in_vals: Vec<f32> = (0..n).map(|i| 0.5 + (i % 11) as f32 * 0.3).collect();
            let q_vals: Vec<f32> = (0..n).map(|i| 0.25 + (i % 7) as f32 * 0.2).collect();
            assert_parity(&enc, &comps, &in_vals, &comps, &q_vals);
        }
    }

    #[test]
    fn parity_query_partial_overlap() {
        const DIM: usize = 200;
        let all: Vec<u16> = (0u16..100).collect();
        let training = build_training_data(DIM, &all);
        let mut enc = DotPacking8Centroid4BitEncoder::new(DIM);
        enc.train_with_params(&training, 0.0, 1.0, 10);

        let comps: Vec<u16> = vec![0, 4, 8, 24, 28, 36, 48, 53, 70, 90];
        let in_vals = [1.0, 3.0, 2.0, 3.5, 1.5, 2.0, 1.0, 2.0, 3.0, 2.5];
        let query_comps: Vec<u16> = vec![2, 4, 8, 24, 28, 70, 90];
        let query_vals = [0.5, 1.5, 2.5, 1.0, 2.0, 1.0, 2.0];
        assert_parity(&enc, &comps, &in_vals, &query_comps, &query_vals);
    }

    fn same_when_quantized(
        before_c: &[u16],
        before_v: &[f32],
        after: &SparseVectorView<u16, f32>,
        encoder: &DotPacking8Centroid4BitEncoder,
    ) {
        assert_eq!(before_c, after.components());
        for (c, (v1, v2)) in before_c
            .iter()
            .zip(before_v.iter().zip(after.values().iter()))
        {
            let encoded = encoder.quantizer.encode_value(*c, *v1);
            let quantized = encoder.quantizer.decode_value(*c, encoded);
            assert!(
                (quantized - v2).abs() < 1e-5,
                "component {c}: {quantized} != {v2}"
            );
        }
    }

    #[test]
    fn decode_roundtrip_various_n() {
        const DIM: usize = 4000;
        let all: Vec<u16> = (0u16..3000).collect();
        let training = build_training_data(DIM, &all);
        let mut enc = DotPacking8Centroid4BitEncoder::new(DIM);
        enc.train_with_params(&training, 0.0, 1.0, 10);

        for &n in &NS {
            let comps: Vec<u16> = (0..n as u16).map(|i| i * 7).collect();
            let in_vals: Vec<f32> = (0..n).map(|i| 0.5 + (i % 13) as f32 * 0.28).collect();
            let input = SparseVectorView::new(&comps, &in_vals);
            let mut buffer = Vec::new();
            enc.push_encoded(input, &mut buffer);
            let decoded = enc.decode_vector(PackedVectorView::new(&buffer));
            same_when_quantized(&comps, &in_vals, &decoded.as_view(), &enc);
        }
    }
}
