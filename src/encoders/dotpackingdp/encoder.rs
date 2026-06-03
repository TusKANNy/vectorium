use crate::SpaceUsage;
use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::encoders::dotpackingdp::common::simd_prefix_sum;
use crate::encoders::dotpacking8::common::fast_lane_gather;
use crate::encoders::dotpackingdp::common::{
    DotPackingDpIter, compute_safe_simd_padding, encode_blocks_dp,
};
use crate::encoders::dotpackingdp::quantizer::DotPackingDpQuantizer;
use bytemuck::cast_slice;
use rusty_perm::PermApply as _;
use rusty_perm::PermFromSorting as _;
use std::simd::StdFloat;
use std::simd::prelude::*;

const N: usize = 8;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct DotPackingDpEncoder<Q, const MAX_BLOCK_LEN: usize = 16>
where
    Q: DotPackingDpQuantizer,
{
    pub dim: usize,
    pub quantizer: Q,
    pub component_mapping: Option<Box<[u16]>>,
    pub inverse_component_mapping: Option<Box<[u16]>>,
}

impl<Q, const MAX_BLOCK_LEN: usize> PartialEq for DotPackingDpEncoder<Q, MAX_BLOCK_LEN>
where
    Q: DotPackingDpQuantizer + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
            && self.quantizer == other.quantizer
            && self.component_mapping == other.component_mapping
            && self.inverse_component_mapping == other.inverse_component_mapping
    }
}

impl<Q, const MAX_BLOCK_LEN: usize> DotPackingDpEncoder<Q, MAX_BLOCK_LEN>
where
    Q: DotPackingDpQuantizer<InputValue = f32>,
{
    pub fn new_with_quantizer(dim: usize, quantizer: Q) -> Self {
        Self {
            dim,
            quantizer,
            component_mapping: None,
            inverse_component_mapping: None,
        }
    }

    #[inline]
    pub fn component_mapping(&self) -> Option<&[u16]> {
        self.component_mapping.as_deref()
    }

    #[inline]
    pub fn inverse_component_mapping(&self) -> Option<&[u16]> {
        self.inverse_component_mapping.as_deref()
    }

    fn compute_inverse_mapping(component_mapping: &[u16]) -> Vec<u16> {
        let dim = component_mapping.len();
        let mut inverse = vec![0u16; dim];
        let mut seen = vec![false; dim];
        for (old, &new) in component_mapping.iter().enumerate() {
            let new = new as usize;
            assert!(new < dim);
            assert!(!seen[new]);
            seen[new] = true;
            inverse[new] = old as u16;
        }
        inverse
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
}

#[derive(Clone)]
pub struct DotPackingDpView<'a> {
    pub n: usize,
    pub n_blocks: usize,
    pub selectors: &'a [u8],
    pub payloads: &'a [u8],
    pub values: &'a [u8],
}

impl<'a> DotPackingDpView<'a> {
    pub unsafe fn from_unchecked_slice(slice: &'a [u64]) -> Self {
        let bytes = cast_slice::<u64, u8>(slice);
        let n = u16::from_le_bytes(bytes[0..2].try_into().unwrap()) as usize;
        let n_blocks = u16::from_le_bytes(bytes[2..4].try_into().unwrap()) as usize;
        let offset = 4;
        let values_start_idx = bytes.len() - n;
        Self {
            n,
            n_blocks,
            selectors: &bytes[offset..offset + n_blocks],
            payloads: &bytes[offset + n_blocks..values_start_idx],
            values: &bytes[values_start_idx..],
        }
    }

    pub fn iter_raw<const MAX_BLOCK_LEN: usize>(&self) -> DotPackingDpIter<'a, MAX_BLOCK_LEN> {
        DotPackingDpIter {
            n: self.n,
            processed_count: 0,
            sel_idx: 0,
            chunk_rem: 0,
            b: 0,
            global_bit_pos: 0,
            overflow_count: 0,
            overflow_reg: Simd::splat(0),
            permuted_reg: Simd::splat(0),
            selectors: self.selectors,
            payload_ptr: self.payloads.as_ptr(),
            val_ptr: self.values.as_ptr(),
        }
    }
}

impl<Q, const MAX_BLOCK_LEN: usize> SparseDataEncoder for DotPackingDpEncoder<Q, MAX_BLOCK_LEN>
where
    Q: DotPackingDpQuantizer<InputValue = f32>,
{
    type InputComponentType = u16;
    type InputValueType = f32;
    type OutputComponentType = u16;
    type OutputValueType = u8;

    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let view = unsafe { DotPackingDpView::from_unchecked_slice(encoded.data()) };

        let mut decoded_comp = Vec::with_capacity(view.n);
        let mut values_raw = Vec::with_capacity(view.n);
        let mut iter = view.iter_raw::<MAX_BLOCK_LEN>();
        let mut last_component = 0u32;
        for (gaps, vals) in &mut iter {
            let components = simd_prefix_sum(gaps);
            let absolute_components = components + Simd::splat(last_component);
            let absolute_arr = absolute_components.to_array();
            let vals_arr = vals.to_array();
            for i in 0..N {
                decoded_comp.push(absolute_arr[i] as u16);
                values_raw.push(vals_arr[i]);
            }
            last_component = absolute_arr[N - 1];
        }
        let (gaps, tail_values, remaining) = iter.decode_tail();
        let components = simd_prefix_sum(gaps);
        let absolute_components = components + Simd::splat(last_component);
        let absolute_arr = absolute_components.to_array();
        for i in 0..remaining {
            decoded_comp.push(absolute_arr[i] as u16);
            values_raw.push(tail_values[i]);
        }
        if let Some(component_mapping) = self.component_mapping() {
            let inverse: std::borrow::Cow<'_, [u16]> = match self.inverse_component_mapping() {
                Some(inverse) => std::borrow::Cow::Borrowed(inverse),
                None => std::borrow::Cow::Owned(Self::compute_inverse_mapping(component_mapping)),
            };

            for c in decoded_comp.iter_mut() {
                *c = inverse[*c as usize];
            }

            let mut values: Vec<f32> = decoded_comp
                .iter()
                .zip(values_raw.iter())
                .map(|(&c, &v)| self.quantizer.decode_value(c, v))
                .collect();

            let permutation = rusty_perm::PermD::from_sort(decoded_comp.as_slice());
            permutation.apply(values.as_mut_slice()).unwrap();
            permutation.apply(decoded_comp.as_mut_slice()).unwrap();

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

impl<Q, const MAX_BLOCK_LEN: usize> PackedSparseVectorEncoder
    for DotPackingDpEncoder<Q, MAX_BLOCK_LEN>
where
    Q: DotPackingDpQuantizer<InputValue = f32>,
{
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

        let mut q_components: Vec<u16> = if let Some(m) = &self.component_mapping {
            input.components().iter().map(|&c| m[c as usize]).collect()
        } else {
            input.components().to_vec()
        };

        if self.component_mapping.is_some() {
            let perm = rusty_perm::PermD::from_sort(q_components.as_slice());
            perm.apply(q_values.as_mut_slice()).unwrap();
            perm.apply(q_components.as_mut_slice()).unwrap();
        }

        let n = q_components.len();
        if n == 0 {
            return;
        }

        let mut gaps = Vec::with_capacity(n);
        let mut last = 0u32;
        for c in q_components {
            gaps.push(c as u32 - last);
            last = c as u32;
        }

        let (selectors, gap_bitstream) = encode_blocks_dp::<MAX_BLOCK_LEN>(&gaps);
        let mut payload = Vec::new();
        payload.extend_from_slice(&(n as u16).to_le_bytes());
        payload.extend_from_slice(&(selectors.len() as u16).to_le_bytes());
        payload.extend_from_slice(&selectors);
        payload.extend_from_slice(&gap_bitstream);

        let current_len = payload.len();
        let padding_needed =
            compute_safe_simd_padding(&selectors, (8 - (current_len + n) % 8) % 8, n);

        for _ in 0..padding_needed {
            payload.push(0);
        }
        payload.extend_from_slice(&q_values);
        output.extend(
            payload
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap())),
        );
    }
}

impl<Q, const MAX_BLOCK_LEN: usize> VectorEncoder for DotPackingDpEncoder<Q, MAX_BLOCK_LEN>
where
    Q: DotPackingDpQuantizer<InputValue = f32>,
{
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u16, f32>;
    type QueryVector<'q> = SparseVectorView<'q, u16, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;
    type Evaluator<'e>
        = DotPackingDpQueryEvaluator<'e, Q, MAX_BLOCK_LEN>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        DotPackingDpQueryEvaluator::new(query, self)
    }
    fn vector_evaluator<'e, 'v>(&'e self, v: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, v);
        DotPackingDpQueryEvaluator::new(decoded.as_view(), self)
    }
    fn output_dim(&self) -> usize {
        self.dim
    }
    fn input_dim(&self) -> usize {
        self.dim
    }
}

pub struct DotPackingDpQueryEvaluator<'a, Q, const MAX_BLOCK_LEN: usize = 16>
where
    Q: DotPackingDpQuantizer,
{
    dense_query: Vec<f32>,
    _encoder: &'a DotPackingDpEncoder<Q, MAX_BLOCK_LEN>,
}

impl<'a, Q, const MAX_BLOCK_LEN: usize> DotPackingDpQueryEvaluator<'a, Q, MAX_BLOCK_LEN>
where
    Q: DotPackingDpQuantizer<InputValue = f32>,
{
    pub fn new(
        query: SparseVectorView<'_, u16, f32>,
        encoder: &'a DotPackingDpEncoder<Q, MAX_BLOCK_LEN>,
    ) -> Self {
        let mut dense_query = vec![0.0f32; encoder.dim];
        for (&c, &v) in query.components().iter().zip(query.values().iter()) {
            let mapped_c = if let Some(m) = &encoder.component_mapping {
                m[c as usize]
            } else {
                c
            };
            dense_query[mapped_c as usize] = encoder.quantizer.query_value(c, v);
        }
        Self {
            dense_query,
            _encoder: encoder,
        }
    }
}

impl<'a, 'v, Q, const MAX_BLOCK_LEN: usize> QueryEvaluator<PackedVectorView<'v, u64>>
    for DotPackingDpQueryEvaluator<'a, Q, MAX_BLOCK_LEN>
where
    Q: DotPackingDpQuantizer<InputValue = f32>,
{
    type Distance = DotProduct;

    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> Self::Distance {
        let view = unsafe { DotPackingDpView::from_unchecked_slice(vector.data()) };
        let mut acc = Simd::<f32, N>::splat(0.0);
        let mut query_slice = self.dense_query.as_slice();

        let mut iter = view.iter_raw::<MAX_BLOCK_LEN>();
        for (gaps, values) in &mut iter {
            let relative_positions = simd_prefix_sum(gaps);
            let q_vals = fast_lane_gather(query_slice, relative_positions);
            acc = q_vals.mul_add(values.cast(), acc);
            let step = relative_positions[N - 1] as usize;
            query_slice = unsafe { query_slice.split_at_unchecked(step).1 };
        }

        let mut total_unscaled = acc.reduce_sum();

        let (gaps, tail_values, remaining) = iter.decode_tail();
        if remaining > 0 {
            let relative_positions = simd_prefix_sum(gaps);
            let comps = relative_positions.as_array();
            for j in 0..remaining {
                let pos = comps[j] as usize;
                let val = tail_values[j] as f32;
                total_unscaled += query_slice[pos] * val;
            }
        }
        DotProduct(total_unscaled)
    }
}

impl<Q, const MAX_BLOCK_LEN: usize> SpaceUsage for DotPackingDpEncoder<Q, MAX_BLOCK_LEN>
where
    Q: DotPackingDpQuantizer,
{
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
