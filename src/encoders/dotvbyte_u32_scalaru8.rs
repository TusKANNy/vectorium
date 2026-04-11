use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::marker::PhantomData;

use crate::core::sealed;
use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::{Dataset, PlainSparseDataset, SpaceUsage, SquaredEuclideanDistance, ValueType};
use num_traits::AsPrimitive;

use rusty_perm::PermApply as _;
use rusty_perm::PermFromSorting as _;

use super::dotvbyte_fixedu8::swizzle::swizzle;
use super::dotvbyte_u32_fixedu8::{MASKS, SCROLLS};
use crate::utils::{permute_components_with_bisection, train_sparse_scalar_quantizer};
use std::simd::StdFloat;
use std::simd::num::{SimdFloat, SimdUint};

use std::{
    mem::transmute_copy,
    simd::{Mask, Simd},
};

use bytemuck::try_cast_slice;
use rusty_perm::*;

/// VectorEncoder for DotVByte-packed sparse vectors with u32 components and scalar quantized u8 values.
///
/// - Encoded vectors are represented as a packed slice of `u64` words.
/// - `output_dim()` is the logical post-quantization dimensionality (typically equal to `input_dim()`),
///   NOT the packed blob length in `u64` words.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DotVByteU32ScalarU8Encoder {
    dim: usize,
    quants: Box<[f32]>,
    component_mapping: Option<Box<[u32]>>,
    #[serde(default)]
    inverse_component_mapping: Option<Box<[u32]>>,
}

impl sealed::Sealed for DotVByteU32ScalarU8Encoder {}

impl DotVByteU32ScalarU8Encoder {
    #[inline]
    pub fn component_mapping(&self) -> Option<&[u32]> {
        let mapping = self.component_mapping.as_deref();
        if let Some(mapping) = mapping {
            assert_eq!(
                mapping.len(),
                self.dim,
                "component mapping length {} does not match input dimension {}",
                mapping.len(),
                self.dim
            );
        }
        mapping
    }

    #[inline]
    pub fn inverse_component_mapping(&self) -> Option<&[u32]> {
        let inverse = self.inverse_component_mapping.as_deref();
        if let Some(inverse) = inverse {
            assert_eq!(
                inverse.len(),
                self.dim,
                "inverse component mapping length {} does not match input dimension {}",
                inverse.len(),
                self.dim
            );
        }
        inverse
    }

    fn compute_inverse_mapping(component_mapping: &[u32]) -> Vec<u32> {
        let dim = component_mapping.len();
        let mut inverse = vec![0u32; dim];
        let mut seen = vec![false; dim];

        for (old, &new) in component_mapping.iter().enumerate() {
            let new = new as usize;
            assert!(
                new < dim,
                "component_mapping maps component {} to out-of-bounds index {} (dim={})",
                old,
                new,
                dim
            );
            assert!(
                !seen[new],
                "component_mapping is not a permutation: duplicate mapped index {}",
                new
            );
            seen[new] = true;
            inverse[new] = old as u32;
        }

        inverse
    }
}

/// Optimistic DotVByte encoder for sparse vectors with `u32` components and scalar quantized u8 values.
///
/// This encoder behaves like `DotVByteU32ScalarU8Encoder`, but in `push_encoded` it injects
/// intermediate zero-valued coordinates whenever a component gap would exceed `u16::MAX`.
/// This keeps most vectors on the fast path while still supporting rare large gaps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimisticDotVByteScalarU8Encoder {
    inner: DotVByteU32ScalarU8Encoder,
}

impl sealed::Sealed for OptimisticDotVByteScalarU8Encoder {}

impl OptimisticDotVByteScalarU8Encoder {
    #[inline]
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            inner: DotVByteU32ScalarU8Encoder::new(input_dim, output_dim),
        }
    }

    #[inline]
    pub fn component_mapping(&self) -> Option<&[u32]> {
        self.inner.component_mapping()
    }

    #[inline]
    pub fn inverse_component_mapping(&self) -> Option<&[u32]> {
        self.inner.inverse_component_mapping()
    }

    pub fn train<V>(
        &mut self,
        training_data: &PlainSparseDataset<u32, f32, SquaredEuclideanDistance>,
    ) where
        V: ValueType,
    {
        self.inner.train::<V>(training_data);
    }

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

        bridged_components.push(components[0]);
        bridged_values.push(values[0]);

        for (&component, &value) in components.iter().zip(values.iter()).skip(1) {
            let mut last = *bridged_components.last().unwrap();
            while component.saturating_sub(last) > u16::MAX as u32 {
                last += u16::MAX as u32;
                bridged_components.push(last);
                bridged_values.push(0);
            }
            bridged_components.push(component);
            bridged_values.push(value);
        }

        (bridged_components, bridged_values)
    }
}

impl SparseDataEncoder for OptimisticDotVByteScalarU8Encoder {
    type InputComponentType = u32;
    type InputValueType = f32;
    type OutputComponentType = u32;
    type OutputValueType = u8;

    #[inline]
    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let optimistic_view =
            unsafe { DotVbyteOptimisticU32ScalarU8::from_unchecked_slice(encoded.data()) };

        let mut components: Vec<u32> = Vec::new();
        let mut values: Vec<f32> = Vec::new();
        for (component, value) in optimistic_view.iter() {
            components.push(component);
            values.push(value as f32 * self.inner.quants[component as usize]);
        }

        if let Some(component_mapping) = self.component_mapping() {
            let inverse: Cow<'_, [u32]> = match self.inverse_component_mapping() {
                Some(inverse) => Cow::Borrowed(inverse),
                None => Cow::Owned(DotVByteU32ScalarU8Encoder::compute_inverse_mapping(
                    component_mapping,
                )),
            };

            for c in components.iter_mut() {
                *c = inverse[*c as usize];
            }

            let permutation = rusty_perm::PermD::from_sort(components.as_slice());
            permutation.apply(values.as_mut_slice()).unwrap();
            permutation.apply(components.as_mut_slice()).unwrap();
        }

        SparseVectorOwned::new(components, values)
    }
}

impl PackedSparseVectorEncoder for OptimisticDotVByteScalarU8Encoder {
    type PackedDataType = u64;

    fn push_encoded<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, Self::InputComponentType, Self::InputValueType>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<Self::PackedDataType>,
    {
        let mut encoded_u8 = Vec::new();

        let mut q_values: Vec<u8> = input
            .components()
            .iter()
            .zip(input.values())
            .map(|(&c, &v)| {
                let idx: usize = c.as_();
                let q = self.inner.quants[idx];
                if q > 0.0 {
                    (v / q).clamp(0.0, 255.0) as u8
                } else {
                    0u8
                }
            })
            .collect();

        let mut q_components = if let Some(component_mapping) = self.component_mapping() {
            input
                .components()
                .iter()
                .map(|&c| component_mapping[c as usize])
                .collect::<Vec<u32>>()
        } else {
            input.components().to_vec()
        };

        if self.component_mapping().is_some() {
            let permutation = rusty_perm::PermD::from_sort(q_components.as_slice());
            permutation.apply(q_values.as_mut_slice()).unwrap();
            permutation.apply(q_components.as_mut_slice()).unwrap();
        }

        let (mut bridged_components, mut bridged_values) =
            Self::inject_zero_bridges(&q_components, &q_values);

        DotVbyteOptimisticU32ScalarU8::push_vector(
            &mut encoded_u8,
            &mut bridged_components,
            &mut bridged_values,
        );

        assert!(
            encoded_u8.len() % std::mem::size_of::<u64>() == 0,
            "encoded_u8 length ({}) is not a multiple of 8",
            encoded_u8.len()
        );

        let data = encoded_u8
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()));
        output.extend(data);
    }
}

impl VectorEncoder for OptimisticDotVByteScalarU8Encoder {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u32, f32>;
    type QueryVector<'q> = SparseVectorView<'q, u32, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;

    type Evaluator<'e>
        = OptimisticDotVByteScalarU8QueryEvaluator<'e>
    where
        Self: 'e;

    #[inline]
    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        OptimisticDotVByteScalarU8QueryEvaluator::new(query, self)
    }

    #[inline]
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        OptimisticDotVByteScalarU8QueryEvaluator::new(decoded.as_view(), self)
    }

    #[inline]
    fn output_dim(&self) -> usize {
        self.inner.output_dim()
    }

    #[inline]
    fn input_dim(&self) -> usize {
        self.inner.input_dim()
    }
}

#[derive(Debug, Clone)]
pub struct OptimisticDotVByteScalarU8QueryEvaluator<'e> {
    dense_query_transformed: Vec<f32>,
    _phantom: PhantomData<&'e ()>,
}

impl<'e> OptimisticDotVByteScalarU8QueryEvaluator<'e> {
    #[inline]
    pub fn new(
        query: SparseVectorView<'_, u32, f32>,
        quantizer: &'e OptimisticDotVByteScalarU8Encoder,
    ) -> Self {
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

        let component_mapping = quantizer.component_mapping();
        let mut transformed = vec![0.0f32; quantizer.input_dim()];
        for (&c, &v) in query.components().iter().zip(query.values()) {
            let idx: usize = c.as_();
            let mapped: usize = match component_mapping {
                Some(mapping) => mapping[idx] as usize,
                None => idx,
            };
            transformed[mapped] = v * quantizer.inner.quants[idx];
        }

        Self {
            dense_query_transformed: transformed,
            _phantom: PhantomData,
        }
    }
}

impl<'e, 'v> QueryEvaluator<PackedVectorView<'v, u64>>
    for OptimisticDotVByteScalarU8QueryEvaluator<'e>
{
    type Distance = DotProduct;

    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> DotProduct {
        let optimistic_view =
            unsafe { DotVbyteOptimisticU32ScalarU8::from_unchecked_slice(vector.data()) };
        DotProduct::from(optimistic_view.dot_product(&self.dense_query_transformed))
    }
}

const N_OPT: usize = u8::BITS as usize;

#[derive(Clone)]
struct DotVbyteOptimisticU32ScalarU8<'a> {
    values: &'a [Simd<u8, N_OPT>],
    bytes_remaining: &'a [u16],
    values_remaining: &'a [u8],
    idx_lens: &'a [u8],
    bytes: &'a [u8],
}

impl<'a> DotVbyteOptimisticU32ScalarU8<'a> {
    #[inline]
    unsafe fn cast_simd_slice<const LANES: usize>(slice: &'a [u8]) -> &'a [Simd<u8, LANES>] {
        let len = slice.len() / std::mem::size_of::<Simd<u8, LANES>>();
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const Simd<u8, LANES>, len) }
    }

    unsafe fn from_unchecked_slice(slice: &'a [u64]) -> Self {
        unsafe {
            let slice = try_cast_slice::<u64, u8>(slice).unwrap_unchecked();

            let original_length =
                u16::from_ne_bytes([*slice.get_unchecked(0), *slice.get_unchecked(1)]);

            let n_packs = original_length as usize / N_OPT;
            let n_remaining = original_length as usize % N_OPT;

            let bytes_remaining_start = std::mem::size_of::<u16>();
            let bytes_remaining_start = bytes_remaining_start.next_multiple_of(align_of::<u16>());
            let bytes_remaining_end = bytes_remaining_start + n_remaining * size_of::<u16>();
            let bytes_remaining =
                try_cast_slice(slice.get_unchecked(bytes_remaining_start..bytes_remaining_end))
                    .unwrap_unchecked();

            let values_remaining_start = bytes_remaining_end.next_multiple_of(align_of::<u8>());
            let values_remaining_end = values_remaining_start + n_remaining * size_of::<u8>();
            let values_remaining =
                try_cast_slice(slice.get_unchecked(values_remaining_start..values_remaining_end))
                    .unwrap_unchecked();

            let idx_lens_start = values_remaining_end.next_multiple_of(align_of::<u8>());
            let idx_lens_end = idx_lens_start + n_packs * size_of::<u8>();
            let idx_lens = try_cast_slice(slice.get_unchecked(idx_lens_start..idx_lens_end))
                .unwrap_unchecked();

            let bytes_start = idx_lens_end;
            let values_start = slice.len() - n_packs * std::mem::size_of::<Simd<u8, N_OPT>>();

            let bytes =
                try_cast_slice(slice.get_unchecked(bytes_start..values_start)).unwrap_unchecked();

            let values_end = slice.len();
            let values =
                Self::cast_simd_slice::<N_OPT>(slice.get_unchecked(values_start..values_end));

            Self {
                values,
                bytes_remaining,
                values_remaining,
                idx_lens,
                bytes,
            }
        }
    }

    fn push_vector(vec: &mut Vec<u8>, converted_components: &mut [u32], values: &mut [u8]) {
        assert!(
            converted_components.len() < u16::MAX as usize,
            "OptimisticDotVByteScalarU8Encoder only supports vectors shorter than 65535."
        );
        assert_eq!(converted_components.len(), values.len());

        let permutation = PermD::from_sort(&*converted_components);
        permutation.apply(values).unwrap();
        permutation.apply(converted_components).unwrap();

        for i in (1..converted_components.len()).rev() {
            converted_components[i] -= converted_components[i - 1];
        }

        let n_chunked = converted_components.len() - converted_components.len() % N_OPT;
        let (components_chunked, components_remaining) =
            unsafe { converted_components.split_at_unchecked(n_chunked) };
        let (values_chunked, values_remaining) = unsafe { values.split_at_unchecked(n_chunked) };

        for &delta in components_chunked.iter() {
            assert!(
                delta <= u16::MAX as u32,
                "delta {} exceeds u16::MAX; optimistic bridge insertion failed",
                delta
            );
        }
        for &delta in components_remaining.iter() {
            assert!(
                delta <= u16::MAX as u32,
                "delta {} exceeds u16::MAX; optimistic bridge insertion failed",
                delta
            );
        }

        let bitvec: Vec<u8> = components_chunked
            .chunks_exact(N_OPT)
            .map(|chunk| {
                let mut byte = 0;
                for (i, b) in chunk.iter().map(|&n| n > u8::MAX as u32).enumerate() {
                    byte |= (b as u8) << (N_OPT - i - 1)
                }
                byte
            })
            .collect();

        let original_length = converted_components.len() as u16;
        unsafe {
            vec.extend_from_slice(&original_length.to_ne_bytes());

            vec.resize(vec.len().next_multiple_of(size_of::<u16>()), 0);
            for &delta in components_remaining.iter() {
                vec.extend_from_slice(&(delta as u16).to_ne_bytes());
            }

            vec.resize(vec.len().next_multiple_of(size_of::<u8>()), 0);
            vec.extend_from_slice(try_cast_slice(values_remaining).unwrap_unchecked());

            vec.resize(vec.len().next_multiple_of(size_of::<u8>()), 0);
            vec.extend_from_slice(try_cast_slice(&bitvec).unwrap_unchecked());

            for &delta in components_chunked.iter() {
                let c = delta as u16;
                if c > u8::MAX as u16 {
                    vec.extend_from_slice(&c.to_ne_bytes());
                } else {
                    vec.extend_from_slice(&[c as u8]);
                }
            }
            vec.resize(vec.len().next_multiple_of(size_of::<Simd<u8, N_OPT>>()), 0);

            vec.extend_from_slice(try_cast_slice(values_chunked).unwrap_unchecked());
            const { assert!(size_of::<u64>().is_multiple_of(size_of::<Simd<u8, N_OPT>>())) };
            vec.resize(vec.len().next_multiple_of(size_of::<Simd<u8, N_OPT>>()), 0);
        }
    }

    fn iter_raw(self) -> impl ExactSizeIterator<Item = (Simd<u16, N_OPT>, Simd<u8, N_OPT>)> {
        let mut total_scroll = 0;
        self.idx_lens
            .iter()
            .map(move |idx_len| {
                let bytes = unsafe {
                    self.bytes
                        .as_ptr()
                        .add(total_scroll)
                        .cast::<Simd<u8, 16>>()
                        .read_unaligned()
                };
                let mask = super::dotvbyte_fixedu8::MASKS[*idx_len as usize];
                total_scroll += 8 + idx_len.count_ones() as usize;
                let result = swizzle(bytes, mask);
                unsafe { transmute_copy(&result) }
            })
            .zip(self.values.iter().cloned())
    }

    fn iter(self) -> impl Iterator<Item = (u32, u8)> {
        let bytes_remaining = self.bytes_remaining;
        let values_remaining = self.values_remaining;
        gen move {
            let mut last_component = 0u32;
            for (delta_u16, v) in self.iter_raw() {
                let c_prefixed = super::dotvbyte_fixedu8::simd_prefix_sum(delta_u16).cast::<u32>();
                let c_prefixed_previous = c_prefixed + Simd::splat(last_component);
                last_component = *c_prefixed_previous.to_array().last().unwrap();

                for (c, v) in c_prefixed_previous
                    .to_array()
                    .into_iter()
                    .zip(v.to_array().into_iter())
                {
                    yield (c, v);
                }
            }

            for (delta, v) in bytes_remaining.iter().zip(values_remaining.iter()) {
                last_component += *delta as u32;
                yield (last_component, *v);
            }
        }
    }

    fn dot_product(&self, mut query: &[f32]) -> f32 {
        let mut result = Simd::<f32, N_OPT>::splat(0.0);

        for (components, values_u8) in self.clone().iter_raw() {
            let components = super::dotvbyte_fixedu8::simd_prefix_sum(components);

            let values = simd_scalaru8_to_f32(values_u8);
            let query_values = unsafe {
                Simd::gather_select_unchecked(
                    query,
                    Mask::splat(true),
                    components.cast(),
                    Simd::splat(0.0),
                )
            };

            result = values.mul_add(query_values, result);

            let last_component = *components.to_array().last().unwrap();
            query = unsafe { query.split_at_unchecked(last_component as usize).1 };
        }

        let simd_result = result.reduce_sum();

        let remaining_result = self
            .bytes_remaining
            .iter()
            .zip(self.values_remaining.iter())
            .scan(0u32, move |acc, (&delta, &v)| {
                *acc += delta as u32;
                let vector_value = v as f32;
                Some(unsafe {
                    query
                        .get_unchecked(*acc as usize)
                        .algebraic_mul(vector_value)
                })
            })
            .fold(0f32, |acc, x| acc.algebraic_add(x));

        simd_result.algebraic_add(remaining_result)
    }
}

impl SpaceUsage for OptimisticDotVByteScalarU8Encoder {
    #[inline]
    fn space_usage_bytes(&self) -> usize {
        self.inner.space_usage_bytes()
    }
}

impl SparseDataEncoder for DotVByteU32ScalarU8Encoder {
    type InputComponentType = u32;
    type InputValueType = f32;
    type OutputComponentType = u32;
    type OutputValueType = u8;

    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let dotvbyte_view = unsafe { DotVbyteU32ScalarU8::from_unchecked_slice(encoded.data()) };

        let mut components: Vec<u32> = Vec::new();
        let mut values: Vec<f32> = Vec::new();
        for (component, value) in dotvbyte_view.iter() {
            components.push(component);
            values.push(value as f32 * self.quants[component as usize]);
        }

        // Components are stored in mapped space if a mapping is present.
        if let Some(component_mapping) = self.component_mapping() {
            let inverse: Cow<'_, [u32]> = match self.inverse_component_mapping() {
                Some(inverse) => Cow::Borrowed(inverse),
                None => Cow::Owned(Self::compute_inverse_mapping(component_mapping)),
            };

            for c in components.iter_mut() {
                *c = inverse[*c as usize];
            }

            // After inverse mapping the order is no longer guaranteed.
            let permutation = rusty_perm::PermD::from_sort(components.as_slice());
            permutation.apply(values.as_mut_slice()).unwrap();
            permutation.apply(components.as_mut_slice()).unwrap();
        }

        SparseVectorOwned::new(components, values)
    }
}

impl PackedSparseVectorEncoder for DotVByteU32ScalarU8Encoder {
    type PackedDataType = u64;

    fn push_encoded<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, Self::InputComponentType, Self::InputValueType>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<Self::PackedDataType>,
    {
        let mut encoded_u8 = Vec::new();

        let mut q_values: Vec<u8> = input
            .components()
            .iter()
            .zip(input.values())
            .map(|(&c, &v)| {
                let idx: usize = c.as_();
                let q = self.quants[idx];
                if q > 0.0 {
                    (v / q).clamp(0.0, 255.0) as u8
                } else {
                    0u8
                }
            })
            .collect();

        //If needed, remap components according to the DotVByte quantizer mapping.
        let mut q_components = if let Some(component_mapping) = self.component_mapping() {
            input
                .components()
                .iter()
                .map(|&c| component_mapping[c as usize])
                .collect::<Vec<u32>>()
        } else {
            input.components().to_vec()
        };

        // Sort components and values by component index.
        if self.component_mapping().is_some() {
            let permutation = rusty_perm::PermD::from_sort(q_components.as_slice());
            permutation.apply(q_values.as_mut_slice()).unwrap();
            permutation.apply(q_components.as_mut_slice()).unwrap();
        }

        DotVbyteU32ScalarU8::push_vector(&mut encoded_u8, &mut q_components, &mut q_values);

        assert!(
            encoded_u8.len() % std::mem::size_of::<u64>() == 0,
            "encoded_u8 length ({}) is not a multiple of 8",
            encoded_u8.len()
        );

        let data = encoded_u8.chunks_exact(8).map(|chunk| {
            u64::from_le_bytes(chunk.try_into().unwrap()) // choose LE/BE/NE
        });
        output.extend(data);
    }
}

impl DotVByteU32ScalarU8Encoder {
    #[inline]
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        assert_eq!(
            input_dim, output_dim,
            "DotVByteU32ScalarU8Encoder requires input_dim == output_dim"
        );
        Self {
            dim: input_dim,
            quants: vec![0_f32].into_boxed_slice(),
            component_mapping: None,
            inverse_component_mapping: None,
        }
    }

    pub fn train<V>(
        &mut self,
        training_data: &PlainSparseDataset<u32, f32, SquaredEuclideanDistance>,
    ) where
        V: ValueType,
    {
        const SAMPLE_RATE: usize = 20;
        let sample_size = if training_data.len() / SAMPLE_RATE < 50_000 {
            training_data.len()
        } else {
            training_data.len() / SAMPLE_RATE
        };

        let components_iter = training_data
            .iter()
            .take(sample_size)
            .map(|v| v.components());
        let permutation = permute_components_with_bisection(self.input_dim(), components_iter);
        let component_mapping: Vec<u32> = permutation.iter().map(|i| *i as u32).collect();
        let inverse = Self::compute_inverse_mapping(&component_mapping);
        self.component_mapping = Some(component_mapping.into_boxed_slice());
        self.inverse_component_mapping = Some(inverse.into_boxed_slice());

        self.quants = train_sparse_scalar_quantizer(training_data, 0.0, 1.0).into_boxed_slice();
    }
}

impl VectorEncoder for DotVByteU32ScalarU8Encoder {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u32, f32>;
    type QueryVector<'q> = SparseVectorView<'q, u32, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;

    type Evaluator<'e>
        = DotVByteU32ScalarU8QueryEvaluator<'e>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        DotVByteU32ScalarU8QueryEvaluator::new(query, self)
    }

    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        DotVByteU32ScalarU8QueryEvaluator::new(decoded.as_view(), self)
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
pub struct DotVByteU32ScalarU8QueryEvaluator<'e> {
    dense_query_transformed: Vec<f32>,
    _phantom: PhantomData<&'e ()>,
}

impl<'e> DotVByteU32ScalarU8QueryEvaluator<'e> {
    #[inline]
    pub fn new(
        query: SparseVectorView<'_, u32, f32>,
        quantizer: &'e DotVByteU32ScalarU8Encoder,
    ) -> Self {
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

        let component_mapping = quantizer.component_mapping();
        let mut transformed = vec![0.0f32; quantizer.dim];
        for (&c, &v) in query.components().iter().zip(query.values()) {
            let idx: usize = c.as_();
            let mapped: usize = match component_mapping {
                Some(mapping) => mapping[idx] as usize,
                None => idx,
            };
            transformed[mapped] = v * quantizer.quants[idx];
        }

        Self {
            dense_query_transformed: transformed,
            _phantom: PhantomData,
        }
    }
}

impl<'e, 'v> QueryEvaluator<PackedVectorView<'v, u64>> for DotVByteU32ScalarU8QueryEvaluator<'e> {
    type Distance = DotProduct;

    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> DotProduct {
        let dotvbyte_view = unsafe { DotVbyteU32ScalarU8::from_unchecked_slice(vector.data()) };
        DotProduct::from(dotvbyte_view.dot_product(&self.dense_query_transformed))
    }
}

// With u32 components and 2 bits per component, N=4 components fit in one control byte (4*2=8 bits).
const N: usize = 4;

#[derive(Clone)]
struct DotVbyteU32ScalarU8<'a> {
    values: &'a [Simd<u8, N>],
    bytes_remaining: &'a [u32],
    values_remaining: &'a [u8],
    idx_lens: &'a [u8],
    bytes: &'a [u8],
}

impl<'a> DotVbyteU32ScalarU8<'a> {
    #[inline]
    unsafe fn cast_simd_slice<const LANES: usize>(slice: &'a [u8]) -> &'a [Simd<u8, LANES>] {
        let len = slice.len() / std::mem::size_of::<Simd<u8, LANES>>();
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const Simd<u8, LANES>, len) }
    }

    unsafe fn from_unchecked_slice(slice: &'a [u64]) -> Self {
        unsafe {
            let slice = try_cast_slice::<u64, u8>(slice).unwrap_unchecked();

            let original_length =
                u16::from_ne_bytes([*slice.get_unchecked(0), *slice.get_unchecked(1)]);

            let n_packs = original_length as usize / N;
            let n_remaining = original_length as usize % N;

            let bytes_remaining_start = std::mem::size_of::<u32>();

            let bytes_remaining_start = bytes_remaining_start.next_multiple_of(align_of::<u32>());
            let bytes_remaining_end = bytes_remaining_start + n_remaining * size_of::<u32>();
            let bytes_remaining =
                try_cast_slice(slice.get_unchecked(bytes_remaining_start..bytes_remaining_end))
                    .unwrap_unchecked();

            let values_remaining_start = bytes_remaining_end.next_multiple_of(align_of::<u8>());
            let values_remaining_end = values_remaining_start + n_remaining * size_of::<u8>();
            let values_remaining =
                try_cast_slice(slice.get_unchecked(values_remaining_start..values_remaining_end))
                    .unwrap_unchecked();

            let idx_lens_start = values_remaining_end.next_multiple_of(align_of::<u8>());
            let idx_lens_end = idx_lens_start + n_packs * size_of::<u8>();
            let idx_lens = try_cast_slice(slice.get_unchecked(idx_lens_start..idx_lens_end))
                .unwrap_unchecked();

            let bytes_start = idx_lens_end;

            let values_raw_len = n_packs * std::mem::size_of::<Simd<u8, N>>();
            let values_padded_len = values_raw_len.next_multiple_of(size_of::<u64>());
            let values_start = slice.len() - values_padded_len;
            let values_end = values_start + values_raw_len;

            let bytes =
                try_cast_slice(slice.get_unchecked(bytes_start..values_start)).unwrap_unchecked();

            let values = Self::cast_simd_slice::<N>(slice.get_unchecked(values_start..values_end));

            Self {
                values,
                bytes_remaining,
                values_remaining,
                idx_lens,
                bytes,
            }
        }
    }

    pub fn push_vector(vec: &mut Vec<u8>, converted_components: &mut [u32], values: &mut [u8]) {
        assert!(
            converted_components.len() < u32::MAX as usize,
            "DotVByteU32ScalarU8Encoder only supports vectors shorter than 4294967296."
        );
        assert_eq!(converted_components.len(), values.len());

        let permutation = PermD::from_sort(&*converted_components);
        permutation.apply(values).unwrap();
        permutation.apply(converted_components).unwrap();

        for i in (1..converted_components.len()).rev() {
            converted_components[i] -= converted_components[i - 1];
        }

        let n_chunked = converted_components.len() - converted_components.len() % N;
        let (components_chunked, components_remaining) =
            unsafe { converted_components.split_at_unchecked(n_chunked) };
        let (values_chunked, values_remaining) = unsafe { values.split_at_unchecked(n_chunked) };

        let first_bound = u8::MAX as u32;
        let second_bound = u16::MAX as u32;
        let third_bound = 2_u32.pow(24) - 1;
        let fourth_bound = u32::MAX;

        let bitvec: Vec<u8> = components_chunked
            .chunks_exact(N)
            .map(|chunk| {
                let mut byte = 0;
                for (i, &value) in chunk.iter().enumerate() {
                    let mut bits = 0_u8;
                    if first_bound < value && value <= second_bound {
                        bits = 1_u8;
                    };
                    if second_bound < value && value <= third_bound {
                        bits = 2_u8;
                    };
                    if third_bound < value && value <= fourth_bound {
                        bits = 3_u8;
                    };
                    byte |= bits << ((N - 1 - i) * 2);
                }
                byte
            })
            .collect();

        let original_length = converted_components.len() as u16;
        unsafe {
            vec.extend_from_slice(&original_length.to_ne_bytes());

            vec.resize(vec.len().next_multiple_of(size_of::<u32>()), 0);
            vec.extend_from_slice(try_cast_slice(components_remaining).unwrap_unchecked());

            vec.resize(vec.len().next_multiple_of(size_of::<u8>()), 0);
            vec.extend_from_slice(try_cast_slice(values_remaining).unwrap_unchecked());

            vec.resize(vec.len().next_multiple_of(size_of::<u8>()), 0);
            vec.extend_from_slice(try_cast_slice(&bitvec).unwrap_unchecked());

            for &c in components_chunked.iter() {
                if c <= u8::MAX as u32 {
                    vec.extend_from_slice(&[c as u8]);
                } else if c <= u16::MAX as u32 {
                    let bytes = c.to_ne_bytes();
                    vec.extend_from_slice(&bytes[..2]);
                } else if c <= (2_u32.pow(24) - 1) {
                    let bytes = c.to_ne_bytes();
                    vec.extend_from_slice(&bytes[..3]);
                } else {
                    vec.extend_from_slice(&c.to_ne_bytes());
                }
            }
            // Pad to u64 alignment BEFORE values so that values are the
            // last meaningful bytes in the buffer.  from_unchecked_slice
            // locates them by walking backwards from the end.
            vec.resize(vec.len().next_multiple_of(size_of::<u64>()), 0);

            vec.extend_from_slice(try_cast_slice(values_chunked).unwrap_unchecked());
            // Pad to u64 after values (needed when n_packs * N is not a multiple of 8).
            vec.resize(vec.len().next_multiple_of(size_of::<u64>()), 0);
        }
    }

    #[allow(dead_code)]
    pub fn iter(self) -> impl Iterator<Item = (u32, u8)> {
        let bytes_remaining = self.bytes_remaining;
        let values_remaining = self.values_remaining;
        gen move {
            let mut last_component = 0;
            for (c, v) in self.iter_raw() {
                let c_prefixed = simd_prefix_sum(c);
                let c_prefixed_previous = c_prefixed + Simd::splat(last_component);
                last_component = *c_prefixed_previous.to_array().last().unwrap();

                for (c, v) in c_prefixed_previous
                    .to_array()
                    .into_iter()
                    .zip(v.to_array().into_iter())
                {
                    yield (c, v);
                }
            }
            for (c, v) in bytes_remaining.iter().zip(values_remaining.iter()) {
                last_component += c;
                yield (last_component, *v);
            }
        }
    }

    fn iter_raw(self) -> impl ExactSizeIterator<Item = (Simd<u32, N>, Simd<u8, N>)> {
        let mut total_scroll = 0;
        self.idx_lens
            .iter()
            .map(move |idx_len| {
                let bytes = unsafe {
                    self.bytes
                        .as_ptr()
                        .add(total_scroll)
                        .cast::<Simd<u8, 16>>()
                        .read_unaligned()
                };
                let mask = MASKS[*idx_len as usize];
                total_scroll += 4 + SCROLLS[*idx_len as usize] as usize;
                let result = swizzle(bytes, mask);
                unsafe { transmute_copy(&result) }
            })
            .zip(self.values.iter().cloned())
    }

    pub(crate) fn dot_product(&self, mut query: &[f32]) -> f32 {
        let mut result = Simd::<f32, 4>::splat(0.0);
        for (components, values) in self.clone().iter_raw() {
            let components = simd_prefix_sum(components);
            let values = simd_scalaru8_to_f32(values);
            let query_values = unsafe {
                Simd::gather_select_unchecked(
                    query,
                    Mask::splat(true),
                    components.cast(),
                    Simd::splat(0.0),
                )
            };

            result = values.mul_add(query_values, result);
            let last_component = *components.to_array().last().unwrap();
            query = unsafe { query.split_at_unchecked(last_component as usize).1 };
        }
        let simd_result = result.reduce_sum();

        let remaining_result = self
            .bytes_remaining
            .iter()
            .zip(self.values_remaining.iter())
            .scan(0, move |acc, (&c, &v)| {
                *acc += c;
                let vector_value = v as f32;

                Some(unsafe {
                    query
                        .get_unchecked(*acc as usize)
                        .algebraic_mul(vector_value)
                })
            })
            .fold(0f32, |acc, x| acc.algebraic_add(x));

        simd_result.algebraic_add(remaining_result)
    }
}

impl SpaceUsage for DotVByteU32ScalarU8Encoder {
    #[inline]
    fn space_usage_bytes(&self) -> usize {
        let size_of_mapping = match &self.component_mapping {
            Some(component_mapping) => component_mapping.space_usage_bytes(),
            None => std::mem::size_of::<Option<Box<[u32]>>>(),
        };
        size_of_mapping + self.dim.space_usage_bytes()
    }
}

fn simd_prefix_sum(mut n: Simd<u32, N>) -> Simd<u32, N> {
    if N > 1 {
        n += n.shift_elements_right::<1>(0);
    }
    if N > 2 {
        n += n.shift_elements_right::<2>(0);
    }
    n
}

fn simd_scalaru8_to_f32<const LANES: usize>(f: Simd<u8, LANES>) -> Simd<f32, LANES>
where
    Simd<u8, LANES>: SimdUint<Cast<f32> = Simd<f32, LANES>>,
{
    f.cast::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;
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

    // -----------------------------------------------------------------------
    // DotVByteU32ScalarU8Encoder tests
    // -----------------------------------------------------------------------

    #[test]
    fn new_creates_encoder() {
        let encoder = DotVByteU32ScalarU8Encoder::new(100, 100);
        assert_eq!(encoder.input_dim(), 100);
        assert_eq!(encoder.output_dim(), 100);
        assert!(encoder.component_mapping().is_none());
        assert!(encoder.inverse_component_mapping().is_none());
    }

    #[test]
    #[should_panic(expected = "requires input_dim == output_dim")]
    fn new_panics_on_dim_mismatch() {
        DotVByteU32ScalarU8Encoder::new(10, 20);
    }

    #[test]
    fn training_sets_component_mapping_and_quants() {
        let mut encoder = DotVByteU32ScalarU8Encoder::new(5, 5);
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
        encoder.train::<f32>(&td);

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
        assert!(encoder.quants.len() > 1);
    }

    #[test]
    fn encode_decode_preserves_structure() {
        let mut encoder = DotVByteU32ScalarU8Encoder::new(10, 10);
        let td = build_training_data(
            10,
            &[
                (&[0_u32, 1, 2, 3], &[255.0_f32, 255.0, 255.0, 255.0]),
                (&[0_u32, 5], &[128.0_f32, 64.0]),
            ],
        );
        encoder.train::<f32>(&td);

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

    #[test]
    fn encode_decode_exactly_n_elements() {
        let mut encoder = DotVByteU32ScalarU8Encoder::new(10, 10);
        let td = build_training_data(
            10,
            &[(&[0_u32, 1, 2, 3], &[255.0_f32, 255.0, 255.0, 255.0])],
        );
        encoder.train::<f32>(&td);

        let values = [1.0_f32, 2.0, 0.5, 3.0];
        let input = SparseVectorView::new(&[0_u32, 1, 2, 3], &values);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components().len(), 4);
        for (&decoded_v, &original_v) in decoded.values().iter().zip(values.iter()) {
            assert!(
                (decoded_v - original_v).abs() <= 1.01,
                "decoded {decoded_v} too far from original {original_v}"
            );
        }
    }

    /// Regression: decode must index quants by component, not by the encoded u8 value.
    /// With dim < 256 and an encoded value >= dim the old code would panic.
    #[test]
    fn decode_small_dim_high_value_no_panic() {
        let mut encoder = DotVByteU32ScalarU8Encoder::new(3, 3);
        // quants[c] = 1.0/255 ≈ 0.00392
        let td = build_training_data(3, &[(&[0_u32, 1, 2], &[1.0_f32, 1.0, 1.0])]);
        encoder.train::<f32>(&td);

        // 0.5 / (1/255) ≈ 127 → encoded u8 = 127 >= dim(3)
        let values = [0.5_f32, 0.8];
        let input = SparseVectorView::new(&[0_u32, 2], &values);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &[0_u32, 2]);
        for (&decoded_v, &original_v) in decoded.values().iter().zip(values.iter()) {
            assert!(
                (decoded_v - original_v).abs() < 0.01,
                "decoded {decoded_v} too far from original {original_v}"
            );
        }
    }

    #[test]
    fn encode_vector_default_matches_push_encoded() {
        let mut encoder = DotVByteU32ScalarU8Encoder::new(10, 10);
        let td = build_training_data(
            10,
            &[(&[0_u32, 1, 2, 3], &[255.0_f32, 255.0, 255.0, 255.0])],
        );
        encoder.train::<f32>(&td);

        let values = [1.0_f32, 2.0_f32];
        let input = SparseVectorView::new(&[0_u32, 3], &values);

        let encoded_default = encoder.encode_vector(input);
        let mut encoded_manual = Vec::new();
        encoder.push_encoded(input, &mut encoded_manual);

        assert_eq!(encoded_default.data(), encoded_manual.as_slice());
    }

    // -----------------------------------------------------------------------
    // OptimisticDotVByteScalarU8Encoder tests
    // -----------------------------------------------------------------------

    #[test]
    fn optimistic_encode_decode_preserves_structure() {
        let mut encoder = OptimisticDotVByteScalarU8Encoder::new(10, 10);
        let td = build_training_data(
            10,
            &[
                (&[0_u32, 1, 2, 3], &[255.0_f32, 255.0, 255.0, 255.0]),
                (&[0_u32, 5], &[128.0_f32, 64.0]),
            ],
        );
        encoder.train::<f32>(&td);

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

    #[test]
    fn optimistic_training_sets_component_mapping() {
        let mut encoder = OptimisticDotVByteScalarU8Encoder::new(5, 5);
        let td = build_training_data(
            5,
            &[(
                &[0_u32, 1, 2, 3, 4],
                &[255.0_f32, 255.0, 255.0, 255.0, 255.0],
            )],
        );
        encoder.train::<f32>(&td);

        let mapping = encoder.component_mapping().unwrap();
        let inverse = encoder.inverse_component_mapping().unwrap();
        assert_eq!(mapping.len(), 5);
        assert_eq!(inverse.len(), 5);
    }

    #[test]
    fn optimistic_inject_zero_bridges_no_gap() {
        let components = vec![0u32, 1, 2, 3];
        let values = vec![10u8, 20, 30, 40];
        let (bc, bv) = OptimisticDotVByteScalarU8Encoder::inject_zero_bridges(&components, &values);
        assert_eq!(bc, components);
        assert_eq!(bv, values);
    }

    #[test]
    fn optimistic_inject_zero_bridges_large_gap() {
        let gap = u16::MAX as u32 + 100;
        let components = vec![0u32, gap];
        let values = vec![10u8, 20];
        let (bc, bv) = OptimisticDotVByteScalarU8Encoder::inject_zero_bridges(&components, &values);
        assert!(bc.len() > 2);
        assert_eq!(*bc.last().unwrap(), gap);
        assert_eq!(*bv.last().unwrap(), 20);
        // Bridge values should be zero
        for &v in &bv[1..bv.len() - 1] {
            assert_eq!(v, 0);
        }
    }

    /// End-to-end test for `DotVByteU32ScalarU8Encoder`:
    /// encode → decode → dot product with analytically known values.
    ///
    /// Training with max=255 per component gives quants[c] = 1.0 exactly.
    /// Integer input values in [0,255] are encoded without any quantization error,
    /// so decoded values must match exactly and the dot product is a known integer sum.
    ///
    /// The previous bug (`quants[value as usize]` instead of
    /// `value as f32 * quants[component as usize]`) would have caused an
    /// index-out-of-bounds panic here: dim=4 but encoded u8 values reach 200,
    /// so `quants[200]` is out of bounds.
    #[test]
    fn encode_decode_dot_product_exact() {
        // dim=4, all components trained to max=255 → quants[c] = 1.0 for all c.
        let mut encoder = DotVByteU32ScalarU8Encoder::new(4, 4);
        let td = build_training_data(4, &[(&[0_u32, 1, 2, 3], &[255.0_f32, 255.0, 255.0, 255.0])]);
        encoder.train::<f32>(&td);
        assert!(
            encoder.quants.iter().all(|&q| (q - 1.0).abs() < 1e-6),
            "expected quants == 1.0 for all components"
        );

        // Vector: integer values in [0,255] → encoded u8 == value, decoded == value.
        let vec_values = [100.0_f32, 150.0, 200.0];
        let vec_input = SparseVectorView::new(&[0_u32, 1, 2], &vec_values);

        let mut buffer = Vec::new();
        encoder.push_encoded(vec_input, &mut buffer);

        // --- decode check ---
        // expected: decoded[c] = encoded_u8[c] * quants[c] = value * 1.0 = value
        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &[0_u32, 1, 2]);
        for (&decoded_v, &original_v) in decoded.values().iter().zip(vec_values.iter()) {
            assert!(
                (decoded_v - original_v).abs() < 1e-4,
                "decoded {decoded_v} != original {original_v}"
            );
        }

        // --- dot product check ---
        // query_transformed[c] = query[c] * quants[c] = query[c] * 1.0
        // dot = sum_c(encoded_u8[c] * query_transformed[c])
        //     = 100*2 + 150*3 + 200*1 = 200 + 450 + 200 = 850
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

    /// Same end-to-end test for `OptimisticDotVByteScalarU8Encoder`.
    #[test]
    fn optimistic_encode_decode_dot_product_exact() {
        let mut encoder = OptimisticDotVByteScalarU8Encoder::new(4, 4);
        let td = build_training_data(4, &[(&[0_u32, 1, 2, 3], &[255.0_f32, 255.0, 255.0, 255.0])]);
        encoder.train::<f32>(&td);
        assert!(
            encoder.inner.quants.iter().all(|&q| (q - 1.0).abs() < 1e-6),
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

        // dot = 100*2 + 150*3 + 200*1 = 850
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
}
