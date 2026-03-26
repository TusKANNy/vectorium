use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::marker::PhantomData;

use crate::core::sealed;
use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::{FixedU8Q, SpaceUsage, ValueType};
use num_traits::{AsPrimitive, ToPrimitive};

use rusty_perm::PermApply as _;
use rusty_perm::PermFromSorting as _;

use super::dotvbyte_fixedu8::swizzle::swizzle;
use crate::utils::permute_components_with_bisection;
use std::simd::StdFloat;
use std::simd::num::{SimdFloat, SimdUint};

use std::{
    mem::transmute_copy,
    simd::{Mask, Simd},
};

use bytemuck::try_cast_slice;
use rusty_perm::*;

/// VectorEncoder for DotVByte-packed sparse vectors with support for u32 with `FixedU8Q` values.
///
/// - Encoded vectors are represented as a packed slice of `u64` words.
/// - `output_dim()` is the logical post-quantization dimensionality (typically equal to `input_dim()`),
///   NOT the packed blob length in `u64` words.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DotVByteU32FixedU8Encoder {
    dim: usize,
    component_mapping: Option<Box<[u32]>>, // Optional component remapping that improves compression. mapping[i] = new_index_of_component_i
    #[serde(default)]
    inverse_component_mapping: Option<Box<[u32]>>, // inverse_mapping[new] = old
}

impl sealed::Sealed for DotVByteU32FixedU8Encoder {}

impl DotVByteU32FixedU8Encoder {
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

/// Optimistic DotVByte encoder for sparse vectors with `u32` components and `FixedU8Q` values.
///
/// This encoder behaves like `DotVByteU32FixedU8Encoder`, but in `push_encoded` it injects
/// intermediate zero-valued coordinates whenever a component gap would exceed `u16::MAX`.
/// This keeps most vectors on the fast path while still supporting rare large gaps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimisticDotVByteFixedU8Encoder {
    inner: DotVByteU32FixedU8Encoder,
}

impl sealed::Sealed for OptimisticDotVByteFixedU8Encoder {}

impl OptimisticDotVByteFixedU8Encoder {
    #[inline]
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            inner: DotVByteU32FixedU8Encoder::new(input_dim, output_dim),
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

    pub fn train<'a, V>(
        &mut self,
        training_data: impl Iterator<Item = SparseVectorView<'a, u32, V>>,
    ) where
        V: ValueType,
    {
        self.inner.train(training_data);
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

impl SparseDataEncoder for OptimisticDotVByteFixedU8Encoder {
    type InputComponentType = u32;
    type InputValueType = FixedU8Q;
    type OutputComponentType = u32;
    type OutputValueType = FixedU8Q;

    #[inline]
    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let optimistic_view =
            unsafe { DotVbyteOptimisticU32Fixedu8::from_unchecked_slice(encoded.data()) };

        let mut components: Vec<u32> = Vec::new();
        let mut values: Vec<f32> = Vec::new();
        for (component, value) in optimistic_view.iter() {
            components.push(component);
            values.push(value.to_f32().expect("Failed to convert value to f32"));
        }

        if let Some(component_mapping) = self.component_mapping() {
            let inverse: Cow<'_, [u32]> = match self.inverse_component_mapping() {
                Some(inverse) => Cow::Borrowed(inverse),
                None => Cow::Owned(DotVByteU32FixedU8Encoder::compute_inverse_mapping(
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

impl PackedSparseVectorEncoder for OptimisticDotVByteFixedU8Encoder {
    type PackedDataType = u64;

    fn push_encoded<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, Self::InputComponentType, Self::InputValueType>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<Self::PackedDataType>,
    {
        let mut encoded_u8 = Vec::new();

        let mut q_values: Vec<_> = input.values().iter().map(|v| v.to_bits()).collect();

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

        DotVbyteOptimisticU32Fixedu8::push_vector(
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

impl VectorEncoder for OptimisticDotVByteFixedU8Encoder {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u32, FixedU8Q>;
    type QueryVector<'q> = SparseVectorView<'q, u32, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;

    type Evaluator<'e>
        = OptimisticDotVByteFixedU8QueryEvaluator<'e>
    where
        Self: 'e;

    #[inline]
    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        OptimisticDotVByteFixedU8QueryEvaluator::new(query, self)
    }

    #[inline]
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        OptimisticDotVByteFixedU8QueryEvaluator::new(decoded.as_view(), self)
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
pub struct OptimisticDotVByteFixedU8QueryEvaluator<'e> {
    dense_query: Vec<f32>,
    _phantom: PhantomData<&'e ()>,
}

impl<'e> OptimisticDotVByteFixedU8QueryEvaluator<'e> {
    #[inline]
    pub fn new(
        query: SparseVectorView<'_, u32, f32>,
        quantizer: &'e OptimisticDotVByteFixedU8Encoder,
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

        let mut vec = vec![0.0; quantizer.input_dim()];
        for (&c, &v) in query.components().iter().zip(query.values().iter()) {
            let mapped_component = if let Some(component_mapping) = quantizer.component_mapping() {
                component_mapping[c as usize]
            } else {
                c
            };
            vec[mapped_component as usize] = v;
        }

        Self {
            dense_query: vec,
            _phantom: PhantomData,
        }
    }
}

impl<'e, 'v> QueryEvaluator<PackedVectorView<'v, u64>>
    for OptimisticDotVByteFixedU8QueryEvaluator<'e>
{
    type Distance = DotProduct;

    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> DotProduct {
        let optimistic_view =
            unsafe { DotVbyteOptimisticU32Fixedu8::from_unchecked_slice(vector.data()) };
        DotProduct::from(optimistic_view.dot_product(&self.dense_query))
    }
}

const N_OPT: usize = u8::BITS as usize;

#[derive(Clone)]
struct DotVbyteOptimisticU32Fixedu8<'a> {
    values: &'a [Simd<u8, N_OPT>],
    bytes_remaining: &'a [u16],
    values_remaining: &'a [u8],
    idx_lens: &'a [u8],
    bytes: &'a [u8],
}

impl<'a> DotVbyteOptimisticU32Fixedu8<'a> {
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
            "OptimisticDotVByteFixedU8Encoder only supports vectors shorter than 65535."
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

    fn iter(self) -> impl Iterator<Item = (u32, FixedU8Q)> {
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
                    .zip(v.to_array().into_iter().map(FixedU8Q::from_bits))
                {
                    yield (c, v);
                }
            }

            for (delta, v) in bytes_remaining.iter().zip(values_remaining.iter()) {
                last_component += *delta as u32;
                yield (last_component, FixedU8Q::from_bits(*v));
            }
        }
    }

    fn dot_product(&self, mut query: &[f32]) -> f32 {
        let mut result = Simd::<f32, N_OPT>::splat(0.0);

        for (components, values_u8) in self.clone().iter_raw() {
            let components = super::dotvbyte_fixedu8::simd_prefix_sum(components);

            let values = super::dotvbyte_fixedu8::simd_fixedu8_to_f32(values_u8);
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
                let vector_value = FixedU8Q::from_bits(v).to_f32().unwrap();
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

impl SpaceUsage for OptimisticDotVByteFixedU8Encoder {
    #[inline]
    fn space_usage_bytes(&self) -> usize {
        self.inner.space_usage_bytes()
    }
}

impl SparseDataEncoder for DotVByteU32FixedU8Encoder {
    type InputComponentType = u32;
    type InputValueType = FixedU8Q;
    type OutputComponentType = u32; // but is this needed?
    type OutputValueType = FixedU8Q;

    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let dotvbyte_view = unsafe { DotVbyteU32Fixedu8::from_unchecked_slice(encoded.data()) };

        let mut components: Vec<u32> = Vec::new();
        let mut values: Vec<f32> = Vec::new();
        for (component, value) in dotvbyte_view.iter() {
            components.push(component);
            values.push(value.to_f32().expect("Failed to convert value to f32"));
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

impl PackedSparseVectorEncoder for DotVByteU32FixedU8Encoder {
    type PackedDataType = u64;

    fn push_encoded<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, Self::InputComponentType, Self::InputValueType>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<Self::PackedDataType>,
    {
        let mut encoded_u8 = Vec::new();

        let mut q_values: Vec<_> = input.values().iter().map(|v| v.to_bits()).collect();

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

        DotVbyteU32Fixedu8::push_vector(&mut encoded_u8, &mut q_components, &mut q_values);

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

// This is pacific, just needs type adjustment.

impl DotVByteU32FixedU8Encoder {
    #[inline]
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        assert_eq!(
            input_dim, output_dim,
            "DotVByteU32FixedU8Encoder requires input_dim == output_dim"
        );
        Self {
            dim: input_dim,
            component_mapping: None,
            inverse_component_mapping: None,
        }
    }

    pub fn train<'a, V>(
        &mut self,
        training_data: impl Iterator<Item = SparseVectorView<'a, u32, V>>,
    ) where
        V: ValueType,
    {
        let components_iter = training_data.map(|v| v.components());
        let permutation = permute_components_with_bisection(self.input_dim(), components_iter);
        let component_mapping: Vec<u32> = permutation.iter().map(|i| *i as u32).collect();
        let inverse = Self::compute_inverse_mapping(&component_mapping);
        self.component_mapping = Some(component_mapping.into_boxed_slice());
        self.inverse_component_mapping = Some(inverse.into_boxed_slice());
    }
}

impl VectorEncoder for DotVByteU32FixedU8Encoder {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u32, FixedU8Q>;
    type QueryVector<'q> = SparseVectorView<'q, u32, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;

    type Evaluator<'e>
        = DotVByteU32FixedU8QueryEvaluator<'e>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        DotVByteU32FixedU8QueryEvaluator::new(query, self)
    }

    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        DotVByteU32FixedU8QueryEvaluator::new(decoded.as_view(), self)
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
pub struct DotVByteU32FixedU8QueryEvaluator<'e> {
    dense_query: Vec<f32>,
    _phantom: PhantomData<&'e ()>,
}

impl<'e> DotVByteU32FixedU8QueryEvaluator<'e> {
    #[inline]
    pub fn new(
        query: SparseVectorView<'_, u32, f32>,
        quantizer: &'e DotVByteU32FixedU8Encoder,
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

        // Densify the remapped query vector
        let mut vec = vec![0.0; quantizer.dim];
        let query_components = query.components();
        let query_values = query.values();

        for (&c, &v) in query_components.iter().zip(query_values.iter()) {
            let mapped_component = if let Some(component_mapping) = quantizer.component_mapping() {
                component_mapping[c as usize]
            } else {
                c
            };
            vec[mapped_component as usize] = v;
        }

        Self {
            dense_query: vec,
            _phantom: PhantomData,
        }
    }
}

impl<'e, 'v> QueryEvaluator<PackedVectorView<'v, u64>> for DotVByteU32FixedU8QueryEvaluator<'e> {
    type Distance = DotProduct;

    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> DotProduct {
        let dotvbyte_view = unsafe { DotVbyteU32Fixedu8::from_unchecked_slice(vector.data()) };
        DotProduct::from(dotvbyte_view.dot_product(&self.dense_query))
    }
}

// With u32 components and 2 bits per component, N=4 components fit in one control byte (4*2=8 bits).
const N: usize = 4;
// MASKS and SCROLLS: each control byte (8 bits = 4 components × 2 bits) encodes the byte-width
// of each component gap (1, 2, 3, or 4 bytes). The mask tells the swizzle how to shuffle bytes
// into u32 lanes, and SCROLLS tells how many extra bytes beyond the base 4 were consumed.
pub(super) const MASKS: [Simd<u8, { N * 4 }>; 256] = generate_masks_u32();
pub(super) const SCROLLS: [u8; 256] = generate_scrolls_u32();

#[derive(Clone)]
struct DotVbyteU32Fixedu8<'a> {
    values: &'a [Simd<u8, N>],
    bytes_remaining: &'a [u32],
    values_remaining: &'a [u8],
    idx_lens: &'a [u8],
    bytes: &'a [u8],
}

impl<'a> DotVbyteU32Fixedu8<'a> {
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

            // These `next_multiple_of` are no-ops, I just want to express the importance of alignment.
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

            // Next part is unaligned by design
            let bytes_start = idx_lens_end;
            // TODO: this unbounded length is "safe": *after the swizzle*, the only read values are of this posting.
            // But *before the swizzle*, for the last document, some values may be out of the slice's bounds.

            // Values are packed and aligned to Simd<u8, N>.  We compute their
            // location from the end of the slice.  The values section occupies
            // n_packs * sizeof(Simd<u8, N>) raw bytes, rounded up to u64 for
            // alignment padding.
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
            "DotVByteU32FixedU8Encoder only supports vectors shorter than 4294967296."
        ); // this is because the encoding of a vector store its original length in u16 as the first packed field
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
        let fourth_bound = u32::MAX as u32;

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
                    byte |= bits << (N - 1 - i) * 2;
                }
                byte
            })
            .collect();

        // The encoded vector need to store aligned values which are read in SIMD.
        // It would be reasonable to store them at the very beginning of the encoding.
        // However, we need to store the length of the original vector first, this is because
        // it is needed to compute the number of packed values.
        // To save spave the lenght is stored as u16. This ruins the alignment of the values.
        // To fix this, we store the values after the rest of the encoding, so that they are aligned.
        // Threfore, the encoding layout is as follows:
        // - original length (u16)
        // - remainning values which are not packed
        // - control bits of Stream VByte (a bit for each component)
        // - encoding bytes for the components dgaps
        // - padding to align to Simd<u8, N>
        // - values packed in Simd<u8, N>)

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
    pub fn iter(self) -> impl Iterator<Item = (u32, FixedU8Q)> {
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
                    .zip(v.to_array().into_iter().map(FixedU8Q::from_bits))
                {
                    yield (c, v);
                }
            }
            for (c, v) in bytes_remaining.iter().zip(values_remaining.iter()) {
                last_component += c;
                yield (last_component, FixedU8Q::from_bits(*v));
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
        // This ugly clone is optimized away
        for (components, values) in self.clone().iter_raw() {
            let components = simd_prefix_sum(components);
            let values = simd_fixedu8_to_f32(values);
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
            // New starting point for the gather
            query = unsafe { query.split_at_unchecked(last_component as usize).1 };
        }
        let simd_result = result.reduce_sum();

        let remaining_result = self
            .bytes_remaining
            .iter()
            .zip(self.values_remaining.iter())
            .scan(0, move |acc, (&c, &v)| {
                *acc += c;
                let vector_value = FixedU8Q::from_bits(v).to_f32().unwrap();

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

impl SpaceUsage for DotVByteU32FixedU8Encoder {
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

fn simd_fixedu8_to_f32(f: Simd<u8, N>) -> Simd<f32, N> {
    let converted_f32 = f.cast();
    let mult = Simd::splat(1.0 / (1 << crate::FixedU8Q::FRAC_NBITS) as f32);
    converted_f32 * mult
}

/// For u32 stream VByte with N=4 components per pack, each component uses 2 control bits
/// encoding byte-widths: 0b00=1 byte, 0b01=2 bytes, 0b10=3 bytes, 0b11=4 bytes.
/// The mask shuffles the variable-length encoded bytes into 4 u32 lanes (N*4 = 16 bytes output).
const fn generate_masks_u32() -> [Simd<u8, { N * 4 }>; 256] {
    let mut masks = [Simd::splat(0); 256];
    let mut i: usize = 0;
    while i < masks.len() {
        masks[i] = generate_mask_u32(i as u8);
        i += 1;
    }
    masks
}

/// Build the swizzle mask for one control byte `i`.
/// Each of the 4 components has a 2-bit code (MSB-first) giving byte_width = code + 1.
/// The mask places each component's stream bytes into the correct positions of a u32 lane,
/// with 0xFF for unused high bytes (swizzle maps 0xFF → 0).
const fn generate_mask_u32(i: u8) -> Simd<u8, { N * 4 }> {
    let mut mask = [u8::MAX; N * 4];
    let mut j: usize = 0;
    let mut scroll: u8 = 0;
    while j < N {
        // Extract 2-bit code for component j (MSB-first in the control byte)
        let code = ((i >> (6 - j * 2)) & 0b11) as u8;
        let byte_width = code + 1; // 1, 2, 3, or 4 bytes

        // Place stream bytes into the u32 lane in native byte order.
        // On LE: low bytes at positions 0..byte_width
        // On BE: low bytes at positions (4-byte_width)..4
        let lane = make_u32_lane(scroll, byte_width);
        mask[j * 4] = lane[0];
        mask[j * 4 + 1] = lane[1];
        mask[j * 4 + 2] = lane[2];
        mask[j * 4 + 3] = lane[3];

        scroll += byte_width;
        j += 1;
    }

    Simd::from_array(mask)
}

/// Build the 4-byte swizzle lane for a single component with `byte_width` valid stream bytes
/// starting at offset `scroll`. Unused bytes are 0xFF (zeroed by swizzle).
const fn make_u32_lane(scroll: u8, byte_width: u8) -> [u8; 4] {
    let mut lane = [0xFFu8; 4];
    let bw = byte_width as usize;
    if cfg!(target_endian = "little") {
        // LE: least significant byte at position 0
        let mut k = 0;
        while k < bw {
            lane[k] = scroll + k as u8;
            k += 1;
        }
    } else {
        // BE: least significant byte at position 3
        let mut k = 0;
        while k < bw {
            lane[4 - bw + k] = scroll + k as u8;
            k += 1;
        }
    }
    lane
}

/// SCROLLS[control_byte] = number of extra bytes consumed beyond the base N (=4) bytes.
/// Each component uses (code + 1) bytes, so total extra = sum of codes.
const fn generate_scrolls_u32() -> [u8; 256] {
    let mut scrolls = [0u8; 256];
    let mut i: usize = 0;
    while i < 256 {
        let mut total = 0u8;
        let mut j: usize = 0;
        while j < N {
            let code = ((i >> (6 - j * 2)) & 0b11) as u8;
            total += code;
            j += 1;
        }
        scrolls[i] = total;
        i += 1;
    }
    scrolls
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FromF32;
    use crate::core::vector::{PackedVectorView, SparseVectorView};
    fn fixed(val: f32) -> FixedU8Q {
        FixedU8Q::from_f32_saturating(val)
    }

    // -----------------------------------------------------------------------
    // Mask and scroll generation
    // -----------------------------------------------------------------------

    #[test]
    fn masks_all_one_byte() {
        // Control byte 0x00: all four components use 1 byte (code=0).
        // Mask should place stream bytes 0,1,2,3 into the LSB of each u32 lane.
        let mask = generate_mask_u32(0x00);
        let arr = mask.to_array();
        // Lane 0: byte at scroll=0
        assert_eq!(arr[0], 0);
        assert_eq!(arr[1], 0xFF);
        assert_eq!(arr[2], 0xFF);
        assert_eq!(arr[3], 0xFF);
        // Lane 1: byte at scroll=1
        assert_eq!(arr[4], 1);
        assert_eq!(arr[5], 0xFF);
        assert_eq!(arr[6], 0xFF);
        assert_eq!(arr[7], 0xFF);
        // Lane 2: byte at scroll=2
        assert_eq!(arr[8], 2);
        assert_eq!(arr[9], 0xFF);
        assert_eq!(arr[10], 0xFF);
        assert_eq!(arr[11], 0xFF);
        // Lane 3: byte at scroll=3
        assert_eq!(arr[12], 3);
        assert_eq!(arr[13], 0xFF);
        assert_eq!(arr[14], 0xFF);
        assert_eq!(arr[15], 0xFF);
    }

    #[test]
    fn masks_all_four_bytes() {
        // Control byte 0xFF (0b11_11_11_11): all four components use 4 bytes.
        // Total stream bytes = 16, each lane copies 4 consecutive bytes.
        let mask = generate_mask_u32(0xFF);
        let arr = mask.to_array();
        for j in 0..4 {
            let base = j * 4;
            for k in 0..4 {
                assert_eq!(arr[j * 4 + k], (base + k) as u8);
            }
        }
    }

    #[test]
    fn masks_mixed_first_two_byte_rest_one_byte() {
        // Control byte 0b01_00_00_00 = 0x40: component 0 uses 2 bytes, rest use 1 byte.
        let mask = generate_mask_u32(0x40);
        let arr = mask.to_array();
        // Lane 0: 2 bytes at scroll=0 → [0, 1, 0xFF, 0xFF]
        assert_eq!(arr[0], 0);
        assert_eq!(arr[1], 1);
        assert_eq!(arr[2], 0xFF);
        assert_eq!(arr[3], 0xFF);
        // Lane 1: 1 byte at scroll=2 → [2, 0xFF, 0xFF, 0xFF]
        assert_eq!(arr[4], 2);
        assert_eq!(arr[5], 0xFF);
        // Lane 2: 1 byte at scroll=3
        assert_eq!(arr[8], 3);
        assert_eq!(arr[9], 0xFF);
        // Lane 3: 1 byte at scroll=4
        assert_eq!(arr[12], 4);
        assert_eq!(arr[13], 0xFF);
    }

    #[test]
    fn masks_mixed_three_byte_component() {
        // Control byte 0b10_00_00_00 = 0x80: component 0 uses 3 bytes, rest use 1 byte.
        let mask = generate_mask_u32(0x80);
        let arr = mask.to_array();
        // Lane 0: 3 bytes at scroll=0 → [0, 1, 2, 0xFF]
        assert_eq!(arr[0], 0);
        assert_eq!(arr[1], 1);
        assert_eq!(arr[2], 2);
        assert_eq!(arr[3], 0xFF);
        // Lane 1: 1 byte at scroll=3
        assert_eq!(arr[4], 3);
        assert_eq!(arr[5], 0xFF);
    }

    #[test]
    fn scrolls_all_one_byte() {
        // All 1-byte: no extra bytes beyond base 4.
        assert_eq!(SCROLLS[0x00], 0);
    }

    #[test]
    fn scrolls_all_four_bytes() {
        // All 4-byte: each code=3, total extra = 3*4 = 12.
        assert_eq!(SCROLLS[0xFF], 12);
    }

    #[test]
    fn scrolls_mixed() {
        // 0b01_00_00_00 = 0x40: codes [1,0,0,0], extra = 1.
        assert_eq!(SCROLLS[0x40], 1);
        // 0b11_01_10_00 = 0xD8: codes [3,1,2,0], extra = 6.
        assert_eq!(SCROLLS[0xD8], 6);
        // 0b01_01_01_01 = 0x55: all code=1, extra = 4.
        assert_eq!(SCROLLS[0x55], 4);
    }

    #[test]
    fn scrolls_symmetry() {
        // Total extra bytes for every control byte = sum of its four 2-bit fields.
        for i in 0..256u16 {
            let expected = ((i >> 6) & 3) + ((i >> 4) & 3) + ((i >> 2) & 3) + (i & 3);
            assert_eq!(SCROLLS[i as usize], expected as u8, "mismatch at i={i}");
        }
    }

    // -----------------------------------------------------------------------
    // make_u32_lane
    // -----------------------------------------------------------------------

    #[test]
    fn make_u32_lane_one_byte() {
        let lane = make_u32_lane(5, 1);
        assert_eq!(lane[0], 5); // LSB on LE
        assert_eq!(lane[1], 0xFF);
        assert_eq!(lane[2], 0xFF);
        assert_eq!(lane[3], 0xFF);
    }

    #[test]
    fn make_u32_lane_two_bytes() {
        let lane = make_u32_lane(3, 2);
        assert_eq!(lane[0], 3);
        assert_eq!(lane[1], 4);
        assert_eq!(lane[2], 0xFF);
        assert_eq!(lane[3], 0xFF);
    }

    #[test]
    fn make_u32_lane_three_bytes() {
        let lane = make_u32_lane(0, 3);
        assert_eq!(lane[0], 0);
        assert_eq!(lane[1], 1);
        assert_eq!(lane[2], 2);
        assert_eq!(lane[3], 0xFF);
    }

    #[test]
    fn make_u32_lane_four_bytes() {
        let lane = make_u32_lane(7, 4);
        assert_eq!(lane, [7, 8, 9, 10]);
    }

    // -----------------------------------------------------------------------
    // simd helpers
    // -----------------------------------------------------------------------

    #[test]
    fn simd_prefix_sum_basic() {
        let input = Simd::from_array([1u32, 2, 3, 4]);
        let result = simd_prefix_sum(input);
        assert_eq!(result.to_array(), [1, 3, 6, 10]);
    }

    #[test]
    fn simd_prefix_sum_zeros() {
        let input = Simd::from_array([0u32, 0, 0, 0]);
        assert_eq!(simd_prefix_sum(input).to_array(), [0, 0, 0, 0]);
    }

    #[test]
    fn simd_prefix_sum_ones() {
        let input = Simd::from_array([1u32, 1, 1, 1]);
        assert_eq!(simd_prefix_sum(input).to_array(), [1, 2, 3, 4]);
    }

    #[test]
    fn simd_fixedu8_to_f32_round_trip() {
        let val = fixed(1.5);
        let bits = val.to_bits();
        let input = Simd::from_array([bits, 0, 0, 0]);
        let result = simd_fixedu8_to_f32(input);
        let expected = val.to_f32().unwrap();
        assert!((result.to_array()[0] - expected).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Encoder construction
    // -----------------------------------------------------------------------

    #[test]
    fn new_creates_encoder() {
        let encoder = DotVByteU32FixedU8Encoder::new(100, 100);
        assert_eq!(encoder.input_dim(), 100);
        assert_eq!(encoder.output_dim(), 100);
        assert!(encoder.component_mapping().is_none());
        assert!(encoder.inverse_component_mapping().is_none());
    }

    #[test]
    #[should_panic(expected = "requires input_dim == output_dim")]
    fn new_panics_on_dim_mismatch() {
        DotVByteU32FixedU8Encoder::new(10, 20);
    }

    #[test]
    fn space_usage_without_mapping() {
        let encoder = DotVByteU32FixedU8Encoder::new(8, 8);
        assert!(encoder.space_usage_bytes() > 0);
    }

    #[test]
    fn space_usage_with_mapping() {
        let mut encoder = DotVByteU32FixedU8Encoder::new(4, 4);
        let training_values = [fixed(1.0), fixed(1.0)];
        let training = vec![SparseVectorView::new(&[0_u32, 1], &training_values)];
        encoder.train(training.into_iter());
        // With mapping allocated, space usage should be larger.
        let with_mapping = encoder.space_usage_bytes();

        let plain = DotVByteU32FixedU8Encoder::new(4, 4);
        let without_mapping = plain.space_usage_bytes();

        assert!(with_mapping > without_mapping);
    }

    // -----------------------------------------------------------------------
    // Encode/decode roundtrip — remaining path (< N elements)
    // -----------------------------------------------------------------------

    #[test]
    fn encode_decode_single_element() {
        let encoder = DotVByteU32FixedU8Encoder::new(10, 10);
        let values = [fixed(2.0)];
        let input = SparseVectorView::new(&[5_u32], &values);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);
        assert!(!buffer.is_empty());

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &[5_u32]);
        assert!((decoded.values()[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn encode_decode_two_elements() {
        let encoder = DotVByteU32FixedU8Encoder::new(100, 100);
        let values = [fixed(1.0), fixed(2.5)];
        let input = SparseVectorView::new(&[0_u32, 50], &values);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &[0_u32, 50]);
        assert!((decoded.values()[0] - 1.0).abs() < 1e-6);
        assert!((decoded.values()[1] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn encode_decode_three_elements() {
        let encoder = DotVByteU32FixedU8Encoder::new(1000, 1000);
        let values = [fixed(0.5), fixed(1.0), fixed(3.0)];
        let input = SparseVectorView::new(&[10_u32, 100, 999], &values);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &[10_u32, 100, 999]);
        assert!((decoded.values()[0] - 0.5).abs() < 1e-6);
        assert!((decoded.values()[1] - 1.0).abs() < 1e-6);
        assert!((decoded.values()[2] - 3.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Encode/decode roundtrip — packed path (>= N elements, small gaps)
    // -----------------------------------------------------------------------

    #[test]
    fn encode_decode_exactly_n_elements() {
        let encoder = DotVByteU32FixedU8Encoder::new(10, 10);
        let values = [fixed(1.0), fixed(2.0), fixed(0.5), fixed(3.0)];
        let input = SparseVectorView::new(&[0_u32, 1, 2, 3], &values);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &[0_u32, 1, 2, 3]);
        assert!((decoded.values()[0] - 1.0).abs() < 1e-6);
        assert!((decoded.values()[1] - 2.0).abs() < 1e-6);
        assert!((decoded.values()[2] - 0.5).abs() < 1e-6);
        assert!((decoded.values()[3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn encode_decode_n_plus_one_elements() {
        // 5 elements = 1 pack of 4 + 1 remaining.
        let encoder = DotVByteU32FixedU8Encoder::new(10, 10);
        let values = [fixed(1.0), fixed(2.0), fixed(0.5), fixed(3.0), fixed(1.5)];
        let input = SparseVectorView::new(&[0_u32, 1, 2, 3, 5], &values);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &[0_u32, 1, 2, 3, 5]);
        assert!((decoded.values()[4] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn encode_decode_two_packs() {
        // 8 elements = exactly 2 packs.
        let encoder = DotVByteU32FixedU8Encoder::new(20, 20);
        let comps = [0_u32, 1, 2, 3, 5, 6, 7, 8];
        let vals = [
            fixed(1.0),
            fixed(2.0),
            fixed(0.5),
            fixed(3.0),
            fixed(1.5),
            fixed(2.5),
            fixed(0.5),
            fixed(1.0),
        ];
        let input = SparseVectorView::new(&comps, &vals);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &comps);
        for (got, &exp) in decoded
            .values()
            .iter()
            .zip([1.0_f32, 2.0, 0.5, 3.0, 1.5, 2.5, 0.5, 1.0].iter())
        {
            assert!((got - exp).abs() < 0.02, "expected {exp}, got {got}");
        }
    }

    #[test]
    fn encode_decode_two_packs_plus_remaining() {
        // 9 elements = 2 packs + 1 remaining.
        let encoder = DotVByteU32FixedU8Encoder::new(20, 20);
        let comps = [0_u32, 1, 2, 3, 5, 6, 7, 8, 10];
        let vals = [
            fixed(1.0),
            fixed(2.0),
            fixed(0.5),
            fixed(3.0),
            fixed(1.5),
            fixed(2.5),
            fixed(0.5),
            fixed(1.0),
            fixed(2.0),
        ];
        let input = SparseVectorView::new(&comps, &vals);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &comps);
        assert!((decoded.values()[8] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn encode_decode_non_contiguous_small_gaps() {
        // Components with varying but small (< 255) gaps.
        let encoder = DotVByteU32FixedU8Encoder::new(500, 500);
        let comps = [10_u32, 30, 100, 200];
        let vals = [fixed(1.0), fixed(0.5), fixed(2.0), fixed(1.5)];
        let input = SparseVectorView::new(&comps, &vals);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &comps);
    }

    #[test]
    fn encode_decode_packed_one_byte_boundary_gaps() {
        // 4 elements => packed path. Gaps equal to 255 must stay in 1-byte bucket.
        let encoder = DotVByteU32FixedU8Encoder::new(1_100, 1_100);
        let comps = [255_u32, 510, 765, 1020];
        let vals = [fixed(1.0), fixed(2.0), fixed(0.5), fixed(3.0)];
        let input = SparseVectorView::new(&comps, &vals);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &comps);
    }

    #[test]
    fn encode_decode_packed_two_byte_gaps() {
        // 4 elements => packed path. Gaps are in (255, 65535], so 2-byte encoding is required.
        let encoder = DotVByteU32FixedU8Encoder::new(100_000, 100_000);
        let comps = [256_u32, 600, 1200, 1900];
        let vals = [fixed(1.0), fixed(2.0), fixed(0.5), fixed(3.0)];
        let input = SparseVectorView::new(&comps, &vals);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &comps);
    }

    #[test]
    fn encode_decode_packed_three_byte_gaps() {
        // 4 elements => packed path. Gaps are in (65535, 2^24-1], so 3-byte encoding is required.
        let encoder = DotVByteU32FixedU8Encoder::new(16_500_000, 16_500_000);
        let comps = [70_000_u32, 140_100, 210_300, 280_600];
        let vals = [fixed(1.0), fixed(1.5), fixed(2.0), fixed(2.5)];
        let input = SparseVectorView::new(&comps, &vals);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &comps);
    }

    #[test]
    fn encode_decode_unsorted_input() {
        // Input not sorted by component — encoder should handle it.
        let encoder = DotVByteU32FixedU8Encoder::new(10, 10);
        let values = [fixed(3.0), fixed(1.0)];
        let input = SparseVectorView::new(&[5_u32, 1], &values);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        // Decoded should be sorted by component.
        assert_eq!(decoded.components(), &[1_u32, 5]);
        assert!((decoded.values()[0] - 1.0).abs() < 1e-6);
        assert!((decoded.values()[1] - 3.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // encode_vector (default trait method)
    // -----------------------------------------------------------------------

    #[test]
    fn encode_vector_default_uses_push_encoded() {
        let encoder = DotVByteU32FixedU8Encoder::new(10, 10);
        let values = [fixed(1.0), fixed(2.0)];
        let input = SparseVectorView::new(&[0_u32, 3], &values);
        let encoded = encoder.encode_vector(input);
        assert!(!encoded.data().is_empty());
        let decoded = encoder.decode_vector(encoded.as_view());
        assert_eq!(decoded.components(), &[0_u32, 3]);
    }

    // -----------------------------------------------------------------------
    // Query evaluator — dot product
    // -----------------------------------------------------------------------

    #[test]
    fn query_evaluator_remaining_path() {
        // 2 elements (remaining path only, no packs).
        let encoder = DotVByteU32FixedU8Encoder::new(10, 10);
        let mut data = Vec::new();
        let encoded_values = [fixed(1.0), fixed(2.0)];
        encoder.push_encoded(
            SparseVectorView::new(&[0_u32, 3], &encoded_values),
            &mut data,
        );

        // query: [1.0 at 0, 1.0 at 3] → dot = 1.0*1.0 + 2.0*1.0 = 3.0
        let query = SparseVectorView::new(&[0_u32, 3], &[1.0_f32, 1.0]);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&data));
        assert!((dist.0 - 3.0).abs() < 0.1, "expected ~3.0, got {:?}", dist);
    }

    #[test]
    fn query_evaluator_packed_path() {
        // 4 elements (one full pack, no remaining).
        let encoder = DotVByteU32FixedU8Encoder::new(10, 10);
        let mut data = Vec::new();
        let encoded_values = [fixed(1.0), fixed(1.0), fixed(1.0), fixed(1.0)];
        encoder.push_encoded(
            SparseVectorView::new(&[0_u32, 1, 2, 3], &encoded_values),
            &mut data,
        );

        // query: all 1.0 at same positions → dot = 4 * 1.0
        let query = SparseVectorView::new(&[0_u32, 1, 2, 3], &[1.0_f32, 1.0, 1.0, 1.0]);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&data));
        assert!((dist.0 - 4.0).abs() < 0.1, "expected ~4.0, got {:?}", dist);
    }

    #[test]
    fn query_evaluator_orthogonal() {
        // No overlap between query and vector → dot product = 0.
        let encoder = DotVByteU32FixedU8Encoder::new(10, 10);
        let mut data = Vec::new();
        let encoded_values = [fixed(2.0)];
        encoder.push_encoded(SparseVectorView::new(&[0_u32], &encoded_values), &mut data);

        let query = SparseVectorView::new(&[5_u32], &[1.0_f32]);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&data));
        assert!(dist.0.abs() < 1e-6, "expected ~0.0, got {:?}", dist);
    }

    #[test]
    fn query_evaluator_partial_overlap() {
        // Vector at [0, 3], query at [0, 5] → only component 0 overlaps.
        let encoder = DotVByteU32FixedU8Encoder::new(10, 10);
        let mut data = Vec::new();
        let encoded_values = [fixed(2.0), fixed(3.0)];
        encoder.push_encoded(
            SparseVectorView::new(&[0_u32, 3], &encoded_values),
            &mut data,
        );

        let query = SparseVectorView::new(&[0_u32, 5], &[1.5_f32, 1.0]);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&data));
        // dot = 2.0 * 1.5 = 3.0
        assert!((dist.0 - 3.0).abs() < 0.1, "expected ~3.0, got {:?}", dist);
    }

    #[test]
    fn vector_evaluator_self_dot() {
        let encoder = DotVByteU32FixedU8Encoder::new(10, 10);
        let mut data = Vec::new();
        let encoded_values = [fixed(1.0), fixed(2.0)];
        encoder.push_encoded(
            SparseVectorView::new(&[0_u32, 3], &encoded_values),
            &mut data,
        );

        // vector_evaluator decodes the vector and uses it as a query.
        // Self-dot = 1.0^2 + 2.0^2 = 5.0
        let evaluator = encoder.vector_evaluator(PackedVectorView::new(&data));
        let dist = evaluator.compute_distance(PackedVectorView::new(&data));
        assert!((dist.0 - 5.0).abs() < 0.1, "expected ~5.0, got {:?}", dist);
    }

    #[test]
    fn query_evaluator_mixed_pack_and_remaining() {
        // 5 elements: 1 pack + 1 remaining.
        let encoder = DotVByteU32FixedU8Encoder::new(20, 20);
        let mut data = Vec::new();
        let encoded_values = [fixed(1.0), fixed(1.0), fixed(1.0), fixed(1.0), fixed(1.0)];
        encoder.push_encoded(
            SparseVectorView::new(&[0_u32, 1, 2, 3, 5], &encoded_values),
            &mut data,
        );

        let query = SparseVectorView::new(&[0_u32, 1, 2, 3, 5], &[1.0_f32, 1.0, 1.0, 1.0, 1.0]);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&data));
        assert!((dist.0 - 5.0).abs() < 0.1, "expected ~5.0, got {:?}", dist);
    }

    // -----------------------------------------------------------------------
    // Training / component mapping
    // -----------------------------------------------------------------------

    #[test]
    fn training_sets_component_mapping() {
        let mut encoder = DotVByteU32FixedU8Encoder::new(5, 5);
        let training_values = [fixed(1.0), fixed(1.0)];
        let training = vec![SparseVectorView::new(&[0_u32, 2], &training_values)];
        encoder.train(training.into_iter());

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
    }

    #[test]
    fn inverse_mapping_roundtrip() {
        let mut encoder = DotVByteU32FixedU8Encoder::new(6, 6);
        let training_values = [fixed(1.0), fixed(1.0), fixed(1.0)];
        let training = vec![SparseVectorView::new(&[0_u32, 3, 5], &training_values)];
        encoder.train(training.into_iter());

        let mapping = encoder.component_mapping().unwrap();
        let inverse = encoder.inverse_component_mapping().unwrap();

        // mapping[inverse[i]] == i for all i
        for i in 0..6 {
            assert_eq!(mapping[inverse[i] as usize], i as u32);
        }
        // inverse[mapping[i]] == i for all i
        for i in 0..6 {
            assert_eq!(inverse[mapping[i] as usize], i as u32);
        }
    }

    #[test]
    fn compute_inverse_mapping_identity() {
        let mapping: Vec<u32> = (0..5).collect();
        let inverse = DotVByteU32FixedU8Encoder::compute_inverse_mapping(&mapping);
        assert_eq!(inverse, mapping);
    }

    #[test]
    fn compute_inverse_mapping_reversal() {
        let mapping = vec![4u32, 3, 2, 1, 0];
        let inverse = DotVByteU32FixedU8Encoder::compute_inverse_mapping(&mapping);
        assert_eq!(inverse, vec![4u32, 3, 2, 1, 0]); // Reversal is self-inverse.
    }

    #[test]
    fn compute_inverse_mapping_permutation() {
        let mapping = vec![2u32, 0, 1];
        let inverse = DotVByteU32FixedU8Encoder::compute_inverse_mapping(&mapping);
        // mapping: 0→2, 1→0, 2→1 ⇒ inverse: 0→1, 1→2, 2→0
        assert_eq!(inverse, vec![1u32, 2, 0]);
    }

    #[test]
    #[should_panic(expected = "not a permutation")]
    fn compute_inverse_mapping_duplicate_panics() {
        let mapping = vec![0u32, 0, 1];
        DotVByteU32FixedU8Encoder::compute_inverse_mapping(&mapping);
    }

    #[test]
    #[should_panic(expected = "out-of-bounds")]
    fn compute_inverse_mapping_oob_panics() {
        let mapping = vec![0u32, 5, 2];
        DotVByteU32FixedU8Encoder::compute_inverse_mapping(&mapping);
    }

    #[test]
    fn encode_decode_with_mapping_remaining_path() {
        let mut encoder = DotVByteU32FixedU8Encoder::new(5, 5);
        let training_values = [fixed(1.0), fixed(1.0)];
        let training = vec![SparseVectorView::new(&[0_u32, 3], &training_values)];
        encoder.train(training.into_iter());

        let values = [fixed(2.0), fixed(1.0)];
        let mut buffer = Vec::new();
        encoder.push_encoded(SparseVectorView::new(&[0_u32, 3], &values), &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        // Decoded components should be in original space, sorted.
        assert_eq!(decoded.components(), &[0_u32, 3]);
        assert!((decoded.values()[0] - 2.0).abs() < 1e-6);
        assert!((decoded.values()[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn encode_decode_with_mapping_packed_path() {
        let mut encoder = DotVByteU32FixedU8Encoder::new(10, 10);
        let training_values = [fixed(1.0); 4];
        let training = vec![SparseVectorView::new(&[0_u32, 2, 5, 8], &training_values)];
        encoder.train(training.into_iter());

        let values = [fixed(1.0), fixed(2.0), fixed(0.5), fixed(3.0)];
        let mut buffer = Vec::new();
        encoder.push_encoded(
            SparseVectorView::new(&[0_u32, 2, 5, 8], &values),
            &mut buffer,
        );

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &[0_u32, 2, 5, 8]);
        assert!((decoded.values()[0] - 1.0).abs() < 1e-6);
        assert!((decoded.values()[1] - 2.0).abs() < 1e-6);
        assert!((decoded.values()[2] - 0.5).abs() < 1e-6);
        assert!((decoded.values()[3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn query_evaluator_with_mapping() {
        let mut encoder = DotVByteU32FixedU8Encoder::new(10, 10);
        let training_values = [fixed(1.0), fixed(1.0)];
        let training = vec![SparseVectorView::new(&[0_u32, 5], &training_values)];
        encoder.train(training.into_iter());

        let mut data = Vec::new();
        let encoded_values = [fixed(2.0), fixed(3.0)];
        encoder.push_encoded(
            SparseVectorView::new(&[0_u32, 5], &encoded_values),
            &mut data,
        );

        // dot = 2.0*1.0 + 3.0*1.0 = 5.0
        let query = SparseVectorView::new(&[0_u32, 5], &[1.0_f32, 1.0]);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&data));
        assert!((dist.0 - 5.0).abs() < 0.1, "expected ~5.0, got {:?}", dist);
    }

    // -----------------------------------------------------------------------
    // Multiple vectors in same buffer
    // -----------------------------------------------------------------------

    #[test]
    fn multiple_vectors_independent_encode() {
        let encoder = DotVByteU32FixedU8Encoder::new(10, 10);

        let mut buf1 = Vec::new();
        let mut buf2 = Vec::new();

        let v1_values = [fixed(1.0)];
        encoder.push_encoded(SparseVectorView::new(&[0_u32], &v1_values), &mut buf1);
        let v2_values = [fixed(2.0)];
        encoder.push_encoded(SparseVectorView::new(&[5_u32], &v2_values), &mut buf2);

        let d1 = encoder.decode_vector(PackedVectorView::new(&buf1));
        let d2 = encoder.decode_vector(PackedVectorView::new(&buf2));

        assert_eq!(d1.components(), &[0_u32]);
        assert_eq!(d2.components(), &[5_u32]);
        assert!((d1.values()[0] - 1.0).abs() < 1e-6);
        assert!((d2.values()[0] - 2.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn encode_decode_large_dim() {
        let dim = 100_000;
        let encoder = DotVByteU32FixedU8Encoder::new(dim, dim);
        // Components spread across the large dimension, but with small gaps (< 255).
        let comps = [0_u32, 100, 200, 300];
        let vals = [fixed(1.0), fixed(2.0), fixed(0.5), fixed(1.5)];
        let input = SparseVectorView::new(&comps, &vals);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &comps);
    }

    #[test]
    fn encode_decode_adjacent_components() {
        // All gaps are 1 (except the first component).
        let encoder = DotVByteU32FixedU8Encoder::new(20, 20);
        let comps = [0_u32, 1, 2, 3, 4, 5, 6, 7];
        let vals = [fixed(1.0); 8];
        let input = SparseVectorView::new(&comps, &vals);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &comps);
    }

    #[test]
    fn encode_decode_component_zero_only() {
        let encoder = DotVByteU32FixedU8Encoder::new(1, 1);
        let values = [fixed(2.5)];
        let input = SparseVectorView::new(&[0_u32], &values);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &[0_u32]);
        assert!((decoded.values()[0] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn fixedu8_quantization_precision() {
        // Verify that values survive the FixedU8Q roundtrip.
        let encoder = DotVByteU32FixedU8Encoder::new(10, 10);
        let test_vals: Vec<f32> = vec![0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];

        for &v in &test_vals {
            let fv = fixed(v);
            let expected = fv.to_f32().unwrap(); // What FixedU8Q actually stores.

            let values = [fv];
            let input = SparseVectorView::new(&[0_u32], &values);
            let mut buffer = Vec::new();
            encoder.push_encoded(input, &mut buffer);

            let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
            assert!(
                (decoded.values()[0] - expected).abs() < 1e-6,
                "roundtrip mismatch for input {v}: expected {expected}, got {}",
                decoded.values()[0]
            );
        }
    }

    #[test]
    fn optimistic_encoder_inserts_zero_bridge_for_large_gap() {
        let encoder = OptimisticDotVByteFixedU8Encoder::new(200_000, 200_000);
        let values = [fixed(1.0), fixed(2.0)];
        let input = SparseVectorView::new(&[0_u32, 100_000], &values);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &[0_u32, 65_535, 100_000]);
        assert!((decoded.values()[0] - 1.0).abs() < 1e-6);
        assert!(decoded.values()[1].abs() < 1e-6);
        assert!((decoded.values()[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn optimistic_encoder_matches_dot_product_with_large_gap() {
        let encoder = OptimisticDotVByteFixedU8Encoder::new(200_000, 200_000);
        let values = [fixed(1.0), fixed(2.0)];
        let input = SparseVectorView::new(&[0_u32, 100_000], &values);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let query = SparseVectorView::new(&[0_u32, 100_000], &[1.0_f32, 1.0]);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        assert!((dist.0 - 3.0).abs() < 0.1, "expected ~3.0, got {:?}", dist);
    }

    #[test]
    fn optimistic_encoder_does_not_insert_when_not_needed() {
        let encoder = OptimisticDotVByteFixedU8Encoder::new(100_000, 100_000);
        let values = [fixed(1.0), fixed(2.0)];
        let input = SparseVectorView::new(&[0_u32, 50_000], &values);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &[0_u32, 50_000]);
        assert!((decoded.values()[0] - 1.0).abs() < 1e-6);
        assert!((decoded.values()[1] - 2.0).abs() < 1e-6);
    }
}
