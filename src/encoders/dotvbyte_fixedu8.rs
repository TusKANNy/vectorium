use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

mod swizzle;

use crate::core::sealed;
use crate::core::vector_encoder::{PackedSparseVectorEncoder, QueryEvaluator, VectorEncoder};
use crate::core::vector1d::{PackedVectorView, SparseVector1DView};
use crate::distances::DotProduct;
use crate::{FixedU8Q, SpaceUsage, ValueType};
use num_traits::{AsPrimitive, ToPrimitive};

use rusty_perm::PermApply as _;
use rusty_perm::PermFromSorting as _;

use self::swizzle::*;
use crate::utils::permute_components_with_bisection;
use std::simd::StdFloat;
use std::simd::num::{SimdFloat, SimdUint};

use std::{
    mem::transmute_copy,
    simd::{Mask, Simd},
};

use bytemuck::try_cast_slice;
use rusty_perm::*;

/// VectorEncoder for DotVByte-packed sparse vectors with `FixedU8Q` values.
///
/// - Encoded vectors are represented as a packed slice of `u64` words.
/// - `output_dim()` is the logical post-quantization dimensionality (typically equal to `input_dim()`),
///   NOT the packed blob length in `u64` words.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DotVByteFixedU8Encoder {
    dim: usize,
    component_mapping: Option<Box<[u16]>>, // Optional component remapping that improves compression. mapping[i] = new_index_of_component_i
}

impl sealed::Sealed for DotVByteFixedU8Encoder {}

impl DotVByteFixedU8Encoder {
    #[inline]
    pub fn component_mapping(&self) -> Option<&[u16]> {
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
}

impl PackedSparseVectorEncoder for DotVByteFixedU8Encoder {
    type InputComponentType = u16;
    type InputValueType = FixedU8Q;
    type PackedValueType = u64;

    fn push_encoded<'a, OutputContainer>(
        &self,
        input: SparseVector1DView<'a, Self::InputComponentType, Self::InputValueType>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<Self::PackedValueType>,
    {
        let mut encoded_u8 = Vec::new();

        let mut q_values: Vec<_> = input.values().iter().map(|v| v.to_bits()).collect();

        //If needed, remap components according to the DotVByte quantizer mapping.
        let mut q_components = if let Some(component_mapping) = self.component_mapping() {
            input
                .components()
                .iter()
                .map(|&c| component_mapping[c as usize])
                .collect::<Vec<u16>>()
        } else {
            input.components().to_vec()
        };

        // Sort components and values by component index.
        if self.component_mapping().is_some() {
            let permutation = rusty_perm::PermD::from_sort(q_components.as_slice());
            permutation.apply(q_values.as_mut_slice()).unwrap();
            permutation.apply(q_components.as_mut_slice()).unwrap();
        }

        DotVbyteFixedu8::push_vector(&mut encoded_u8, &mut q_components, &mut q_values);

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

impl DotVByteFixedU8Encoder {
    #[inline]
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        assert_eq!(
            input_dim, output_dim,
            "DotVByteFixedU8Encoder requires input_dim == output_dim"
        );
        Self {
            dim: input_dim,
            component_mapping: None,
        }
    }

    pub fn train<'a, V>(
        &mut self,
        training_data: impl Iterator<Item = SparseVector1DView<'a, u16, V>>,
    ) where
        V: ValueType,
    {
        let components_iter = training_data.map(|v| v.components());
        let permutation = permute_components_with_bisection(self.input_dim(), components_iter);
        let component_mapping: Vec<u16> = permutation.iter().map(|i| *i as u16).collect();
        self.component_mapping = Some(component_mapping.into_boxed_slice());
    }
}

impl VectorEncoder for DotVByteFixedU8Encoder {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVector1DView<'a, u16, FixedU8Q>;
    type QueryVector<'a, V>
        = SparseVector1DView<'a, u16, V>
    where
        V: ValueType;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;

    type Evaluator<'a, V>
        = DotVByteFixedU8QueryEvaluator<'a, V>
    where
        V: ValueType,
        Self: 'a;

    fn query_evaluator<'a, V>(&'a self, query: Self::QueryVector<'a, V>) -> Self::Evaluator<'a, V>
    where
        V: ValueType,
    {
        DotVByteFixedU8QueryEvaluator::new(query, self)
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
pub struct DotVByteFixedU8QueryEvaluator<'a, V: ValueType> {
    dense_query: Vec<f32>,
    _phantom: PhantomData<&'a V>,
}

impl<'a, V: ValueType> DotVByteFixedU8QueryEvaluator<'a, V> {
    #[inline]
    pub fn new(query: SparseVector1DView<'a, u16, V>, quantizer: &DotVByteFixedU8Encoder) -> Self {
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
            vec[mapped_component as usize] = v.to_f32().unwrap();
        }

        Self {
            dense_query: vec,
            _phantom: PhantomData,
        }
    }
}

impl<'a, 'v, V: ValueType> QueryEvaluator<PackedVectorView<'v, u64>>
    for DotVByteFixedU8QueryEvaluator<'a, V>
{
    type Distance = DotProduct;

    #[inline]
    fn compute_distance(&mut self, vector: PackedVectorView<'v, u64>) -> DotProduct {
        let dotvbyte_view = unsafe { DotVbyteFixedu8::from_unchecked_slice(vector.data()) };
        DotProduct::from(dotvbyte_view.dot_product(&self.dense_query))
    }
}

const N: usize = u8::BITS as usize;
pub(super) const MASKS: [Simd<u8, { N * 2 }>; 256] = generate_masks_u16();

#[derive(Clone)]
struct DotVbyteFixedu8<'a> {
    values: &'a [Simd<u8, N>],
    bytes_remaining: &'a [u16],
    values_remaining: &'a [u8],
    idx_lens: &'a [u8],
    bytes: &'a [u8],
}

impl<'a> DotVbyteFixedu8<'a> {
    unsafe fn from_unchecked_slice(slice: &'a [u64]) -> Self {
        unsafe {
            let slice = try_cast_slice::<u64, u8>(slice).unwrap_unchecked();

            let original_length =
                u16::from_ne_bytes([*slice.get_unchecked(0), *slice.get_unchecked(1)]);
            // println!("Original length: {}", original_length);

            let n_packs = original_length as usize / N;
            let n_remaining = original_length as usize % N;

            let bytes_remaining_start = std::mem::size_of::<u16>();

            // These `next_multiple_of` are no-ops, I just want to express the importance of alignment.
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

            // Next part is unaligned by design
            let bytes_start = idx_lens_end;
            // TODO: this unbounded length is "safe": *after the swizzle*, the only read values are of this posting.
            // But *before the swizzle*, for the last document, some values may be out of the slice's bounds.

            // Values are packed and aligned to <Simd<u8, N>. We compute its starting location
            // form the end of the slice. This is beacuse vbyte encodings in bytes are unaligned by design and of variable unknown length.
            let values_start = slice.len() - n_packs * std::mem::size_of::<Simd<u8, N>>();

            let bytes =
                try_cast_slice(slice.get_unchecked(bytes_start..values_start)).unwrap_unchecked();

            let values_end = slice.len();
            let values =
                try_cast_slice(slice.get_unchecked(values_start..values_end)).unwrap_unchecked();

            Self {
                values,
                bytes_remaining,
                values_remaining,
                idx_lens,
                bytes,
            }
        }
    }

    pub fn push_vector(vec: &mut Vec<u8>, converted_components: &mut [u16], values: &mut [u8]) {
        assert!(
            converted_components.len() < u16::MAX as usize,
            "DotVByteFixedU8Encoder only supports vectors shorter than 65535."
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

        let bitvec: Vec<u8> = components_chunked
            .chunks_exact(N)
            .map(|chunk| {
                let mut byte = 0;
                for (i, b) in chunk.iter().map(|&n| n > u8::MAX as u16).enumerate() {
                    byte |= (b as u8) << (N - i - 1)
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

            vec.resize(vec.len().next_multiple_of(size_of::<u16>()), 0);
            vec.extend_from_slice(try_cast_slice(components_remaining).unwrap_unchecked());

            vec.resize(vec.len().next_multiple_of(size_of::<u8>()), 0);
            vec.extend_from_slice(try_cast_slice(values_remaining).unwrap_unchecked());

            vec.resize(vec.len().next_multiple_of(size_of::<u8>()), 0);
            vec.extend_from_slice(try_cast_slice(&bitvec).unwrap_unchecked());

            for &c in components_chunked.iter() {
                if c > u8::MAX as u16 {
                    vec.extend_from_slice(&c.to_ne_bytes());
                } else {
                    vec.extend_from_slice(&[c as u8]);
                }
            }
            vec.resize(vec.len().next_multiple_of(size_of::<Simd<u8, N>>()), 0);

            vec.extend_from_slice(try_cast_slice(values_chunked).unwrap_unchecked());
            // The vector is aligned to a u64
            const { assert!(size_of::<u64>().is_multiple_of(size_of::<Simd<u8, N>>())) };
            vec.resize(vec.len().next_multiple_of(size_of::<Simd<u8, N>>()), 0);
        }
    }

    fn iter_raw(self) -> impl ExactSizeIterator<Item = (Simd<u16, N>, Simd<u8, N>)> {
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
                total_scroll += 8 + idx_len.count_ones() as usize;
                let result = swizzle(bytes, mask);
                unsafe { transmute_copy(&result) }
            })
            .zip(self.values.iter().cloned())
    }

    #[allow(dead_code)]
    pub fn iter(self) -> impl Iterator<Item = (u16, FixedU8Q)> {
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

    pub(crate) fn dot_product(&self, mut query: &[f32]) -> f32 {
        let mut result = Simd::<f32, 8>::splat(0.0);
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

const fn generate_masks_u16() -> [Simd<u8, { N * 2 }>; 256] {
    let mut masks = [Simd::splat(0); 256];
    let mut i: usize = 0;
    while i < masks.len() {
        let mask = generate_mask_u16(i as u8);
        masks[i] = mask;
        i += 1;
    }
    masks
}

const fn generate_mask_u16(i: u8) -> Simd<u8, { N * 2 }> {
    let mut mask = [u8::MAX; N * 2];
    let mut j = 0;
    let mut scroll = 0;
    while j < 8 {
        let bytes = if (i & (0b1000_0000 >> j)) > 0 {
            // If two bytes, they are already in the correct endianness
            let n = [scroll, scroll + 1];
            scroll += 2;
            n
        } else {
            // If one byte, swizzle to the correct endianness
            let n = u16::from_be_bytes([u8::MAX, scroll]).to_ne_bytes();
            scroll += 1;
            n
        };
        mask[j * 2] = bytes[0];
        mask[j * 2 + 1] = bytes[1];

        j += 1;
    }

    Simd::from_array(mask)
}

fn simd_prefix_sum<const N: usize>(mut n: Simd<u16, N>) -> Simd<u16, N>
where
    std::simd::LaneCount<N>: std::simd::SupportedLaneCount,
{
    // I'd use a for loop, but the const argument prevents doing that...
    // God I wish there was an easier way to do this
    if N > 1 {
        n += n.shift_elements_right::<1>(0);
    }
    if N > 2 {
        n += n.shift_elements_right::<2>(0);
    }
    if N > 4 {
        n += n.shift_elements_right::<4>(0);
    }
    // TODO: N more than 8
    n
}

fn simd_fixedu8_to_f32<const N: usize>(f: Simd<u8, N>) -> Simd<f32, N>
where
    std::simd::LaneCount<N>: std::simd::SupportedLaneCount,
{
    let converted_f32 = f.cast();
    // This is *so* hardcoded
    let mult = Simd::splat(1.0 / (1 << crate::FixedU8Q::FRAC_NBITS) as f32);
    converted_f32 * mult
}

impl SpaceUsage for DotVByteFixedU8Encoder {
    #[inline]
    fn space_usage_bytes(&self) -> usize {
        let size_of_mapping = match &self.component_mapping {
            Some(component_mapping) => component_mapping.space_usage_bytes(),
            None => std::mem::size_of::<Option<Box<[u16]>>>(),
        };
        size_of_mapping + self.dim.space_usage_bytes()
    }
}
