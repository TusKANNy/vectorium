use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::marker::PhantomData;

use crate::core::sealed;
use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::encoders::dotvbyte_fixedu8::swizzle::swizzle;
use crate::{Dataset, PlainSparseDataset, SpaceUsage, SquaredEuclideanDistance, ValueType};
use num_traits::AsPrimitive;

use rusty_perm::PermApply as _;
use rusty_perm::PermFromSorting as _;

use crate::utils::{permute_components_with_bisection, train_sparse_scalar_quantizer};
use std::simd::StdFloat;
use std::simd::num::{SimdFloat, SimdUint};

use std::{
    mem::transmute_copy,
    simd::{Mask, Simd},
};

use bytemuck::try_cast_slice;
use rusty_perm::*;

/// VectorEncoder for DotVByte-packed sparse vectors with scalar quantized u8 values.
///
/// - Encoded vectors are represented as a packed slice of `u64` words.
/// - `output_dim()` is the logical post-quantization dimensionality (typically equal to `input_dim()`),
///   NOT the packed blob length in `u64` words.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DotVByteScalarU8Encoder {
    dim: usize,
    quants: Box<[f32]>,
    component_mapping: Option<Box<[u16]>>, // Optional component remapping that improves compression. mapping[i] = new_index_of_component_i
    #[serde(default)]
    inverse_component_mapping: Option<Box<[u16]>>, // inverse_mapping[new] = old
}

impl sealed::Sealed for DotVByteScalarU8Encoder {}

impl DotVByteScalarU8Encoder {
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

    #[inline]
    pub fn inverse_component_mapping(&self) -> Option<&[u16]> {
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

    fn compute_inverse_mapping(component_mapping: &[u16]) -> Vec<u16> {
        let dim = component_mapping.len();
        let mut inverse = vec![0u16; dim];
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
            inverse[new] = old as u16;
        }

        inverse
    }
}

impl SparseDataEncoder for DotVByteScalarU8Encoder {
    type InputComponentType = u16;
    type InputValueType = f32;
    type OutputComponentType = u16;
    type OutputValueType = u8;

    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let dotvbyte_view = unsafe { DotVbyteScalarU8::from_unchecked_slice(encoded.data()) };

        let mut components: Vec<u16> = Vec::new();
        let mut values: Vec<f32> = Vec::new();
        for (component, value) in dotvbyte_view.iter() {
            components.push(component);
            values.push(self.quants[value as usize]);
        }

        // Components are stored in mapped space if a mapping is present.
        if let Some(component_mapping) = self.component_mapping() {
            let inverse: Cow<'_, [u16]> = match self.inverse_component_mapping() {
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

impl PackedSparseVectorEncoder for DotVByteScalarU8Encoder {
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

        DotVbyteScalarU8::push_vector(&mut encoded_u8, &mut q_components, &mut q_values);

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

impl DotVByteScalarU8Encoder {
    #[inline]
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        assert_eq!(
            input_dim, output_dim,
            "DotVByteFixedU8Encoder requires input_dim == output_dim"
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
        //training_data: impl Iterator<Item = SparseVectorView<'a, u16, V>>,
        training_data: &PlainSparseDataset<u16, f32, SquaredEuclideanDistance>,
    ) where
        V: ValueType,
    {
        // Mirror the sampling done in `From<SparseDatasetGeneric> for
        // PackedSparseDataset<DotVByteFixedU8Encoder>`: bisection on the full
        // dataset is expensive (O(nnz * iters)) and a small sample gives an
        // essentially identical permutation.
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
        let component_mapping: Vec<u16> = permutation.iter().map(|i| *i as u16).collect();
        let inverse = Self::compute_inverse_mapping(&component_mapping);
        self.component_mapping = Some(component_mapping.into_boxed_slice());
        self.inverse_component_mapping = Some(inverse.into_boxed_slice());

        self.quants = train_sparse_scalar_quantizer(training_data, 0.0, 1.0).into_boxed_slice();
    }
}

impl VectorEncoder for DotVByteScalarU8Encoder {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u16, f32>;
    type QueryVector<'q> = SparseVectorView<'q, u16, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;

    type Evaluator<'e>
        = DotVByteScalarU8QueryEvaluator<'e>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        DotVByteScalarU8QueryEvaluator::new(query, self)
    }

    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        DotVByteScalarU8QueryEvaluator::new(decoded.as_view(), self)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PlainSparseDatasetGrowable;
    use crate::core::dataset::DatasetGrowable;
    use crate::core::vector::{PackedVectorView, SparseVectorView};

    fn build_training_data(
        dim: usize,
        vectors: &[(&[u16], &[f32])],
    ) -> PlainSparseDataset<u16, f32, SquaredEuclideanDistance> {
        let encoder = crate::encoders::sparse_scalar::PlainSparseQuantizer::<
            u16,
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
    fn dotvbyte_encode_decode_preserves_structure() {
        let mut encoder = DotVByteScalarU8Encoder::new(4, 4);
        let td = build_training_data(
            4,
            &[
                (&[0_u16, 1, 2, 3], &[255.0_f32, 255.0, 255.0, 255.0]),
                (&[0_u16, 3], &[128.0_f32, 64.0]),
            ],
        );
        encoder.train::<f32>(&td);

        let values = [1.0_f32, 2.0_f32];
        let input = SparseVectorView::new(&[0_u16, 3], &values);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);
        assert!(!buffer.is_empty());

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &[0_u16, 3]);
        assert_eq!(decoded.values().len(), 2);
        assert!(decoded.values().iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    #[test]
    fn dotvbyte_encode_vector_default_matches_push_encoded() {
        let mut encoder = DotVByteScalarU8Encoder::new(3, 3);
        let td = build_training_data(3, &[(&[0_u16, 1, 2], &[255.0_f32, 255.0, 255.0])]);
        encoder.train::<f32>(&td);

        let values = [1.0_f32, 2.0_f32];
        let input = SparseVectorView::new(&[0_u16, 1], &values);

        let encoded_default = encoder.encode_vector(input);
        let mut encoded_manual = Vec::new();
        encoder.push_encoded(input, &mut encoded_manual);

        assert_eq!(encoded_default.data(), encoded_manual.as_slice());
    }

    #[test]
    fn dotvbyte_training_sets_component_mapping() {
        let mut encoder = DotVByteScalarU8Encoder::new(3, 3);
        let training = build_training_data(3, &[(&[0_u16, 1], &[1.0_f32, 1.0])]);
        encoder.train::<f32>(&training);

        let mapping = encoder.component_mapping().unwrap();
        let inverse = encoder.inverse_component_mapping().unwrap();
        assert_eq!(mapping.len(), 3);
        assert_eq!(inverse.len(), 3);
        let mut seen = [false; 3];
        for &m in mapping {
            assert!((m as usize) < 3);
            assert!(!seen[m as usize]);
            seen[m as usize] = true;
        }
        assert!(seen.into_iter().all(|v| v));
        for &value in inverse {
            assert!(value < 3);
        }
    }

    #[test]
    fn dotvbyte_decode_vector_with_mapping() {
        let mut encoder = DotVByteScalarU8Encoder::new(3, 3);
        let training = build_training_data(
            3,
            &[
                (&[0_u16, 1, 2], &[255.0_f32, 255.0, 255.0]),
                (&[0_u16, 2], &[32.0_f32, 16.0]),
            ],
        );
        encoder.train::<f32>(&training);

        let values = [1.0_f32, 2.0];
        let mut buffer = Vec::new();
        encoder.push_encoded(SparseVectorView::new(&[0_u16, 1], &values), &mut buffer);
        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &[0_u16, 1]);
        assert_eq!(decoded.values().len(), 2);
        assert!(decoded.values().iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(encoder.component_mapping().is_some());
        assert!(encoder.inverse_component_mapping().is_some());
    }
}

#[derive(Debug, Clone)]
pub struct DotVByteScalarU8QueryEvaluator<'e> {
    dense_query_transformed: Vec<f32>,
    _phantom: PhantomData<&'e ()>,
}

impl<'e> DotVByteScalarU8QueryEvaluator<'e> {
    #[inline]
    pub fn new(
        query: SparseVectorView<'_, u16, f32>,
        quantizer: &'e DotVByteScalarU8Encoder,
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

        // Densify the query into the *mapped* component space, with each entry
        // pre-multiplied by the per-component quantization scale (indexed by the
        // original component, since `quants` is trained on the original space).
        // The encoded vectors store components in the mapped space, so the SIMD
        // gather inside `dot_product` reads the dense query at the mapped index.
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

impl<'e, 'v> QueryEvaluator<PackedVectorView<'v, u64>> for DotVByteScalarU8QueryEvaluator<'e> {
    type Distance = DotProduct;

    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> DotProduct {
        let dotvbyte_view = unsafe { DotVbyteScalarU8::from_unchecked_slice(vector.data()) };
        DotProduct::from(dotvbyte_view.dot_product(&self.dense_query_transformed))
    }
}

const N: usize = u8::BITS as usize;
pub(super) const MASKS: [Simd<u8, { N * 2 }>; 256] = generate_masks_u16();

#[derive(Clone)]
struct DotVbyteScalarU8<'a> {
    values: &'a [Simd<u8, N>], //TODO: we may not need to load this as SIMD.
    bytes_remaining: &'a [u16],
    values_remaining: &'a [u8],
    idx_lens: &'a [u8],
    bytes: &'a [u8],
}

impl<'a> DotVbyteScalarU8<'a> {
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
    pub fn iter(self) -> impl Iterator<Item = (u16, u8)> {
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

    pub(crate) fn dot_product(&self, mut query: &[f32]) -> f32 {
        let mut result = Simd::<f32, 8>::splat(0.0);
        // This ugly clone is optimized away
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
                // Plain u8 -> f32: per-component scale already folded into the query.
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

pub(super) fn simd_prefix_sum<const N: usize>(mut n: Simd<u16, N>) -> Simd<u16, N> {
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

/// Decode the packed u8 values for the scalar-quantized encoder.
///
/// Unlike `dotvbyte_fixedu8`, the values stored here are NOT `FixedU8Q`: they
/// are per-component scalar-quantized integers in `[0, 255]`, and the matching
/// per-component scale (`quants[c]`) has already been folded into the dense
/// query inside `DotVByteScalarU8QueryEvaluator::new`. So the right decoding is
/// a plain `u8 -> f32` cast with no additional scaling.
pub(super) fn simd_scalaru8_to_f32<const N: usize>(f: Simd<u8, N>) -> Simd<f32, N>
where
    Simd<u8, N>: std::simd::num::SimdUint<Cast<f32> = Simd<f32, N>>,
{
    f.cast::<f32>()
}

impl SpaceUsage for DotVByteScalarU8Encoder {
    #[inline]
    fn space_usage_bytes(&self) -> usize {
        let size_of_mapping = match &self.component_mapping {
            Some(component_mapping) => component_mapping.space_usage_bytes(),
            None => std::mem::size_of::<Option<Box<[u16]>>>(),
        };
        size_of_mapping + self.dim.space_usage_bytes()
    }
}
