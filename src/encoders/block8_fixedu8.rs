/*
Block8 SIMD sparse decoder/encoder for `FixedU8Q` values.
This implementation decodes 8 gap values in parallel using a 3-step SIMD pipeline:
1) byte-level shuffle to place the relevant 4-byte window per lane,
2) per-lane variable right shift to bit-align each element,
3) bitmasking to keep only `b` significant bits.

The constants table (`BLOCK8_TABLE`) precomputes shuffle indices, shifts, and masks
for bit-widths `b in [1, 16]`, enabling branch-free bulk decoding. Tails are handled
separately in scalar form
*/

use rusty_perm::PermApply as _;
use rusty_perm::PermFromSorting as _;

use crate::core::sealed;
use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::{FixedU8Q, SpaceUsage};
mod swizzle;
use self::swizzle::swizzle;
use bytemuck::cast_slice;
use num_traits::ToPrimitive;
use std::borrow::Cow;
use std::marker::PhantomData;
use std::simd::StdFloat;
use std::simd::prelude::*;

const FIXED_U8_SCALE: f32 = 1.0 / ((1u32 << FixedU8Q::FRAC_NBITS) as f32);

const N: usize = 8;

// Unified Constants for Block8 Encoding to maximize cache locality
#[repr(C, align(32))]
#[derive(Copy, Clone)]
struct Block8Constants {
    shuffle: [u8; 32],
    shifts: [u32; 8],
    masks: [u32; 8],
}

static BLOCK8_TABLE: [Block8Constants; 16] = gen_block8_table();

const fn gen_block8_table() -> [Block8Constants; 16] {
    let mut all = [Block8Constants {
        shuffle: [0u8; 32],
        shifts: [0u32; 8],
        masks: [0u32; 8],
    }; 16];
    let mut b_idx = 0;
    while b_idx < 16 {
        let b = b_idx + 1;
        // 1. Generate Shuffle Mask
        let mut m = [0u8; 32];
        let mut i = 0;
        while i < 8 {
            let pos_bits = i * b;
            let byte_off = (pos_bits / 8) as u8;
            let dest_idx = if i < 4 { i * 4 } else { (i - 4) * 4 + 16 };
            m[dest_idx] = byte_off;
            m[dest_idx + 1] = byte_off + 1;
            m[dest_idx + 2] = byte_off + 2;
            m[dest_idx + 3] = byte_off + 3;
            i += 1;
        }
        all[b_idx].shuffle = m;

        // 2. Generate Bit Shifts
        let mut s = [0u32; 8];
        let mut i = 0;
        while i < 8 {
            s[i] = (i * b % 8) as u32;
            i += 1;
        }
        all[b_idx].shifts = s;

        // 3. Generate Width Masks
        let mask = (1u32 << b) - 1;
        all[b_idx].masks = [mask; 8];

        b_idx += 1;
    }
    all
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Block8FixedU8Encoder {
    dim: usize,
    component_mapping: Option<Box<[u16]>>,
    #[serde(default)]
    inverse_component_mapping: Option<Box<[u16]>>,
}

impl PartialEq for Block8FixedU8Encoder {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
            && self.component_mapping == other.component_mapping
            && self.inverse_component_mapping == other.inverse_component_mapping
    }
}

impl sealed::Sealed for Block8FixedU8Encoder {}

impl Block8FixedU8Encoder {
    pub fn new(input_dim: usize) -> Self {
        Self {
            dim: input_dim,
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

    pub fn train<'a, V>(
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
struct Block8Fixedu8<'a> {
    selectors: &'a [u8],
    payloads_start: *const u8,
    tail_gaps: &'a [u16],
    tail_values: &'a [u8],
    bulk_values: &'a [Simd<u8, N>],
}

impl<'a> Block8Fixedu8<'a> {
    #[inline]
    unsafe fn cast_simd_slice<const LANES: usize>(slice: &'a [u8]) -> &'a [Simd<u8, LANES>] {
        let len = slice.len() / std::mem::size_of::<Simd<u8, LANES>>();
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const Simd<u8, LANES>, len) }
    }

    unsafe fn from_unchecked_slice(slice: &'a [u64]) -> Self {
        let bytes = cast_slice::<u64, u8>(slice);
        let n = u16::from_le_bytes(bytes[0..2].try_into().unwrap());
        let n_bulk = (n as usize / N) * N;
        let n_tail = n as usize % N;

        let mut offset = 2; // Header (2 bytes)

        // Tails: gaps aligned to 16-bit boundaries for efficient loading, followed by values as u8
        let tail_gaps: &[u16] = cast_slice(&bytes[offset..offset + n_tail * 2]);
        offset += n_tail * 2;
        let tail_values = &bytes[offset..offset + n_tail];
        offset += n_tail;

        // Selectors: Each byte encodes the bit-widths for two blocks (4 bits each), so we need
        //(n_bulk / N + 1) / 2 bytes to cover all bulk blocks
        let n_blocks = n_bulk / N;
        let selectors_size = (n_blocks + 1) / 2;
        let selectors = &bytes[offset..offset + selectors_size];
        offset += selectors_size;

        // Payloads
        let payloads_start = unsafe { bytes.as_ptr().add(offset) };

        // Bulk Values: block of 8 u8 are aligned for SIMD access from the end so we can easily cast them to Simd<u8, 8> slices
        let bulk_values_start = bytes.len() - n_bulk;
        let bulk_values = unsafe { Self::cast_simd_slice::<N>(&bytes[bulk_values_start..]) };

        Self {
            selectors,
            payloads_start,
            tail_gaps,
            tail_values,
            bulk_values,
        }
    }

    fn iter_raw(self) -> impl ExactSizeIterator<Item = (Simd<u32, N>, Simd<u8, N>)> {
        let mut payload_ptr = self.payloads_start;
        const BROADCAST_MASK: [usize; 32] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15,
        ];
        (0..self.bulk_values.len()).map(move |i| {
            // Selector stores width as (b - 1) in 4 bits, so decode back to [1, 16]
            let b = (((self.selectors[i / 2] >> ((i & 1) << 2)) & 0x0F) as usize) + 1;
            let raw = unsafe { payload_ptr.cast::<Simd<u8, 16>>().read_unaligned() };
            payload_ptr = unsafe { payload_ptr.add(b) };
            let v: Simd<u8, 32> = simd_swizzle!(raw, BROADCAST_MASK);

            let table_ptr = unsafe { BLOCK8_TABLE.get_unchecked(b - 1) };
            let shuffled_simd_u8 = swizzle(v, Simd::from_array(table_ptr.shuffle));

            // Use Portable SIMD for the rest
            let shuffled: Simd<u32, N> = unsafe { std::mem::transmute(shuffled_simd_u8) };
            let shifts = Simd::from_array(table_ptr.shifts);
            let masks = Simd::from_array(table_ptr.masks);

            let gaps = (shuffled >> shifts) & masks;
            (gaps, self.bulk_values[i])
        })
    }

    pub fn iter(self) -> impl Iterator<Item = (u16, FixedU8Q)> {
        let tail_gaps = self.tail_gaps;
        let tail_values = self.tail_values;
        gen move {
            let mut last_component = 0u32;
            for (gaps, values) in self.iter_raw() {
                let components = simd_prefix_sum(gaps);
                let absolute_components = components + Simd::splat(last_component);
                last_component = absolute_components[N - 1];

                for (c, v) in absolute_components
                    .to_array()
                    .into_iter()
                    .zip(values.to_array().into_iter().map(FixedU8Q::from_bits))
                {
                    yield (c as u16, v);
                }
            }
            for (&gap, &val) in tail_gaps.iter().zip(tail_values.iter()) {
                last_component += gap as u32;
                yield (last_component as u16, FixedU8Q::from_bits(val));
            }
        }
    }
}

#[inline(always)]
fn simd_prefix_sum(mut n: Simd<u32, 8>) -> Simd<u32, 8> {
    n += n.shift_elements_right::<1>(0);
    n += n.shift_elements_right::<2>(0);
    n += n.shift_elements_right::<4>(0);
    n
}

impl SparseDataEncoder for Block8FixedU8Encoder {
    type InputComponentType = u16;
    type InputValueType = FixedU8Q;
    type OutputComponentType = u16;
    type OutputValueType = FixedU8Q;

    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let view = unsafe { Block8Fixedu8::from_unchecked_slice(encoded.data()) };

        let mut components: Vec<u16> = Vec::new();
        let mut values: Vec<f32> = Vec::new();
        for (component, value) in view.iter() {
            components.push(component);
            values.push(value.to_f32().expect("Failed to convert value to f32"));
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

impl PackedSparseVectorEncoder for Block8FixedU8Encoder {
    type PackedDataType = u64;
    fn push_encoded<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, u16, FixedU8Q>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<u64>,
    {
        let mut q_values: Vec<u8> = input.values().iter().map(|v| v.to_bits()).collect();
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
        let n_bulk = (n / N) * N;
        let _n_tail = n % N;

        let mut gaps = Vec::with_capacity(n);
        let mut last = 0u32;
        for &comp in &q_components {
            gaps.push(comp as u32 - last);
            last = comp as u32;
        }

        let bulk_gaps = &gaps[..n_bulk];
        let tail_gaps = &gaps[n_bulk..];
        let bulk_values = &q_values[..n_bulk];
        let tail_values = &q_values[n_bulk..];

        // Helper function to encode d-gaps into bit-packed blocks.
        let encode_blocks = |gaps: &[u32]| {
            let n_blocks = gaps.len() / N;
            let mut selectors = vec![0u8; (n_blocks + 1) / 2];
            let mut payloads = Vec::new();
            for (idx, chunk) in gaps.chunks_exact(N).enumerate() {
                let max_val = chunk.iter().cloned().max().unwrap_or(0);
                let b = if max_val == 0 {
                    1
                } else {
                    32 - max_val.leading_zeros()
                } as u8;
                // Store (b - 1) in 4 bits of the selector byte in order to be sure to also fit b = 16 in
                // the 4 bits (since b is in [1, 16])
                let selector_b = b - 1;
                if idx % 2 == 0 {
                    selectors[idx / 2] |= selector_b & 0x0F;
                } else {
                    selectors[idx / 2] |= (selector_b & 0x0F) << 4;
                }
                let mut acc = 0u128;
                for (i, &g) in chunk.iter().enumerate() {
                    acc |= (g as u128) << (i as u8 * b);
                }
                // 8 values of b bits occupy exacly 8*b = b bytes
                payloads.extend_from_slice(&acc.to_le_bytes()[..b as usize]);
            }
            (selectors, payloads)
        };

        let (selectors, gap_payloads) = encode_blocks(bulk_gaps);

        let mut payload = Vec::new();

        // Header (2 bytes): Total number of elements in the vector.
        payload.extend_from_slice(&(n as u16).to_le_bytes());

        // Tail section: Gaps and values that don't fit into a full SIMD block.
        // - Tail gaps are aligned to 16-bit boundaries in order to use cast_slice.
        // - Tail values are stored directly as u8.
        let tail_gaps_u16: Vec<u16> = tail_gaps.iter().map(|&g| g as u16).collect();
        payload.extend_from_slice(cast_slice(&tail_gaps_u16));
        payload.extend_from_slice(tail_values);

        // Selectors: Bit-widths for each bulk gap block.
        // - Each selector byte stores the bit-widths for two blocks (4 bits each).
        // - Storing them separately enables efficient 128-bit loads of gap payloads.
        payload.extend_from_slice(&selectors);

        // Gap Payloads: The actual bit-packed d-gaps for bulk blocks.
        payload.extend_from_slice(&gap_payloads);

        // Bulk Values: Values for bulk blocks, aligned for SIMD access.
        // - Padding is added to ensure bulk values start at an 8-byte boundary.
        // - This allows reinterpreting the data as a Simd<u8, N> slice during decoding.
        payload.resize(payload.len().next_multiple_of(8), 0);
        payload.extend_from_slice(bulk_values);

        let data = payload
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()));

        output.extend(data);
    }

}

impl VectorEncoder for Block8FixedU8Encoder {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u16, FixedU8Q>;
    type QueryVector<'q> = SparseVectorView<'q, u16, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;
    type Evaluator<'e>
        = Block8FixedU8QueryEvaluator<'e>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        Block8FixedU8QueryEvaluator::new(query, self)
    }
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        Block8FixedU8QueryEvaluator::new(decoded.as_view(), self)
    }
    fn output_dim(&self) -> usize {
        self.dim
    }
    fn input_dim(&self) -> usize {
        self.dim
    }
}

pub struct Block8FixedU8QueryEvaluator<'a> {
    dense_query: Vec<f32>,
    _phantom: PhantomData<&'a ()>,
}

impl<'a, 'v> Block8FixedU8QueryEvaluator<'a> {
    pub fn new(query: SparseVectorView<'_, u16, f32>, encoder: &'a Block8FixedU8Encoder) -> Self {
        let mut dense_query = vec![0.0f32; encoder.dim];
        for (&c, &v) in query.components().iter().zip(query.values().iter()) {
            let mapped_c = if let Some(mapping) = &encoder.component_mapping {
                mapping[c as usize]
            } else {
                c
            };
            dense_query[mapped_c as usize] = v;
        }
        Self {
            dense_query,
            _phantom: PhantomData,
        }
    }

    #[inline]
    unsafe fn simd_compute_distance(&self, vector: PackedVectorView<'v, u64>) -> DotProduct {
        let view = unsafe { Block8Fixedu8::from_unchecked_slice(vector.data()) };

        let mut acc = Simd::<f32, N>::splat(0.0);
        let mut query_ptr = self.dense_query.as_slice();
        // --- BULK PHASE ---
        for (components, values) in view.clone().iter_raw() {
            let components = simd_prefix_sum(components);
            let query_values = unsafe {
                Simd::gather_select_unchecked(
                    query_ptr,
                    Mask::splat(true),
                    components.cast(),
                    Simd::splat(0.0),
                )
            };

            acc = query_values.mul_add(values.cast(), acc);
            let last_component = components[N - 1];
            query_ptr = unsafe { query_ptr.split_at_unchecked(last_component as usize).1 };
        }

        let mut total_unscaled = acc.reduce_sum();

        // --- TAIL PHASE ---
        total_unscaled += view
            .tail_gaps
            .iter()
            .zip(view.tail_values.iter())
            .scan(0, move |curr_pos, (&gap, &val)| {
                *curr_pos += gap as usize;
                let val_bits = val as f32;
                Some(unsafe { query_ptr.get_unchecked(*curr_pos).algebraic_mul(val_bits) })
            })
            .fold(0f32, |acc, x| acc.algebraic_add(x));

        // Final Scale Application
        DotProduct(total_unscaled.algebraic_mul(FIXED_U8_SCALE))
    }
}

impl<'a, 'v> QueryEvaluator<PackedVectorView<'v, u64>> for Block8FixedU8QueryEvaluator<'a> {
    type Distance = DotProduct;
    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> Self::Distance {
        unsafe { self.simd_compute_distance(vector) }
    }
}

impl SpaceUsage for Block8FixedU8Encoder {
    fn space_usage_bytes(&self) -> usize {
        let size_of_mapping = match &self.component_mapping {
            Some(component_mapping) => component_mapping.space_usage_bytes(),
            None => std::mem::size_of::<Option<Box<[u16]>>>(),
        };
        let size_of_inverse_mapping = match &self.inverse_component_mapping {
            Some(inverse_component_mapping) => inverse_component_mapping.space_usage_bytes(),
            None => std::mem::size_of::<Option<Box<[u16]>>>(),
        };
        size_of_mapping + size_of_inverse_mapping + self.dim.space_usage_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FromF32;
    use crate::core::distances::Distance;
    use crate::core::vector::{PackedVectorView, SparseVectorView};
    use num_traits::ToPrimitive;

    fn fixed(val: f32) -> FixedU8Q {
        FixedU8Q::from_f32_saturating(val)
    }

    #[test]
    fn block8_decode_only_bulk() {
        let encoder = Block8FixedU8Encoder::new(100);
        let values: Vec<_> = (0..8).map(|i| fixed(1.0 + i as f32)).collect();
        let components: Vec<_> = (0..8).map(|i| (i * 10) as u16).collect();
        let input = SparseVectorView::new(&components, &values);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &components);
        for (i, &v) in values.iter().enumerate() {
            assert!((decoded.values()[i] - v.to_f32().unwrap()).abs() < 1e-6);
        }
    }

    #[test]
    fn block8_decode_bulk_and_tail() {
        let encoder = Block8FixedU8Encoder::new(100);
        let values: Vec<_> = (0..10).map(|i| fixed(1.0 + i as f32)).collect();
        let components: Vec<_> = (0..10).map(|i| (i * 5) as u16).collect();
        let input = SparseVectorView::new(&components, &values);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &components);
        for (i, &v) in values.iter().enumerate() {
            assert!((decoded.values()[i] - v.to_f32().unwrap()).abs() < 1e-6);
        }
    }

    #[test]
    fn block8_encode_decode_roundtrip() {
        let encoder = Block8FixedU8Encoder::new(100);
        let values = [fixed(1.0), fixed(2.5), fixed(3.0)];
        let components = [10u16, 20, 30];
        let input = SparseVectorView::new(&components, &values);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);
        assert!(!buffer.is_empty());

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &components);
        let decoded_vals = decoded.values();
        assert_eq!(decoded_vals.len(), 3);
        assert!((decoded_vals[0] - 1.0).abs() < 1e-6);
        assert!((decoded_vals[1] - 2.5).abs() < 1e-6);
        assert!((decoded_vals[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn compute_distance_only_bulk() {
        let encoder = Block8FixedU8Encoder::new(100);
        let binding = [
            fixed(1.0),
            fixed(3.0),
            fixed(2.0),
            fixed(3.5),
            fixed(1.5),
            fixed(2.0),
            fixed(1.0),
            fixed(2.0),
        ];
        let input = SparseVectorView::new(&[2, 4, 8, 24, 36, 48, 53, 90], &binding);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);
        let query = SparseVectorView::new(
            &[2, 4, 6, 8, 24, 70, 90],
            &[0.5, 1.5, 2.5, 1.0, 2.0, 1.0, 2.0],
        );
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        // expected = 1.0*0.5 + 3.0*1.5 + 2.0*1.0 + 3.5*2.0 + 2.0*2.0
        let expected = 1.0 * 0.5 + 3.0 * 1.5 + 2.0 * 1.0 + 3.5 * 2.0 + 2.0 * 2.0;
        assert!((dist.distance() - expected).abs() < 1e-5);
    }

    #[test]
    fn compute_distance_bulk_and_tail() {
        let encoder = Block8FixedU8Encoder::new(100);
        let binding = [
            fixed(1.0),
            fixed(3.0),
            fixed(2.0),
            fixed(3.5),
            fixed(1.5),
            fixed(2.0),
            fixed(1.0),
            fixed(2.0),
            fixed(3.0),
            fixed(2.5),
        ];
        let input = SparseVectorView::new(&[2, 4, 8, 24, 28, 36, 48, 53, 70, 90], &binding);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);
        let query = SparseVectorView::new(
            &[2, 4, 8, 24, 28, 70, 90],
            &[0.5, 1.5, 2.5, 1.0, 2.0, 1.0, 2.0],
        );
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let expected =
            1.0 * 0.5 + 3.0 * 1.5 + 2.0 * 2.5 + 3.5 * 1.0 + 1.5 * 2.0 + 3.0 * 1.0 + 2.5 * 2.0;
        assert!((dist.distance() - expected).abs() < 1e-5);
    }

    fn compute_distance_with_gaps(gaps: &[u32]) {
        let num_vals = gaps.len();
        let mut components = Vec::new();
        let mut curr = 0u32;
        for &g in gaps {
            curr += g;
            components.push(curr as u16);
        }
        println!("Testing with components: {:?}", components);

        let encoder = Block8FixedU8Encoder::new(u16::MAX as usize + 1);
        let values: Vec<_> = (0..num_vals)
            .map(|i| fixed(1.0 + (i % 7) as f32 / 10.0))
            .collect();
        let input = SparseVectorView::new(&components, &values);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let query_vals: Vec<_> = (0..num_vals).map(|i| 0.1 + (i % 5) as f32 / 10.0).collect();
        let query = SparseVectorView::new(&components, &query_vals);

        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let mut expected = 0.0f32;
        for i in 0..num_vals {
            expected += values[i].to_f32().unwrap() * query_vals[i];
        }

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
        compute_distance_with_gaps(&gaps);
    }

    #[test]
    fn test_plain_with_tail() {
        let gaps = vec![1u32; 11];
        compute_distance_with_gaps(&gaps);
    }

    #[test]
    fn test_plain_large_gaps() {
        let gaps = vec![1000u32, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000];
        compute_distance_with_gaps(&gaps);
    }

    #[test]
    fn test_plain_u16_gaps() {
        let gaps = vec![60000, 1, 1, 2, 3, 200, 450, 2000, 3];
        compute_distance_with_gaps(&gaps);
    }
}
