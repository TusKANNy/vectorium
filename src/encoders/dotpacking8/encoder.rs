//! The DotPacking8 encoder, generic over the component type `C` (`u16`, `u32`) and the value
//! codec `V` (see `values.rs`). The public encoders (`DotPacking8U16FixedU8Encoder`, ...) are
//! type aliases of it.

use rusty_perm::{PermApply as _, PermFromSorting as _};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::simd::{Simd, StdFloat, prelude::*};

use crate::core::sealed;
use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::encoders::dotpacking8::bridges::{drop_bridges, insert_bridges};
use crate::encoders::dotpacking8::common::{GapDecoder, fast_lane_gather, simd_prefix_sum};
use crate::encoders::dotpacking8::layout::{EncodedView, write_encoded};
pub use crate::encoders::dotpacking8::mapping::{ComponentMapping, PackedComponent};
use crate::encoders::dotpacking8::values::{ValueCodec, ValueReader};
use crate::{PlainSparseDataset, ScalarSparseSupportedDistance, SpaceUsage};

const N: usize = 8;

/// Packed sparse dot-product encoder: components gap-coded in bit-packed blocks of 8 (see
/// `common.rs`), values stored by the codec `V`. `u32` components insert zero-coded bridges
/// where a gap exceeds `u16::MAX` (`PackedComponent::MAY_NEED_BRIDGES`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound = "C: PackedComponent, V: ValueCodec")]
pub struct DotPacking8Encoder<C, V> {
    dim: usize,
    values: V,
    mapping: Option<ComponentMapping<C>>,
}

impl<C, V> sealed::Sealed for DotPacking8Encoder<C, V> {}

impl<C: PackedComponent, V: ValueCodec> DotPacking8Encoder<C, V> {
    /// Untrained encoder: original component ids, untrained value codec.
    pub fn new(dim: usize) -> Self {
        Self::with_values(dim, V::new_untrained(dim))
    }

    pub fn with_values(dim: usize, values: V) -> Self {
        Self { dim, values, mapping: None }
    }

    pub fn values(&self) -> &V {
        &self.values
    }

    #[inline]
    pub fn component_mapping(&self) -> Option<&[C]> {
        self.mapping.as_ref().map(|m| m.forward())
    }

    #[inline]
    pub fn inverse_component_mapping(&self) -> Option<&[C]> {
        self.mapping.as_ref().map(|m| m.inverse())
    }

    /// Installs a component permutation (e.g. one computed by another ordering).
    pub fn set_component_mapping(&mut self, mapping: ComponentMapping<C>) {
        assert_eq!(mapping.forward().len(), self.dim, "mapping dimension mismatch");
        self.mapping = Some(mapping);
    }

    /// Reorders components by decreasing frequency in `training_data` (shorter gaps), then
    /// trains the value codec on the original component ids. The distance does not matter.
    pub fn train<D>(&mut self, training_data: &PlainSparseDataset<C, f32, D>)
    where
        D: ScalarSparseSupportedDistance,
    {
        self.mapping = Some(ComponentMapping::by_decreasing_frequency(
            self.dim,
            training_data,
        ));
        self.values.train(training_data);
    }
}

pub struct DotPacking8QueryEvaluator<'e, C, V> {
    dense_query: Vec<f32>,
    _encoder: PhantomData<&'e DotPacking8Encoder<C, V>>,
}

impl<'e, C: PackedComponent, V: ValueCodec> DotPacking8QueryEvaluator<'e, C, V> {
    pub fn new(query: SparseVectorView<'_, C, f32>, encoder: &'e DotPacking8Encoder<C, V>) -> Self {
        let mut dense_query = vec![0.0f32; encoder.dim];
        for (&c, &v) in query.components().iter().zip(query.values()) {
            let idx: usize = c.as_();
            assert!(
                idx < encoder.dim,
                "Query component {idx} exceeds dimension {}",
                encoder.dim
            );
            let mapped: usize = match &encoder.mapping {
                Some(m) => m.map(c).as_(),
                None => idx,
            };
            dense_query[mapped] = encoder.values.query_value(idx, v);
        }
        Self {
            dense_query,
            _encoder: PhantomData,
        }
    }
}

impl<'e, 'v, C: PackedComponent, V: ValueCodec> QueryEvaluator<PackedVectorView<'v, u64>>
    for DotPacking8QueryEvaluator<'e, C, V>
{
    type Distance = DotProduct;

    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> Self::Distance {
        // SAFETY: datasets only hold vectors written by `push_encoded` with the same codec.
        let view = unsafe { EncodedView::new::<V>(vector.data()) };
        // SAFETY: the view comes from `push_encoded` output, invariants (1)-(3) in `common.rs`.
        let mut gaps = unsafe { GapDecoder::new(view.n, view.selectors, view.payload) };
        let mut values = V::reader(view.values);
        let mut acc = Simd::<f32, N>::splat(0.0);
        let mut query_ptr = self.dense_query.as_slice();
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

impl<C: PackedComponent, V: ValueCodec> VectorEncoder for DotPacking8Encoder<C, V> {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, C, f32>;
    type QueryVector<'q> = SparseVectorView<'q, C, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;
    type Evaluator<'e>
        = DotPacking8QueryEvaluator<'e, C, V>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        DotPacking8QueryEvaluator::new(query, self)
    }

    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        DotPacking8QueryEvaluator::new(decoded.as_view(), self)
    }

    fn output_dim(&self) -> usize {
        self.dim
    }

    fn input_dim(&self) -> usize {
        self.dim
    }
}

impl<C: PackedComponent, V: ValueCodec> SparseDataEncoder for DotPacking8Encoder<C, V> {
    type InputComponentType = C;
    type InputValueType = f32;
    type OutputComponentType = C;
    type OutputValueType = V::Output;

    fn decode_vector<'a>(
        &self,
        encoded: Self::EncodedVector<'a>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        // SAFETY: as in `compute_distance`.
        let view = unsafe { EncodedView::new::<V>(encoded.data()) };
        // SAFETY: as in `compute_distance`.
        let mut gaps = unsafe { GapDecoder::new(view.n, view.selectors, view.payload) };
        let mut values = V::reader(view.values);
        let mut comps: Vec<u32> = Vec::with_capacity(view.n);
        let mut codes: Vec<V::Code> = Vec::with_capacity(view.n);
        let mut last_component = 0u32;
        while let Some(block_gaps) = gaps.next_block() {
            let abs = (simd_prefix_sum(block_gaps) + Simd::splat(last_component)).to_array();
            comps.extend_from_slice(&abs);
            codes.extend_from_slice(&values.next_block_codes());
            last_component = abs[N - 1];
        }
        let (tail_gaps, rem) = gaps.tail();
        let abs = (simd_prefix_sum(tail_gaps) + Simd::splat(last_component)).to_array();
        comps.extend_from_slice(&abs[..rem]);
        codes.extend_from_slice(&values.tail(rem)[..rem]);

        if C::MAY_NEED_BRIDGES {
            drop_bridges(&mut comps, &mut codes, V::zero_code());
        }

        let mut components: Vec<C> = match &self.mapping {
            Some(m) => comps.iter().map(|&c| m.unmap(C::from_usize(c as usize))).collect(),
            None => comps.iter().map(|&c| C::from_usize(c as usize)).collect(),
        };
        if self.mapping.is_some() {
            // Decode after the inverse mapping, so each code uses its original component's
            // parameters; keep codes aligned with their components.
            let permutation = rusty_perm::PermD::from_sort(components.as_slice());
            permutation.apply(codes.as_mut_slice()).unwrap();
            permutation.apply(components.as_mut_slice()).unwrap();
        }
        let decoded: Vec<f32> = components
            .iter()
            .zip(&codes)
            .map(|(&c, &code)| self.values.decode(c.as_(), code))
            .collect();
        SparseVectorOwned::new(components, decoded)
    }
}

impl<C: PackedComponent, V: ValueCodec> PackedSparseVectorEncoder for DotPacking8Encoder<C, V> {
    type PackedDataType = u64;

    fn push_encoded<'a, OutputContainer>(
        &self,
        input: SparseVectorView<'a, C, f32>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<u64>,
    {
        let mut codes: Vec<V::Code> = input
            .components()
            .iter()
            .zip(input.values())
            .map(|(&c, &v)| self.values.encode(c.as_(), v))
            .collect();
        let mut comps: Vec<u32> = match &self.mapping {
            Some(m) => input.components().iter().map(|&c| m.map(c).as_() as u32).collect(),
            None => input.components().iter().map(|&c| c.as_() as u32).collect(),
        };
        if self.mapping.is_some() {
            let permutation = rusty_perm::PermD::from_sort(comps.as_slice());
            permutation.apply(codes.as_mut_slice()).unwrap();
            permutation.apply(comps.as_mut_slice()).unwrap();
        }
        if C::MAY_NEED_BRIDGES {
            debug_assert!(
                codes.iter().all(|&code| code != V::zero_code()),
                "a real coordinate quantized to code 0, which is reserved for bridges: every value \
                 must be at least its component's quantization step"
            );
            if let Some((bridged_comps, bridged_codes)) =
                insert_bridges(&comps, &codes, V::zero_code())
            {
                write_encoded::<V>(&bridged_comps, &bridged_codes, output);
                return;
            }
        }
        write_encoded::<V>(&comps, &codes, output);
    }
}

impl<C: PackedComponent, V: ValueCodec> SpaceUsage for DotPacking8Encoder<C, V> {
    fn space_usage_bytes(&self) -> usize {
        self.values.space_usage_bytes() + self.mapping.as_ref().map_or(0, |m| m.space_usage_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoders::dotpacking8::values::*;
    use crate::core::vector::{PackedVectorView, SparseVectorView};

    fn encode<C: PackedComponent, V: ValueCodec>(
        enc: &DotPacking8Encoder<C, V>, comps: &[C], vals: &[f32],
    ) -> Vec<u64> {
        let mut out = Vec::new();
        enc.push_encoded(SparseVectorView::new(comps, vals), &mut out);
        out
    }

    fn check_roundtrip<V: ValueCodec>(enc: &DotPacking8Encoder<u16, V>, tol: f32) {
        for n in [0usize, 1, 7, 8, 9, 15, 16, 17, 40] {
            let comps: Vec<u16> = (0..n as u16).map(|i| i * 37).collect();
            let vals: Vec<f32> = (0..n).map(|i| 0.25 + (i % 13) as f32 * 0.25).collect();
            let blob = encode(enc, &comps, &vals);
            let d = enc.decode_vector(PackedVectorView::new(&blob));
            assert_eq!(d.components(), comps.as_slice(), "n={n}");
            for (x, y) in d.values().iter().zip(&vals) {
                assert!((x - y).abs() <= tol, "n={n}: {x} vs {y}");
            }
        }
    }

    #[test]
    fn roundtrip_all_codecs_all_lengths() {
        let dim = 2000;
        check_roundtrip(&DotPacking8Encoder::<u16, FixedU8Values>::new(dim), 1.0 / 64.0);
        check_roundtrip(&DotPacking8Encoder::<u16, F16Values>::new(dim), 0.002);
        check_roundtrip(
            &DotPacking8Encoder::with_values(dim, ScalarU8Values::from_steps(vec![4.0 / 255.0; dim].into())),
            4.0 / 255.0,
        );
        check_roundtrip(
            &DotPacking8Encoder::with_values(dim, Scalar4BitValues::from_steps(vec![4.0 / 15.0; dim].into())),
            4.0 / 15.0,
        );
    }

    #[test]
    fn empty_vector_all_codecs() {
        let q = SparseVectorView::new(&[1u16, 5], &[1.0f32, 2.0]);
        let e1 = DotPacking8Encoder::<u16, FixedU8Values>::new(10);
        let e2 = DotPacking8Encoder::<u16, F16Values>::new(10);
        assert!(encode(&e1, &[], &[]).is_empty());
        assert!(encode(&e2, &[], &[]).is_empty());
        assert_eq!(e1.query_evaluator(q).compute_distance(PackedVectorView::new(&[])).0, 0.0);
        assert_eq!(e2.query_evaluator(q).compute_distance(PackedVectorView::new(&[])).0, 0.0);
    }

    #[test]
    fn u16_decode_keeps_zero_codes() {
        // Value 0.001 is below the 1/255 step: code 0. u16 never has bridges, so it must survive.
        let enc = DotPacking8Encoder::with_values(10, ScalarU8Values::from_steps(vec![1.0; 10].into()));
        let blob = encode(&enc, &[2u16, 3], &[0.001, 5.0]);
        let d = enc.decode_vector(PackedVectorView::new(&blob));
        assert_eq!(d.components(), &[2, 3]);
        assert_eq!(d.values(), &[0.0, 5.0]);
    }

    #[test]
    #[should_panic(expected = "exceeds")]
    fn query_component_out_of_range_panics() {
        let enc = DotPacking8Encoder::<u16, FixedU8Values>::new(10);
        let _ = enc.query_evaluator(SparseVectorView::new(&[10u16], &[1.0f32]));
    }
}
