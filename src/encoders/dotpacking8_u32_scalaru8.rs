use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::marker::PhantomData;
use std::simd::num::{SimdFloat, SimdUint};
use std::simd::{Mask, Simd, StdFloat};

use crate::core::sealed;
use crate::core::vector::{PackedVectorView, SparseVectorView};
use crate::core::vector_encoder::{
    PackedSparseVectorEncoder, QueryEvaluator, SparseDataEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::DotProduct;
use crate::encoders::dotpacking8::common::{
    compute_safe_simd_padding, encode_blocks, simd_prefix_sum,
};
use crate::encoders::dotpacking8::encoder::DotPacking8View;
use crate::utils::{permute_components_with_bisection, train_sparse_scalar_quantizer};
use crate::{Dataset, PlainSparseDataset, SpaceUsage, SquaredEuclideanDistance};

use rusty_perm::PermApply as _;
use rusty_perm::PermFromSorting as _;

const N: usize = 8;

/// Optimistic block-8 encoder for sparse vectors with `u32` components and scalar
/// quantized `u8` values.
///
/// Mirrors [`OptimisticDotVByteScalarU8Encoder`] but emits the block-8 packed format
/// from [`crate::encoders::dotpacking8`] instead of the VByte format. The block-8
/// encoding stores per-block gap bit-widths in a 4-bit selector field, so individual
/// gaps must fit in 16 bits (`b <= 16`). To support `u32` components, this encoder
/// injects synthetic zero-valued "bridge" coordinates whenever a delta would exceed
/// `u16::MAX`, keeping the encoding on the fast SIMD path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimisticDotPacking8U32ScalarU8Encoder {
    dim: usize,
    quants: Box<[f32]>,
    component_mapping: Option<Box<[u32]>>,
    #[serde(default)]
    inverse_component_mapping: Option<Box<[u32]>>,
}

impl sealed::Sealed for OptimisticDotPacking8U32ScalarU8Encoder {}

impl OptimisticDotPacking8U32ScalarU8Encoder {
    #[inline]
    pub fn new(input_dim: usize) -> Self {
        Self {
            dim: input_dim,
            quants: vec![0.0_f32; input_dim].into_boxed_slice(),
            component_mapping: None,
            inverse_component_mapping: None,
        }
    }

    #[inline]
    pub fn component_mapping(&self) -> Option<&[u32]> {
        self.component_mapping.as_deref()
    }

    #[inline]
    pub fn inverse_component_mapping(&self) -> Option<&[u32]> {
        self.inverse_component_mapping.as_deref()
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

    pub fn train(
        &mut self,
        training_data: &PlainSparseDataset<u32, f32, SquaredEuclideanDistance>,
    ) {
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

    /// Inserts synthetic zero-valued coordinates so that every consecutive component
    /// delta is at most `u16::MAX`. Components must be sorted ascending. The trailing
    /// real coordinate is preserved as-is.
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

impl SparseDataEncoder for OptimisticDotPacking8U32ScalarU8Encoder {
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

        let mut iter = view.iter_raw();
        let mut last_component = 0u32;

        for (gaps, vals) in &mut iter {
            let components = simd_prefix_sum(gaps);
            let absolute_components = components + Simd::splat(last_component);
            let absolute_arr = absolute_components.to_array();
            let vals_arr = vals.to_array();
            for i in 0..N {
                decoded_comp.push(absolute_arr[i]);
                values_raw.push(vals_arr[i]);
            }
            last_component = absolute_arr[N - 1];
        }

        let (gaps, tail_values, remaining) = iter.decode_tail();
        if remaining > 0 {
            let components = simd_prefix_sum(gaps);
            let absolute_components = components + Simd::splat(last_component);
            let absolute_arr = absolute_components.to_array();
            for i in 0..remaining {
                decoded_comp.push(absolute_arr[i]);
                values_raw.push(tail_values[i]);
            }
        }

        if let Some(component_mapping) = self.component_mapping() {
            let inverse: Cow<'_, [u32]> = match self.inverse_component_mapping() {
                Some(inverse) => Cow::Borrowed(inverse),
                None => Cow::Owned(Self::compute_inverse_mapping(component_mapping)),
            };

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

impl PackedSparseVectorEncoder for OptimisticDotPacking8U32ScalarU8Encoder {
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

        let mut q_components: Vec<u32> = if let Some(mapping) = self.component_mapping() {
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

        let (bridged_components, bridged_values) =
            Self::inject_zero_bridges(&q_components, &q_values);

        let n = bridged_components.len();
        if n == 0 {
            return;
        }
        assert!(
            n < u16::MAX as usize,
            "OptimisticDotPacking8U32ScalarU8Encoder only supports vectors (after bridging) shorter than 65535, got {n}"
        );

        let mut gaps: Vec<u32> = Vec::with_capacity(n);
        let mut last = 0u32;
        for &comp in &bridged_components {
            let g = comp - last;
            assert!(
                g <= u16::MAX as u32,
                "gap {g} exceeds u16::MAX; optimistic bridge insertion failed"
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
        for _ in 0..padding_needed {
            payload.push(0);
        }
        payload.extend_from_slice(bridged_values.as_slice());

        let data = payload
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()));
        output.extend(data);
    }
}

impl VectorEncoder for OptimisticDotPacking8U32ScalarU8Encoder {
    type Distance = DotProduct;
    type InputVector<'a> = SparseVectorView<'a, u32, f32>;
    type QueryVector<'q> = SparseVectorView<'q, u32, f32>;
    type EncodedVector<'a> = PackedVectorView<'a, u64>;

    type Evaluator<'e>
        = OptimisticDotPacking8U32ScalarU8QueryEvaluator<'e>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        OptimisticDotPacking8U32ScalarU8QueryEvaluator::new(query, self)
    }

    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        OptimisticDotPacking8U32ScalarU8QueryEvaluator::new(decoded.as_view(), self)
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
pub struct OptimisticDotPacking8U32ScalarU8QueryEvaluator<'e> {
    dense_query_transformed: Vec<f32>,
    _phantom: PhantomData<&'e ()>,
}

impl<'e> OptimisticDotPacking8U32ScalarU8QueryEvaluator<'e> {
    pub fn new(
        query: SparseVectorView<'_, u32, f32>,
        encoder: &'e OptimisticDotPacking8U32ScalarU8Encoder,
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
    for OptimisticDotPacking8U32ScalarU8QueryEvaluator<'e>
{
    type Distance = DotProduct;

    #[inline]
    fn compute_distance(&self, vector: PackedVectorView<'v, u64>) -> Self::Distance {
        let view = unsafe { DotPacking8View::from_unchecked_slice(vector.data()) };
        let mut acc = Simd::<f32, N>::splat(0.0);
        let mut query_ptr = self.dense_query_transformed.as_slice();

        let mut iter = view.iter_raw();
        for (gaps, values) in &mut iter {
            let components = simd_prefix_sum(gaps);
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
        let mut total = acc.reduce_sum();

        let (gaps, tail_values, remaining) = iter.decode_tail();
        if remaining > 0 {
            let components = simd_prefix_sum(gaps);
            let components_arr = components.to_array();
            for i in 0..remaining {
                total += query_ptr[components_arr[i] as usize] * tail_values[i] as f32;
            }
        }
        DotProduct(total)
    }
}

impl SpaceUsage for OptimisticDotPacking8U32ScalarU8Encoder {
    fn space_usage_bytes(&self) -> usize {
        let mapping_size = self
            .component_mapping
            .as_ref()
            .map_or(0, |m| m.len() * std::mem::size_of::<u32>());
        let inverse_mapping_size = self
            .inverse_component_mapping
            .as_ref()
            .map_or(0, |m| m.len() * std::mem::size_of::<u32>());
        let quants_size = self.quants.len() * std::mem::size_of::<f32>();
        mapping_size + inverse_mapping_size + quants_size
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
        let encoder = OptimisticDotPacking8U32ScalarU8Encoder::new(100);
        assert_eq!(encoder.input_dim(), 100);
        assert_eq!(encoder.output_dim(), 100);
        assert!(encoder.component_mapping().is_none());
        assert!(encoder.inverse_component_mapping().is_none());
    }

    #[test]
    fn training_sets_component_mapping_and_quants() {
        let mut encoder = OptimisticDotPacking8U32ScalarU8Encoder::new(5);
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
            OptimisticDotPacking8U32ScalarU8Encoder::inject_zero_bridges(&components, &values);
        assert_eq!(bc, components);
        assert_eq!(bv, values);
    }

    #[test]
    fn inject_zero_bridges_large_gap() {
        let gap = u16::MAX as u32 + 100;
        let components = vec![0u32, gap];
        let values = vec![10u8, 20];
        let (bc, bv) =
            OptimisticDotPacking8U32ScalarU8Encoder::inject_zero_bridges(&components, &values);
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
        let mut encoder = OptimisticDotPacking8U32ScalarU8Encoder::new(10);
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
        let mut encoder = OptimisticDotPacking8U32ScalarU8Encoder::new(4);
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
        let mut encoder = OptimisticDotPacking8U32ScalarU8Encoder::new(dim);
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

    /// Vector length not a multiple of 8 → exercises `decode_tail`.
    #[test]
    fn bulk_and_tail_blocks() {
        let dim = 256;
        let mut encoder = OptimisticDotPacking8U32ScalarU8Encoder::new(dim);
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
