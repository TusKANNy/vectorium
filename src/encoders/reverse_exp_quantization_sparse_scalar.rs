use std::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::core::vector_encoder::{
    QueryEvaluator, SparseDataEncoder, SparseVectorEncoder, SparseVectorOwned, VectorEncoder,
};
use crate::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use crate::utils::is_strictly_sorted;
use crate::{ComponentType, Dataset, PlainSparseDataset, SpaceUsage, SparseVectorView};

const EPS: f32 = 1e-9;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReverseExpSparseQuantizer<C, D> {
    dim: usize,
    /// Per-component max values
    maxs: Box<[f32]>,
    /// Per-component f base: f = ((max - min + eps) / eps)^(1/255)
    fs: Box<[f32]>,
    /// Per-component log(f) for encoding
    log_fs: Box<[f32]>,
    _phantom: PhantomData<(C, D)>,
}

impl<C, D> PartialEq for ReverseExpSparseQuantizer<C, D> {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
    }
}

impl<C, D> ReverseExpSparseQuantizer<C, D> {
    #[inline]
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        assert_eq!(
            input_dim, output_dim,
            "ReverseExpSparseQuantizer requires input_dim == output_dim"
        );
        Self {
            dim: input_dim,
            maxs: vec![0.0; output_dim].into_boxed_slice(),
            fs: vec![1.0; output_dim].into_boxed_slice(),
            log_fs: vec![0.0; output_dim].into_boxed_slice(),
            _phantom: PhantomData,
        }
    }

    pub fn maxs(&self) -> &[f32] {
        &self.maxs
    }

    pub fn fs(&self) -> &[f32] {
        &self.fs
    }

    /// Dequantize a single component value.
    /// x = max - eps * f^v_int + eps
    #[inline]
    fn dequant(&self, component_idx: usize, v_int: u8) -> f32 {
        self.maxs[component_idx] + EPS
            - EPS * self.fs[component_idx].powi(v_int as i32)
    }

    pub fn train(training_data: &PlainSparseDataset<C, f32, SquaredEuclideanDistance>) -> Self
    where
        C: ComponentType,
    {
        let dim = training_data.output_dim();

        let mut maxs = vec![f32::MIN; dim];
        let mut mins = vec![f32::MAX; dim];

        for doc in training_data.iter() {
            for (&c, &v) in doc.components().iter().zip(doc.values()) {
                let idx: usize = c.as_();
                maxs[idx] = maxs[idx].max(v);
                mins[idx] = mins[idx].min(v);
            }
        }

        let mut fs = vec![1.0f32; dim];
        let mut log_fs = vec![0.0f32; dim];

        for i in 0..dim {
            if maxs[i] > mins[i] {
                let rev_max = maxs[i] - mins[i] + EPS;
                let rev_min = EPS;
                let f = (rev_max / rev_min).powf(1.0 / 255.0);
                fs[i] = f;
                log_fs[i] = f.ln();
            }
            // else: f=1, log_f=0 → all values encode to 0, decode to max
        }

        Self {
            dim,
            maxs: maxs.into_boxed_slice(),
            fs: fs.into_boxed_slice(),
            log_fs: log_fs.into_boxed_slice(),
            _phantom: PhantomData,
        }
    }
}

/// Distance dispatch trait for reverse-exponential quantized sparse vectors.
pub trait ReverseExpQuantizedSparseSupportedDistance: Distance {
    fn requires_dot_query() -> bool {
        false
    }

    fn compute_dense<C: ComponentType>(
        dense_query: &[f32],
        maxs: &[f32],
        fs: &[f32],
        vector: SparseVectorView<'_, C, u8>,
        dot_query: Option<f32>,
    ) -> Self;

    fn compute_sparse<C: ComponentType>(
        query: &SparseVectorOwned<C, f32>,
        maxs: &[f32],
        fs: &[f32],
        vector: SparseVectorView<'_, C, u8>,
        dot_query: Option<f32>,
    ) -> Self;
}

fn compute_query_squared_norm(values: &[f32]) -> f32 {
    values
        .iter()
        .fold(0.0f32, |acc, &v| acc.algebraic_add(v.algebraic_mul(v)))
}

/// Dequantize: x = max + eps - eps * f^v_int
#[inline]
fn dequant_value(max: f32, f: f32, v_int: u8) -> f32 {
    max + EPS - EPS * f.powi(v_int as i32)
}

impl ReverseExpQuantizedSparseSupportedDistance for DotProduct {
    #[inline]
    fn compute_dense<C: ComponentType>(
        dense_query: &[f32],
        maxs: &[f32],
        fs: &[f32],
        vector: SparseVectorView<'_, C, u8>,
        _dot_query: Option<f32>,
    ) -> Self {
        let result =
            vector
                .components()
                .iter()
                .zip(vector.values())
                .fold(0.0f32, |acc, (&c, &v)| {
                    let idx: usize = c.as_();
                    let v_real = dequant_value(maxs[idx], fs[idx], v);
                    acc.algebraic_add(unsafe {
                        dense_query.get_unchecked(idx).algebraic_mul(v_real)
                    })
                });
        DotProduct::from(result)
    }

    #[inline]
    fn compute_sparse<C: ComponentType>(
        query: &SparseVectorOwned<C, f32>,
        maxs: &[f32],
        fs: &[f32],
        vector: SparseVectorView<'_, C, u8>,
        _dot_query: Option<f32>,
    ) -> Self {
        let result =
            sparse_merge_dot_product_with_rev_exp_dequant(query.as_view(), vector, maxs, fs);
        DotProduct::from(result)
    }
}

impl ReverseExpQuantizedSparseSupportedDistance for SquaredEuclideanDistance {
    #[inline]
    fn requires_dot_query() -> bool {
        true
    }

    #[inline]
    fn compute_dense<C: ComponentType>(
        dense_query: &[f32],
        maxs: &[f32],
        fs: &[f32],
        vector: SparseVectorView<'_, C, u8>,
        dot_query: Option<f32>,
    ) -> Self {
        let dot_query =
            dot_query.expect("SquaredEuclideanDistance requires a precomputed ||q||² value");

        let mut dot_qv = 0.0f32;
        let mut v_norm_sq = 0.0f32;
        for (&c, &v) in vector.components().iter().zip(vector.values()) {
            let idx: usize = c.as_();
            let v_real = dequant_value(maxs[idx], fs[idx], v);
            dot_qv = dot_qv.algebraic_add(unsafe {
                dense_query.get_unchecked(idx).algebraic_mul(v_real)
            });
            v_norm_sq = v_norm_sq.algebraic_add(v_real.algebraic_mul(v_real));
        }

        let dist = dot_query
            .algebraic_add(v_norm_sq)
            .algebraic_sub(2.0f32.algebraic_mul(dot_qv));
        SquaredEuclideanDistance::from(dist)
    }

    #[inline]
    fn compute_sparse<C: ComponentType>(
        query: &SparseVectorOwned<C, f32>,
        maxs: &[f32],
        fs: &[f32],
        vector: SparseVectorView<'_, C, u8>,
        dot_query: Option<f32>,
    ) -> Self {
        let dot_query =
            dot_query.expect("SquaredEuclideanDistance requires a precomputed ||q||² value");

        let dot_qv =
            sparse_merge_dot_product_with_rev_exp_dequant(query.as_view(), vector, maxs, fs);

        let v_norm_sq =
            vector
                .components()
                .iter()
                .zip(vector.values())
                .fold(0.0f32, |acc, (&c, &v)| {
                    let idx: usize = c.as_();
                    let v_real = dequant_value(maxs[idx], fs[idx], v);
                    acc.algebraic_add(v_real.algebraic_mul(v_real))
                });

        let dist = dot_query
            .algebraic_add(v_norm_sq)
            .algebraic_sub(2.0f32.algebraic_mul(dot_qv));
        SquaredEuclideanDistance::from(dist)
    }
}

/// Merge-sort style dot product between a sparse f32 query and a sparse u8 vector,
/// dequantizing with reverse exponential on the fly.
#[inline]
fn sparse_merge_dot_product_with_rev_exp_dequant<C: ComponentType>(
    query: SparseVectorView<'_, C, f32>,
    vector: SparseVectorView<'_, C, u8>,
    maxs: &[f32],
    fs: &[f32],
) -> f32 {
    let q_components = query.components();
    let q_values = query.values();
    let v_components = vector.components();
    let v_values = vector.values();

    let mut qi = 0;
    let mut vi = 0;
    let mut result = 0.0f32;

    while qi < q_components.len() && vi < v_components.len() {
        let qc: usize = q_components[qi].as_();
        let vc: usize = v_components[vi].as_();
        if qc == vc {
            let v_real = dequant_value(maxs[vc], fs[vc], v_values[vi]);
            result = result.algebraic_add(q_values[qi].algebraic_mul(v_real));
            qi += 1;
            vi += 1;
        } else if qc < vc {
            qi += 1;
        } else {
            vi += 1;
        }
    }

    result
}

impl<C, D> SparseDataEncoder for ReverseExpSparseQuantizer<C, D>
where
    C: ComponentType,
    D: ReverseExpQuantizedSparseSupportedDistance,
{
    type InputComponentType = C;
    type InputValueType = f32;
    type OutputComponentType = C;
    type OutputValueType = u8;

    #[inline]
    fn decode_vector<'a>(
        &self,
        encoded: SparseVectorView<'a, Self::OutputComponentType, Self::OutputValueType>,
    ) -> SparseVectorOwned<Self::InputComponentType, f32> {
        let components = encoded.components().to_vec();
        let values: Vec<f32> = encoded
            .components()
            .iter()
            .zip(encoded.values())
            .map(|(&c, &v)| {
                let idx: usize = c.as_();
                self.dequant(idx, v)
            })
            .collect();
        SparseVectorOwned::new(components, values)
    }
}

impl<C, D> SparseVectorEncoder for ReverseExpSparseQuantizer<C, D>
where
    C: ComponentType,
    D: ReverseExpQuantizedSparseSupportedDistance,
{
    fn push_encoded<'a, ComponentContainer, ValueContainer>(
        &self,
        input: SparseVectorView<'a, C, f32>,
        components: &mut ComponentContainer,
        values: &mut ValueContainer,
    ) where
        ComponentContainer: Extend<Self::OutputComponentType>,
        ValueContainer: Extend<Self::OutputValueType>,
    {
        components.extend(input.components().iter().cloned());
        values.extend(
            input
                .components()
                .iter()
                .zip(input.values())
                .map(|(&c, &v)| {
                    let idx: usize = c.as_();
                    let log_f = self.log_fs[idx];
                    if log_f > 0.0 {
                        // rev_val = max - v + eps
                        let rev_val = self.maxs[idx] - v + EPS;
                        // x_q = round(log(rev_val / eps) / log(f))
                        let x_q = (rev_val / EPS).ln() / log_f;
                        x_q.round().clamp(0.0, 255.0) as u8
                    } else {
                        0u8
                    }
                }),
        );
    }

    fn encode_vector<'a>(
        &self,
        input: Self::InputVector<'a>,
    ) -> SparseVectorOwned<Self::OutputComponentType, Self::OutputValueType> {
        let mut components = Vec::new();
        let mut values = Vec::new();
        self.push_encoded(input, &mut components, &mut values);
        SparseVectorOwned::new(components, values)
    }
}

impl<C, D> VectorEncoder for ReverseExpSparseQuantizer<C, D>
where
    C: ComponentType,
    D: ReverseExpQuantizedSparseSupportedDistance,
{
    type Distance = D;
    type InputVector<'a> = SparseVectorView<'a, C, f32>;
    type QueryVector<'q> = SparseVectorView<'q, C, f32>;
    type EncodedVector<'a> = SparseVectorView<'a, C, u8>;

    type Evaluator<'e>
        = ReverseExpSparseQueryEvaluator<'e, C, D>
    where
        Self: 'e;

    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        ReverseExpSparseQueryEvaluator::new(query, self)
    }

    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        let decoded = <Self as SparseDataEncoder>::decode_vector(self, vector);
        ReverseExpSparseQueryEvaluator::new_from_owned_query(decoded, self)
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
pub struct ReverseExpSparseQueryEvaluator<'e, C, D>
where
    C: ComponentType,
    D: ReverseExpQuantizedSparseSupportedDistance,
{
    /// Dense query values (dim < 2^20)
    dense_query: Option<Vec<f32>>,
    /// Sparse query fallback (dim >= 2^20)
    sparse_query: Option<SparseVectorOwned<C, f32>>,
    /// Precomputed ||q||² for Euclidean
    dot_query: Option<f32>,
    maxs: &'e [f32],
    fs: &'e [f32],
    _phantom: PhantomData<D>,
}

impl<'e, C, D> ReverseExpSparseQueryEvaluator<'e, C, D>
where
    C: ComponentType,
    D: ReverseExpQuantizedSparseSupportedDistance,
{
    pub fn new(
        query: SparseVectorView<'_, C, f32>,
        quantizer: &'e ReverseExpSparseQuantizer<C, D>,
    ) -> Self {
        let dot_query = if D::requires_dot_query() {
            Some(compute_query_squared_norm(query.values()))
        } else {
            None
        };

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

        let small_dim = quantizer.input_dim() < 2_usize.pow(20);

        if small_dim {
            let mut dense = vec![0.0f32; quantizer.dim];
            for (&c, &v) in query.components().iter().zip(query.values()) {
                let idx: usize = c.as_();
                dense[idx] = v;
            }

            Self {
                dense_query: Some(dense),
                sparse_query: None,
                dot_query,
                maxs: &quantizer.maxs,
                fs: &quantizer.fs,
                _phantom: PhantomData,
            }
        } else {
            assert!(
                is_strictly_sorted(query.components()),
                "Query components must be sorted in strictly ascending order."
            );

            Self {
                dense_query: None,
                sparse_query: Some(SparseVectorOwned::new(
                    query.components().to_vec(),
                    query.values().to_vec(),
                )),
                dot_query,
                maxs: &quantizer.maxs,
                fs: &quantizer.fs,
                _phantom: PhantomData,
            }
        }
    }

    pub fn new_from_owned_query(
        query: SparseVectorOwned<C, f32>,
        quantizer: &'e ReverseExpSparseQuantizer<C, D>,
    ) -> Self {
        let dot_query = if D::requires_dot_query() {
            Some(compute_query_squared_norm(query.values()))
        } else {
            None
        };

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

        let small_dim = quantizer.input_dim() < 2_usize.pow(20);

        if small_dim {
            let mut dense = vec![0.0f32; quantizer.dim];
            for (&c, &v) in query.components().iter().zip(query.values()) {
                let idx: usize = c.as_();
                dense[idx] = v;
            }

            Self {
                dense_query: Some(dense),
                sparse_query: None,
                dot_query,
                maxs: &quantizer.maxs,
                fs: &quantizer.fs,
                _phantom: PhantomData,
            }
        } else {
            assert!(
                is_strictly_sorted(query.components()),
                "Query components must be sorted in strictly ascending order."
            );

            Self {
                dense_query: None,
                sparse_query: Some(query),
                dot_query,
                maxs: &quantizer.maxs,
                fs: &quantizer.fs,
                _phantom: PhantomData,
            }
        }
    }
}

impl<'e, 'v, C, D> QueryEvaluator<SparseVectorView<'v, C, u8>>
    for ReverseExpSparseQueryEvaluator<'e, C, D>
where
    C: ComponentType,
    D: ReverseExpQuantizedSparseSupportedDistance,
{
    type Distance = D;

    #[inline]
    fn compute_distance(&self, vector: SparseVectorView<'v, C, u8>) -> D {
        if let Some(dense) = &self.dense_query {
            D::compute_dense(dense, self.maxs, self.fs, vector, self.dot_query)
        } else {
            D::compute_sparse(
                self.sparse_query.as_ref().unwrap(),
                self.maxs,
                self.fs,
                vector,
                self.dot_query,
            )
        }
    }
}

impl<C, D> SpaceUsage for ReverseExpSparseQuantizer<C, D>
where
    C: ComponentType,
    D: ReverseExpQuantizedSparseSupportedDistance,
{
    fn space_usage_bytes(&self) -> usize {
        self.dim.space_usage_bytes()
            + self.maxs.space_usage_bytes()
            + self.fs.space_usage_bytes()
            + self.log_fs.space_usage_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PlainSparseDatasetGrowable;
    use crate::core::dataset::DatasetGrowable;
    use crate::core::vector::SparseVectorView;
    use crate::encoders::sparse_scalar::PlainSparseQuantizer;

    type DotQuantizer = ReverseExpSparseQuantizer<u16, DotProduct>;

    fn build_training_data(
        dim: usize,
        vectors: &[(&[u16], &[f32])],
    ) -> PlainSparseDataset<u16, f32, SquaredEuclideanDistance> {
        let q = PlainSparseQuantizer::<u16, f32, SquaredEuclideanDistance>::new(dim, dim);
        let mut g = PlainSparseDatasetGrowable::new(q);
        for &(c, v) in vectors {
            g.push(SparseVectorView::new(c, v));
        }
        g.into()
    }

    #[test]
    fn encode_max_gives_zero_min_gives_255() {
        let td = build_training_data(
            3,
            &[
                (&[0, 1, 2], &[0.0, 10.0, -5.0]),
                (&[0, 1, 2], &[1.0, 20.0, 5.0]),
            ],
        );
        let q = DotQuantizer::train(&td);

        // max values should encode to 0 (rev_val = 0 + eps → q=0)
        let enc_max = <DotQuantizer as SparseVectorEncoder>::encode_vector(
            &q,
            SparseVectorView::new(&[0_u16, 1, 2], &[1.0_f32, 20.0, 5.0]),
        );
        assert_eq!(enc_max.values(), &[0_u8, 0, 0]);

        // min values should encode to 255 (rev_val = max-min+eps → q=255)
        let enc_min = <DotQuantizer as SparseVectorEncoder>::encode_vector(
            &q,
            SparseVectorView::new(&[0_u16, 1, 2], &[0.0_f32, 10.0, -5.0]),
        );
        for &v in enc_min.values() {
            assert!(v >= 254, "min value should encode to 254 or 255, got {v}");
        }
    }

    #[test]
    fn decode_reconstructs_within_tolerance() {
        let td = build_training_data(2, &[(&[0, 1], &[0.0, 0.0]), (&[0, 1], &[10.0, 10.0])]);
        let q = DotQuantizer::train(&td);

        let input = SparseVectorView::new(&[0_u16, 1], &[3.7_f32, 8.2]);
        let enc = <DotQuantizer as SparseVectorEncoder>::encode_vector(&q, input);
        let dec = <DotQuantizer as SparseDataEncoder>::decode_vector(
            &q,
            SparseVectorView::new(enc.components(), enc.values()),
        );

        // Exponential quantization has larger steps at the low end.
        // Tolerance is generous since it's not uniform.
        for (&orig, &got) in input.values().iter().zip(dec.values()) {
            assert!(
                (orig - got).abs() < 1.0,
                "orig={orig}, got={got}, diff={}",
                (orig - got).abs()
            );
        }
    }

    #[test]
    fn dot_product_matches_dequantized_reference() {
        let td = build_training_data(
            4,
            &[(&[0, 1, 2, 3], &[0.0; 4]), (&[0, 1, 2, 3], &[10.0; 4])],
        );
        let q = DotQuantizer::train(&td);

        let query = SparseVectorView::new(&[0_u16, 2], &[1.0_f32, 3.0]);
        let doc = SparseVectorView::new(&[0_u16, 1, 2], &[5.0_f32, 7.0, 9.0]);

        let enc = <DotQuantizer as SparseVectorEncoder>::encode_vector(&q, doc);
        let enc_view = SparseVectorView::new(enc.components(), enc.values());

        let evaluator = q.query_evaluator(query);
        let got: f32 = evaluator.compute_distance(enc_view).distance();

        let dec = <DotQuantizer as SparseDataEncoder>::decode_vector(&q, enc_view);
        let mut ref_dot = 0.0f32;
        for (&qc, &qv) in [0_u16, 2].iter().zip(&[1.0_f32, 3.0]) {
            for (&dc, &dv) in dec.components().iter().zip(dec.values()) {
                if qc == dc {
                    ref_dot += qv * dv;
                }
            }
        }

        assert!(
            (got - ref_dot).abs() < 1e-4,
            "got {got}, expected {ref_dot}"
        );
    }

    #[test]
    fn dot_product_no_overlap_is_zero() {
        let td = build_training_data(
            4,
            &[(&[0, 1, 2, 3], &[0.0; 4]), (&[0, 1, 2, 3], &[10.0; 4])],
        );
        let q = DotQuantizer::train(&td);

        let query = SparseVectorView::new(&[0_u16, 1], &[5.0_f32, 5.0]);
        let doc = SparseVectorView::new(&[2_u16, 3], &[5.0_f32, 5.0]);

        let enc = <DotQuantizer as SparseVectorEncoder>::encode_vector(&q, doc);
        let evaluator = q.query_evaluator(query);
        let got: f32 = evaluator
            .compute_distance(SparseVectorView::new(enc.components(), enc.values()))
            .distance();

        assert!(got.abs() < 1e-6, "no overlap should give 0, got {got}");
    }
}
