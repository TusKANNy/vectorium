use crate::encoders::dotpacking8::cencoder::{CDotPacking8Encoder, CDotPacking8QueryEvaluator};
use crate::encoders::dotpacking8::quantizer::ScalarU8Quantizer;
use crate::{Dataset, PlainSparseDataset, SquaredEuclideanDistance};

pub type CDotPacking8ScalarU8Encoder = CDotPacking8Encoder<ScalarU8Quantizer>;
pub type CDotPacking8Scalaru8QueryEvaluator<'e> = CDotPacking8QueryEvaluator<'e, ScalarU8Quantizer>;

impl CDotPacking8ScalarU8Encoder {
    pub fn new_with_references(
        input_dim: usize,
        reference_lists: Vec<Vec<u16>>,
        max_ref_size: usize,
    ) -> Self {
        CDotPacking8Encoder::new(
            input_dim,
            ScalarU8Quantizer::new(vec![0.0f32; input_dim].into_boxed_slice()),
            reference_lists,
            max_ref_size,
        )
    }

    pub fn train(
        &mut self,
        training_data: &PlainSparseDataset<u16, f32, SquaredEuclideanDistance>,
    ) {
        const SAMPLE_RATE: usize = 20;
        let sample_size = if training_data.len() / SAMPLE_RATE < 50_000 {
            training_data.len()
        } else {
            training_data.len() / SAMPLE_RATE
        };
        self.train_components(training_data.iter().take(sample_size));
        self.quantizer.train(training_data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::distances::Distance;
    use crate::core::vector::{PackedVectorView, SparseVectorView};
    use crate::encoders::dotpacking8::quantizer::DotPacking8Quantizer;
    use crate::vector_encoder::SparseDataEncoder;
    use crate::{
        CDotPacking8ScalarU8Encoder, DatasetGrowable,
        PackedSparseVectorEncoder, PlainSparseDatasetGrowable, PlainSparseQuantizer,
        QueryEvaluator, VectorEncoder,
    };

    fn build_training_data(
        dim: usize,
        components: &[u16],
    ) -> PlainSparseDataset<u16, f32, SquaredEuclideanDistance> {
        let quantizer = PlainSparseQuantizer::<u16, f32, SquaredEuclideanDistance>::new(dim, dim);
        let mut growable = PlainSparseDatasetGrowable::new(quantizer);
        let values0: Vec<f32> = components
            .iter()
            .enumerate()
            .map(|(idx, _)| 10.0 + (idx % 7) as f32 * 3.5)
            .collect();
        let values1: Vec<f32> = components
            .iter()
            .enumerate()
            .map(|(idx, _)| 25.0 + (idx % 5) as f32 * 2.25)
            .collect();
        growable.push(SparseVectorView::new(components, &values0));
        if components.len() > 1 {
            growable.push(SparseVectorView::new(components, &values1));
        }
        growable.into()
    }

    fn quantized_values(
        encoder: &CDotPacking8ScalarU8Encoder,
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
        let mut i = 0usize;
        let mut j = 0usize;
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

    #[test]
    fn compute_distance_with_only_mapped_bulk() {
        const DIM: usize = 100;
        let comps: Vec<u16> = (0u16..100).collect();
        let training = build_training_data(DIM, &comps);
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let mut encoder =
            CDotPacking8ScalarU8Encoder::new_with_references(DIM, reference_lists, 128);
        encoder.train(&training);
        let binding = [1.0, 3.0, 2.0, 3.5, 1.5, 2.0, 1.0, 2.0];
        let input = SparseVectorView::new(&[0, 4, 8, 24, 36, 48, 53, 90], &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);
        let query = SparseVectorView::new(
            &[2, 4, 6, 8, 24, 70, 90],
            &[0.5, 1.5, 2.5, 1.0, 2.0, 1.0, 2.0],
        );
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
        let q_values = quantized_values(&encoder, &input.components(), &input.values());
        let expected = dot_with_query(
            &input.components(),
            &q_values,
            &query.components(),
            &query.values(),
        );

        assert!((dist.distance() - expected).abs() < 1e-4);
    }

    #[test]
    fn compute_distance_with_only_mapped_tail() {
        const DIM: usize = 100;
        let comps: Vec<u16> = (0u16..100).collect();
        let training = build_training_data(DIM, &comps);
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let mut encoder =
            CDotPacking8ScalarU8Encoder::new_with_references(DIM, reference_lists, 128);
        encoder.train(&training);
        let binding = [1.0, 3.0, 2.0, 3.5];
        let input = SparseVectorView::new(&[0, 4, 8, 24], &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);
        let query = SparseVectorView::new(&[2, 4, 6, 8], &[0.5, 1.5, 2.5, 1.0]);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
        let q_values = quantized_values(&encoder, &input.components(), &input.values());
        let expected = dot_with_query(
            &input.components(),
            &q_values,
            &query.components(),
            &query.values(),
        );

        assert!((dist.distance() - expected).abs() < 1e-4);
    }

    #[test]
    fn compute_distance_with_mapped_both() {
        const DIM: usize = 100;
        let comps: Vec<u16> = (0u16..100).collect();
        let training = build_training_data(DIM, &comps);
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let mut encoder =
            CDotPacking8ScalarU8Encoder::new_with_references(DIM, reference_lists, 128);
        encoder.train(&training);
        let binding = [1.0, 3.0, 2.0, 3.5, 1.5, 2.0, 1.0, 2.0, 3.0, 2.5];
        let input = SparseVectorView::new(&[0, 4, 8, 24, 28, 36, 48, 53, 70, 90], &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);
        // query = (2, 0.5), (4, 1.5), (8, 2.5), (24, 1.0), (28, 2.0), (70, 1.0), (90, 2.0)
        let query = SparseVectorView::new(
            &[2, 4, 8, 24, 28, 70, 90],
            &[0.5, 1.5, 2.5, 1.0, 2.0, 1.0, 2.0],
        );
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
        let q_values = quantized_values(&encoder, &input.components(), &input.values());
        let expected = dot_with_query(
            &input.components(),
            &q_values,
            &query.components(),
            &query.values(),
        );

        assert!((dist.distance() - expected).abs() < 1e-4);
    }

    #[test]
    fn compute_distance_only_res_tail() {
        const DIM: usize = 100;
        let comps: Vec<u16> = (0u16..100).collect();
        let training = build_training_data(DIM, &comps);
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let mut encoder =
            CDotPacking8ScalarU8Encoder::new_with_references(DIM, reference_lists, 128);
        encoder.train(&training);
        let binding = [1.0, 3.0];
        let input = SparseVectorView::new(&[5, 12], &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);
        let query = SparseVectorView::new(&[5, 9, 12], &[0.5f32, 1.5, 2.5]);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
        let q_values = quantized_values(&encoder, &input.components(), &input.values());
        let expected = dot_with_query(
            &input.components(),
            &q_values,
            &query.components(),
            &query.values(),
        );

        assert!((dist.distance() - expected).abs() < 1e-4);
    }

    #[test]
    fn compute_distance_only_res_bulk() {
        const DIM: usize = 100;
        let comps: Vec<u16> = (0u16..100).collect();
        let training = build_training_data(DIM, &comps);
        let reference_lists = vec![vec![0, 2, 4, 6, 8]];
        let mut encoder =
            CDotPacking8ScalarU8Encoder::new_with_references(DIM, reference_lists, 128);
        encoder.train(&training);
        let binding = [1.0, 2.5, 2.0, 3.5, 1.0, 2.0, 1.0, 2.0];
        let input = SparseVectorView::new(&[5, 12, 15, 18, 21, 31, 35, 60], &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);
        let query = SparseVectorView::new(&[5, 9, 12, 21, 60], &[0.5f32, 1.5, 2.5, 1.0, 2.0]);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
        let q_values = quantized_values(&encoder, &input.components(), &input.values());
        let expected = dot_with_query(
            &input.components(),
            &q_values,
            &query.components(),
            &query.values(),
        );

        assert!((dist.distance() - expected).abs() < 1e-4);
    }

    #[test]
    fn compute_distance_only_res_both() {
        const DIM: usize = 100;
        let comps: Vec<u16> = (0u16..100).collect();
        let training = build_training_data(DIM, &comps);
        let reference_lists = vec![vec![0, 2, 4, 6, 8]];
        let mut encoder =
            CDotPacking8ScalarU8Encoder::new_with_references(DIM, reference_lists, 128);
        encoder.train(&training);
        let comps = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23];
        let binding = [1.0, 2.0, 1.5, 2.5, 3.0, 1.0, 2.0, 1.5, 2.5, 3.0];
        let input = SparseVectorView::new(&comps, &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);

        let query_vals = [0.5, 1.5, 0.5, 2.0, 1.0, 1.5, 2.0, 0.5, 1.5, 2.0];
        let query = SparseVectorView::new(&comps, &query_vals);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
        let q_values = quantized_values(&encoder, &input.components(), &input.values());
        let expected = dot_with_query(
            &input.components(),
            &q_values,
            &query.components(),
            &query.values(),
        );

        assert!((dist.distance() - expected).abs() < 1e-4);
    }

    #[test]
    fn compute_distance_mapped_tail_residual_tail() {
        const DIM: usize = 100;
        let comps: Vec<u16> = (0u16..100).collect();
        let training = build_training_data(DIM, &comps);
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let mut encoder =
            CDotPacking8ScalarU8Encoder::new_with_references(DIM, reference_lists, 128);
        encoder.train(&training);
        let binding = [1.0, 2.5, 2.0, 3.5, 1.0, 2.0, 1.0, 2.0];
        let input = SparseVectorView::new(&[5, 12, 15, 18, 21, 31, 35, 60], &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);
        let query = SparseVectorView::new(&[5, 9, 12, 21, 60], &[0.5f32, 1.5, 2.5, 1.0, 2.0]);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
        let q_values = quantized_values(&encoder, &input.components(), &input.values());
        let expected = dot_with_query(
            &input.components(),
            &q_values,
            &query.components(),
            &query.values(),
        );

        assert!((dist.distance() - expected).abs() < 1e-4);
    }

    #[test]
    fn compute_distance_mapped_bulk_residual_bulk() {
        const DIM: usize = 100;
        let comps: Vec<u16> = (0u16..100).collect();
        let training = build_training_data(DIM, &comps);
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let mut encoder =
            CDotPacking8ScalarU8Encoder::new_with_references(DIM, reference_lists, 128);
        encoder.train(&training);

        let comps = [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 24, 36, 48, 53];
        let binding = [
            1.0, 2.0, 1.5, 2.5, 3.0, 1.0, 2.0, 1.5, 2.5, 3.0, 1.0, 2.0, 1.5, 2.5, 3.0, 1.0,
        ];
        let input = SparseVectorView::new(&comps, &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);

        let query_vals = [
            0.5, 1.5, 0.5, 2.0, 1.0, 1.5, 2.0, 0.5, 1.5, 2.0, 1.0, 0.5, 2.5, 1.0, 1.5, 2.0,
        ];
        let query = SparseVectorView::new(&comps, &query_vals);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
        let q_values = quantized_values(&encoder, &input.components(), &input.values());
        let expected = dot_with_query(
            &input.components(),
            &q_values,
            &query.components(),
            &query.values(),
        );

        assert!((dist.distance() - expected).abs() < 1e-4);
    }

    #[test]
    fn compute_distance_mapped_bulk_tail_residual_bulk_tail() {
        const DIM: usize = 100;
        let comps: Vec<u16> = (0u16..100).collect();
        let training = build_training_data(DIM, &comps);
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let mut encoder =
            CDotPacking8ScalarU8Encoder::new_with_references(DIM, reference_lists, 128);
        encoder.train(&training);

        let comps = [
            0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 17, 19, 24, 36, 48, 53, 87, 90,
        ];
        let binding = [
            1.0, 2.0, 1.5, 2.5, 3.0, 1.0, 2.0, 1.5, 2.5, 3.0, 1.0, 2.0, 1.5, 2.5, 3.0, 1.0, 2.0,
            1.5, 2.5, 3.0,
        ];
        let input = SparseVectorView::new(&comps, &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);

        let query_vals = [
            0.5, 1.5, 0.5, 2.0, 1.0, 1.5, 2.0, 0.5, 1.5, 2.0, 1.0, 0.5, 2.5, 1.0, 1.5, 2.0, 1.0,
            0.5, 2.5, 1.0,
        ];
        let query = SparseVectorView::new(&comps, &query_vals);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
        let q_values = quantized_values(&encoder, &input.components(), &input.values());
        let expected = dot_with_query(
            &input.components(),
            &q_values,
            &query.components(),
            &query.values(),
        );

        assert!((dist.distance() - expected).abs() < 1e-4);
    }

    #[test]
    fn test_all_bit_widths() {
        const DIM: usize = 100_000;
        for b in 1..=16 {
            let max_val = (1u32 << b) - 1;
            let num_vals = 40;
            let mut gaps = Vec::new();
            let mut refs = Vec::new();
            let mut curr = 0u32;
            for i in 0..num_vals {
                let g = (i as u32 % (max_val + 1)).max(1);
                if curr + g > u16::MAX as u32 {
                    break;
                }
                curr += g;
                gaps.push(g);
                refs.push(curr as u16);
            }
            let n = gaps.len();
            if n == 0 {
                continue;
            }

            let training = build_training_data(DIM, &refs);
            let reference_lists = vec![refs.clone()];
            let mut encoder =
                CDotPacking8ScalarU8Encoder::new_with_references(DIM, reference_lists, 128);
            encoder.train(&training);

            let values: Vec<_> = (0..n).map(|i| 1.0 + (i % 7) as f32 / 10.0).collect();
            let input = SparseVectorView::new(&refs, &values);

            let mut buffer = Vec::new();
            encoder.push_vector(input, 0, &mut buffer);

            let query_vals: Vec<_> = (0..n).map(|i| 0.1 + (i % 5) as f32 / 10.0).collect();
            let query = SparseVectorView::new(&refs, &query_vals);

            let evaluator = encoder.query_evaluator(query);
            let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
            let q_values = quantized_values(&encoder, &input.components(), &input.values());
            let expected = dot_with_query(
                &input.components(),
                &q_values,
                &query.components(),
                &query.values(),
            );

            assert!((dist.distance() - expected).abs() < 1e-4);
        }
    }

    fn same_when_quantized(
        before: &SparseVectorView<u16, f32>,
        after: &SparseVectorView<u16, f32>,
        encoder: &CDotPacking8ScalarU8Encoder,
    ) {
        assert_eq!(before.components(), after.components());
        for (c, (v1, v2)) in before
            .components()
            .iter()
            .zip(before.values().iter().zip(after.values().iter()))
        {
            let encoded = encoder.quantizer.encode_value(*c, *v1);
            let quantized = encoder.quantizer.decode_value(*c, encoded);
            assert!((quantized - v2).abs() < 1e-5);
        }
    }

    #[test]
    fn test_cdot8_decode_roundtrip() {
        const DIM: usize = 100;
        let comps: Vec<u16> = (0u16..100).collect();
        let training = build_training_data(DIM, &comps);
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let mut encoder =
            CDotPacking8ScalarU8Encoder::new_with_references(DIM, reference_lists, 128);
        encoder.train(&training);

        let components = vec![0, 4, 8, 24, 28, 36, 48, 53, 70, 90];
        let binding = [1.0, 3.0, 2.0, 3.5, 1.5, 2.0, 1.0, 2.0, 3.0, 2.5];
        let input = SparseVectorView::new(&components, &binding);

        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        same_when_quantized(&input, &decoded.as_view(), &encoder);
    }

    #[test]
    fn test_cdot8_decode_train_roundtrip() {
        const DIM: usize = 100;
        let comps: Vec<u16> = (0u16..100).collect();
        let training = build_training_data(DIM, &comps);
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let mut encoder =
            CDotPacking8ScalarU8Encoder::new_with_references(DIM, reference_lists, 128);
        encoder.train(&training);
        let binding = [1.0, 3.0, 2.0, 3.5, 2.5, 1.5];
        let input_components = [0, 4, 8, 24, 60, 87];
        let input0 = SparseVectorView::new(&input_components, &binding);
        let binding1 = [3.0, 2.0, 3.5, 2.5, 1.5];
        let input_components1 = [4, 8, 24, 50, 55];
        let input1 = SparseVectorView::new(&input_components1, &binding1);

        let mut buffer0 = Vec::new();
        encoder.push_encoded(input0, &mut buffer0);

        let mut buffer1 = Vec::new();
        encoder.push_encoded(input1, &mut buffer1);

        let decoded0 = encoder.decode_vector(PackedVectorView::new(&buffer0));
        let decoded1 = encoder.decode_vector(PackedVectorView::new(&buffer1));

        same_when_quantized(&input0, &decoded0.as_view(), &encoder);
        same_when_quantized(&input1, &decoded1.as_view(), &encoder);
    }
}
