use crate::encoders::dotpackingdp::cencoder::{CDotPackingDpEncoder, CDotPackingDpQueryEvaluator};
use crate::encoders::dotpackingdp::quantizer::FixedU8Quantizer;
use crate::{Dataset, PlainSparseDataset, SquaredEuclideanDistance};

pub type CDotPackingDpFixedu8Encoder<const MAX_LEN: usize> =
    CDotPackingDpEncoder<FixedU8Quantizer, MAX_LEN>;
pub type CDotPackingDpFixedu8QueryEvaluator<'e, const MAX_LEN: usize> =
    CDotPackingDpQueryEvaluator<'e, FixedU8Quantizer, MAX_LEN>;

pub type CDotPackingDp8FixedU8Encoder = CDotPackingDpFixedu8Encoder<8>;
pub type CDotPackingDp16FixedU8Encoder = CDotPackingDpFixedu8Encoder<16>;

impl<const MAX_LEN: usize> CDotPackingDpFixedu8Encoder<MAX_LEN> {
    pub fn new_with_references(
        input_dim: usize,
        reference_lists: Vec<Vec<u16>>,
        max_ref_size: usize,
    ) -> Self {
        CDotPackingDpEncoder::new(input_dim, FixedU8Quantizer, reference_lists, max_ref_size)
    }

    pub fn train<'a>(
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::distances::Distance;
    use crate::core::vector::{PackedVectorView, SparseVectorView};
    use crate::vector_encoder::SparseDataEncoder;
    use crate::{
        DatasetGrowable, FixedU8Q, FromF32, PackedSparseVectorEncoder, PlainSparseDatasetGrowable,
        PlainSparseQuantizer, QueryEvaluator, VectorEncoder,
    };
    use num_traits::ToPrimitive;

    fn calculate_expected_distance(
        vector: &SparseVectorView<u16, f32>,
        query: &SparseVectorView<u16, f32>,
    ) -> f32 {
        let mut expected = 0.0f32;
        let mut vec_iter = vector.iter().peekable(); // Usiamo peekable per non "perdere" l'elemento

        for (comp, val) in query.iter() {
            while let Some(&(vec_comp, _)) = vec_iter.peek() {
                if vec_comp < comp {
                    vec_iter.next(); // Salta questo elemento del vettore, è troppo indietro
                } else {
                    break;
                }
            }

            if let Some(&(vec_comp, vec_val)) = vec_iter.peek() {
                if vec_comp == comp {
                    expected += FixedU8Q::from_f32_saturating(vec_val).to_f32().unwrap() * val;
                    vec_iter.next();
                }
            }
        }
        expected
    }

    #[test]
    fn compute_distance_with_only_mapped_bulk() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let encoder = CDotPackingDp16FixedU8Encoder::new_with_references(100, reference_lists, 512);
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
        let expected = calculate_expected_distance(&input, &query);
        assert_eq!(dist.distance(), expected);
    }

    #[test]
    fn compute_distance_with_only_mapped_tail() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let encoder = CDotPackingDp16FixedU8Encoder::new_with_references(100, reference_lists, 512);
        let binding = [1.0, 3.0, 2.0, 3.5];
        let input = SparseVectorView::new(&[0, 4, 8, 24], &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);
        let query = SparseVectorView::new(&[2, 4, 6, 8], &[0.5, 1.5, 2.5, 1.0]);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let expected = calculate_expected_distance(&input, &query);
        assert_eq!(dist.distance(), expected);
    }

    #[test]
    fn compute_distance_with_mapped_both() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let encoder = CDotPackingDp16FixedU8Encoder::new_with_references(100, reference_lists, 512);
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
        let expected = calculate_expected_distance(&input, &query);
        assert_eq!(dist.distance(), expected);
    }

    #[test]
    fn compute_distance_only_res_tail() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8]];
        let encoder = CDotPackingDp16FixedU8Encoder::new_with_references(25, reference_lists, 512);
        let binding = [1.0, 3.0];
        let input = SparseVectorView::new(&[5, 12], &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);
        let query = SparseVectorView::new(&[5, 9, 12], &[0.5f32, 1.5, 2.5]);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));
        let expected = calculate_expected_distance(&input, &query);

        assert_eq!(dist.distance(), expected);
    }

    #[test]
    fn compute_distance_only_res_bulk() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8]];
        let encoder = CDotPackingDp16FixedU8Encoder::new_with_references(70, reference_lists, 512);
        let binding = [1.0, 2.5, 2.0, 3.5, 1.0, 2.0, 1.0, 2.0];
        let input = SparseVectorView::new(&[5, 12, 15, 18, 21, 31, 35, 60], &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);
        let query = SparseVectorView::new(&[5, 9, 12, 21, 60], &[0.5f32, 1.5, 2.5, 1.0, 2.0]);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let expected = calculate_expected_distance(&input, &query);
        assert_eq!(dist.distance(), expected);
    }

    #[test]
    fn compute_distance_only_res_both() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8]];
        let encoder = CDotPackingDp16FixedU8Encoder::new_with_references(80, reference_lists, 512);
        let comps = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23];
        let binding = [1.0, 2.0, 1.5, 2.5, 3.0, 1.0, 2.0, 1.5, 2.5, 3.0];
        let input = SparseVectorView::new(&comps, &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);

        let query_vals = [0.5, 1.5, 0.5, 2.0, 1.0, 1.5, 2.0, 0.5, 1.5, 2.0];
        let query = SparseVectorView::new(&comps, &query_vals);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let expected = calculate_expected_distance(&input, &query);
        assert_eq!(dist.distance(), expected);
    }

    #[test]
    fn compute_distance_mapped_tail_residual_tail() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let encoder = CDotPackingDp16FixedU8Encoder::new_with_references(100, reference_lists, 512);
        let binding = [1.0, 2.5, 2.0, 3.5, 1.0, 2.0, 1.0, 2.0];
        let input = SparseVectorView::new(&[5, 12, 15, 18, 21, 31, 35, 60], &binding);
        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);
        let query = SparseVectorView::new(&[5, 9, 12, 21, 60], &[0.5f32, 1.5, 2.5, 1.0, 2.0]);
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let expected = calculate_expected_distance(&input, &query);
        assert_eq!(dist.distance(), expected);
    }

    #[test]
    fn compute_distance_mapped_bulk_residual_bulk() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let encoder = CDotPackingDp16FixedU8Encoder::new_with_references(100, reference_lists, 512);

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

        let expected = calculate_expected_distance(&input, &query);
        assert_eq!(dist.distance(), expected);
    }

    #[test]
    fn compute_distance_mapped_bulk_tail_residual_bulk_tail() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let encoder = CDotPackingDp16FixedU8Encoder::new_with_references(120, reference_lists, 512);

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

        let expected = calculate_expected_distance(&input, &query);
        assert_eq!(dist.distance(), expected);
    }

    #[test]
    fn test_all_bit_widths() {
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

            let reference_lists = vec![refs.clone()];
            let encoder = CDotPackingDp16FixedU8Encoder::new_with_references(
                u16::MAX as usize + 1,
                reference_lists,
                512,
            );

            let values: Vec<_> = (0..n).map(|i| 1.0 + (i % 7) as f32 / 10.0).collect();
            let input = SparseVectorView::new(&refs, &values);

            let mut buffer = Vec::new();
            encoder.push_vector(input, 0, &mut buffer);

            let query_vals: Vec<_> = (0..n).map(|i| 0.1 + (i % 5) as f32 / 10.0).collect();
            let query = SparseVectorView::new(&refs, &query_vals);

            let evaluator = encoder.query_evaluator(query);
            let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

            let expected = calculate_expected_distance(&input, &query);
            assert_eq!(dist.distance(), expected);
        }
    }

    fn same_when_quantized(
        before: &SparseVectorView<u16, f32>,
        after: &SparseVectorView<u16, f32>,
    ) {
        assert_eq!(before.components(), after.components());
        for (v1, v2) in before.values().iter().zip(after.values().iter()) {
            let q1 = FixedU8Q::from_f32_saturating(*v1).to_f32().unwrap();
            let q2 = FixedU8Q::from_f32_saturating(*v2).to_f32().unwrap();
            assert_eq!(q1, q2);
        }
    }

    #[test]
    fn test_cdotdp_decode_roundtrip() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let encoder = CDotPackingDp16FixedU8Encoder::new_with_references(100, reference_lists, 512);

        let components = vec![0, 4, 8, 24, 28, 36, 48, 53, 70, 90];
        let binding = [1.0, 3.0, 2.0, 3.5, 1.5, 2.0, 1.0, 2.0, 3.0, 2.5];
        let input = SparseVectorView::new(&components, &binding);

        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        same_when_quantized(&input, &decoded.as_view());
    }

    #[test]
    fn test_cdotdp_decode_train_roundtrip() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let mut encoder =
            CDotPackingDp16FixedU8Encoder::new_with_references(100, reference_lists, 512);
        let binding = [1.0, 3.0, 2.0, 3.5, 2.5, 1.5];
        let input_components = [0, 4, 8, 24, 60, 87];
        let input0 = SparseVectorView::new(&input_components, &binding);
        let binding1 = [3.0, 2.0, 3.5, 2.5, 1.5];
        let input_components1 = [4, 8, 24, 50, 55];
        let input1 = SparseVectorView::new(&input_components1, &binding1);

        let quantizer = PlainSparseQuantizer::<u16, f32, SquaredEuclideanDistance>::new(100, 100);

        let mut growable = PlainSparseDatasetGrowable::new(quantizer);

        growable.push(input0.clone());
        growable.push(input1.clone());

        let training_data = growable.into();

        encoder.train(&training_data);

        let mut buffer0 = Vec::new();
        encoder.push_encoded(input0, &mut buffer0);

        let mut buffer1 = Vec::new();
        encoder.push_encoded(input1, &mut buffer1);

        let decoded0 = encoder.decode_vector(PackedVectorView::new(&buffer0));
        let decoded1 = encoder.decode_vector(PackedVectorView::new(&buffer1));

        same_when_quantized(&input0, &decoded0.as_view());
        same_when_quantized(&input1, &decoded1.as_view());
    }
}
