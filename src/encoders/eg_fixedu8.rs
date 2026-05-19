use crate::encoders::eg::encoder::{EgEncoder, EgQueryEvaluator};
use crate::encoders::eg::quantizer::FixedU8Quantizer;
use crate::{Dataset, PlainSparseDataset, SquaredEuclideanDistance};

pub type EgFixedU8Encoder = EgEncoder<FixedU8Quantizer>;
pub type EgFixedU8QueryEvaluator<'e> = EgQueryEvaluator<'e, FixedU8Quantizer>;

impl EgFixedU8Encoder {
    pub fn new(input_dim: usize) -> Self {
        EgEncoder::new_with_quantizer(input_dim, FixedU8Quantizer)
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
    use crate::{FixedU8Q, FromF32, PackedSparseVectorEncoder, QueryEvaluator, VectorEncoder};
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
    fn eg_compute_distance_basic() {
        let encoder = EgFixedU8Encoder::new(100);
        let binding = [1.0, 3.0, 2.0, 3.5, 1.5, 2.0, 1.0, 2.0];
        let input = SparseVectorView::new(&[0, 4, 8, 24, 36, 48, 53, 90], &binding);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let query = SparseVectorView::new(
            &[2, 4, 6, 8, 24, 70, 90],
            &[0.5, 1.5, 2.5, 1.0, 2.0, 1.0, 2.0],
        );
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let expected = calculate_expected_distance(&input, &query);
        assert!((dist.distance() - expected).abs() < 1e-5);
    }

    fn verify_eg(gaps: &[u32]) {
        let num_vals = gaps.len();
        let mut components = Vec::new();
        let mut curr = 0u32;
        for &g in gaps {
            curr += g;
            assert!(curr < u16::MAX as u32);
            components.push(curr as u16);
        }

        let encoder = EgFixedU8Encoder::new(u16::MAX as usize + 1);
        let values: Vec<_> = (0..num_vals).map(|i| 1.0 + (i % 7) as f32 / 10.0).collect();
        let input = SparseVectorView::new(&components, &values);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let query_vals: Vec<_> = (0..num_vals).map(|i| 0.1 + (i % 5) as f32 / 10.0).collect();
        let query = SparseVectorView::new(&components, &query_vals);

        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let expected = calculate_expected_distance(&input, &query);
        assert!((dist.distance() - expected).abs() < 1e-3);
    }

    #[test]
    fn test_eg_dist_various() {
        // Just a few selected ones to verify logic
        verify_eg(&[1; 64]);
        verify_eg(&[3; 64]);
        verify_eg(&[7; 64]);
        verify_eg(&[15; 64]);
        verify_eg(&[31; 64]);
        verify_eg(&[127; 8]);
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
    fn eg_decode_roundtrip() {
        let mut encoder = EgFixedU8Encoder::new(100);
        let binding = [1.0, 3.0, 2.0, 3.5];
        let input_components = [0, 4, 8, 24];
        let input0 = SparseVectorView::new(&input_components, &binding);
        let binding1 = [3.0, 2.0, 3.5];
        let input_components1 = [4, 8, 24];
        let input1 = SparseVectorView::new(&input_components1, &binding1);

        encoder.train_components([input0.clone(), input1.clone()].into_iter());

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
