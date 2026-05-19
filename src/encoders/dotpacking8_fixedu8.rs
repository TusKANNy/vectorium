use crate::encoders::dotpacking8::encoder::{DotPacking8Encoder, DotPacking8QueryEvaluator};
use crate::encoders::dotpacking8::quantizer::FixedU8Quantizer;
use crate::{Dataset, PlainSparseDataset, SquaredEuclideanDistance};

pub type DotPacking8FixedU8Encoder = DotPacking8Encoder<FixedU8Quantizer>;
pub type DotPacking8Fixedu8QueryEvaluator<'e> = DotPacking8QueryEvaluator<'e, FixedU8Quantizer>;

impl DotPacking8Encoder<FixedU8Quantizer> {
    pub fn new(input_dim: usize) -> Self {
        DotPacking8Encoder::new_with_quantizer(input_dim, FixedU8Quantizer)
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
        DatasetGrowable, FixedU8Q, FromF32, PackedSparseVectorEncoder, PlainSparseDatasetGrowable, PlainSparseQuantizer, QueryEvaluator, VectorEncoder
    };
    use num_traits::ToPrimitive;

    #[test]
    fn compute_distance_only_bulk() {
        // 1. Istanziamo l'encoder passando proprio FixedU8Quantizer!
        let encoder = DotPacking8Encoder {
            dim: 100,
            quantizer: FixedU8Quantizer,
            component_mapping: None,
            inverse_component_mapping: None,
        };

        let binding = [1.0, 3.0, 2.0, 3.5, 1.5, 2.0, 1.0, 2.0];
        let input = SparseVectorView::new(&[2, 4, 8, 24, 36, 48, 53, 90], &binding);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let query = SparseVectorView::new(
            &[2, 4, 6, 8, 24, 70, 90],
            &[0.5, 1.5, 2.5, 1.0, 2.0, 1.0, 2.0],
        );
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        // 2. Il calcolo dell'atteso rispecchia la quantizzazione a punto fisso di FixedU8Q
        let expected = FixedU8Q::from_f32_saturating(1.0).to_f32().unwrap() * 0.5
            + FixedU8Q::from_f32_saturating(3.0).to_f32().unwrap() * 1.5
            + FixedU8Q::from_f32_saturating(2.0).to_f32().unwrap() * 1.0
            + FixedU8Q::from_f32_saturating(3.5).to_f32().unwrap() * 2.0
            + FixedU8Q::from_f32_saturating(2.0).to_f32().unwrap() * 2.0;

        assert!((dist.distance() - expected).abs() < 1e-5);
    }

    #[test]
    fn compute_distance_bulk_and_tail() {
        let encoder = DotPacking8Encoder {
            dim: 100,
            quantizer: FixedU8Quantizer,
            component_mapping: None,
            inverse_component_mapping: None,
        };

        let binding = [1.0, 3.0, 2.0, 3.5, 1.5, 2.0, 1.0, 2.0, 3.0, 2.5];
        let input = SparseVectorView::new(&[2, 4, 8, 24, 28, 36, 48, 53, 70, 90], &binding);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let query = SparseVectorView::new(
            &[2, 4, 8, 24, 28, 70, 90],
            &[0.5, 1.5, 2.5, 1.0, 2.0, 1.0, 2.0],
        );
        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let expected = FixedU8Q::from_f32_saturating(1.0).to_f32().unwrap() * 0.5
            + FixedU8Q::from_f32_saturating(3.0).to_f32().unwrap() * 1.5
            + FixedU8Q::from_f32_saturating(2.0).to_f32().unwrap() * 2.5
            + FixedU8Q::from_f32_saturating(3.5).to_f32().unwrap() * 1.0
            + FixedU8Q::from_f32_saturating(1.5).to_f32().unwrap() * 2.0
            + FixedU8Q::from_f32_saturating(3.0).to_f32().unwrap() * 1.0
            + FixedU8Q::from_f32_saturating(2.5).to_f32().unwrap() * 2.0;

        assert!((dist.distance() - expected).abs() < 1e-5);
    }

    fn verify_gaps(gaps: &[u32]) {
        let num_vals = gaps.len();
        let mut components = Vec::new();
        let mut curr = 0u32;
        for &g in gaps {
            curr += g;
            components.push(curr as u16);
        }

        let encoder = DotPacking8Encoder {
            dim: u16::MAX as usize + 1,
            quantizer: FixedU8Quantizer,
            component_mapping: None,
            inverse_component_mapping: None,
        };

        let values: Vec<_> = (0..num_vals).map(|i| 1.0 + (i % 7) as f32 / 10.0).collect();
        let input = SparseVectorView::new(&components, &values);

        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);

        let query_vals: Vec<_> = (0..num_vals).map(|i| 0.1 + (i % 5) as f32 / 10.0).collect();
        let query = SparseVectorView::new(&components, &query_vals);

        let evaluator = encoder.query_evaluator(query);
        let dist = evaluator.compute_distance(PackedVectorView::new(&buffer));

        let mut expected = 0.0f32;
        for i in 0..num_vals {
            let quantized_val = FixedU8Q::from_f32_saturating(values[i]).to_f32().unwrap();
            expected += quantized_val * query_vals[i];
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
        verify_gaps(&gaps);
    }

    #[test]
    fn test_plain_small_gaps() {
        let gaps = vec![1u32; 1];
        verify_gaps(&gaps);
    }

    #[test]
    fn test_plain_with_tail() {
        let gaps = vec![1u32; 11];
        verify_gaps(&gaps);
    }

    #[test]
    fn test_plain_large_gaps() {
        let gaps = vec![1000u32, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000];
        verify_gaps(&gaps);
    }

    #[test]
    fn test_plain_u16_gaps() {
        let gaps = vec![60000, 1, 1, 2, 3, 200, 450, 2000, 3];
        verify_gaps(&gaps);
    }

    #[test]
    fn block8_decode_roundtrip() {
        let mut encoder = DotPacking8Encoder {
            dim: 100,
            quantizer: FixedU8Quantizer,
            component_mapping: None,
            inverse_component_mapping: None,
        };

        let binding = [1.0, 3.0, 2.0, 3.5];
        let input_components = [0, 4, 8, 24];
        let input0 = SparseVectorView::new(&input_components, &binding);
        let binding1 = [3.0, 2.0, 3.5];
        let input_components1 = [4, 8, 24];
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

        assert_eq!(decoded0.components(), &input_components);
        let decoded_vals0 = decoded0.values();
        assert_eq!(decoded_vals0.len(), 4);

        // Usiamo una tolleranza di 1e-3 perché FixedU8Q introduce piccoli scostamenti di precisione decimale
        assert!((decoded_vals0[0] - 1.0).abs() < 1e-3);
        assert!((decoded_vals0[1] - 3.0).abs() < 1e-3);
        assert!((decoded_vals0[2] - 2.0).abs() < 1e-3);
        assert!((decoded_vals0[3] - 3.5).abs() < 1e-3);

        assert_eq!(decoded1.components(), &[4, 8, 24]);
        let decoded_vals1 = decoded1.values();
        assert_eq!(decoded_vals1.len(), 3);
        assert!((decoded_vals1[0] - 3.0).abs() < 1e-3);
        assert!((decoded_vals1[1] - 2.0).abs() < 1e-3);
        assert!((decoded_vals1[2] - 3.5).abs() < 1e-3);
    }
}
