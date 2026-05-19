use crate::encoders::eg::cencoder::{CegEncoder, CegQueryEvaluator};
use crate::encoders::eg::quantizer::FixedU8Quantizer;
use crate::{PlainSparseDataset, SquaredEuclideanDistance, Dataset};

pub type CegFixedU8Encoder = CegEncoder<FixedU8Quantizer>;
pub type CegFixedU8QueryEvaluator<'e> = CegQueryEvaluator<'e, FixedU8Quantizer>;

impl CegFixedU8Encoder {
    pub fn new_with_references(input_dim: usize, reference_lists: Vec<Vec<u16>>, max_ref_size: usize) -> Self {
        CegEncoder::new(input_dim, FixedU8Quantizer, reference_lists, max_ref_size)
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
        FixedU8Q, FromF32, QueryEvaluator, VectorEncoder
    };
    use num_traits::ToPrimitive;

    #[test]
    fn cef_compute_distance_with_only_mapped_bulk() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let encoder = CegFixedU8Encoder::new_with_references(100, reference_lists, 512);
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
        
        let q1 = FixedU8Q::from_f32_saturating(3.0).to_f32().unwrap();
        let q2 = FixedU8Q::from_f32_saturating(2.0).to_f32().unwrap();
        let q3 = FixedU8Q::from_f32_saturating(3.5).to_f32().unwrap();
        let q4 = FixedU8Q::from_f32_saturating(2.0).to_f32().unwrap();
        let expected = 1.5 * q1 + 1.0 * q2 + 2.0 * q3 + 2.0 * q4;
        assert!((dist.distance() - expected).abs() < 1e-5);
    }

    #[test]
    fn test_ceg_decode_roundtrip() {
        let reference_lists = vec![vec![0, 2, 4, 6, 8, 12, 24, 36, 48, 50, 53, 87, 90]];
        let encoder = CegFixedU8Encoder::new_with_references(100, reference_lists, 512);

        let components = vec![0, 4, 8, 24, 28, 36, 48, 53, 70, 90];
        let binding = [1.0, 3.0, 2.0, 3.5, 1.5, 2.0, 1.0, 2.0, 3.0, 2.5];
        let input = SparseVectorView::new(&components, &binding);

        let mut buffer = Vec::new();
        encoder.push_vector(input, 0, &mut buffer);

        let decoded = encoder.decode_vector(PackedVectorView::new(&buffer));
        assert_eq!(decoded.components(), &components);
        for (v1, v2) in decoded.values().iter().zip(binding.iter()) {
            let quantized = FixedU8Q::from_f32_saturating(*v2).to_f32().unwrap();
            assert!((v1 - quantized).abs() < 1e-5);
        }
    }
}
