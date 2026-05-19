use crate::encoders::dotpackingdp::encoder::{DotPackingDpEncoder, DotPackingDpQueryEvaluator};
use crate::encoders::dotpackingdp::quantizer::FixedU8Quantizer;
use crate::{Dataset, PlainSparseDataset, SquaredEuclideanDistance};

pub type DotPackingDp8FixedU8Encoder = DotPackingDpFixedU8Encoder<8>;
pub type DotPackingDp16FixedU8Encoder = DotPackingDpFixedU8Encoder<16>;

pub type DotPackingDpFixedU8Encoder<const MAX_BLOCK_LEN: usize> =
    DotPackingDpEncoder<FixedU8Quantizer, MAX_BLOCK_LEN>;
pub type DotPackingDpFixedu8QueryEvaluator<'e, const MAX_BLOCK_LEN: usize> =
    DotPackingDpQueryEvaluator<'e, FixedU8Quantizer, MAX_BLOCK_LEN>;

impl<const MAX_BLOCK_LEN: usize> DotPackingDpFixedU8Encoder<MAX_BLOCK_LEN> {
    pub fn new(input_dim: usize) -> Self {
        DotPackingDpEncoder::new_with_quantizer(input_dim, FixedU8Quantizer)
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DatasetGrowable, FixedU8Q, PlainSparseDatasetGrowable, PlainSparseQuantizer};
    use crate::utils::numeric_markers::FromF32;
    use crate::core::distances::Distance;
    use crate::core::vector::{PackedVectorView, SparseVectorView};
    use crate::vector_encoder::{PackedSparseVectorEncoder, SparseDataEncoder};
    use bytemuck::cast_slice;
    use num_traits::ToPrimitive;

    fn verify_dot_product_only<const M: usize>(
        components: &[u16],
        values: &[f32],
        dim: usize,
    ) {
        let encoder = DotPackingDpFixedU8Encoder::<M>::new(dim);
        let input = SparseVectorView::new(components, values);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);
        let packed_view = PackedVectorView::new(&buffer);
        let query_values: Vec<f32> = (0..components.len())
            .map(|i| ((i + 1) as f32) * 0.25)
            .collect();
        let query = SparseVectorView::new(components, &query_values);
        let evaluator = crate::VectorEncoder::query_evaluator(&encoder, query);
        let dist = crate::QueryEvaluator::compute_distance(&evaluator, packed_view);
        let expected: f32 = values
            .iter()
            .zip(query_values.iter())
            .map(|(&v, &q)| {
                FixedU8Q::from_f32_saturating(v).to_f32().unwrap() * q
            })
            .sum();
        assert!((dist.distance() - expected).abs() < 1e-4);
    }

    #[test]
    fn test_block_dp_n_less_than_8() {
        let components = vec![10, 20, 30, 40, 50];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        verify_dot_product_only::<16>(&components, &values, 100);
    }

    #[test]
    fn test_block_dp_n_equal_8() {
        let components: Vec<u16> = (0..8).map(|i| (i * 10) as u16).collect();
        let values: Vec<f32> = (0..8).map(|i| i as f32).collect();
        verify_dot_product_only::<16>(&components, &values, 100);
    }

    #[test]
    fn test_block_dp_n_greater_than_8() {
        let components: Vec<u16> = (0..12).map(|i| (i * 5) as u16).collect();
        let values: Vec<f32> = (0..12).map(|i| i as f32 / 10.0).collect();
        verify_dot_product_only::<16>(&components, &values, 100);
    }

    #[test]
    fn test_block_dp_max8() {
        let components: Vec<u16> = (0..20).map(|i| (i * 5) as u16).collect();
        let values: Vec<f32> = (0..20).map(|i| i as f32 / 10.0).collect();
        let encoder = DotPackingDpFixedU8Encoder::<8>::new(1000);
        let input = SparseVectorView::new(&components, &values);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);
        let bytes = cast_slice::<u64, u8>(&buffer);
        let n_blocks = u16::from_le_bytes(bytes[2..4].try_into().unwrap()) as usize;
        let selectors = &bytes[4..4 + n_blocks];
        for &sel in selectors {
            let l = (sel & 0x0F) as usize + 1;
            assert!(l <= 8);
        }
        verify_dot_product_only::<8>(&components, &values, 1000);
    }

    #[test]
    fn test_block_dp_max16() {
        let components: Vec<u16> = (0..20).map(|i| (i * 5) as u16).collect();
        let values: Vec<f32> = (0..20).map(|i| i as f32 / 10.0).collect();
        let encoder = DotPackingDpFixedU8Encoder::<16>::new(1000);
        let input = SparseVectorView::new(&components, &values);
        let mut buffer = Vec::new();
        encoder.push_encoded(input, &mut buffer);
        let bytes = cast_slice::<u64, u8>(&buffer);
        let n_blocks = u16::from_le_bytes(bytes[2..4].try_into().unwrap()) as usize;
        let selectors = &bytes[4..4 + n_blocks];
        for &sel in selectors {
            let l = (sel & 0x0F) as usize + 1;
            assert!(l <= 16);
        }
        verify_dot_product_only::<16>(&components, &values, 1000);
    }

    #[test]
    fn blockdp_decode_roundtrip() {
        let mut encoder = DotPackingDp16FixedU8Encoder::new(100);

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
