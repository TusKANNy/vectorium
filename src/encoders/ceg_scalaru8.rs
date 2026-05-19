use crate::encoders::eg::cencoder::{CegEncoder, CegQueryEvaluator};
use crate::encoders::eg::quantizer::ScalarU8Quantizer;
use crate::{Dataset, PlainSparseDataset, SquaredEuclideanDistance};

pub type CegScalarU8Encoder = CegEncoder<ScalarU8Quantizer>;
pub type CegScalarU8QueryEvaluator<'e> = CegQueryEvaluator<'e, ScalarU8Quantizer>;

impl CegScalarU8Encoder {
    pub fn new_with_references(input_dim: usize, reference_lists: Vec<Vec<u16>>, max_ref_size: usize) -> Self {
        CegEncoder::new(
            input_dim,
            ScalarU8Quantizer {
                quants: vec![0.0; input_dim].into_boxed_slice(),
            },
            reference_lists,
            max_ref_size,
        )
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
        self.quantizer.train(training_data);
    }
}
