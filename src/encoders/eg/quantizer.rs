use crate::utils::train_sparse_scalar_quantizer;
use crate::{FixedU8Q, FromF32, PlainSparseDataset, SpaceUsage, SquaredEuclideanDistance, ValueType};

pub trait EgQuantizer: Send + Sync + SpaceUsage {
    type InputValue: ValueType;

    fn encode_value(&self, component: u16, input: Self::InputValue) -> u8;
    fn decode_value(&self, component: u16, output: u8) -> f32;
    fn query_value(&self, component: u16, input: Self::InputValue) -> f32;
    fn scale(&self) -> f32;
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FixedU8Quantizer;

impl SpaceUsage for FixedU8Quantizer {
    fn space_usage_bytes(&self) -> usize {
        0
    }
}

const FIXED_U8_SCALE: f32 = 1.0 / ((1u32 << FixedU8Q::FRAC_NBITS) as f32);

impl EgQuantizer for FixedU8Quantizer {
    type InputValue = f32;

    #[inline(always)]
    fn encode_value(&self, _component: u16, input: f32) -> u8 {
        FixedU8Q::from_f32_saturating(input).to_bits()
    }

    #[inline(always)]
    fn decode_value(&self, _component: u16, output: u8) -> f32 {
        output as f32 * FIXED_U8_SCALE
    }

    #[inline(always)]
    fn query_value(&self, _component: u16, input: f32) -> f32 {
        input
    }

    #[inline(always)]
    fn scale(&self) -> f32 {
        FIXED_U8_SCALE
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScalarU8Quantizer {
    pub quants: Box<[f32]>,
}

impl SpaceUsage for ScalarU8Quantizer {
    fn space_usage_bytes(&self) -> usize {
        self.quants.len() * std::mem::size_of::<f32>()
    }
}

impl ScalarU8Quantizer {
    pub fn new(quants: Box<[f32]>) -> Self {
        Self { quants }
    }

    pub fn train(&mut self, training_data: &PlainSparseDataset<u16, f32, SquaredEuclideanDistance>) {
        self.quants = train_sparse_scalar_quantizer(training_data, 0.0, 1.0).into_boxed_slice();
    }
}

impl EgQuantizer for ScalarU8Quantizer {
    type InputValue = f32;

    #[inline(always)]
    fn encode_value(&self, component: u16, value: f32) -> u8 {
        let q = self.quants[component as usize];
        if q > 0.0 {
            (value / q).clamp(0.0, 255.0) as u8
        } else {
            0u8
        }
    }

    #[inline(always)]
    fn decode_value(&self, component: u16, output: u8) -> f32 {
        output as f32 * self.quants[component as usize]
    }

    #[inline(always)]
    fn query_value(&self, component: u16, input: f32) -> f32 {
        input * self.quants[component as usize]
    }

    #[inline(always)]
    fn scale(&self) -> f32 {
        1.0
    }
}
