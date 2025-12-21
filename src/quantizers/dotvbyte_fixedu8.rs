use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::datasets::sparse_dataset::sparse_dot_vbyte_dataset::dot_vbyte_fixedu8::StreamVbyteFixedu8;
use crate::distances::DotProduct;
use crate::packed_vector::PackedVector;
use crate::quantizers::{PackedQuantizer, Quantizer, QueryEvaluator, QueryVectorFor};
use crate::{FixedU8Q, SpaceUsage, SparseVector1D, Vector1D};

/// Quantizer for DotVByte-packed sparse vectors with `FixedU8Q` values.
///
/// - Encoded vectors are represented as a packed slice of `u64` words.
/// - `output_dim()` is the logical post-quantization dimensionality (typically equal to `input_dim()`),
///   NOT the packed blob length in `u64` words.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DotVByteFixedU8Quantizer {
    dim: usize,
    component_mapping: Option<Box<[u16]>>, // Optional component remapping that improves compression. mapping[i] = new_index_of_component_i
}

impl DotVByteFixedU8Quantizer {
    #[inline]
    pub fn component_mapping(&self) -> Option<&[u16]> {
        self.component_mapping.as_deref()
    }

    /// Encode a sparse vector (components + `FixedU8Q` values) into the packed DotVByte format
    /// and append it to `out_words`.
    ///
    /// The packed representation is appended as a sequence of `u64` words; its length depends
    /// on the vector sparsity and is therefore variable.
    pub fn extend_with_encode(
        // XXXXX_TODO: NOT WORKING PLEASE FIX &self,
        input_vector: SparseVector1D<u16, FixedU8Q, impl AsRef<[u16]>, impl AsRef<[FixedU8Q]>>,
        out_words: &mut Vec<u64>,
    ) {
        todo!("Fix me");
        let components_in = input_vector.components_as_slice();
        let values_in = input_vector.values_as_slice();
        assert_eq!(components_in.len(), values_in.len());

        // NOTE: this method expects `components_in` to already be in the final component space
        // (i.e., remapped if `component_mapping` is used). This keeps encoding flexible and
        // lets callers decide when/how to apply remapping + sorting.
        let mut components: Vec<u16> = components_in.to_vec();
        let mut values: Vec<u8> = values_in.iter().map(|v| v.to_bits()).collect();

        let mut posting: Vec<u8> = Vec::new();
        StreamVbyteFixedu8::push_posting(&mut posting, &mut components, &mut values);

        // Prefix with the number of non-zeros so decoding knows how to interpret the packed layout.
        out_words.push(values_in.len() as u64);

        // `push_posting` guarantees the byte buffer is padded to a multiple of 8 bytes.
        debug_assert_eq!(posting.len() % 8, 0);
        for chunk in posting.chunks_exact(8) {
            let arr: [u8; 8] = chunk.try_into().unwrap();
            out_words.push(u64::from_ne_bytes(arr));
        }
    }
}

impl PackedQuantizer for DotVByteFixedU8Quantizer {
    type EncodingType = u64;
}

impl Quantizer for DotVByteFixedU8Quantizer {
    type Distance = DotProduct;

    type QueryValueType = f32;
    type QueryComponentType = u16;

    type InputValueType = FixedU8Q;
    type InputComponentType = u16;

    type OutputValueType = FixedU8Q;
    type OutputComponentType = u16;

    type Evaluator<'a>
        = DotVByteFixedU8QueryEvaluator<'a>
    where
        Self: 'a;

    type EncodedVector<'a> = PackedVector<u64, &'a [u64]>;

    #[inline]
    fn new(input_dim: usize, output_dim: usize) -> Self {
        assert_eq!(
            input_dim, output_dim,
            "DotVByteFixedU8Quantizer requires input_dim == output_dim"
        );
        Self {
            dim: input_dim,
            component_mapping: None,
        }
    }

    #[inline]
    fn get_query_evaluator<'a, QueryVector>(&'a self, query: &'a QueryVector) -> Self::Evaluator<'a>
    where
        QueryVector: QueryVectorFor<Self> + ?Sized,
    {
        DotVByteFixedU8QueryEvaluator::new(query, self)
    }

    #[inline]
    fn output_dim(&self) -> usize {
        self.dim
    }

    #[inline]
    fn input_dim(&self) -> usize {
        self.dim
    }

    fn train<'a>(&mut self, _training_data: impl Iterator<Item = Self::EncodedVector<'a>>) {
        todo!("learn optional component remapping to improve dotvbyte compression");
    }
}

/// Allow sparse query vectors (`SparseVector1D<u16, f32, ..>`) for DotVByte quantization.
impl<AC, AV> QueryVectorFor<DotVByteFixedU8Quantizer> for SparseVector1D<u16, f32, AC, AV>
where
    AC: AsRef<[u16]>,
    AV: AsRef<[f32]>,
{
}

pub struct DotVByteFixedU8QueryEvaluator<'a> {
    dense_query: Vec<f32>,
    _phantom: PhantomData<&'a ()>,
}

impl<'a> DotVByteFixedU8QueryEvaluator<'a> {
    #[inline]
    pub fn new<QueryVector>(query: &'a QueryVector, quantizer: &DotVByteFixedU8Quantizer) -> Self
    where
        QueryVector: Vector1D<ComponentType = u16, ValueType = f32> + ?Sized,
    {
        // Densify the remapped query vector
        let mut vec = vec![0.0; quantizer.dim];
        println!(
            "Creating dense query vector for DotVByteFixedU8QueryEvaluator with dim {}",
            quantizer.dim
        );
        let query_components = query.components_as_slice();
        let query_values = query.values_as_slice();

        for (&c, &v) in query_components.iter().zip(query_values.iter()) {
            let mapped_component = if let Some(component_mapping) = &quantizer.component_mapping {
                component_mapping[c as usize]
            } else {
                c
            };
            vec[mapped_component as usize] = v;
        }

        Self {
            dense_query: vec,
            _phantom: PhantomData,
        }
    }
}

impl QueryEvaluator<DotVByteFixedU8Quantizer> for DotVByteFixedU8QueryEvaluator<'_> {
    #[inline]
    fn compute_distance(
        &self,
        vector: <DotVByteFixedU8Quantizer as Quantizer>::EncodedVector<'_>,
    ) -> <DotVByteFixedU8Quantizer as Quantizer>::Distance {
        let packed_words: &[u64] = vector.as_slice();
        let dotvbyte_view = unsafe { StreamVbyteFixedu8::from_unchecked_slice(vector.as_slice()) };
        DotProduct::from(dotvbyte_view.dot_product(&self.dense_query))
    }
}

impl SpaceUsage for DotVByteFixedU8Quantizer {
    #[inline]
    fn space_usage_byte(&self) -> usize {
        let size_of_mapping = match &self.component_mapping {
            Some(component_mapping) => component_mapping.space_usage_byte(),
            None => 0,
        };
        size_of_mapping + std::mem::size_of::<Self>()
    }
}
