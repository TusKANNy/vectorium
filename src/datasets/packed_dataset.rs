use serde::{Deserialize, Serialize};

use crate::SpaceUsage;
use crate::datasets::{Dataset, VectorId, VectorKey};
use crate::packed_vector::PackedEncoded;
use crate::quantizers::PackedQuantizer;
use crate::utils::prefetch_read_slice;

use rayon::prelude::*;

/// A growable packed dataset.
pub type PackedDatasetGrowable<Q> =
    PackedDatasetGeneric<Q, Vec<usize>, Vec<<Q as PackedQuantizer>::EncodingType>>;

/// An immutable packed dataset.
pub type PackedDataset<Q> =
    PackedDatasetGeneric<Q, Box<[usize]>, Box<[<Q as PackedQuantizer>::EncodingType]>>;

/// Dataset storing variable-length packed encodings in a single concatenated `data` array.
///
/// Vector boundaries are stored in `offsets`, exactly like in `SparseDataset`:
/// - `offsets.len() == len() + 1`
/// - `offsets[0] == 0`
/// - vector `i` lives in `data[offsets[i]..offsets[i+1]]`
#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct PackedDatasetGeneric<Q, Offsets, Data>
where
    Q: PackedQuantizer,
    Offsets: AsRef<[usize]>,
    Data: AsRef<[Q::EncodingType]>,
    for<'a> Q::EncodedVector<'a>: PackedEncoded<'a, Q::EncodingType>,
{
    offsets: Offsets,
    data: Data,
    quantizer: Q,
}

impl<Q, Offsets, Data> PackedDatasetGeneric<Q, Offsets, Data>
where
    Q: PackedQuantizer,
    Offsets: AsRef<[usize]>,
    Data: AsRef<[Q::EncodingType]>,
    for<'a> Q::EncodedVector<'a>: PackedEncoded<'a, Q::EncodingType>,
{
    #[inline]
    pub fn from_raw(offsets: Offsets, data: Data, quantizer: Q) -> Self {
        let offsets_ref = offsets.as_ref();
        assert!(!offsets_ref.is_empty(), "offsets must contain at least [0]");
        assert_eq!(offsets_ref[0], 0, "offsets[0] must be 0");
        assert!(
            offsets_ref.windows(2).all(|w| w[0] <= w[1]),
            "offsets must be non-decreasing"
        );
        let total = *offsets_ref.last().unwrap();
        assert_eq!(
            total,
            data.as_ref().len(),
            "last offset must equal data length"
        );

        Self {
            offsets,
            data,
            quantizer,
        }
    }

    #[inline]
    pub fn offsets(&self) -> &[usize] {
        self.offsets.as_ref()
    }

    #[inline]
    pub fn data(&self) -> &[Q::EncodingType] {
        self.data.as_ref()
    }

    /// Parallel iterator over dataset encoded vectors.
    ///
    /// Each item is a `Q::EncodedVector<'_>` borrowing its slice from the dataset `data`.
    #[inline]
    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = Q::EncodedVector<'_>> + '_ {
        let offsets = self.offsets.as_ref();
        let data = self.data.as_ref();

        // https://github.com/rayon-rs/rayon/pull/789
        offsets.par_windows(2).map(move |window| {
            let start = window[0];
            let end = window[1];
            Q::EncodedVector::from_slice(&data[start..end])
        })
    }
}

impl<Q> PackedDatasetGeneric<Q, Vec<usize>, Vec<Q::EncodingType>>
where
    Q: PackedQuantizer,
    for<'a> Q::EncodedVector<'a>: PackedEncoded<'a, Q::EncodingType>,
{
    #[inline]
    pub fn new_growable(quantizer: Q) -> Self {
        Self {
            offsets: vec![0],
            data: Vec::new(),
            quantizer,
        }
    }

    /// Push an already-packed representation.
    #[inline]
    pub fn push_encoded(&mut self, encoded: impl AsRef<[Q::EncodingType]>) {
        let encoded = encoded.as_ref();
        self.data.extend_from_slice(encoded);
        self.offsets.push(self.data.len());
    }
}

impl<Q, Offsets, Data> SpaceUsage for PackedDatasetGeneric<Q, Offsets, Data>
where
    Q: PackedQuantizer + SpaceUsage,
    Offsets: AsRef<[usize]> + SpaceUsage,
    Data: AsRef<[Q::EncodingType]> + SpaceUsage,
    for<'a> Q::EncodedVector<'a>: PackedEncoded<'a, Q::EncodingType>,
{
    fn space_usage_byte(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.quantizer.space_usage_byte()
            + self.offsets.space_usage_byte()
            + self.data.space_usage_byte()
    }
}

impl<Q, Offsets, Data> Dataset<Q> for PackedDatasetGeneric<Q, Offsets, Data>
where
    Q: PackedQuantizer + SpaceUsage,
    Offsets: AsRef<[usize]> + SpaceUsage,
    Data: AsRef<[Q::EncodingType]> + SpaceUsage,
    Q::EncodingType: SpaceUsage,
    for<'a> Q::EncodedVector<'a>: PackedEncoded<'a, Q::EncodingType>,
{
    #[inline]
    fn quantizer(&self) -> &Q {
        &self.quantizer
    }

    #[inline]
    fn len(&self) -> usize {
        self.offsets.as_ref().len().saturating_sub(1)
    }

    #[inline]
    fn nnz(&self) -> usize {
        self.data.as_ref().len()
    }

    #[inline]
    fn key_from_id(&self, id: VectorId) -> VectorKey {
        id
    }

    #[inline]
    fn id_from_key(&self, key: VectorKey) -> VectorId {
        key
    }

    #[inline]
    fn get<'a>(&'a self, key: VectorKey) -> Q::EncodedVector<'a> {
        let idx = key as usize;
        let offsets = self.offsets.as_ref();
        assert!(idx + 1 < offsets.len(), "Index out of bounds.");

        let start = offsets[idx];
        let end = offsets[idx + 1];
        let slice = &self.data.as_ref()[start..end];
        Q::EncodedVector::from_slice(slice)
    }

    #[inline]
    fn prefetch(&self, key: VectorKey) {
        let idx = key as usize;
        let offsets = self.offsets.as_ref();
        assert!(idx + 1 < offsets.len(), "Index out of bounds.");

        let start = offsets[idx];
        let end = offsets[idx + 1];
        prefetch_read_slice(&self.data.as_ref()[start..end]);
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = Q::EncodedVector<'a>> {
        let offsets = self.offsets.as_ref();
        let data = self.data.as_ref();
        offsets
            .windows(2)
            .map(move |w| Q::EncodedVector::from_slice(&data[w[0]..w[1]]))
    }
}

impl<Q> From<PackedDatasetGrowable<Q>> for PackedDataset<Q>
where
    Q: PackedQuantizer,
    for<'a> Q::EncodedVector<'a>: PackedEncoded<'a, Q::EncodingType>,
{
    fn from(dataset: PackedDatasetGrowable<Q>) -> Self {
        PackedDatasetGeneric {
            offsets: dataset.offsets.into_boxed_slice(),
            data: dataset.data.into_boxed_slice(),
            quantizer: dataset.quantizer,
        }
    }
}

impl<Q> From<PackedDataset<Q>> for PackedDatasetGrowable<Q>
where
    Q: PackedQuantizer,
    for<'a> Q::EncodedVector<'a>: PackedEncoded<'a, Q::EncodingType>,
{
    fn from(dataset: PackedDataset<Q>) -> Self {
        PackedDatasetGeneric {
            offsets: dataset.offsets.to_vec(),
            data: dataset.data.to_vec(),
            quantizer: dataset.quantizer,
        }
    }
}

#[cfg(feature = "dotvbyte")]
use crate::datasets::sparse_dataset::sparse_dot_vbyte_dataset::dot_vbyte_fixedu8::StreamVbyteFixedu8;

#[cfg(feature = "dotvbyte")]
impl<QIn> From<crate::datasets::sparse_dataset::SparseDataset<QIn>>
    for PackedDataset<crate::quantizers::dotvbyte_fixedu8::DotVByteFixedU8Quantizer>
where
    QIn: crate::quantizers::SparseQuantizer<OutputComponentType = u16> + SpaceUsage,
    <QIn as crate::quantizers::Quantizer>::OutputValueType: crate::ValueType + crate::Float,
    for<'a> QIn: crate::quantizers::Quantizer<
            EncodedVector<'a> = crate::SparseVector1D<
                u16,
                <QIn as crate::quantizers::Quantizer>::OutputValueType,
                &'a [u16],
                &'a [<QIn as crate::quantizers::Quantizer>::OutputValueType],
            >,
        >,
{
    fn from(dataset: crate::datasets::sparse_dataset::SparseDataset<QIn>) -> Self {
        use crate::quantizers::Quantizer;
        use crate::quantizers::SparseQuantizer;
        use crate::quantizers::sparse_scalar::ScalarSparseQuantizer;
        use crate::{DotProduct, FixedU8Q, SparseVector1D, Vector1D};
        use rusty_perm::PermApply as _;
        use rusty_perm::PermFromSorting as _;

        let dim = dataset.output_dim();
        let mut dotvbyte_quantizer =
            <crate::quantizers::dotvbyte_fixedu8::DotVByteFixedU8Quantizer as Quantizer>::new(
                dim, dim,
            );

        // NOTE: `Quantizer::train` consumes `Self::EncodedVector` items. Training a DotVByte
        // quantizer on a sparse dataset requires a (potentially) two-pass conversion (or a
        // dedicated training iterator) and is left for later.
        // TODO! dotvbyte_quantizer.train(std::iter::empty());

        // Use a scalar quantizer to map values from `QIn::OutputValueType` into `FixedU8Q`.
        let scalar =
            <ScalarSparseQuantizer<u16, QIn::OutputValueType, FixedU8Q, DotProduct> as Quantizer>::new(
                dim, dim,
            );

        let mut offsets: Vec<usize> = Vec::with_capacity(dataset.len() + 1);
        offsets.push(0);
        let mut encoded_u8: Vec<u8> = Vec::new();

        // TODO: we have two datasets in parallel here. We could convert in place, but this is not easy with current API.
        for v in dataset.iter() {
            let components: &[u16] = v.components_as_slice();
            let values_in: &[QIn::OutputValueType] = v.values_as_slice();

            // Step 1: quantize sparse values to FixedU8Q (same components).
            let mut q_components: Vec<u16> = Vec::with_capacity(components.len());
            let mut q_values: Vec<FixedU8Q> = Vec::with_capacity(values_in.len());
            scalar.extend_with_encode(
                SparseVector1D::new(components, values_in),
                &mut q_components,
                &mut q_values,
            );

            // XXXXX_TODO: consider doing this directly in push_posting to avoid allocations
            let mut q_values: Vec<_> = q_values.into_iter().map(|v| v.to_bits()).collect(); // comvert to u8 representation

            // Step 2: If needed, remap components according to DotVByte quantizer mapping.
            let mut q_components =
                if let Some(component_mapping) = &dotvbyte_quantizer.component_mapping() {
                    q_components
                        .iter()
                        .map(|&c| component_mapping[c as usize])
                        .collect::<Vec<u16>>()
                } else {
                    q_components
                };

            //XXXXX_TODO: the permutation is applied inside push_posting now. move this code?
            // if dotvbyte_quantizer.component_mapping().is_some() {
            //     let permutation = rusty_perm::PermD::from_sort(q_components.as_slice());
            //     permutation.apply(q_values.as_mut_slice()).unwrap();
            //     permutation.apply(q_components.as_mut_slice()).unwrap();
            // }

            // Step 3: pack into DotVByteFixedU8 representation.
            // dotvbyte_quantizer
            //     .extend_with_encode(SparseVector1D::new(&q_components, &q_values), &mut data);

            StreamVbyteFixedu8::push_posting(&mut encoded_u8, &mut q_components, &mut q_values);
            assert!(
                encoded_u8.len() % std::mem::size_of::<u64>() == 0,
                "Encoded data length must be multiple of 8 bytes."
            );

            offsets.push(encoded_u8.len() / std::mem::size_of::<u64>());
        }

        // We have to reallocate to guarantee alignment, Rust is currently really bad at these kinds of things.
        assert!(
            encoded_u8.len() % std::mem::size_of::<u64>() == 0,
            "Encoded data length must be multiple of 8 bytes."
        );
        let mut data = Vec::with_capacity(encoded_u8.len() / std::mem::size_of::<u64>());

        unsafe {
            std::ptr::copy_nonoverlapping(
                encoded_u8.as_mut_ptr(),
                data.spare_capacity_mut().as_mut_ptr() as *mut u8,
                encoded_u8.len(),
            );
            data.set_len(data.capacity());
        };

        Self {
            offsets: offsets.into_boxed_slice(),
            data: data.into_boxed_slice(),
            quantizer: dotvbyte_quantizer,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_dataset_basic() {
        #[derive(Clone, Debug)]
        struct TestPackedQuantizer;

        impl crate::quantizers::PackedQuantizer for TestPackedQuantizer {
            type EncodingType = u64;
        }

        impl crate::quantizers::Quantizer for TestPackedQuantizer {
            type Distance = crate::DotProduct;
            type QueryValueType = f32;
            type QueryComponentType = crate::num_marker::DenseComponent;
            type InputValueType = f32;
            type InputComponentType = crate::num_marker::DenseComponent;
            type OutputValueType = f32;
            type OutputComponentType = crate::num_marker::DenseComponent;
            type Evaluator<'a>
                = DummyEvaluator
            where
                Self: 'a;
            type EncodedVector<'a> = crate::PackedVector<u64, &'a [u64]>;

            fn new(_input_dim: usize, _output_dim: usize) -> Self {
                Self
            }

            fn get_query_evaluator<'a, QueryVector>(
                &'a self,
                _query: &'a QueryVector,
            ) -> Self::Evaluator<'a>
            where
                QueryVector: crate::quantizers::QueryVectorFor<Self> + ?Sized,
            {
                DummyEvaluator
            }

            fn output_dim(&self) -> usize {
                0
            }

            fn input_dim(&self) -> usize {
                0
            }
        }

        impl crate::SpaceUsage for TestPackedQuantizer {
            fn space_usage_byte(&self) -> usize {
                std::mem::size_of::<Self>()
            }
        }

        struct DummyEvaluator;

        impl crate::quantizers::QueryEvaluator<TestPackedQuantizer> for DummyEvaluator {
            fn compute_distance(
                &self,
                _vector: <TestPackedQuantizer as crate::quantizers::Quantizer>::EncodedVector<'_>,
            ) -> crate::DotProduct {
                0.0f32.into()
            }
        }

        let quantizer = <TestPackedQuantizer as crate::quantizers::Quantizer>::new(0, 0);
        let dataset =
            PackedDatasetGeneric::from_raw(vec![0_usize, 3, 5], vec![1_u64, 2, 3, 4, 5], quantizer);
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.get(0).as_slice(), &[1, 2, 3]);
        assert_eq!(dataset.get(1).as_slice(), &[4, 5]);
    }

    #[cfg(feature = "dotvbyte")]
    #[test]
    fn conversion_and_dot_product() {
        use crate::datasets::GrowableDataset;
        use crate::distances::Distance as _;
        use crate::quantizers::Quantizer as _;
        use crate::quantizers::QueryEvaluator as _;
        use crate::{
            DotProduct, DotVByteFixedU8Quantizer, FixedU8Q, FromF32 as _, PlainSparseDataset,
            PlainSparseDatasetGrowable, SparseVector1D,
        };
        use num_traits::ToPrimitive as _;

        let dim = 505;

        let mut growable: PlainSparseDatasetGrowable<u16, f32, DotProduct> =
            PlainSparseDatasetGrowable::new(
                <crate::PlainSparseQuantizer<u16, f32, DotProduct> as crate::quantizers::Quantizer>::new(
                    dim, dim,
                ),
            );

        let v0_components = vec![1_u16, 10, 100];
        let v0_values = vec![1.5_f32, 2.0, 2.5];

        growable.push(SparseVector1D::new(&v0_components, &v0_values));

        let v1_components = vec![2_u16, 11];
        let v1_values = vec![0.5_f32, 1.0];

        growable.push(SparseVector1D::new(&v1_components, &v1_values));

        let frozen: PlainSparseDataset<u16, f32, DotProduct> = growable.into();

        let dataset: PackedDataset<DotVByteFixedU8Quantizer> = frozen.into();

        let query = SparseVector1D::new(vec![1_u16, 10, 11], vec![2.0_f32, 3.0, 4.0]);
        let evaluator = dataset.quantizer().get_query_evaluator(&query);

        let d0 = evaluator.compute_distance(dataset.get(0)).distance();
        let d1 = evaluator.compute_distance(dataset.get(1)).distance();

        let expected0 = FixedU8Q::from_f32_saturating(1.5).to_f32().unwrap() * 2.0
            + FixedU8Q::from_f32_saturating(2.0).to_f32().unwrap() * 3.0;
        let expected1 = FixedU8Q::from_f32_saturating(1.0).to_f32().unwrap() * 4.0;

        assert_eq!(d0, expected0);
        assert_eq!(d1, expected1);
    }
}
