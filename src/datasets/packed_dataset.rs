use serde::{Deserialize, Serialize};

use crate::datasets::{Dataset, VectorId, VectorKey};
use crate::packed_vector::PackedEncoded;
use crate::quantizers::PackedQuantizer;
use crate::utils::prefetch_read_slice;
use crate::SpaceUsage;

use rayon::prelude::*;

/// A growable packed dataset.
pub type PackedDatasetGrowable<Q> = PackedDatasetGeneric<Q, Vec<usize>, Vec<<Q as PackedQuantizer>::EncodingType>>;

/// An immutable packed dataset.
pub type PackedDataset<Q> = PackedDatasetGeneric<Q, Box<[usize]>, Box<[<Q as PackedQuantizer>::EncodingType]>>;

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
    pub fn par_iter(
        &self,
    ) -> impl IndexedParallelIterator<Item = Q::EncodedVector<'_>> + '_ {
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
        use crate::quantizers::sparse_scalar::ScalarSparseQuantizer;
        use crate::quantizers::Quantizer;
        use crate::quantizers::SparseQuantizer;
        use crate::{DotProduct, FixedU8Q, SparseVector1D, Vector1D};

        let dim = dataset.output_dim();
        let packed_quantizer =
            <crate::quantizers::dotvbyte_fixedu8::DotVByteFixedU8Quantizer as Quantizer>::new(
                dim, dim,
            );

        // Use a scalar quantizer to map values from `QIn::OutputValueType` into `FixedU8Q`.
        let scalar =
            <ScalarSparseQuantizer<u16, QIn::OutputValueType, FixedU8Q, DotProduct> as Quantizer>::new(
                dim, dim,
            );

        let mut offsets: Vec<usize> = Vec::with_capacity(dataset.len() + 1);
        offsets.push(0);
        let mut data: Vec<u64> = Vec::new();

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

            // Step 2: pack `q_components` and quantized values into DotVByte words (`u64`) and append to `data`.
            // Note: `PackedDataset` stores word-aligned `u64` data; the packed length can vary per vector.
            todo!("encode one posting into `data: Vec<u64>` and push new offset");
        }

        PackedDatasetGeneric::from_raw(
            offsets.into_boxed_slice(),
            data.into_boxed_slice(),
            packed_quantizer,
        )
    }
}
