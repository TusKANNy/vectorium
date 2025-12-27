use serde::{Deserialize, Serialize};

use crate::PackedQuantizer;
use crate::SpaceUsage;
use crate::packed_vector::PackedEncoded;
use crate::utils::prefetch_read_slice;
use crate::{Dataset, VectorId};

use rayon::prelude::*;

/// A growable packed dataset.
pub type PackedDatasetGrowable<E> =
    PackedDatasetGeneric<E, Vec<usize>, Vec<<E as PackedQuantizer>::EncodingType>>;

/// An immutable packed dataset.
pub type PackedDataset<E> =
    PackedDatasetGeneric<E, Box<[usize]>, Box<[<E as PackedQuantizer>::EncodingType]>>;

/// Dataset storing variable-length packed encodings in a single concatenated `data` array.
///
/// Vector boundaries are stored in `offsets`, exactly like in `SparseDataset`:
/// - `offsets.len() == len() + 1`
/// - `offsets[0] == 0`
/// - vector `i` lives in `data[offsets[i]..offsets[i+1]]`
/// Packed dataset storing variable-length encodings with offsets.
///
/// # Example
/// ```
/// use vectorium::{
///     Dataset, DotProduct, DotVByteFixedU8Quantizer, GrowableDataset, PackedDataset,
///     PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVector1D, VectorEncoder,
/// };
///
/// let quantizer = PlainSparseQuantizer::<u16, f32, DotProduct>::new(5, 5);
/// let mut sparse = PlainSparseDatasetGrowable::new(quantizer);
/// sparse.push(SparseVector1D::new(vec![1_u16, 3], vec![1.0, 2.0]));
///
/// let frozen: vectorium::PlainSparseDataset<u16, f32, DotProduct> = sparse.into();
/// let packed: PackedDataset<DotVByteFixedU8Quantizer> = frozen.into();
/// let range = packed.range_from_id(0);
/// let v = packed.get_by_range(range);
/// assert!(!v.as_slice().is_empty());
/// ```
#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct PackedDatasetGeneric<E, Offsets, Data>
where
    E: PackedQuantizer,
    Offsets: AsRef<[usize]>,
    Data: AsRef<[E::EncodingType]>,
    for<'a> E::EncodedVector<'a>: PackedEncoded<'a, E::EncodingType>,
{
    offsets: Offsets,
    data: Data,
    quantizer: E,
    nnz: usize,
}

impl<E, Offsets, Data> PackedDatasetGeneric<E, Offsets, Data>
where
    E: PackedQuantizer,
    Offsets: AsRef<[usize]>,
    Data: AsRef<[E::EncodingType]>,
    for<'a> E::EncodedVector<'a>: PackedEncoded<'a, E::EncodingType>,
{
    #[inline]
    pub fn offsets(&self) -> &[usize] {
        self.offsets.as_ref()
    }

    #[inline]
    pub fn data(&self) -> &[E::EncodingType] {
        self.data.as_ref()
    }

    /// Parallel iterator over dataset encoded vectors.
    ///
    /// Each item is a `E::EncodedVector<'_>` borrowing its slice from the dataset `data`.
    #[inline]
    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = E::EncodedVector<'_>> + '_ {
        let offsets = self.offsets.as_ref();
        let data = self.data.as_ref();

        // https://github.com/rayon-rs/rayon/pull/789
        offsets.par_windows(2).map(move |window| {
            let start = window[0];
            let end = window[1];
            E::EncodedVector::from_slice(&data[start..end])
        })
    }
}

// impl<E> PackedDatasetGeneric<E, Vec<usize>, Vec<E::EncodingType>>
// where
//     E: PackedQuantizer,
//     for<'a> E::EncodedVector<'a>: PackedEncoded<'a, E::EncodingType>,
// {
//     #[inline]
//     pub fn new_growable(quantizer: E) -> Self {
//         Self {
//             offsets: vec![0],
//             data: Vec::new(),
//             quantizer,
//             nnz: 0,
//         }
//     }

//     /// Push an already-packed representation.
//     #[inline]
//     pub fn push_encoded(&mut self, encoded: impl AsRef<[E::EncodingType]>) {
//         let encoded = encoded.as_ref();
//         self.data.extend_from_slice(encoded);
//         self.offsets.push(self.data.len());
//         self.nnz += XXX; (how to compute nnz?)
//     }
// }

impl<E, Offsets, Data> SpaceUsage for PackedDatasetGeneric<E, Offsets, Data>
where
    E: PackedQuantizer + SpaceUsage,
    Offsets: AsRef<[usize]> + SpaceUsage,
    Data: AsRef<[E::EncodingType]> + SpaceUsage,
    for<'a> E::EncodedVector<'a>: PackedEncoded<'a, E::EncodingType>,
{
    fn space_usage_byte(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.quantizer.space_usage_byte()
            + self.offsets.space_usage_byte()
            + self.data.space_usage_byte()
    }
}

impl<E, Offsets, Data> Dataset<E> for PackedDatasetGeneric<E, Offsets, Data>
where
    E: PackedQuantizer + SpaceUsage,
    Offsets: AsRef<[usize]> + SpaceUsage,
    Data: AsRef<[E::EncodingType]> + SpaceUsage,
    E::EncodingType: SpaceUsage,
    for<'a> E::EncodedVector<'a>: PackedEncoded<'a, E::EncodingType>,
{
    #[inline]
    fn quantizer(&self) -> &E {
        &self.quantizer
    }

    #[inline]
    fn len(&self) -> usize {
        self.offsets.as_ref().len().saturating_sub(1)
    }

    #[inline]
    fn nnz(&self) -> usize {
        self.nnz
    }

    #[inline]
    fn range_from_id(&self, id: VectorId) -> std::ops::Range<usize> {
        let index = id as usize;
        let offsets = self.offsets.as_ref();
        assert!(index + 1 < offsets.len(), "Index out of bounds.");
        offsets[index]..offsets[index + 1]
    }

    #[inline]
    fn id_from_range(&self, range: std::ops::Range<usize>) -> VectorId {
        let offsets = self.offsets.as_ref();
        let idx = offsets.binary_search(&range.start).unwrap();
        assert_eq!(
            offsets[idx + 1],
            range.end,
            "Range does not match vector boundaries."
        );
        idx as VectorId
    }

    #[inline]
    fn get_by_range<'a>(&'a self, range: std::ops::Range<usize>) -> E::EncodedVector<'a> {
        let slice = &self.data.as_ref()[range];
        E::EncodedVector::from_slice(slice)
    }

    #[inline]
    fn prefetch(&self, range: std::ops::Range<usize>) {
        prefetch_read_slice(&self.data.as_ref()[range]);
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = E::EncodedVector<'a>> {
        let offsets = self.offsets.as_ref();
        let data = self.data.as_ref();
        offsets
            .windows(2)
            .map(move |w| E::EncodedVector::from_slice(&data[w[0]..w[1]]))
    }
}

impl<E> From<PackedDatasetGrowable<E>> for PackedDataset<E>
where
    E: PackedQuantizer,
    for<'a> E::EncodedVector<'a>: PackedEncoded<'a, E::EncodingType>,
{
    fn from(dataset: PackedDatasetGrowable<E>) -> Self {
        PackedDatasetGeneric {
            offsets: dataset.offsets.into_boxed_slice(),
            data: dataset.data.into_boxed_slice(),
            quantizer: dataset.quantizer,
            nnz: dataset.nnz,
        }
    }
}

impl<E> From<PackedDataset<E>> for PackedDatasetGrowable<E>
where
    E: PackedQuantizer,
    for<'a> E::EncodedVector<'a>: PackedEncoded<'a, E::EncodingType>,
{
    fn from(dataset: PackedDataset<E>) -> Self {
        PackedDatasetGeneric {
            offsets: dataset.offsets.to_vec(),
            data: dataset.data.to_vec(),
            quantizer: dataset.quantizer,
            nnz: dataset.nnz,
        }
    }
}

impl<QIn> From<crate::datasets::sparse_dataset::SparseDataset<QIn>>
    for PackedDataset<crate::DotVByteFixedU8Quantizer>
where
    QIn: crate::SparseQuantizer<OutputComponentType = u16> + SpaceUsage,
    <QIn as crate::VectorEncoder>::OutputValueType: crate::ValueType + crate::Float,
    for<'a> QIn: crate::VectorEncoder<
            EncodedVector<'a> = crate::SparseVector1D<
                u16,
                <QIn as crate::VectorEncoder>::OutputValueType,
                &'a [u16],
                &'a [<QIn as crate::VectorEncoder>::OutputValueType],
            >,
        >,
{
    fn from(dataset: crate::datasets::sparse_dataset::SparseDataset<QIn>) -> Self {
        use crate::SparseQuantizer;
        use crate::VectorEncoder;
        use crate::encoders::sparse_scalar::ScalarSparseQuantizer;
        use crate::{DotProduct, FixedU8Q};

        let dim = dataset.output_dim();
        let mut dotvbyte_quantizer =
            <crate::DotVByteFixedU8Quantizer as VectorEncoder>::new(dim, dim);

        // NOTE: `VectorEncoder::train` consumes input vectors. Training a DotVByte quantizer on a
        // sparse dataset requires a (potentially) two-pass conversion (or a
        // dedicated training iterator) and is left for later.
        dotvbyte_quantizer.train(dataset.iter());

        // Use a scalar quantizer to map values from `QIn::OutputValueType` into `FixedU8Q`.
        let scalar =
            <ScalarSparseQuantizer<u16, QIn::OutputValueType, FixedU8Q, DotProduct> as VectorEncoder>::new(
                dim, dim,
            );

        let mut offsets: Vec<usize> = Vec::with_capacity(dataset.len() + 1);
        offsets.push(0);
        let mut data = Vec::with_capacity(dataset.nnz() / 3); // overallocate, estimated 21 bits per entry

        for v in dataset.iter() {
            let v_fixedu8 = scalar.quantize_vector(v); // convert to FixedU8Q representation

            dotvbyte_quantizer.extend_with_encode(v_fixedu8, &mut data);

            offsets.push(data.len());
        }

        Self {
            offsets: offsets.into_boxed_slice(),
            data: data.into_boxed_slice(),
            quantizer: dotvbyte_quantizer,
            nnz: dataset.nnz(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversion_and_dot_product() {
        use crate::GrowableDataset;
        use crate::QueryEvaluator as _;
        use crate::VectorEncoder as _;
        use crate::distances::Distance as _;
        use crate::{
            DotProduct, DotVByteFixedU8Quantizer, FixedU8Q, FromF32 as _, PlainSparseDataset,
            PlainSparseDatasetGrowable, SparseVector1D,
        };
        use num_traits::ToPrimitive as _;

        let dim = 505;

        let mut growable: PlainSparseDatasetGrowable<u16, f32, DotProduct> =
            PlainSparseDatasetGrowable::new(
                <crate::PlainSparseQuantizer<u16, f32, DotProduct> as crate::VectorEncoder>::new(
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
