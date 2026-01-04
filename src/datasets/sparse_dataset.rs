use serde::{Deserialize, Serialize};
use std::hint::assert_unchecked;

use crate::SpaceUsage;
use crate::core::storage::{GrowableSparseStorage, ImmutableSparseStorage, SparseStorage};
use crate::utils::{is_strictly_sorted, prefetch_read_slice};
use crate::{ComponentType, Float, FromF32, ValueType, VectorId};
use crate::core::dataset::ConvertFrom;
use num_traits::AsPrimitive;
use crate::{Dataset, GrowableDataset};
use crate::{SparseVectorEncoder, VectorEncoder};
use crate::{SparseVector1D, Vector1D};

use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSlice};

type SparseEncodedVector<'a, E> = SparseVector1D<
    <E as VectorEncoder>::OutputComponentType,
    <E as VectorEncoder>::OutputValueType,
    &'a [<E as VectorEncoder>::OutputComponentType],
    &'a [<E as VectorEncoder>::OutputValueType],
>;

/// A growable representation of a sparse dataset.
///
/// # Examples
///
/// ```rust
/// use vectorium::{Dataset, GrowableDataset};
/// use vectorium::{DotProduct, PlainSparseDatasetGrowable, PlainSparseQuantizer};
///
/// // Create a new empty dataset
/// let quantizer = <PlainSparseQuantizer<u32, f32, DotProduct> as vectorium::VectorEncoder>::new(0, 0);
/// let dataset = PlainSparseDatasetGrowable::new(quantizer);
/// assert_eq!(dataset.len(), 0);
/// assert_eq!(dataset.nnz(), 0);
/// ```
///
pub type SparseDatasetGrowable<E> = SparseDatasetGeneric<E, GrowableSparseStorage<E>>;

// Implementation of a (immutable) sparse dataset.
pub type SparseDataset<E> = SparseDatasetGeneric<E, ImmutableSparseStorage<E>>;

/// Sparse dataset storing variable-length vectors with offsets.
///
/// # Type Parameters
///
/// - `E`: The encoder/quantizer type (must implement `SparseVectorEncoder`)
/// - `S`: The storage backend (defaults to `GrowableSparseStorage<E>`)
///
/// # Example
/// ```
/// use vectorium::{
///     Dataset, DotProduct, GrowableDataset, PlainSparseDataset, PlainSparseQuantizer,
///     SparseDatasetGrowable, SparseVector1D, Vector1D, VectorEncoder,
/// };
///
/// let quantizer = PlainSparseQuantizer::<u16, f32, DotProduct>::new(5, 5);
/// let mut dataset = SparseDatasetGrowable::new(quantizer);
/// dataset.push(SparseVector1D::new(vec![1_u16, 3], vec![1.0, 2.0]));
///
/// let frozen: PlainSparseDataset<u16, f32, DotProduct> = dataset.into();
/// let v = frozen.get(0);
/// assert_eq!(v.components_as_slice(), &[1_u16, 3]);
/// assert_eq!(v.values_as_slice(), &[1.0, 2.0]);
///
/// let range = frozen.range_from_id(0);
/// let v = frozen.get_by_range(range);
/// assert_eq!(v.components_as_slice(), &[1_u16, 3]);
/// assert_eq!(v.values_as_slice(), &[1.0, 2.0]);
/// ```
#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct SparseDatasetGeneric<E, S = GrowableSparseStorage<E>>
where
    E: SparseVectorEncoder,
    for<'a> E: VectorEncoder<EncodedVector<'a> = SparseEncodedVector<'a, E>>, // Ensure that the encoded vector is a sparse vector
    S: SparseStorage<E>,
{
    storage: S,
    quantizer: E,
}

impl<E, S> SparseDatasetGeneric<E, S>
where
    E: SparseVectorEncoder,
    for<'a> E: VectorEncoder<EncodedVector<'a> = SparseEncodedVector<'a, E>>,
    S: SparseStorage<E>,
{
    /// Parallel iterator over all vectors as slice-backed `SparseVector1D`.
    #[inline]
    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = SparseEncodedVector<'_, E>> + '_ {
        let offsets = self.storage.offsets().as_ref();
        let components = self.storage.components().as_ref();
        let values = self.storage.values().as_ref();

        // https://github.com/rayon-rs/rayon/pull/789
        offsets.par_windows(2).map(move |window| {
            let &[start, end] = window else {
                unsafe { std::hint::unreachable_unchecked() }
            };
            SparseVector1D::new(&components[start..end], &values[start..end])
        })
    }
}

impl<E, S> Dataset for SparseDatasetGeneric<E, S>
where
    E: SparseVectorEncoder,
    for<'a> E: VectorEncoder<EncodedVector<'a> = SparseEncodedVector<'a, E>>,
    S: SparseStorage<E>,
{
    type VectorEncoder = E;

    /// Retrieves the components and values of the sparse vector at the specified index.
    ///
    /// This method returns a tuple containing slices of components and values
    /// of the sparse vector located at the given index.
    ///
    /// # Panics
    ///
    /// Panics if the specified index is out of range.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectorium::{Dataset, GrowableDataset};
    /// use vectorium::Vector1D;
    /// use vectorium::{DotProduct, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVector1D};
    ///
    /// let quantizer = <PlainSparseQuantizer<u32, f32, DotProduct> as vectorium::VectorEncoder>::new(5, 5);
    /// let mut dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    ///
    /// dataset.push(SparseVector1D::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0]));
    /// dataset.push(SparseVector1D::new(vec![1, 3], vec![4.0, 5.0]));
    ///
    /// let v = dataset.get(1);
    /// assert_eq!(v.components_as_slice(), &[1, 3]);
    /// assert_eq!(v.values_as_slice(), &[4.0, 5.0]);
    /// ```
    #[inline]
    fn get_by_range<'a>(&'a self, range: std::ops::Range<usize>) -> E::EncodedVector<'a> {
        let components = self.storage.components().as_ref();
        let values = self.storage.values().as_ref();
        unsafe { assert_unchecked(components.len() == values.len()) };

        let v_components = &components[range.clone()];
        let v_values = &values[range];

        SparseVector1D::new(v_components, v_values)
    }

    // NOTE: `get_with_offset` was used by an older API but is not currently exposed.
    // #[inline]
    // pub fn get_with_offset(&self, offset: usize, len: usize) -> (&[C], &[V]) {
    //     unsafe { assert_unchecked(self.components.as_ref().len() == self.values.as_ref().len()) };

    //     let v_components = &self.components.as_ref()[offset..offset + len];
    //     let v_values = &self.values.as_ref()[offset..offset + len];

    //     (v_components, v_values)
    // }

    // Returns the range of positions of the slice with the given `id`.
    //
    // ### Panics
    // Panics if the `id` is out of range.
    // #[inline]
    // pub fn offset_range(&self, id: usize) -> Range<usize> {
    //     let offsets = self.offsets.as_ref();
    //     assert!(id < offsets.len() - 1, "{id} is out of range");

    //     // Safety: safe accesses due to the check above
    //     unsafe {
    //         Range {
    //             start: *offsets.get_unchecked(id),
    //             end: *offsets.get_unchecked(id + 1),
    //         }
    //     }
    // }

    /// Converts the `offset` of a vector within the dataset to its id, i.e., the position
    /// of the vector within the dataset.
    ///
    /// # Panics
    /// Panics if the `offset` is not the first position of a vector in the dataset.
    #[inline]
    fn id_from_range(&self, range: std::ops::Range<usize>) -> VectorId {
        let offsets = self.storage.offsets().as_ref();
        let idx = offsets.binary_search(&range.start).unwrap();
        assert_eq!(
            offsets[idx + 1],
            range.end,
            "Range does not match vector boundaries."
        );
        idx as VectorId
    }

    #[inline]
    fn range_from_id(&self, id: VectorId) -> std::ops::Range<usize> {
        let offsets = self.storage.offsets().as_ref();
        let index = id as usize;
        assert!(index + 1 < offsets.len(), "Index out of bounds.");
        offsets[index]..offsets[index + 1]
    }

    fn quantizer(&self) -> &E {
        &self.quantizer
    }

    /// Prefetches the components and values of an encoded vector into CPU cache.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectorium::{Dataset, GrowableDataset};
    /// use vectorium::Vector1D;
    /// use vectorium::{DotProduct, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVector1D};
    ///
    /// let quantizer = <PlainSparseQuantizer<u32, f32, DotProduct> as vectorium::VectorEncoder>::new(5, 5);
    /// let mut dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    /// dataset.push(SparseVector1D::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0]));
    ///
    /// dataset.prefetch(dataset.range_from_id(0));
    /// ```
    #[inline]
    fn prefetch(&self, range: std::ops::Range<usize>) {
        let sparse_vector = self.get_by_range(range);

        prefetch_read_slice(sparse_vector.components_as_slice());
        prefetch_read_slice(sparse_vector.values_as_slice());
    }

    /// Returns an iterator over the vectors of the dataset.
    ///
    /// This method returns an iterator that yields references to each vector in the dataset.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectorium::{Dataset, GrowableDataset};
    /// use vectorium::Vector1D;
    /// use vectorium::{DotProduct, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVector1D};
    ///
    /// let data = vec![
    ///     (vec![0_u32, 2, 4], vec![1.0, 2.0, 3.0]),
    ///     (vec![1_u32, 3], vec![4.0, 5.0]),
    ///     (vec![0_u32, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0]),
    /// ];
    ///
    /// let quantizer = <PlainSparseQuantizer<u32, f32, DotProduct> as vectorium::VectorEncoder>::new(5, 5);
    /// let mut dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    /// for (c, v) in data.iter() {
    ///     dataset.push(SparseVector1D::new(c.clone(), v.clone()));
    /// }
    ///
    /// for (vec, (c, v)) in dataset.iter().zip(data.iter()) {
    ///     assert_eq!(vec.components_as_slice(), c.as_slice());
    ///     assert_eq!(vec.values_as_slice(), v.as_slice());
    /// }
    /// ```
    fn iter<'a>(&'a self) -> impl Iterator<Item = E::EncodedVector<'a>> {
        SparseDatasetIter::new(self)
    }

    // /// Returns an iterator over the sparse vector with id `vec_id`.
    // ///
    // /// # Panics
    // /// Panics if the `vec_id` is out of bounds.
    // fn iter(
    //     &self,
    //     vec_id: usize,
    // ) -> Zip<std::slice::Iter<C>, std::slice::Iter<V>> {
    //     let (v_components, v_values) = self.get(vec_id);

    //     v_components.iter().zip(v_values.iter())
    // }

    /// Returns the number of vectors in the dataset.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectorium::{Dataset, GrowableDataset};
    /// use vectorium::{DotProduct, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVector1D};
    ///
    /// let quantizer = <PlainSparseQuantizer<u32, f32, DotProduct> as vectorium::VectorEncoder>::new(5, 5);
    /// let mut dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    ///
    /// dataset.push(SparseVector1D::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0]));
    /// dataset.push(SparseVector1D::new(vec![1, 3], vec![4.0, 5.0]));
    /// dataset.push(SparseVector1D::new(vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0]));
    ///
    /// assert_eq!(dataset.len(), 3);
    /// ```
    fn len(&self) -> usize {
        self.storage.offsets().as_ref().len() - 1
    }

    /// Returns the number of components of the dataset, i.e., it returns one plus the ID of the largest component.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectorium::{Dataset, GrowableDataset};
    /// use vectorium::{DotProduct, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVector1D};
    ///
    /// let quantizer = <PlainSparseQuantizer<u32, f32, DotProduct> as vectorium::VectorEncoder>::new(5, 5);
    /// let mut dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    ///
    /// dataset.push(SparseVector1D::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0]));
    /// assert_eq!(dataset.input_dim(), 5);
    /// ```
    ///
    /// Returns the number of non-zero components in the dataset.
    ///
    /// This function returns the total count of non-zero components across all vectors
    /// currently stored in the dataset.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectorium::{Dataset, GrowableDataset};
    /// use vectorium::{DotProduct, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVector1D};
    ///
    /// let quantizer = <PlainSparseQuantizer<u32, f32, DotProduct> as vectorium::VectorEncoder>::new(5, 5);
    /// let mut dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    ///
    /// dataset.push(SparseVector1D::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0]));
    /// dataset.push(SparseVector1D::new(vec![1, 3], vec![4.0, 5.0]));
    /// dataset.push(SparseVector1D::new(vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0]));
    ///
    /// assert_eq!(dataset.nnz(), 9);
    /// ```
    fn nnz(&self) -> usize {
        self.storage.components().as_ref().len()
    }

    // pub fn from_dataset_f32<D, P, AD, AU>(dataset: SparseDatasetGeneric<D, f32, P, AD, AU>) -> Self
    // where
    //     D: ComponentType,
    //     O: From<P>,
    //     P: AsRef<[usize]>,
    //     AC: From<AD>,
    //     AD: AsRef<[D]>,
    //     AV: From<Vec<V>>,
    //     AU: AsRef<[f32]>,
    // {
    //     Self {
    //         dim: dataset.dim,
    //         offsets: dataset.offsets.into(),
    //         components: dataset.components.into(),
    //         values: dataset
    //             .values
    //             .as_ref()
    //             .iter()
    //             .map(|&v| V::from_f32_saturating(v))
    //             .collect_vec()
    //             .into(),
    //         _phantom: PhantomData,
    //     }
    // }
}

// impl<E, AC, AV> FromIterator<(AC, AV)> for SparseDatasetGrowable<E>
// where
//     E: SparseVectorEncoder,
//     AC: AsRef<[E::OutputComponentType]>,
//     AV: AsRef<[E::OutputValueType]>,
// {
//     /// Constructs a `SparseDatasetGrowable<V>` from an iterator over pairs of references to `[u16]` and `[V]`.
//     ///
//     /// This function consumes the provided iterator and constructs a new `SparseDatasetGrowable<V>`.
//     /// Each pair in the iterator represents a pair of vectors, where the first vector contains
//     /// the components and the second vector contains their corresponding values.
//     ///
//     /// # Parameters
//     ///
//     /// * `iter`: An iterator over pairs of vectors `(AsRef<[C]>, AsRef<[V]>)`.
//     ///
//     /// # Returns
//     ///
//     /// A new instance of `SparseDatasetGrowable<C, V>` populated with the pairs from the iterator.
//     ///
//     /// # Examples
//     ///
//     /// ```
//     /// use seismic::SparseDatasetGrowable;
//     ///
//     /// let data = vec![
//     ///                 (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
//     ///                 (vec![1, 3],       vec![4.0, 5.0]),
//     ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
//     ///                 ];
//     ///
//     /// let dataset: SparseDatasetGrowable<u16, f32> = data.into_iter().collect();
//     ///
//     /// assert_eq!(dataset.nnz(), 9); // Total non-zero components across all vectors
//     /// ```
//     fn from_iter<I>(iter: I) -> Self
//     where
//         I: IntoIterator<Item = (AC, AV)>,
//     {
//         let mut dataset = SparseDatasetGrowable::<E>::new(E);

//         for (components, values) in iter {
//             dataset.push(components.as_ref(), values.as_ref());
//         }

//         dataset
//     }
// }

// Unfortunately, Rust doesn't yet support specialization, meaning that we can't use From too generically (otherwise it fails due to reimplementing `From<T> for T`)

impl<E> ConvertFrom<SparseDatasetGrowable<E>> for SparseDataset<E>
where
    E: SparseVectorEncoder,
    for<'a> E: VectorEncoder<EncodedVector<'a> = SparseEncodedVector<'a, E>>,
{
    /// Converts a mutable sparse dataset into an immutable one.
    ///
    /// This function consumes the provided `SparseDatasetGrowable<C, V>` and produces
    /// a corresponding immutable `SparseDataset<C, V>` instance. The conversion is performed
    /// by transferring ownership of the internal data structures from the mutable dataset
    /// to the immutable one.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectorium::{Dataset, GrowableDataset};
    /// use vectorium::{DotProduct, PlainSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVector1D};
    ///
    /// let quantizer = <PlainSparseQuantizer<u32, f32, DotProduct> as vectorium::VectorEncoder>::new(5, 5);
    /// let mut growable_dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    ///
    /// growable_dataset.push(SparseVector1D::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0]));
    /// growable_dataset.push(SparseVector1D::new(vec![1, 3], vec![4.0, 5.0]));
    /// growable_dataset.push(SparseVector1D::new(vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0]));
    ///
    /// let immutable_dataset: PlainSparseDataset<u32, f32, DotProduct> = growable_dataset.into();
    /// assert_eq!(immutable_dataset.nnz(), 9);
    /// ```
    fn convert_from(dataset: SparseDatasetGrowable<E>) -> Self {
        Self {
            storage: dataset.storage.into(),
            quantizer: dataset.quantizer,
        }
    }
}

impl<E> From<SparseDatasetGrowable<E>> for SparseDataset<E>
where
    E: SparseVectorEncoder,
    for<'a> E: VectorEncoder<EncodedVector<'a> = SparseEncodedVector<'a, E>>,
{
    fn from(dataset: SparseDatasetGrowable<E>) -> Self {
        Self::convert_from(dataset)
    }
}

impl<E> ConvertFrom<SparseDataset<E>> for SparseDatasetGrowable<E>
where
    E: SparseVectorEncoder,
    for<'a> E: VectorEncoder<EncodedVector<'a> = SparseEncodedVector<'a, E>>,
{
    /// Converts an immutable sparse dataset into a mutable one.
    ///
    /// This function consumes the provided `SparseDataset<C, V>` and produces
    /// a corresponding mutable `SparseDatasetGrowable<C, V>` instance. The conversion is performed
    /// by transferring ownership of the internal data structures from the immutable dataset
    /// to the mutable one.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectorium::{Dataset, GrowableDataset};
    /// use vectorium::{DotProduct, PlainSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVector1D};
    ///
    /// let quantizer = <PlainSparseQuantizer<u32, f32, DotProduct> as vectorium::VectorEncoder>::new(5, 5);
    /// let mut growable_dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    ///
    /// growable_dataset.push(SparseVector1D::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0]));
    /// growable_dataset.push(SparseVector1D::new(vec![1, 3], vec![4.0, 5.0]));
    ///
    /// let immutable_dataset: PlainSparseDataset<u32, f32, DotProduct> = growable_dataset.into();
    ///
    /// // Convert immutable dataset back to mutable
    /// let mut growable_dataset_again: PlainSparseDatasetGrowable<u32, f32, DotProduct> = immutable_dataset.into();
    /// growable_dataset_again.push(SparseVector1D::new(vec![1, 4], vec![1.0, 3.0]));
    ///
    /// assert_eq!(growable_dataset_again.nnz(), 7);
    /// ```
    fn convert_from(dataset: SparseDataset<E>) -> Self {
        Self {
            storage: dataset.storage.into(),
            quantizer: dataset.quantizer,
        }
    }
}

impl<E> From<SparseDataset<E>> for SparseDatasetGrowable<E>
where
    E: SparseVectorEncoder,
    for<'a> E: VectorEncoder<EncodedVector<'a> = SparseEncodedVector<'a, E>>,
{
    fn from(dataset: SparseDataset<E>) -> Self {
        Self::convert_from(dataset)
    }
}

impl<C, InValue, OutValue, D>
    SparseDatasetGeneric<
        crate::ScalarSparseQuantizer<C, InValue, OutValue, D>,
        GrowableSparseStorage<crate::ScalarSparseQuantizer<C, InValue, OutValue, D>>,
    >
where
    C: ComponentType,
    InValue: ValueType + Float,
    OutValue: ValueType + Float + FromF32,
    D: crate::ScalarSparseSupportedDistance,
{
    /// Relabels a scalar-quantized growable dataset as plain without re-encoding.
    pub fn relabel_as_plain(
        self,
    ) -> SparseDatasetGeneric<
        crate::PlainSparseQuantizer<C, OutValue, D>,
        GrowableSparseStorage<crate::PlainSparseQuantizer<C, OutValue, D>>,
    > {
        let dim = self.quantizer.output_dim();
        let storage = self
            .storage
            .relabel::<crate::PlainSparseQuantizer<C, OutValue, D>>();

        SparseDatasetGeneric {
            storage,
            quantizer: crate::PlainSparseQuantizer::<C, OutValue, D>::new(dim, dim),
        }
    }
}

impl<C, OutValue, D>
    SparseDatasetGeneric<
        crate::PlainSparseQuantizer<C, OutValue, D>,
        GrowableSparseStorage<crate::PlainSparseQuantizer<C, OutValue, D>>,
    >
where
    C: ComponentType,
    OutValue: ValueType + Float + FromF32,
    D: crate::ScalarSparseSupportedDistance,
{
    /// Relabels a plain-quantized growable dataset as scalar without re-encoding.
    pub fn relabel_as_scalar<InValue>(
        self,
    ) -> SparseDatasetGeneric<
        crate::ScalarSparseQuantizer<C, InValue, OutValue, D>,
        GrowableSparseStorage<crate::ScalarSparseQuantizer<C, InValue, OutValue, D>>,
    >
    where
        InValue: ValueType + Float,
    {
        let dim = self.quantizer.output_dim();
        let storage = self
            .storage
            .relabel::<crate::ScalarSparseQuantizer<C, InValue, OutValue, D>>();

        SparseDatasetGeneric {
            storage,
            quantizer: crate::ScalarSparseQuantizer::<C, InValue, OutValue, D>::new(dim, dim),
        }
    }
}

impl<C, InValue, OutValue, D>
    SparseDatasetGeneric<
        crate::ScalarSparseQuantizer<C, InValue, OutValue, D>,
        ImmutableSparseStorage<crate::ScalarSparseQuantizer<C, InValue, OutValue, D>>,
    >
where
    C: ComponentType,
    InValue: ValueType + Float,
    OutValue: ValueType + Float + FromF32,
    D: crate::ScalarSparseSupportedDistance,
{
    /// Relabels a scalar-quantized immutable dataset as plain without re-encoding.
    pub fn relabel_as_plain(
        self,
    ) -> SparseDatasetGeneric<
        crate::PlainSparseQuantizer<C, OutValue, D>,
        ImmutableSparseStorage<crate::PlainSparseQuantizer<C, OutValue, D>>,
    > {
        let dim = self.quantizer.output_dim();
        let storage = self
            .storage
            .relabel::<crate::PlainSparseQuantizer<C, OutValue, D>>();

        SparseDatasetGeneric {
            storage,
            quantizer: crate::PlainSparseQuantizer::<C, OutValue, D>::new(dim, dim),
        }
    }
}

impl<C, OutValue, D>
    SparseDatasetGeneric<
        crate::PlainSparseQuantizer<C, OutValue, D>,
        ImmutableSparseStorage<crate::PlainSparseQuantizer<C, OutValue, D>>,
    >
where
    C: ComponentType,
    OutValue: ValueType + Float + FromF32,
    D: crate::ScalarSparseSupportedDistance,
{
    /// Relabels a plain-quantized immutable dataset as scalar without re-encoding.
    pub fn relabel_as_scalar<InValue>(
        self,
    ) -> SparseDatasetGeneric<
        crate::ScalarSparseQuantizer<C, InValue, OutValue, D>,
        ImmutableSparseStorage<crate::ScalarSparseQuantizer<C, InValue, OutValue, D>>,
    >
    where
        InValue: ValueType + Float,
    {
        let dim = self.quantizer.output_dim();
        let storage = self
            .storage
            .relabel::<crate::ScalarSparseQuantizer<C, InValue, OutValue, D>>();

        SparseDatasetGeneric {
            storage,
            quantizer: crate::ScalarSparseQuantizer::<C, InValue, OutValue, D>::new(dim, dim),
        }
    }
}

impl<C, SrcIn, Mid, DstOut, D, SrcStorage, DstStorage>
    ConvertFrom<
        &SparseDatasetGeneric<
            crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>,
            SrcStorage,
        >,
    >
    for SparseDatasetGeneric<
        crate::ScalarSparseQuantizer<C, Mid, DstOut, D>,
        DstStorage,
    >
where
    C: ComponentType,
    SrcIn: ValueType + Float,
    Mid: ValueType + Float + FromF32,
    DstOut: ValueType + Float + FromF32,
    D: crate::ScalarSparseSupportedDistance,
    crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>: SparseVectorEncoder<
        InputComponentType = C,
        InputValueType = SrcIn,
        OutputComponentType = C,
        OutputValueType = Mid,
    >,
    crate::ScalarSparseQuantizer<C, Mid, DstOut, D>: SparseVectorEncoder<
        InputComponentType = C,
        InputValueType = Mid,
        OutputComponentType = C,
        OutputValueType = DstOut,
    >,
    SrcStorage: SparseStorage<crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>>,
    for<'a> crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>: VectorEncoder<
        EncodedVector<'a> =
            SparseEncodedVector<'a, crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>>,
    >,
    for<'a> crate::ScalarSparseQuantizer<C, Mid, DstOut, D>: VectorEncoder<
        EncodedVector<'a> =
            SparseEncodedVector<'a, crate::ScalarSparseQuantizer<C, Mid, DstOut, D>>,
    >,
    DstStorage: SparseStorage<crate::ScalarSparseQuantizer<C, Mid, DstOut, D>>
        + From<GrowableSparseStorage<crate::ScalarSparseQuantizer<C, Mid, DstOut, D>>>,
{
    fn convert_from(
        source: &SparseDatasetGeneric<
            crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>,
            SrcStorage,
        >,
    ) -> Self {
        let n_vecs = source.len();
        let nnz = source.nnz();
        let dim = source.quantizer.output_dim();
        let quantizer = crate::ScalarSparseQuantizer::<C, Mid, DstOut, D>::new(dim, dim);

        let mut storage =
            GrowableSparseStorage::<crate::ScalarSparseQuantizer<C, Mid, DstOut, D>>::with_capacity(
                n_vecs, nnz,
            );

        for src_vec in source.iter() {
            quantizer.extend_with_encode(
                SparseVector1D::<C, Mid, _, _>::new(
                    src_vec.components_as_slice(),
                    src_vec.values_as_slice(),
                ),
                &mut storage.components,
                &mut storage.values,
            );
            storage.offsets.push(storage.components.len());
        }

        SparseDatasetGeneric {
            storage: storage.into(),
            quantizer,
        }
    }
}

impl<C, SrcIn, Mid, DstOut, D>
    ConvertFrom<SparseDataset<crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>>>
    for SparseDataset<crate::ScalarSparseQuantizer<C, Mid, DstOut, D>>
where
    C: ComponentType,
    SrcIn: ValueType + Float,
    Mid: ValueType + Float + FromF32,
    DstOut: ValueType + Float + FromF32,
    D: crate::ScalarSparseSupportedDistance,
    crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>: SparseVectorEncoder<
        InputComponentType = C,
        InputValueType = SrcIn,
        OutputComponentType = C,
        OutputValueType = Mid,
    >,
    crate::ScalarSparseQuantizer<C, Mid, DstOut, D>: SparseVectorEncoder<
        InputComponentType = C,
        InputValueType = Mid,
        OutputComponentType = C,
        OutputValueType = DstOut,
    >,
    for<'a> crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>: VectorEncoder<
        EncodedVector<'a> =
            SparseEncodedVector<'a, crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>>,
    >,
    for<'a> crate::ScalarSparseQuantizer<C, Mid, DstOut, D>: VectorEncoder<
        EncodedVector<'a> =
            SparseEncodedVector<'a, crate::ScalarSparseQuantizer<C, Mid, DstOut, D>>,
    >,
{
    fn convert_from(
        source: SparseDataset<crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>>,
    ) -> Self {
        Self::convert_from(&source)
    }
}

impl<C, SrcIn, Mid, DstOut, D>
    ConvertFrom<SparseDatasetGrowable<crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>>>
    for SparseDatasetGrowable<crate::ScalarSparseQuantizer<C, Mid, DstOut, D>>
where
    C: ComponentType,
    SrcIn: ValueType + Float,
    Mid: ValueType + Float + FromF32,
    DstOut: ValueType + Float + FromF32,
    D: crate::ScalarSparseSupportedDistance,
    crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>: SparseVectorEncoder<
        InputComponentType = C,
        InputValueType = SrcIn,
        OutputComponentType = C,
        OutputValueType = Mid,
    >,
    crate::ScalarSparseQuantizer<C, Mid, DstOut, D>: SparseVectorEncoder<
        InputComponentType = C,
        InputValueType = Mid,
        OutputComponentType = C,
        OutputValueType = DstOut,
    >,
    for<'a> crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>: VectorEncoder<
        EncodedVector<'a> =
            SparseEncodedVector<'a, crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>>,
    >,
    for<'a> crate::ScalarSparseQuantizer<C, Mid, DstOut, D>: VectorEncoder<
        EncodedVector<'a> =
            SparseEncodedVector<'a, crate::ScalarSparseQuantizer<C, Mid, DstOut, D>>,
    >,
{
    fn convert_from(
        source: SparseDatasetGrowable<crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>>,
    ) -> Self {
        Self::convert_from(&source)
    }
}

/// A struct to iterate over the *raw* sparse dataset storage as slice-backed `SparseVector1D`.
///
/// This iterator is independent from the quantizer’s `EncodedVector` choice and is
/// used by internal utilities (including the parallel iterator plumbing).
#[derive(Clone)]
pub struct SparseDatasetIter<'a, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    // Last offset consumed from the front; used to rebase remaining offsets to the current slices.
    front_base_offset: usize,
    offsets: &'a [usize],
    components: &'a [C],
    values: &'a [V],
}

impl<'a, C, V> SparseDatasetIter<'a, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    #[inline]
    pub fn new<E, S>(dataset: &'a SparseDatasetGeneric<E, S>) -> Self
    where
        E: SparseVectorEncoder<OutputComponentType = C, OutputValueType = V>,
        for<'b> E: VectorEncoder<EncodedVector<'b> = SparseEncodedVector<'b, E>>,
        S: SparseStorage<E>,
    {
        Self {
            front_base_offset: 0,
            offsets: &dataset.storage.offsets().as_ref()[1..],
            components: dataset.storage.components().as_ref(),
            values: dataset.storage.values().as_ref(),
        }
    }
}

impl<E> GrowableDataset for SparseDatasetGrowable<E>
where
    E: SparseVectorEncoder,
    for<'a> E: VectorEncoder<EncodedVector<'a> = SparseEncodedVector<'a, E>>,
{
    /// For SparseDataset, the dimensionality `d` may be 0 if unknown when creating a new dataset.
    fn new(quantizer: E) -> Self {
        Self {
            storage: GrowableSparseStorage::new(),
            quantizer,
        }
    }

    /// Adds a new sparse vector to the dataset.
    ///
    /// The `components` parameter is assumed to be a strictly increasing sequence
    /// representing the indices of non-zero values in the vector, and `values`
    /// holds the corresponding values. Both `components` and `values` must have
    /// the same size and cannot be empty. Additionally, `components` must be sorted in strictly ascending order.
    ///
    /// # Parameters
    ///
    /// * `components`: A slice containing the indices of non-zero values in the vector.
    /// * `values`: A slice containing the corresponding values for each index in `components`.
    ///
    /// # Panics
    ///
    /// This function will panic if:
    ///
    /// * The sizes of `components` and `values` are different.
    /// * The size of either `components` or `values` is 0.
    /// * `components` is not sorted in strictly ascending order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectorium::{Dataset, GrowableDataset};
    /// use vectorium::{DotProduct, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVector1D};
    ///
    /// let quantizer = <PlainSparseQuantizer<u32, f32, DotProduct> as vectorium::VectorEncoder>::new(5, 5);
    /// let mut dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    /// dataset.push(SparseVector1D::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0]));
    ///
    /// assert_eq!(dataset.len(), 1);
    /// assert_eq!(dataset.input_dim(), 5);
    /// assert_eq!(dataset.nnz(), 3);
    /// ```
    fn push(
        &mut self,
        vec: impl Vector1D<Component = E::InputComponentType, Value = E::InputValueType>,
    ) {
        let components = vec.components_as_slice();
        let values = vec.values_as_slice();

        assert_eq!(
            components.len(),
            values.len(),
            "Vectors have different sizes"
        );

        assert!(
            is_strictly_sorted(components),
            "Components must be given in strictly ascending order"
        );

        let cur_dim = self.quantizer.input_dim();
        if let Some(last_component) = components.last().map(|l| l.as_()) {
            assert!(
                last_component < cur_dim,
                "Component index {} is out of bounds for input_dim {}",
                last_component,
                cur_dim
            );
        }

        self.quantizer.extend_with_encode(
            SparseVector1D::new(components, values),
            &mut self.storage.components,
            &mut self.storage.values,
        );

        self.storage.offsets.push(self.storage.components.len());
    }
}

impl<'a, C, V> Iterator for SparseDatasetIter<'a, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    type Item = SparseVector1D<C, V, &'a [C], &'a [V]>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (&next_offset, rest) = self.offsets.split_first()?;
        self.offsets = rest;

        let (cur_components, rest) = self
            .components
            .split_at(next_offset - self.front_base_offset);
        self.components = rest;

        let (cur_values, rest) = self.values.split_at(next_offset - self.front_base_offset);
        self.values = rest;

        self.front_base_offset = next_offset;

        Some(SparseVector1D::new(cur_components, cur_values))
    }
}

impl<C, V> ExactSizeIterator for SparseDatasetIter<'_, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    fn len(&self) -> usize {
        self.offsets.len()
    }
}

impl<C, V> DoubleEndedIterator for SparseDatasetIter<'_, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    /// Retrieves the next vector from the end of the iterator.
    ///
    /// # Returns
    ///
    /// An option containing a tuple of slices representing the components and values of the next vector,
    /// or `None` if the end of the iterator is reached.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectorium::{Dataset, GrowableDataset};
    /// use vectorium::datasets::sparse_dataset::SparseDatasetIter;
    /// use vectorium::{DotProduct, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVector1D, Vector1D};
    ///
    /// let data = vec![
    ///     (vec![0_u32, 2, 4], vec![1.0, 2.0, 3.0]),
    ///     (vec![1_u32, 3], vec![4.0, 5.0]),
    ///     (vec![0_u32, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0]),
    /// ];
    ///
    /// let quantizer = <PlainSparseQuantizer<u32, f32, DotProduct> as vectorium::VectorEncoder>::new(5, 5);
    /// let mut dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    /// for (c, v) in data.iter() {
    ///     dataset.push(SparseVector1D::new(c.clone(), v.clone()));
    /// }
    ///
    /// // Use the concrete iterator type so `next_back()` is available.
    /// let mut iter = SparseDatasetIter::new(&dataset);
    /// let last = iter.next_back().unwrap();
    /// assert_eq!(last.components_as_slice(), &[0, 1, 2, 3]);
    /// assert_eq!(last.values_as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    /// ```
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.offsets.is_empty() {
            return None;
        }
        let (&tail_offset, rest) = self.offsets.split_last()?;
        self.offsets = rest;

        let tail_offset = tail_offset - self.front_base_offset;
        let next_offset =
            *self.offsets.last().unwrap_or(&self.front_base_offset) - self.front_base_offset;

        let len = tail_offset - next_offset;

        let (rest, cur_components) = self.components.split_at(tail_offset - len);
        self.components = rest;

        let (rest, cur_values) = self.values.split_at(tail_offset - len);
        self.values = rest;

        Some(SparseVector1D::new(cur_components, cur_values))
    }
}

impl<E, S> SpaceUsage for SparseDatasetGeneric<E, S>
where
    E: SparseVectorEncoder,
    for<'a> E: VectorEncoder<EncodedVector<'a> = SparseEncodedVector<'a, E>>,
    E: SpaceUsage,
    S: SparseStorage<E> + SpaceUsage,
{
    /// Returns the size of the dataset in bytes.
    fn space_usage_bytes(&self) -> usize {
        self.storage.space_usage_bytes() + self.quantizer.space_usage_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::SparseDatasetIter;
    use crate::core::dataset::ConvertInto;
    use crate::core::dataset::GrowableDataset;
    use crate::{
        Dataset, DotProduct, PlainSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer,
        SparseDataset, SparseDatasetGrowable, SparseVector1D,
    };
    use crate::{Vector1D, VectorEncoder};
    use half::f16;

    #[test]
    fn sparse_dataset_iter_next_then_next_back_returns_last_vector() {
        let quantizer = <PlainSparseQuantizer<u16, f32, DotProduct> as VectorEncoder>::new(6, 6);
        let mut dataset = PlainSparseDatasetGrowable::new(quantizer);

        dataset.push(SparseVector1D::new(vec![0_u16, 1], vec![1.0_f32, 1.0]));
        dataset.push(SparseVector1D::new(vec![2_u16, 3], vec![2.0_f32, 2.0]));
        dataset.push(SparseVector1D::new(vec![4_u16, 5], vec![3.0_f32, 3.0]));

        let mut iter = SparseDatasetIter::new(&dataset);
        let _ = iter.next().unwrap();
        let back = iter.next_back().unwrap();

        assert_eq!(back.components_as_slice(), &[4_u16, 5]);
        assert_eq!(back.values_as_slice(), &[3.0_f32, 3.0]);
        let back = iter.next_back().unwrap();
        assert_eq!(back.components_as_slice(), &[2_u16, 3]);
        assert_eq!(back.values_as_slice(), &[2.0_f32, 2.0]);

        assert!(iter.next_back().is_none());
    }

    #[test]
    fn sparse_growable_immutable_roundtrip() {
        let quantizer = <PlainSparseQuantizer<u16, f32, DotProduct> as VectorEncoder>::new(6, 6);
        let mut growable = PlainSparseDatasetGrowable::new(quantizer);

        growable.push(SparseVector1D::new(vec![0_u16, 2], vec![1.0_f32, 2.0]));
        growable.push(SparseVector1D::new(vec![1_u16, 3], vec![3.0_f32, 4.0]));

        let frozen: PlainSparseDataset<u16, f32, DotProduct> = growable.into();
        assert_eq!(frozen.len(), 2);
        assert_eq!(frozen.nnz(), 4);

        let first = frozen.get(0);
        assert_eq!(first.components_as_slice(), &[0_u16, 2]);
        assert_eq!(first.values_as_slice(), &[1.0_f32, 2.0]);

        let mut growable_again: PlainSparseDatasetGrowable<u16, f32, DotProduct> = frozen.into();
        growable_again.push(SparseVector1D::new(vec![4_u16], vec![5.0_f32]));

        assert_eq!(growable_again.len(), 3);
        assert_eq!(growable_again.nnz(), 5);
    }

    #[test]
    fn sparse_scalar_plain_roundtrip_without_reencode() {
        let quantizer =
            <crate::ScalarSparseQuantizer<u16, f32, f16, DotProduct> as VectorEncoder>::new(6, 6);
        let mut growable = SparseDatasetGrowable::new(quantizer);

        growable.push(SparseVector1D::new(vec![0_u16, 2], vec![1.0_f32, 2.0]));
        growable.push(SparseVector1D::new(vec![1_u16, 3], vec![3.0_f32, 4.0]));

        let growable_plain: SparseDatasetGrowable<PlainSparseQuantizer<u16, f16, DotProduct>> =
            growable.relabel_as_plain();
        let first = growable_plain.get(0);
        assert_eq!(first.values_as_slice(), &[f16::from_f32(1.0), f16::from_f32(2.0)]);

        let frozen_plain: PlainSparseDataset<u16, f16, DotProduct> = growable_plain.into();
        let frozen_scalar: SparseDataset<
            crate::ScalarSparseQuantizer<u16, f32, f16, DotProduct>,
        > = frozen_plain.relabel_as_scalar();
        let second = frozen_scalar.get(1);
        assert_eq!(second.values_as_slice(), &[f16::from_f32(3.0), f16::from_f32(4.0)]);
    }

    #[test]
    fn sparse_scalar_reencode_convert_into_owned() {
        type SrcQuant = crate::ScalarSparseQuantizer<u16, f32, f16, DotProduct>;
        type DstQuant = crate::ScalarSparseQuantizer<u16, f16, f32, DotProduct>;

        let quantizer = <SrcQuant as VectorEncoder>::new(6, 6);
        let mut growable = SparseDatasetGrowable::new(quantizer);

        growable.push(SparseVector1D::new(vec![0_u16, 2], vec![1.25_f32, 2.5]));
        growable.push(SparseVector1D::new(vec![1_u16, 3], vec![3.5_f32, 4.25]));

        let frozen: SparseDataset<SrcQuant> = growable.into();

        let converted: SparseDataset<DstQuant> = frozen.convert_into();

        let first = converted.get(0);
        assert_eq!(first.components_as_slice(), &[0_u16, 2]);
        assert_eq!(
            first.values_as_slice(),
            &[
                f32::from(f16::from_f32(1.25)),
                f32::from(f16::from_f32(2.5)),
            ]
        );

        let second = converted.get(1);
        assert_eq!(second.components_as_slice(), &[1_u16, 3]);
        assert_eq!(
            second.values_as_slice(),
            &[
                f32::from(f16::from_f32(3.5)),
                f32::from(f16::from_f32(4.25)),
            ]
        );
    }
}
