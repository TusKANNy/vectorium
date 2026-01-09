use serde::{Deserialize, Serialize};
use std::hint::assert_unchecked;

use crate::SpaceUsage;
use crate::core::sealed;
use crate::core::storage::{GrowableSparseStorage, ImmutableSparseStorage, SparseStorage};
use crate::core::vector_encoder::{SparseVectorEncoder, VectorEncoder};
use crate::core::vector::{SparseVectorView, VectorView};
use crate::utils::is_strictly_sorted;
use crate::{ComponentType, Float, FromF32, ValueType, VectorId};
use crate::{Dataset, GrowableDataset, SparseData};
use num_traits::AsPrimitive;

use crate::dataset::ConvertFrom;

use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSlice};

/// A growable representation of a sparse dataset.
///
/// # Examples
///
/// ```rust
/// use vectorium::{Dataset, GrowableDataset};
/// use vectorium::{DotProduct, PlainSparseDatasetGrowable, PlainSparseQuantizer};
///
/// // Create a new empty dataset
/// let quantizer = PlainSparseQuantizer::<u32, f32, DotProduct>::new(0, 0);
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
/// - `E`: The encoder type (quantizer types are encoders)
/// - `S`: The storage backend (defaults to `GrowableSparseStorage<E>`)
///
/// # Example
/// ```
/// use vectorium::{
///     Dataset, DotProduct, GrowableDataset, PlainSparseDataset, PlainSparseQuantizer,
///     SparseDatasetGrowable, SparseVectorView, VectorView, VectorEncoder,
/// };
///
/// let quantizer = PlainSparseQuantizer::<u16, f32, DotProduct>::new(5, 5);
/// let mut dataset = SparseDatasetGrowable::new(quantizer);
/// dataset.push(SparseVectorView::new(&[1_u16, 3], &[1.0, 2.0]));
///
/// let frozen: PlainSparseDataset<u16, f32, DotProduct> = dataset.into();
/// let v = frozen.get(0);
/// assert_eq!(v.components(), &[1_u16, 3]);
/// assert_eq!(v.values(), &[1.0, 2.0]);
///
/// let range = frozen.range_from_id(0);
/// let v = frozen.get_with_range(range);
/// assert_eq!(v.components(), &[1_u16, 3]);
/// assert_eq!(v.values(), &[1.0, 2.0]);
/// ```
#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct SparseDatasetGeneric<E, S = GrowableSparseStorage<E>>
where
    E: SparseVectorEncoder,
    S: SparseStorage<E>,
{
    storage: S,
    encoder: E,
}

impl<E, S> sealed::Sealed for SparseDatasetGeneric<E, S>
where
    E: SparseVectorEncoder,
    S: SparseStorage<E>,
{
}

impl<E, S> SparseDatasetGeneric<E, S>
where
    E: SparseVectorEncoder,
    S: SparseStorage<E>,
{
    #[inline]
    pub fn nnz(&self) -> usize {
        self.storage.components().as_ref().len()
    }

    #[inline]
    pub fn range_from_id(&self, id: VectorId) -> std::ops::Range<usize> {
        let offsets = self.storage.offsets().as_ref();
        let index = id as usize;
        assert!(index + 1 < offsets.len(), "Index out of bounds.");
        offsets[index]..offsets[index + 1]
    }

    #[inline]
    pub fn id_from_range(&self, range: std::ops::Range<usize>) -> VectorId {
        let offsets = self.storage.offsets().as_ref();
        let idx = offsets.binary_search(&range.start).unwrap();
        assert_eq!(
            offsets[idx + 1],
            range.end,
            "Range does not match vector boundaries."
        );
        idx as VectorId
    }

    /// Parallel iterator over all vectors as slice-backed `SparseVectorView`.
    #[inline]
    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = E::EncodedVector<'_>> + '_
    where
        for<'a> E::EncodedVector<'a>: Send,
        S: Sync,
    {
        let offsets = self.storage.offsets().as_ref();
        let components = self.storage.components().as_ref();
        let values = self.storage.values().as_ref();

        // https://github.com/rayon-rs/rayon/pull/789
        offsets.par_windows(2).map(move |window| {
            let &[start, end] = window else {
                unsafe { std::hint::unreachable_unchecked() }
            };
            SparseVectorView::new(&components[start..end], &values[start..end])
        })
    }
}

impl<E, S> Dataset for SparseDatasetGeneric<E, S>
where
    E: SparseVectorEncoder,
    S: SparseStorage<E>,
    for<'a> E::EncodedVector<'a>: VectorView,
{
    type Encoder = E;

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
    /// use vectorium::VectorView;
    /// use vectorium::{DotProduct, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView};
    ///
    /// let quantizer = PlainSparseQuantizer::<u32, f32, DotProduct>::new(5, 5);
    /// let mut dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    ///
    /// dataset.push(SparseVectorView::new(&[0, 2, 4], &[1.0, 2.0, 3.0]));
    /// dataset.push(SparseVectorView::new(&[1, 3], &[4.0, 5.0]));
    ///
    /// let v = dataset.get(1);
    /// assert_eq!(v.components(), &[1, 3]);
    /// assert_eq!(v.values(), &[4.0, 5.0]);
    /// ```
    #[inline]
    fn get(&self, index: usize) -> E::EncodedVector<'_> {
        let range = self.range_from_id(index as VectorId);
        self.get_with_range(range)
    }

    #[inline]
    fn get_with_range<'a>(&'a self, range: std::ops::Range<usize>) -> E::EncodedVector<'a> {
        let components = self.storage.components().as_ref();
        let values = self.storage.values().as_ref();
        unsafe { assert_unchecked(components.len() == values.len()) };

        let v_components = &components[range.clone()];
        let v_values = &values[range];

        SparseVectorView::new(v_components, v_values)
    }

    #[inline]
    fn prefetch_with_range(&self, _range: std::ops::Range<usize>) {}

    fn encoder(&self) -> &E {
        &self.encoder
    }

    /// Returns an iterator over the vectors of the dataset.
    ///
    /// This method returns an iterator that yields references to each vector in the dataset.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectorium::{Dataset, GrowableDataset};
    /// use vectorium::VectorView;
    /// use vectorium::{DotProduct, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView};
    ///
    /// let data = vec![
    ///     (vec![0_u32, 2, 4], vec![1.0_f32, 2.0, 3.0]),
    ///     (vec![1_u32, 3], vec![4.0_f32, 5.0]),
    ///     (vec![0_u32, 1, 2, 3], vec![1.0_f32, 2.0, 3.0, 4.0]),
    /// ];
    ///
    /// let quantizer = PlainSparseQuantizer::<u32, f32, DotProduct>::new(5, 5);
    /// let mut dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    /// for (c, v) in data.iter() {
    ///     dataset.push(SparseVectorView::new(c.as_slice(), v.as_slice()));
    /// }
    ///
    /// for (vec, (c, v)) in dataset.iter().zip(data.iter()) {
    ///     assert_eq!(vec.components(), c.as_slice());
    ///     assert_eq!(vec.values(), v.as_slice());
    /// }
    /// ```
    fn iter<'a>(&'a self) -> impl Iterator<Item = E::EncodedVector<'a>> {
        let offsets = self.storage.offsets().as_ref();
        let components = self.storage.components().as_ref();
        let values = self.storage.values().as_ref();
        offsets.windows(2).map(move |window| {
            let start = window[0];
            let end = window[1];
            SparseVectorView::new(&components[start..end], &values[start..end])
        })
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
    /// use vectorium::{DotProduct, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView};
    ///
    /// let quantizer = PlainSparseQuantizer::<u32, f32, DotProduct>::new(5, 5);
    /// let mut dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    ///
    /// dataset.push(SparseVectorView::new(&[0, 2, 4], &[1.0, 2.0, 3.0]));
    /// dataset.push(SparseVectorView::new(&[1, 3], &[4.0, 5.0]));
    /// dataset.push(SparseVectorView::new(&[0, 1, 2, 3], &[1.0, 2.0, 3.0, 4.0]));
    ///
    /// assert_eq!(dataset.len(), 3);
    /// ```
    fn len(&self) -> usize {
        self.storage.offsets().as_ref().len() - 1
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
//     ///                 (vec![0, 1, 2, 3], &[1.0, 2.0, 3.0, 4.0])
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

impl<E> From<SparseDatasetGrowable<E>> for SparseDataset<E>
where
    E: SparseVectorEncoder,
    for<'a> E::EncodedVector<'a>: VectorView,
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
    /// use vectorium::{DotProduct, PlainSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView};
    ///
    /// let quantizer = PlainSparseQuantizer::<u32, f32, DotProduct>::new(5, 5);
    /// let mut growable_dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    ///
    /// growable_dataset.push(SparseVectorView::new(&[0, 2, 4], &[1.0, 2.0, 3.0]));
    /// growable_dataset.push(SparseVectorView::new(&[1, 3], &[4.0, 5.0]));
    /// growable_dataset.push(SparseVectorView::new(&[0, 1, 2, 3], &[1.0, 2.0, 3.0, 4.0]));
    ///
    /// let immutable_dataset: PlainSparseDataset<u32, f32, DotProduct> = growable_dataset.into();
    /// assert_eq!(immutable_dataset.nnz(), 9);
    /// ```
    fn from(dataset: SparseDatasetGrowable<E>) -> Self {
        Self {
            storage: dataset.storage.into(),
            encoder: dataset.encoder,
        }
    }
}

impl<E> From<SparseDataset<E>> for SparseDatasetGrowable<E>
where
    E: SparseVectorEncoder,
    for<'a> E::EncodedVector<'a>: VectorView,
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
    /// use vectorium::{DotProduct, PlainSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView};
    ///
    /// let quantizer = PlainSparseQuantizer::<u32, f32, DotProduct>::new(5, 5);
    /// let mut growable_dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    ///
    /// growable_dataset.push(SparseVectorView::new(&[0, 2, 4], &[1.0, 2.0, 3.0]));
    /// growable_dataset.push(SparseVectorView::new(&[1, 3], &[4.0, 5.0]));
    ///
    /// let immutable_dataset: PlainSparseDataset<u32, f32, DotProduct> = growable_dataset.into();
    ///
    /// // Convert immutable dataset back to mutable
    /// let mut growable_dataset_again: PlainSparseDatasetGrowable<u32, f32, DotProduct> = immutable_dataset.into();
    /// growable_dataset_again.push(SparseVectorView::new(&[1, 4], &[1.0, 3.0]));
    ///
    /// assert_eq!(growable_dataset_again.nnz(), 7);
    /// ```
    fn from(dataset: SparseDataset<E>) -> Self {
        Self {
            storage: dataset.storage.into(),
            encoder: dataset.encoder,
        }
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
        let dim = self.encoder.output_dim();
        let storage = self
            .storage
            .relabel::<crate::PlainSparseQuantizer<C, OutValue, D>>();

        SparseDatasetGeneric {
            storage,
            encoder: crate::PlainSparseQuantizer::<C, OutValue, D>::new(dim, dim),
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
        let dim = self.encoder.output_dim();
        let storage = self
            .storage
            .relabel::<crate::ScalarSparseQuantizer<C, InValue, OutValue, D>>();

        SparseDatasetGeneric {
            storage,
            encoder: crate::ScalarSparseQuantizer::<C, InValue, OutValue, D>::new(dim, dim),
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
        let dim = self.encoder.output_dim();
        let storage = self
            .storage
            .relabel::<crate::PlainSparseQuantizer<C, OutValue, D>>();

        SparseDatasetGeneric {
            storage,
            encoder: crate::PlainSparseQuantizer::<C, OutValue, D>::new(dim, dim),
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
        let dim = self.encoder.output_dim();
        let storage = self
            .storage
            .relabel::<crate::ScalarSparseQuantizer<C, InValue, OutValue, D>>();

        SparseDatasetGeneric {
            storage,
            encoder: crate::ScalarSparseQuantizer::<C, InValue, OutValue, D>::new(dim, dim),
        }
    }
}

impl<C, SrcIn, Mid, DstOut, D, SrcStorage, DstStorage>
    ConvertFrom<&SparseDatasetGeneric<crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>, SrcStorage>>
    for SparseDatasetGeneric<crate::ScalarSparseQuantizer<C, Mid, DstOut, D>, DstStorage>
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
    DstStorage: SparseStorage<crate::ScalarSparseQuantizer<C, Mid, DstOut, D>>
        + From<GrowableSparseStorage<crate::ScalarSparseQuantizer<C, Mid, DstOut, D>>>,
{
    fn convert_from(
        source: &SparseDatasetGeneric<crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>, SrcStorage>,
    ) -> Self {
        let n_vecs = source.len();
        let nnz = source.nnz();
        let dim = source.encoder.output_dim();
        let encoder = crate::ScalarSparseQuantizer::<C, Mid, DstOut, D>::new(dim, dim);

        let mut storage =
            GrowableSparseStorage::<crate::ScalarSparseQuantizer<C, Mid, DstOut, D>>::with_capacity(
                n_vecs, nnz,
            );

        for src_vec in crate::datasets::sparse_dataset::SparseDatasetIter::new(source) {
            let encoded = encoder.encode_vector(SparseVectorView::<C, Mid>::new(
                src_vec.components(),
                src_vec.values(),
            ));
            storage.components.extend_from_slice(encoded.components());
            storage.values.extend_from_slice(encoded.values());
            storage.offsets.push(storage.components.len());
        }

        SparseDatasetGeneric {
            storage: storage.into(),
            encoder,
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
{
    fn convert_from(source: SparseDataset<crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>>) -> Self {
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
{
    fn convert_from(
        source: SparseDatasetGrowable<crate::ScalarSparseQuantizer<C, SrcIn, Mid, D>>,
    ) -> Self {
        Self::convert_from(&source)
    }
}

/// A struct to iterate over the *raw* sparse dataset storage as slice-backed `SparseVectorView`.
///
/// This iterator is independent from the encoder’s `EncodedVector` choice and is
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
{
    /// For SparseDataset, the dimensionality `d` may be 0 if unknown when creating a new dataset.
    fn new(encoder: E) -> Self {
        Self {
            storage: GrowableSparseStorage::new(),
            encoder,
        }
    }

    fn with_capacity(encoder: E, capacity: usize) -> Self {
        Self {
            storage: GrowableSparseStorage::with_capacity(capacity, 0),
            encoder,
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
    /// use vectorium::{DotProduct, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView};
    ///
    /// let quantizer = PlainSparseQuantizer::<u32, f32, DotProduct>::new(5, 5);
    /// let mut dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    /// dataset.push(SparseVectorView::new(&[0, 2, 4], &[1.0, 2.0, 3.0]));
    ///
    /// assert_eq!(dataset.len(), 1);
    /// assert_eq!(dataset.input_dim(), 5);
    /// assert_eq!(dataset.nnz(), 3);
    /// ```
    fn push<'a>(&mut self, vec: E::InputVector<'a>) {
        let components = vec.components();
        let values = vec.values();

        assert_eq!(
            components.len(),
            values.len(),
            "Vectors have different sizes"
        );

        assert!(
            is_strictly_sorted(components),
            "Components must be given in strictly ascending order"
        );

        let cur_dim = self.encoder.input_dim();
        if let Some(last_component) = components.last().map(|l| l.as_()) {
            assert!(
                last_component < cur_dim,
                "Component index {} is out of bounds for input_dim {}",
                last_component,
                cur_dim
            );
        }

        self.encoder
            .push_encoded(vec, &mut self.storage.components, &mut self.storage.values);

        self.storage.offsets.push(self.storage.components.len());
    }
}

impl<'a, C, V> Iterator for SparseDatasetIter<'a, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    type Item = SparseVectorView<'a, C, V>;

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

        Some(SparseVectorView::new(cur_components, cur_values))
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
    /// use vectorium::{DotProduct, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView, VectorView};
    ///
    /// let data = vec![
    ///     (vec![0_u32, 2, 4], vec![1.0_f32, 2.0, 3.0]),
    ///     (vec![1_u32, 3], vec![4.0_f32, 5.0]),
    ///     (vec![0_u32, 1, 2, 3], vec![1.0_f32, 2.0, 3.0, 4.0]),
    /// ];
    ///
    /// let quantizer = PlainSparseQuantizer::<u32, f32, DotProduct>::new(5, 5);
    /// let mut dataset = PlainSparseDatasetGrowable::<u32, f32, DotProduct>::new(quantizer);
    /// for (c, v) in data.iter() {
    ///     dataset.push(SparseVectorView::new(c.as_slice(), v.as_slice()));
    /// }
    ///
    /// // Use the concrete iterator type so `next_back()` is available.
    /// let mut iter = SparseDatasetIter::new(&dataset);
    /// let last = iter.next_back().unwrap();
    /// assert_eq!(last.components(), &[0, 1, 2, 3]);
    /// assert_eq!(last.values(), &[1.0, 2.0, 3.0, 4.0]);
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

        Some(SparseVectorView::new(cur_components, cur_values))
    }
}

impl<E, S> SpaceUsage for SparseDatasetGeneric<E, S>
where
    E: SparseVectorEncoder,
    E: SpaceUsage,
    S: SparseStorage<E> + SpaceUsage,
{
    /// Returns the size of the dataset in bytes.
    fn space_usage_bytes(&self) -> usize {
        self.storage.space_usage_bytes() + self.encoder.space_usage_bytes()
    }
}

impl<C, V, D>
    SparseDatasetGrowable<crate::encoders::sparse_scalar::ScalarSparseQuantizer<C, V, V, D>>
where
    C: ComponentType,
    V: ValueType + Float + FromF32,
    D: crate::encoders::sparse_scalar::ScalarSparseSupportedDistance,
{
    pub fn new(dim: usize) -> Self {
        let encoder = crate::encoders::sparse_scalar::ScalarSparseQuantizer::new(dim, dim);
        crate::GrowableDataset::new(encoder)
    }

    pub fn with_capacity(dim: usize, capacity: usize) -> Self {
        let encoder = crate::encoders::sparse_scalar::ScalarSparseQuantizer::new(dim, dim);
        Self {
            storage: GrowableSparseStorage::with_capacity(capacity, 0),
            encoder,
        }
    }
}

impl<E, S> SparseData for SparseDatasetGeneric<E, S>
where
    E: SparseVectorEncoder,
    S: SparseStorage<E>,
{
}

#[cfg(test)]
mod tests {
    use super::SparseDatasetIter;
    use crate::core::dataset::ConvertInto;
    use crate::core::dataset::GrowableDataset;
    use crate::core::vector::SparseVectorView;
    use crate::{
        Dataset, DotProduct, PlainSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer,
        SparseDataset, SparseDatasetGrowable,
    };
    use half::f16;

    #[test]
    fn sparse_dataset_iter_next_then_next_back_returns_last_vector() {
        let quantizer = PlainSparseQuantizer::<u16, f32, DotProduct>::new(6, 6);
        let mut dataset = PlainSparseDatasetGrowable::new(quantizer);

        dataset.push(SparseVectorView::new(&[0_u16, 1], &[1.0_f32, 1.0]));
        dataset.push(SparseVectorView::new(&[2_u16, 3], &[2.0_f32, 2.0]));
        dataset.push(SparseVectorView::new(&[4_u16, 5], &[3.0_f32, 3.0]));

        let mut iter = SparseDatasetIter::new(&dataset);
        let _ = iter.next().unwrap();
        let back = iter.next_back().unwrap();

        assert_eq!(back.components(), &[4_u16, 5]);
        assert_eq!(back.values(), &[3.0_f32, 3.0]);
        let back = iter.next_back().unwrap();
        assert_eq!(back.components(), &[2_u16, 3]);
        assert_eq!(back.values(), &[2.0_f32, 2.0]);

        assert!(iter.next_back().is_none());
    }

    #[test]
    fn sparse_growable_immutable_roundtrip() {
        let quantizer = PlainSparseQuantizer::<u16, f32, DotProduct>::new(6, 6);
        let mut growable = PlainSparseDatasetGrowable::new(quantizer);

        growable.push(SparseVectorView::new(&[0_u16, 2], &[1.0_f32, 2.0]));
        growable.push(SparseVectorView::new(&[1_u16, 3], &[3.0_f32, 4.0]));

        let frozen: PlainSparseDataset<u16, f32, DotProduct> = growable.into();
        assert_eq!(frozen.len(), 2);
        assert_eq!(frozen.nnz(), 4);

        let first = frozen.get(0);
        assert_eq!(first.components(), &[0_u16, 2]);
        assert_eq!(first.values(), &[1.0_f32, 2.0]);

        let mut growable_again: PlainSparseDatasetGrowable<u16, f32, DotProduct> = frozen.into();
        growable_again.push(SparseVectorView::new(&[4_u16], &[5.0_f32]));

        assert_eq!(growable_again.len(), 3);
        assert_eq!(growable_again.nnz(), 5);
    }

    #[test]
    fn sparse_scalar_plain_roundtrip_without_reencode() {
        let quantizer = crate::ScalarSparseQuantizer::<u16, f32, f16, DotProduct>::new(6, 6);
        let mut growable = SparseDatasetGrowable::new(quantizer);

        growable.push(SparseVectorView::new(&[0_u16, 2], &[1.0_f32, 2.0]));
        growable.push(SparseVectorView::new(&[1_u16, 3], &[3.0_f32, 4.0]));

        let growable_plain: SparseDatasetGrowable<PlainSparseQuantizer<u16, f16, DotProduct>> =
            growable.relabel_as_plain();
        let first = growable_plain.get(0);
        assert_eq!(first.values(), &[f16::from_f32(1.0), f16::from_f32(2.0)]);

        let frozen_plain: PlainSparseDataset<u16, f16, DotProduct> = growable_plain.into();
        let frozen_scalar: SparseDataset<crate::ScalarSparseQuantizer<u16, f32, f16, DotProduct>> =
            frozen_plain.relabel_as_scalar();
        let second = frozen_scalar.get(1);
        assert_eq!(second.values(), &[f16::from_f32(3.0), f16::from_f32(4.0)]);
    }

    #[test]
    fn sparse_scalar_reencode_convert_into_owned() {
        type SrcQuant = crate::ScalarSparseQuantizer<u16, f32, f16, DotProduct>;
        type DstQuant = crate::ScalarSparseQuantizer<u16, f16, f32, DotProduct>;

        let quantizer = SrcQuant::new(6, 6);
        let mut growable = SparseDatasetGrowable::new(quantizer);

        growable.push(SparseVectorView::new(&[0_u16, 2], &[1.25_f32, 2.5]));
        growable.push(SparseVectorView::new(&[1_u16, 3], &[3.5_f32, 4.25]));

        let frozen: SparseDataset<SrcQuant> = growable.into();

        let converted: SparseDataset<DstQuant> = frozen.convert_into();

        let first = converted.get(0);
        assert_eq!(first.components(), &[0_u16, 2]);
        assert_eq!(
            first.values(),
            &[
                f32::from(f16::from_f32(1.25)),
                f32::from(f16::from_f32(2.5)),
            ]
        );

        let second = converted.get(1);
        assert_eq!(second.components(), &[1_u16, 3]);
        assert_eq!(
            second.values(),
            &[
                f32::from(f16::from_f32(3.5)),
                f32::from(f16::from_f32(4.25)),
            ]
        );
    }
}
