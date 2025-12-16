use serde::{Deserialize, Serialize};
use std::hint::assert_unchecked;

use crate::SpaceUsage;
use crate::datasets::{Dataset, GrowableDataset};
use crate::quantizers::{Quantizer, SparseQuantizer};
use crate::utils::prefetch_read_slice;
use crate::{ComponentType, ValueType, VectorId, VectorKey};
use crate::{SparseVector1D, Vector1D};

use rayon::iter::plumbing::ProducerCallback;
use rayon::iter::plumbing::{Consumer, Producer, UnindexedConsumer, bridge};
use rayon::prelude::{IndexedParallelIterator, ParallelIterator};

/// A growable representation of a sparse dataset.

///
/// # Examples
///
/// ```
/// use seismic::SparseDatasetGrowable;
///
/// // Create a new empty dataset
/// let mut dataset = SparseDatasetGrowable::<u16, f32>::default();
/// ```
///
// TODO: decidere se vogliamo growable dataset solo per plain o sparse o anche per quantizzati
pub type SparseDatasetGrowable<Q> = SparseDatasetGeneric<
    Q,
    Vec<usize>,
    Vec<<Q as Quantizer>::OutputComponentType>,
    Vec<<Q as Quantizer>::OutputValueType>,
>;

// Implementation of a (immutable) sparse dataset.
pub type SparseDataset<Q> = SparseDatasetGeneric<
    Q,
    Box<[usize]>,
    Box<[<Q as Quantizer>::OutputComponentType]>,
    Box<[<Q as Quantizer>::OutputValueType]>,
>;

#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct SparseDatasetGeneric<Q, O, AC, AV>
where
    Q: SparseQuantizer,
    O: AsRef<[usize]>,
    AC: AsRef<[Q::OutputComponentType]>,
    AV: AsRef<[Q::OutputValueType]>,
{
    dim: usize,
    dim_bits: u32, // Number of bits required to represent dim. Needed to pack/umpack offset and lenght in VectorKey
    offsets: O,
    components: AC,
    values: AV,
    quantizer: Q,
}

impl<Q, O, AC, AV> SparseDatasetGeneric<Q, O, AC, AV>
where
    Q: SparseQuantizer,
    O: AsRef<[usize]>,
    AC: AsRef<[Q::OutputComponentType]>,
    AV: AsRef<[Q::OutputValueType]>,
{
    fn key_to_range(&self, key: VectorKey) -> std::ops::Range<usize> {
        let length = (key & ((1u64 << self.dim_bits) - 1)) as usize;
        let offset = (key >> self.dim_bits) as usize;

        std::ops::Range {
            start: offset,
            end: offset + length,
        }
    }

    pub fn par_iter<'a>(
        &'a self,
    ) -> ParSparseDatasetIter<'a, Q::OutputComponentType, Q::OutputValueType> {
        ParSparseDatasetIter::new(self)
    }
}

impl<Q, O, AC, AV> Dataset<Q> for SparseDatasetGeneric<Q, O, AC, AV>
where
    Q: SparseQuantizer,
    O: AsRef<[usize]>,
    AC: AsRef<[Q::OutputComponentType]>,
    AV: AsRef<[Q::OutputValueType]>,
{
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
    /// ```
    /// use seismic::SparseDatasetGrowable;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3],       vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                 ];
    ///
    /// let dataset: SparseDatasetGrowable<u16, f32> = data.into_iter().collect();
    ///
    /// let (components, values) = dataset.get(1);
    /// assert_eq!(components, &[1, 3]);
    /// assert_eq!(values, &[4.0, 5.0]);
    /// ```
    #[inline]
    fn get(
        &self,
        key: VectorKey,
    ) -> impl Vector1D<ComponentType = Q::OutputComponentType, ValueType = Q::OutputValueType> {
        let range = self.key_to_range(key);
        let offset = range.start;
        let len = range.end - range.start;

        unsafe { assert_unchecked(self.components.as_ref().len() == self.values.as_ref().len()) };

        let v_components = &self.components.as_ref()[offset..offset + len];
        let v_values = &self.values.as_ref()[offset..offset + len];

        SparseVector1D::new(v_components, v_values)
    }

    /// Returns a vector of the dataset at the specified `offset` and `len`.
    ///
    /// This method returns slices of components and values for the dataset starting at the specified `offset`
    /// and with the specified `len`.
    ///
    /// This method is needed by [`InvertedIndex`] which often already knows the offset of the required vector. This speeds up the access.
    ///
    /// # Panics
    ///
    /// Panics if the `offset` + `len` is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDatasetGrowable;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3],       vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                 ];
    ///
    /// let dataset: SparseDatasetGrowable<u16, f32> = data.into_iter().collect();
    ///
    /// let (components, values) = dataset.get_with_offset(3, 2);
    /// assert_eq!(components, &[1, 3]);
    /// assert_eq!(values, &[4.0, 5.0]);
    /// ```
    // #[inline]
    // pub fn get_with_offset(&self, offset: usize, len: usize) -> (&[C], &[V]) {
    //     unsafe { assert_unchecked(self.components.as_ref().len() == self.values.as_ref().len()) };

    //     let v_components = &self.components.as_ref()[offset..offset + len];
    //     let v_values = &self.values.as_ref()[offset..offset + len];

    //     (v_components, v_values)
    // }

    /// Returns the range of positions of the slice with the given `id`.
    ///
    /// ### Panics
    /// Panics if the `id` is out of range.
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
    fn id_from_key(&self, key: VectorKey) -> VectorId {
        let offset = self.key_to_range(key).start;
        self.offsets.as_ref().binary_search(&offset).unwrap() as VectorId
    }

    #[inline]
    fn key_from_id(&self, id: VectorId) -> VectorKey {
        let offsets = self.offsets.as_ref();
        let start = offsets[id as usize];
        let end = offsets[id as usize + 1];
        let length = end - start;

        ((start as u64) << self.dim_bits) | (length as u64)
    }

    fn quantizer(&self) -> &Q {
        &self.quantizer
    }

    /// Prefetches the components and values of a vector with the specified offset and length into the CPU cache.
    ///
    /// This method prefetches the components and values of a vector starting at the specified `offset``
    /// and with the specified length `len` into the CPU cache, which can improve performance by reducing
    /// cache misses during subsequent accesses.
    ///
    /// # Parameters
    ///
    /// * `offset`: The starting index of the vector to prefetch.
    /// * `len`: The length of the vector to prefetch.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDatasetGrowable;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3],       vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                 ];
    ///
    /// let dataset: SparseDatasetGrowable<u16, f32> = data.into_iter().collect();
    ///
    /// // Prefetch components and values of the vector starting at index 1 with length 3
    /// dataset.prefetch_vec_with_offset(1, 3);
    /// ```
    #[inline]
    fn prefetch(&self, key: VectorKey) {
        let sparse_vector = self.get(key);

        prefetch_read_slice(sparse_vector.components_as_slice());
        prefetch_read_slice(sparse_vector.values_as_slice());
    }

    /// Returns an iterator over the vectors of the dataset.
    ///
    /// This method returns an iterator that yields references to each vector in the dataset.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDatasetGrowable;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3],       vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                 ];
    ///
    /// let dataset: SparseDatasetGrowable<u16, f32> = data.clone().into_iter().collect();
    ///
    /// for ((c0, v0), (c1,v1)) in dataset.iter().zip(data.iter()) {
    ///     assert_eq!(c0, c1);
    ///     assert_eq!(v0, v1);
    /// }
    /// ```
    fn iter(
        &self,
    ) -> impl Iterator<
        Item = impl Vector1D<ComponentType = Q::OutputComponentType, ValueType = Q::OutputValueType>,
    > {
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
    /// ```
    /// use seismic::SparseDatasetGrowable;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3],       vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                ];
    ///
    /// let dataset: SparseDatasetGrowable<u16, f32> = data.into_iter().collect();
    ///
    /// assert_eq!(dataset.len(), 3);
    /// ```
    fn len(&self) -> usize {
        self.offsets.as_ref().len() - 1
    }

    /// Returns the number of components of the dataset, i.e., it returns one plus the ID of the largest component.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDatasetGrowable;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4], vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3], vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                ];
    ///
    /// let dataset: SparseDatasetGrowable<u16, f32> = data.into_iter().collect();
    ///
    /// assert_eq!(dataset.dim(), 5); // Largest component ID is 4, so dim() returns 5
    /// ```
    fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    fn shape(&self) -> (usize, usize) {
        (self.len(), self.dim())
    }

    /// Returns the number of non-zero components in the dataset.
    ///
    /// This function returns the total count of non-zero components across all vectors
    /// currently stored in the dataset.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDatasetGrowable;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4], vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3], vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                ];
    ///
    /// let dataset: SparseDatasetGrowable<u16, f32> = data.into_iter().collect();
    ///
    /// assert_eq!(dataset.nnz(), 9); // Total non-zero components across all vectors
    /// ```
    fn nnz(&self) -> usize {
        self.components.as_ref().len()
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

// impl<Q, AC, AV> FromIterator<(AC, AV)> for SparseDatasetGrowable<Q>
// where
//     Q: SparseQuantizer,
//     AC: AsRef<[Q::OutputComponentType]>,
//     AV: AsRef<[Q::OutputValueType]>,
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
//         let mut dataset = SparseDatasetGrowable::<Q>::new(Q);

//         for (components, values) in iter {
//             dataset.push(components.as_ref(), values.as_ref());
//         }

//         dataset
//     }
// }

// Unfortunately, Rust doesn't yet support specialization, meaning that we can't use From too generically (otherwise it fails due to reimplementing `From<T> for T`)

impl<Q> From<SparseDatasetGrowable<Q>> for SparseDataset<Q>
where
    Q: SparseQuantizer,
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
    /// ```
    /// use seismic::{SparseDatasetGrowable, SparseDataset};
    ///
    /// let mut mutable_dataset = SparseDatasetGrowable::<u16, f32>::new();
    /// // Populate mutable dataset...
    /// mutable_dataset.push(&[0, 2, 4],    &[1.0, 2.0, 3.0]);
    /// mutable_dataset.push(&[1, 3],       &[4.0, 5.0]);
    /// mutable_dataset.push(&[0, 1, 2, 3], &[1.0, 2.0, 3.0, 4.0]);
    ///
    /// let immutable_dataset: SparseDataset<u16, f32> = mutable_dataset.into();
    ///
    /// assert_eq!(immutable_dataset.nnz(), 9); // Total non-zero components across all vectors
    /// ```
    fn from(dataset: SparseDatasetGrowable<Q>) -> Self {
        Self {
            dim: dataset.dim,
            dim_bits: (dataset.dim - 1).ilog2() + 1,
            offsets: dataset.offsets.into(),
            components: dataset.components.into(),
            values: dataset.values.into(),
            quantizer: dataset.quantizer,
        }
    }
}

impl<Q> From<SparseDataset<Q>> for SparseDatasetGrowable<Q>
where
    Q: SparseQuantizer,
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
    /// ```
    /// use seismic::{SparseDatasetGrowable, SparseDataset};
    ///
    /// let mut mutable_dataset = SparseDatasetGrowable::<u16, f32>::new();
    /// // Populate mutable dataset...
    /// mutable_dataset.push(&[0, 2, 4],    &[1.0, 2.0, 3.0]);
    /// mutable_dataset.push(&[1, 3],       &[4.0, 5.0]);
    /// mutable_dataset.push(&[0, 1, 2, 3], &[1.0, 2.0, 3.0, 4.0]);
    ///
    /// let immutable_dataset: SparseDataset<u16, f32> = mutable_dataset.into();
    ///
    /// // Convert immutable dataset back to mutable
    /// let mut mutable_dataset_again: SparseDatasetGrowable<u16, f32> = immutable_dataset.into();
    ///
    /// mutable_dataset_again.push(&[1, 7], &[1.0, 3.0]);
    ///
    /// assert_eq!(mutable_dataset_again.nnz(), 11); // Total non-zero components across all vectors
    /// ```
    fn from(dataset: SparseDataset<Q>) -> Self {
        Self {
            dim: dataset.dim,
            dim_bits: (dataset.dim - 1).ilog2() + 1,
            offsets: dataset.offsets.into(),
            components: dataset.components.into(),
            values: dataset.values.into(),
            quantizer: dataset.quantizer,
        }
    }
}

/// A struct to iterate over a sparse dataset. It assumes the dataset can be represented as a pair of slices.
#[derive(Clone)]
pub struct SparseDatasetIter<'a, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    last_offset: usize,
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
    fn new<Q, O, AC, AV>(dataset: &'a SparseDatasetGeneric<Q, O, AC, AV>) -> Self
    where
        Q: SparseQuantizer<OutputComponentType = C, OutputValueType = V>,
        O: AsRef<[usize]>,
        AC: AsRef<[C]>,
        AV: AsRef<[V]>,
    {
        Self {
            last_offset: 0,
            offsets: &dataset.offsets.as_ref()[1..],
            components: dataset.components.as_ref(),
            values: dataset.values.as_ref(),
        }
    }
}

impl<Q> GrowableDataset<Q> for SparseDatasetGrowable<Q>
where
    Q: SparseQuantizer,
{
    /// For SparseDataset, thw dimensionality `d` may be 0 if unkwown when creating a new dataset.
    fn new(quantizer: Q, d: usize) -> Self {
        Self {
            dim: d,
            dim_bits: if d == 0 { 0 } else { (d - 1).ilog2() + 1 },
            offsets: vec![0; 1],
            components: Vec::new(),
            values: Vec::new(),
            quantizer,
        }
    }

    /// Adds a new sparse vector to the dataset.
    ///
    /// The `components` parameter is assumed to be a strictly increasing sequence
    /// representing the indices of non-zero values in the vector, and `values`
    /// holds the corresponding values. Both `components` and `values` must have
    /// the same size and cannot be empty. Additionally, `components` must be sorted.
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
    /// * `components` is not sorted in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use seismic::SparseDatasetMut;
    ///
    /// let mut dataset = SparseDatasetMut::<u16, f32>::default();
    /// dataset.push(&[0, 2, 4], &[1.0, 2.0, 3.0]);
    ///
    /// assert_eq!(dataset.len(), 1);
    /// assert_eq!(dataset.dim(), 5);
    /// assert_eq!(dataset.nnz(), 3);
    /// ```
    fn push(
        &mut self,
        vec: impl Vector1D<ComponentType = Q::InputComponentType, ValueType = Q::InputValueType>,
    ) {
        let components = vec.components_as_slice();
        let values = vec.values_as_slice();

        assert_eq!(
            components.len(),
            values.len(),
            "Vectors have different sizes"
        );

        assert!(
            components.is_sorted(),
            "Components must be given in sorted order"
        );

        if let Some(last_component) = components.last().map(|l| l.as_())
            && last_component >= self.dim
        {
            self.dim = last_component + 1;
            self.dim_bits = (self.dim - 1).ilog2() + 1;
        }

        self.quantizer
            .extend_with_encode(vec, &mut self.components, &mut self.values);

        self.offsets.push(self.components.len());
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

        let (cur_components, rest) = self.components.split_at(next_offset - self.last_offset);
        self.components = rest;

        let (cur_values, rest) = self.values.split_at(next_offset - self.last_offset);
        self.values = rest;

        self.last_offset = next_offset;

        Some(SparseVector1D::new(cur_components, cur_values))
    }
}

/// A struct to iterate over a sparse dataset in parallel.
/// It assumes the dataset can be represented as a pair of slices.
#[derive(Clone)]
pub struct ParSparseDatasetIter<'a, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    last_offset: usize,
    offsets: &'a [usize],
    components: &'a [C],
    values: &'a [V],
}

impl<'a, C, V> ParSparseDatasetIter<'a, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    #[inline]
    fn new<Q, O, AC, AV>(dataset: &'a SparseDatasetGeneric<Q, O, AC, AV>) -> Self
    where
        Q: SparseQuantizer<OutputComponentType = C, OutputValueType = V>,
        O: AsRef<[usize]>,
        AC: AsRef<[C]>,
        AV: AsRef<[V]>,
    {
        Self {
            last_offset: 0,
            offsets: &dataset.offsets.as_ref()[1..],
            components: dataset.components.as_ref(),
            values: dataset.values.as_ref(),
        }
    }
}

impl<'a, C, V> ParallelIterator for ParSparseDatasetIter<'a, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    type Item = SparseVector1D<C, V, &'a [C], &'a [V]>;

    fn drive_unindexed<CS>(self, consumer: CS) -> CS::Result
    where
        CS: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.offsets.len())
    }
}

impl<C, V> IndexedParallelIterator for ParSparseDatasetIter<'_, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        let producer = SparseDatasetProducer::from(self);
        callback.callback(producer)
    }

    fn drive<CS: Consumer<Self::Item>>(self, consumer: CS) -> CS::Result {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.offsets.len()
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
    /// ```
    /// use seismic::SparseDatasetGrowable;
    ///
    /// let data = vec![
    ///                 (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    ///                 (vec![1, 3],       vec![4.0, 5.0]),
    ///                 (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
    ///                 ];
    ///
    /// let dataset: SparseDatasetGrowable<u16, f32> = data.clone().into_iter().collect();
    ///
    /// let data_rev: Vec<_> =  dataset.iter().rev().collect();
    ///
    /// for ((c0,v0), (c1, v1)) in data.into_iter().zip(data_rev.into_iter().rev()) {
    ///     assert_eq!(c0.as_slice(), c1);
    ///     assert_eq!(v0.as_slice(), v1);
    /// }
    /// ```
    fn next_back(&mut self) -> Option<Self::Item> {
        let (&last_offset, rest) = self.offsets.split_last()?;
        self.offsets = rest;

        let next_offset = *self.offsets.last().unwrap_or(&self.last_offset);

        let len = last_offset - next_offset;

        let (rest, cur_components) = self.components.split_at(last_offset - len);
        self.components = rest;

        let (rest, cur_values) = self.values.split_at(last_offset - len);
        self.values = rest;

        Some(SparseVector1D::new(cur_components, cur_values))
    }
}
struct SparseDatasetProducer<'a, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    last_offset: usize,
    offsets: &'a [usize],
    components: &'a [C],
    values: &'a [V],
}

impl<'a, C, V> Producer for SparseDatasetProducer<'a, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    type Item = SparseVector1D<C, V, &'a [C], &'a [V]>;
    type IntoIter = SparseDatasetIter<'a, C, V>;

    fn into_iter(self) -> Self::IntoIter {
        SparseDatasetIter {
            last_offset: self.last_offset,
            offsets: self.offsets,
            components: self.components,
            values: self.values,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let left_last_offset = self.last_offset;

        let (left_offsets, right_offsets) = self.offsets.split_at(index);
        let right_last_offset = *left_offsets.last().unwrap();

        let (left_components, right_components) = self
            .components
            .split_at(right_last_offset - left_last_offset);
        let (left_values, right_values) =
            self.values.split_at(right_last_offset - left_last_offset);

        (
            SparseDatasetProducer {
                last_offset: left_last_offset,
                offsets: left_offsets,
                components: left_components,
                values: left_values,
            },
            SparseDatasetProducer {
                last_offset: right_last_offset,
                offsets: right_offsets,
                components: right_components,
                values: right_values,
            },
        )
    }
}

impl<'a, C, V> From<ParSparseDatasetIter<'a, C, V>> for SparseDatasetProducer<'a, C, V>
where
    C: ComponentType,
    V: ValueType,
{
    fn from(other: ParSparseDatasetIter<'a, C, V>) -> Self {
        Self {
            last_offset: other.last_offset,
            offsets: other.offsets,
            components: other.components,
            values: other.values,
        }
    }
}

impl<Q, O, AC, AV> SpaceUsage for SparseDatasetGeneric<Q, O, AC, AV>
where
    Q: SparseQuantizer + SpaceUsage,
    O: AsRef<[usize]> + SpaceUsage,
    AC: AsRef<[Q::OutputComponentType]> + SpaceUsage,
    AV: AsRef<[Q::OutputValueType]> + SpaceUsage,
{
    /// Returns the size of the dataset in bytes.
    fn space_usage_byte(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.offsets.space_usage_byte()
            + self.components.space_usage_byte()
            + self.values.space_usage_byte()
            + self.quantizer.space_usage_byte()
    }
}
