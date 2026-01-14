//! Storage abstractions for datasets.
//!
//! This module provides storage traits that bundle the storage requirements
//! for sparse datasets, reducing generic parameter explosion.
//!
//! # Storage Trait
//!
//! - [`SparseStorage`]: Storage for sparse datasets (offsets + components + values)
//!
//! Currently the trait has two implementations:
//! - `GrowableSparseStorage`: Uses `Vec<T>` for dynamic growth
//! - `ImmutableSparseStorage`: Uses `Box<[T]>` for frozen/immutable data

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::SpaceUsage;
use crate::SparseVectorEncoder;

// =============================================================================
// Sparse Storage
// =============================================================================

/// Trait that bundles the storage requirements for a sparse dataset.
///
/// This abstracts over the concrete storage types (`Vec`, `Box<[T]>`, etc.)
/// reducing the generic parameter count from 4 (`O, AC, AV` + encoder) to 2
/// (encoder + storage).
pub trait SparseStorage<E: SparseVectorEncoder>: Clone {
    /// Type for storing offsets (e.g., `Vec<usize>` or `Box<[usize]>`)
    type Offsets: AsRef<[usize]>;

    /// Type for storing components (e.g., `Vec<C>` or `Box<[C]>`)
    type Components: AsRef<[E::OutputComponentType]>;

    /// Type for storing values (e.g., `Vec<V>` or `Box<[V]>`)
    type Values: AsRef<[E::OutputValueType]>;

    /// Returns a reference to the offsets array.
    fn offsets(&self) -> &Self::Offsets;

    /// Returns a reference to the components array.
    fn components(&self) -> &Self::Components;

    /// Returns a reference to the values array.
    fn values(&self) -> &Self::Values;
}

/// Extension trait for mutable sparse storage operations.
pub trait SparseStorageMut<E: SparseVectorEncoder>: SparseStorage<E> {
    /// Returns a mutable reference to the offsets array.
    fn offsets_mut(&mut self) -> &mut Self::Offsets;

    /// Returns a mutable reference to the components array.
    fn components_mut(&mut self) -> &mut Self::Components;

    /// Returns a mutable reference to the values array.
    fn values_mut(&mut self) -> &mut Self::Values;
}

/// Growable sparse storage using `Vec` for all arrays.
#[derive(Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::OutputComponentType: Serialize, E::OutputValueType: Serialize",
    deserialize = "E::OutputComponentType: Deserialize<'de>, E::OutputValueType: Deserialize<'de>"
))]
pub struct GrowableSparseStorage<E>
where
    E: SparseVectorEncoder,
{
    pub(crate) offsets: Vec<usize>,
    pub(crate) components: Vec<E::OutputComponentType>,
    pub(crate) values: Vec<E::OutputValueType>,
    #[serde(skip)]
    _phantom: PhantomData<E>,
}

impl<E> Clone for GrowableSparseStorage<E>
where
    E: SparseVectorEncoder,
    E::OutputComponentType: Clone,
    E::OutputValueType: Clone,
{
    fn clone(&self) -> Self {
        Self {
            offsets: self.offsets.clone(),
            components: self.components.clone(),
            values: self.values.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<E> GrowableSparseStorage<E>
where
    E: SparseVectorEncoder,
{
    /// Creates a new empty growable storage.
    #[inline]
    pub fn new() -> Self {
        Self {
            offsets: vec![0],
            components: Vec::new(),
            values: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Creates a new growable storage with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(n_vecs: usize, nnz: usize) -> Self {
        let mut offsets = Vec::with_capacity(n_vecs + 1);
        offsets.push(0);
        Self {
            offsets,
            components: Vec::with_capacity(nnz),
            values: Vec::with_capacity(nnz),
            _phantom: PhantomData,
        }
    }

    pub(crate) fn relabel<E2>(self) -> GrowableSparseStorage<E2>
    where
        E2: SparseVectorEncoder<
            OutputComponentType = E::OutputComponentType,
            OutputValueType = E::OutputValueType,
        >,
    {
        GrowableSparseStorage {
            offsets: self.offsets,
            components: self.components,
            values: self.values,
            _phantom: PhantomData,
        }
    }
}

impl<E> SpaceUsage for GrowableSparseStorage<E>
where
    E: SparseVectorEncoder,
    E::OutputComponentType: SpaceUsage,
    E::OutputValueType: SpaceUsage,
{
    fn space_usage_bytes(&self) -> usize {
        self.offsets.space_usage_bytes()
            + self.components.space_usage_bytes()
            + self.values.space_usage_bytes()
    }
}

impl<E> SparseStorage<E> for GrowableSparseStorage<E>
where
    E: SparseVectorEncoder,
{
    type Offsets = Vec<usize>;
    type Components = Vec<E::OutputComponentType>;
    type Values = Vec<E::OutputValueType>;

    #[inline]
    fn offsets(&self) -> &Self::Offsets {
        &self.offsets
    }

    #[inline]
    fn components(&self) -> &Self::Components {
        &self.components
    }

    #[inline]
    fn values(&self) -> &Self::Values {
        &self.values
    }
}

impl<E> SparseStorageMut<E> for GrowableSparseStorage<E>
where
    E: SparseVectorEncoder,
{
    #[inline]
    fn offsets_mut(&mut self) -> &mut Self::Offsets {
        &mut self.offsets
    }

    #[inline]
    fn components_mut(&mut self) -> &mut Self::Components {
        &mut self.components
    }

    #[inline]
    fn values_mut(&mut self) -> &mut Self::Values {
        &mut self.values
    }
}

/// Immutable sparse storage using `Box<[T]>` for all arrays.
#[derive(Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "E::OutputComponentType: Serialize, E::OutputValueType: Serialize",
    deserialize = "E::OutputComponentType: Deserialize<'de>, E::OutputValueType: Deserialize<'de>"
))]
pub struct ImmutableSparseStorage<E>
where
    E: SparseVectorEncoder,
{
    pub(crate) offsets: Box<[usize]>,
    pub(crate) components: Box<[E::OutputComponentType]>,
    pub(crate) values: Box<[E::OutputValueType]>,
    #[serde(skip)]
    _phantom: PhantomData<E>,
}

impl<E> Clone for ImmutableSparseStorage<E>
where
    E: SparseVectorEncoder,
    E::OutputComponentType: Clone,
    E::OutputValueType: Clone,
{
    fn clone(&self) -> Self {
        Self {
            offsets: self.offsets.clone(),
            components: self.components.clone(),
            values: self.values.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<E> SpaceUsage for ImmutableSparseStorage<E>
where
    E: SparseVectorEncoder,
    E::OutputComponentType: SpaceUsage,
    E::OutputValueType: SpaceUsage,
{
    fn space_usage_bytes(&self) -> usize {
        self.offsets.space_usage_bytes()
            + self.components.space_usage_bytes()
            + self.values.space_usage_bytes()
    }
}

impl<E> SparseStorage<E> for ImmutableSparseStorage<E>
where
    E: SparseVectorEncoder,
{
    type Offsets = Box<[usize]>;
    type Components = Box<[E::OutputComponentType]>;
    type Values = Box<[E::OutputValueType]>;

    #[inline]
    fn offsets(&self) -> &Self::Offsets {
        &self.offsets
    }

    #[inline]
    fn components(&self) -> &Self::Components {
        &self.components
    }

    #[inline]
    fn values(&self) -> &Self::Values {
        &self.values
    }
}

impl<E> ImmutableSparseStorage<E>
where
    E: SparseVectorEncoder,
{
    pub(crate) fn relabel<E2>(self) -> ImmutableSparseStorage<E2>
    where
        E2: SparseVectorEncoder<
            OutputComponentType = E::OutputComponentType,
            OutputValueType = E::OutputValueType,
        >,
    {
        ImmutableSparseStorage {
            offsets: self.offsets,
            components: self.components,
            values: self.values,
            _phantom: PhantomData,
        }
    }
}

impl<E> From<GrowableSparseStorage<E>> for ImmutableSparseStorage<E>
where
    E: SparseVectorEncoder,
{
    fn from(storage: GrowableSparseStorage<E>) -> Self {
        Self {
            offsets: storage.offsets.into_boxed_slice(),
            components: storage.components.into_boxed_slice(),
            values: storage.values.into_boxed_slice(),
            _phantom: PhantomData,
        }
    }
}

impl<E> From<ImmutableSparseStorage<E>> for GrowableSparseStorage<E>
where
    E: SparseVectorEncoder,
{
    fn from(storage: ImmutableSparseStorage<E>) -> Self {
        Self {
            offsets: storage.offsets.into_vec(),
            components: storage.components.into_vec(),
            values: storage.values.into_vec(),
            _phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DotProduct, PlainSparseQuantizer};

    #[test]
    fn growable_storage_roundtrip_preserves_data() {
        type Encoder = PlainSparseQuantizer<u16, f32, DotProduct>;

        let mut storage = GrowableSparseStorage::<Encoder>::new();
        storage.components.extend_from_slice(&[1_u16, 2]);
        storage.values.extend_from_slice(&[1.0_f32, 2.0]);
        storage.offsets.push(storage.components.len());

        let frozen: ImmutableSparseStorage<Encoder> = storage.clone().into();
        assert_eq!(frozen.components().as_ref(), &[1_u16, 2]);
        assert_eq!(frozen.values().as_ref(), &[1.0_f32, 2.0]);

        let restored: GrowableSparseStorage<Encoder> = frozen.into();
        assert_eq!(restored, storage);
        assert!(restored.space_usage_bytes() > 0);
    }

    #[test]
    fn growable_relabel_preserves_space_usage() {
        type Encoder = PlainSparseQuantizer<u16, f32, DotProduct>;

        let mut storage = GrowableSparseStorage::<Encoder>::new();
        storage.components.extend_from_slice(&[0_u16, 3]);
        storage.values.extend_from_slice(&[0.5_f32, 1.5]);
        storage.offsets.push(storage.components.len());

        let original_space = storage.space_usage_bytes();
        let relabeled: GrowableSparseStorage<Encoder> = storage.clone().relabel();
        assert_eq!(relabeled, storage);
        let relabeled_space = relabeled.space_usage_bytes();
        assert!(relabeled_space > 0);
    }

    #[test]
    fn immutable_relabel_preserves_data() {
        type Encoder = PlainSparseQuantizer<u16, f32, DotProduct>;

        let mut storage = GrowableSparseStorage::<Encoder>::new();
        storage.components.extend_from_slice(&[5_u16]);
        storage.values.extend_from_slice(&[7.0_f32]);
        storage.offsets.push(storage.components.len());

        let frozen: ImmutableSparseStorage<Encoder> = storage.into();
        let relabeled: ImmutableSparseStorage<Encoder> = frozen.clone().relabel();
        assert_eq!(relabeled, frozen);
        assert_eq!(relabeled.space_usage_bytes(), frozen.space_usage_bytes());
    }

    #[test]
    fn growable_storage_mutable_accessors_allow_editing() {
        type Encoder = PlainSparseQuantizer<u16, f32, DotProduct>;

        let mut storage = GrowableSparseStorage::<Encoder>::new();
        storage.components.extend_from_slice(&[2_u16]);
        storage.values.extend_from_slice(&[3.0_f32]);

        storage.offsets_mut().push(1);
        storage.offsets_mut().pop();
        storage.components_mut().push(4_u16);
        storage.values_mut().push(5.0_f32);

        assert_eq!(storage.offsets(), &[0]);
        assert_eq!(storage.components(), &[2_u16, 4_u16]);
        assert_eq!(storage.values(), &[3.0_f32, 5.0_f32]);
    }
}
