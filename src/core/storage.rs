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
