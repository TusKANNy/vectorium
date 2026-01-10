use crate::{ComponentType, SpaceUsage, ValueType};

/// Marker trait for vector views.
/// Does not mandate any accessor methods, just marks the type as a vector view.
pub trait VectorView {}

impl<T: VectorView> VectorView for &T {}
impl<T: VectorView> VectorView for &mut T {}

/// Marker trait for *plain* (not encoded) vector views.
///
/// Intended for values supplied by the user (e.g. query vectors), as opposed to vectors
/// already in an encoder's output representation.
pub trait PlainVectorView<V: ValueType>: VectorView {}

impl<V, T> PlainVectorView<V> for &T
where
    V: ValueType,
    T: PlainVectorView<V>,
{
}

impl<V, T> PlainVectorView<V> for &mut T
where
    V: ValueType,
    T: PlainVectorView<V>,
{
}

/// A view over a dense vector (slice of values).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseVectorView<'a, V: ValueType> {
    values: &'a [V],
}

impl<'a, V: ValueType> DenseVectorView<'a, V> {
    #[inline]
    pub fn new(values: &'a [V]) -> Self {
        Self { values }
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    #[inline]
    pub fn values(&self) -> &'a [V] {
        self.values
    }
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = V> + '_ {
        self.values.iter().copied()
    }
    #[inline]
    pub fn to_owned(&self) -> DenseVectorOwned<V> {
        DenseVectorOwned::new(self.values.to_vec())
    }
}

impl<'a, V: ValueType> VectorView for DenseVectorView<'a, V> {}

impl<'a, V: ValueType> PlainVectorView<V> for DenseVectorView<'a, V> {}

/// A view over a sparse vector (slices of components and values).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseVectorView<'a, C: ComponentType, V: ValueType> {
    components: &'a [C],
    values: &'a [V],
}

impl<'a, C: ComponentType, V: ValueType> SparseVectorView<'a, C, V> {
    #[inline]
    pub fn new(components: &'a [C], values: &'a [V]) -> Self {
        debug_assert_eq!(components.len(), values.len());
        Self { components, values }
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.components.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }
    #[inline]
    pub fn components(&self) -> &'a [C] {
        self.components
    }
    #[inline]
    pub fn values(&self) -> &'a [V] {
        self.values
    }
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (C, V)> + '_ {
        self.components
            .iter()
            .copied()
            .zip(self.values.iter().copied())
    }
    #[inline]
    pub fn to_owned(&self) -> SparseVectorOwned<C, V> {
        SparseVectorOwned::new(self.components.to_vec(), self.values.to_vec())
    }
}

impl<'a, C: ComponentType, V: ValueType> VectorView for SparseVectorView<'a, C, V> {}

impl<'a, C: ComponentType, V: ValueType> PlainVectorView<V> for SparseVectorView<'a, C, V> {}

/// A view over a packed vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PackedVectorView<'a, T: SpaceUsage + Copy> {
    data: &'a [T],
}

impl<'a, T: SpaceUsage + Copy> PackedVectorView<'a, T> {
    #[inline]
    pub fn new(data: &'a [T]) -> Self {
        Self { data }
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }
    #[inline]
    pub fn data(&self) -> &'a [T] {
        self.data
    }
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        self.data.iter().copied()
    }
    #[inline]
    pub fn to_owned(&self) -> PackedVectorOwned<T> {
        PackedVectorOwned::new(self.data.to_vec())
    }
}
impl<'a, T: SpaceUsage + Copy> VectorView for PackedVectorView<'a, T> {}

// --- Owned Structs ---

#[derive(Debug, Clone, PartialEq)]
pub struct DenseVectorOwned<V: ValueType> {
    values: Vec<V>,
}

impl<V: ValueType> DenseVectorOwned<V> {
    #[inline]
    pub fn new(values: Vec<V>) -> Self {
        Self { values }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    #[inline]
    pub fn values(&self) -> &[V] {
        &self.values
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = V> + '_ {
        self.values.iter().copied()
    }

    #[inline]
    pub fn as_view(&self) -> DenseVectorView<'_, V> {
        DenseVectorView::new(&self.values)
    }
}
impl<V: ValueType> VectorView for DenseVectorOwned<V> {}

impl<'a, V: ValueType> From<&'a DenseVectorOwned<V>> for DenseVectorView<'a, V> {
    fn from(owned: &'a DenseVectorOwned<V>) -> Self {
        owned.as_view()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseVectorOwned<C: ComponentType, V: ValueType> {
    components: Vec<C>,
    values: Vec<V>,
}

impl<C: ComponentType, V: ValueType> SparseVectorOwned<C, V> {
    #[inline]
    pub fn new(components: Vec<C>, values: Vec<V>) -> Self {
        assert!(
            components.len() == values.len(),
            "Components and values must have the same length"
        );
        Self { components, values }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.components.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    #[inline]
    pub fn components(&self) -> &[C] {
        &self.components
    }

    #[inline]
    pub fn values(&self) -> &[V] {
        &self.values
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (C, V)> + '_ {
        self.components
            .iter()
            .copied()
            .zip(self.values.iter().copied())
    }

    #[inline]
    pub fn as_view(&self) -> SparseVectorView<'_, C, V> {
        SparseVectorView::new(&self.components, &self.values)
    }
}
impl<C: ComponentType, V: ValueType> VectorView for SparseVectorOwned<C, V> {}

impl<'a, C: ComponentType, V: ValueType> From<&'a SparseVectorOwned<C, V>>
    for SparseVectorView<'a, C, V>
{
    fn from(owned: &'a SparseVectorOwned<C, V>) -> Self {
        owned.as_view()
    }
}

// PackedVectorOwned
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedVectorOwned<T>
where
    T: SpaceUsage + Copy,
{
    data: Vec<T>,
}

impl<T> PackedVectorOwned<T>
where
    T: SpaceUsage + Copy,
{
    #[inline]
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub fn as_view(&self) -> PackedVectorView<'_, T> {
        PackedVectorView::new(&self.data)
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        self.data.iter().copied()
    }
}
impl<T> VectorView for PackedVectorOwned<T> where T: SpaceUsage + Copy {}

impl<'a, T> From<&'a PackedVectorOwned<T>> for PackedVectorView<'a, T>
where
    T: SpaceUsage + Copy,
{
    fn from(owned: &'a PackedVectorOwned<T>) -> Self {
        owned.as_view()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_vector_owned_basic() {
        let values = vec![1.0f32, 2.0, 3.0];
        let v = DenseVectorOwned::new(values.clone());
        assert_eq!(v.len(), 3);
        assert_eq!(v.values(), values.as_slice());

        let iter_vals: Vec<_> = v.iter().collect();
        assert_eq!(iter_vals, values);

        let view = v.as_view();
        assert_eq!(view.values(), values.as_slice());
    }

    #[test]
    fn dense_vector_view_basic() {
        let values = vec![1.0f32, 2.0, 3.0];
        let v = DenseVectorView::new(&values);
        assert_eq!(v.len(), 3);
        assert_eq!(v.values(), values.as_slice());

        let owned = v.to_owned();
        assert_eq!(owned.values(), values.as_slice());
    }

    #[test]
    fn sparse_vector_owned_basic() {
        let comps = vec![0usize, 2usize];
        let vals = vec![1.0f32, 3.0f32];
        let v = SparseVectorOwned::new(comps.clone(), vals.clone());
        assert_eq!(v.len(), 2);
        assert_eq!(v.components(), comps.as_slice());
        assert_eq!(v.values(), vals.as_slice());

        let iter_vals: Vec<_> = v.iter().collect();
        assert_eq!(iter_vals, vec![(0, 1.0), (2, 3.0)]);

        let view = v.as_view();
        assert_eq!(view.components(), comps.as_slice());
        assert_eq!(view.values(), vals.as_slice());
    }

    #[test]
    fn sparse_vector_view_basic() {
        let comps = vec![0usize, 2usize];
        let vals = vec![1.0f32, 3.0f32];
        let v = SparseVectorView::new(&comps, &vals);
        assert_eq!(v.len(), 2);
        assert_eq!(v.components(), comps.as_slice());
        assert_eq!(v.values(), vals.as_slice());

        let owned = v.to_owned();
        assert_eq!(owned.components(), comps.as_slice());
        assert_eq!(owned.values(), vals.as_slice());
    }
}
