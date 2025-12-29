use crate::numeric_markers::DenseComponent;
use crate::{ComponentType, SpaceUsage, ValueType};

pub trait Vector1D {
    type Component: ComponentType;
    type Value: ValueType;

    fn len(&self) -> usize;

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn components_as_slice(&self) -> &[Self::Component];
    fn values_as_slice(&self) -> &[Self::Value];
}

impl<T> Vector1D for &T
where
    T: Vector1D + ?Sized,
{
    type Component = T::Component;
    type Value = T::Value;

    #[inline]
    fn len(&self) -> usize {
        (**self).len()
    }

    #[inline]
    fn components_as_slice(&self) -> &[Self::Component] {
        (**self).components_as_slice()
    }

    #[inline]
    fn values_as_slice(&self) -> &[Self::Value] {
        (**self).values_as_slice()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseVector1D<V, AV>
where
    V: ValueType,
    AV: AsRef<[V]>,
{
    values: AV,
    _phantom: std::marker::PhantomData<V>,
}

impl<V, AV> DenseVector1D<V, AV>
where
    V: ValueType,
    AV: AsRef<[V]>,
{
    #[inline]
    pub fn new(values: AV) -> Self {
        Self {
            values,
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = V> + '_ {
        self.values.as_ref().iter().copied()
    }
}

impl<V, AV> Vector1D for DenseVector1D<V, AV>
where
    V: ValueType,
    AV: AsRef<[V]>,
{
    type Component = DenseComponent;
    type Value = V;

    #[inline]
    fn len(&self) -> usize {
        self.values.as_ref().len()
    }

    #[inline]
    fn values_as_slice(&self) -> &[Self::Value] {
        self.values.as_ref()
    }

    #[inline(always)]
    fn components_as_slice(&self) -> &[Self::Component] {
        // DenseComponent is a zero-sized type; return empty slice
        &[]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseVector1D<C, V, AC, AV>
where
    C: ComponentType,
    V: ValueType,
    AC: AsRef<[C]>,
    AV: AsRef<[V]>,
{
    components: AC,
    values: AV,
    _phantom: std::marker::PhantomData<(C, V)>,
}

impl<C, V, AC, AV> SparseVector1D<C, V, AC, AV>
where
    C: ComponentType,
    V: ValueType,
    AC: AsRef<[C]>,
    AV: AsRef<[V]>,
{
    #[inline]
    pub fn new(components: AC, values: AV) -> Self {
        assert!(
            components.as_ref().len() == values.as_ref().len(),
            "Components and values must have the same length"
        );

        SparseVector1D {
            components,
            values,
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (C, V)> + '_ {
        self.components
            .as_ref()
            .iter()
            .copied()
            .zip(self.values.as_ref().iter().copied())
    }
}

mod packed_sealed {
    pub trait Sealed {}
}

/// Implemented only for the crate-provided packed vector view.
///
/// This is a sealed trait: external types cannot implement it. It is used to
/// enforce that `PackedDataset` can only expose `PackedVector` as encoded vectors.
pub trait PackedEncoded<'a, T>: packed_sealed::Sealed + Send {
    fn from_slice(slice: &'a [T]) -> Self;
}

/// A packed vector view/container.
///
/// This is intentionally minimal: it only provides access to the underlying
/// packed representation as a slice of `T`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedVector<T, AT>
where
    T: SpaceUsage + Copy,
    AT: AsRef<[T]>,
{
    data: AT,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, AT> PackedVector<T, AT>
where
    T: SpaceUsage + Copy,
    AT: AsRef<[T]>,
{
    #[inline]
    pub fn new(data: AT) -> Self {
        Self {
            data,
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data.as_ref()
    }

    /// Returns the length of the packed version of the sparse vector. It's the length of the packed data, not the number of components or values in the original vector.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.as_ref().len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.as_ref().is_empty()
    }
}

impl<T, AT> AsRef<[T]> for PackedVector<T, AT>
where
    T: SpaceUsage + Copy,
    AT: AsRef<[T]>,
{
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.data.as_ref()
    }
}

impl<T> packed_sealed::Sealed for PackedVector<T, &[T]> where T: SpaceUsage + Copy {}

impl<'a, T> PackedEncoded<'a, T> for PackedVector<T, &'a [T]>
where
    T: SpaceUsage + Copy + Send + Sync,
{
    #[inline]
    fn from_slice(slice: &'a [T]) -> Self {
        PackedVector::new(slice)
    }
}

impl<C, V, AC, AV> Vector1D for SparseVector1D<C, V, AC, AV>
where
    C: ComponentType,
    V: ValueType,
    AC: AsRef<[C]>,
    AV: AsRef<[V]>,
{
    type Component = C;
    type Value = V;

    /// Returns the length of the sparse array.
    #[inline(always)]
    fn len(&self) -> usize {
        self.components.as_ref().len()
    }

    #[inline(always)]
    fn components_as_slice(&self) -> &[Self::Component] {
        self.components.as_ref()
    }

    #[inline(always)]
    fn values_as_slice(&self) -> &[Self::Value] {
        self.values.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_vector_basic() {
        let values = vec![1.0f32, 2.0, 3.0];
        let v = DenseVector1D::new(values.clone());
        assert_eq!(v.len(), 3);
        assert_eq!(v.values_as_slice(), values.as_slice());
        assert!(v.components_as_slice().is_empty());

        let v = DenseVector1D::new(&values);
        assert_eq!(v.len(), 3);
        assert_eq!(v.values_as_slice(), values.as_slice());
        assert!(v.components_as_slice().is_empty());
    }

    #[test]
    fn sparse_vector_basic() {
        let comps = vec![0usize, 2usize];
        let vals = vec![1.0f32, 3.0f32];
        let v = SparseVector1D::new(comps.clone(), vals.clone());
        assert_eq!(v.len(), 2);
        assert_eq!(v.components_as_slice(), comps.as_slice());
        assert_eq!(v.values_as_slice(), vals.as_slice());
    }
}
