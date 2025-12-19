use crate::num_marker::DenseComponent;
use crate::{ComponentType, PackedType, ValueType};

pub trait Vector1D {
    type ComponentType;
    type ValueType;

    fn len(&self) -> usize;

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn components_as_slice(&self) -> &[Self::ComponentType];
    fn values_as_slice(&self) -> &[Self::ValueType];
}

impl<T> Vector1D for &T
where
    T: Vector1D + ?Sized,
{
    type ComponentType = T::ComponentType;
    type ValueType = T::ValueType;

    #[inline]
    fn len(&self) -> usize {
        (**self).len()
    }

    #[inline]
    fn components_as_slice(&self) -> &[Self::ComponentType] {
        (**self).components_as_slice()
    }

    #[inline]
    fn values_as_slice(&self) -> &[Self::ValueType] {
        (**self).values_as_slice()
    }
}

pub trait MutableVector1D: Vector1D {
    fn values_as_mut_slice(&mut self) -> &mut [Self::ValueType];
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseVector1D<V, AV>
where
    V: ValueType,
    AV: AsRef<[V]>,
{
    components: DenseComponent,
    values: AV,
    phantom: std::marker::PhantomData<V>,
}

impl<V, AV> DenseVector1D<V, AV>
where
    V: ValueType,
    AV: AsRef<[V]>,
{
    #[inline]
    pub fn new(values: AV) -> Self {
        Self {
            components: DenseComponent,
            values,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<V, AV> Vector1D for DenseVector1D<V, AV>
where
    V: ValueType,
    AV: AsRef<[V]>,
{
    type ComponentType = DenseComponent;
    type ValueType = V;

    #[inline]
    fn len(&self) -> usize {
        self.values.as_ref().len()
    }

    #[inline]
    fn values_as_slice(&self) -> &[Self::ValueType] {
        self.values.as_ref()
    }

    #[inline(always)]
    fn components_as_slice(&self) -> &[Self::ComponentType] {
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
}

impl<C, V, AC, AV> Vector1D for SparseVector1D<C, V, AC, AV>
where
    C: ComponentType,
    V: ValueType,
    AC: AsRef<[C]>,
    AV: AsRef<[V]>,
{
    type ComponentType = C;
    type ValueType = V;

    /// Returns the length of the sparse array.
    #[inline(always)]
    fn len(&self) -> usize {
        self.components.as_ref().len()
    }

    #[inline(always)]
    fn components_as_slice(&self) -> &[Self::ComponentType] {
        self.components.as_ref()
    }

    #[inline(always)]
    fn values_as_slice(&self) -> &[Self::ValueType] {
        self.values.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackedSparseVector1D<C, V, D, AD>
where
    C: ComponentType,
    V: ValueType,
    D: PackedType,
    AD: AsRef<[D]>,
{
    data: AD,
    _phantom: std::marker::PhantomData<(C, V, D)>,
}

/// In PackedSparseVector1D, the components and values are stored in contiguous memory blocks, potentially compressed for saving space.
/// We need the proper Encoder type to interpret the packed data.

impl<C, V, D, AD> PackedSparseVector1D<C, V, D, AD>
where
    C: ComponentType,
    V: ValueType,
    D: PackedType,
    AD: AsRef<[D]>,
{
    #[inline]
    pub fn new(data: AD) -> Self {
        PackedSparseVector1D {
            data,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<C, V, D, AD> Vector1D for PackedSparseVector1D<C, V, D, AD>
where
    C: ComponentType,
    V: ValueType,
    D: PackedType,
    AD: AsRef<[D]>,
{
    type ComponentType = C;
    type ValueType = V;

    /// Returns the length of the sparse array. It's the length of the packed data, not the number of components or values in the original vector.
    #[inline(always)]
    fn len(&self) -> usize {
        self.data.as_ref().len()
    }

    #[inline(always)]
    fn components_as_slice(&self) -> &[Self::ComponentType] {
        &[] // Packed representation does not expose components directly
    }

    #[inline(always)]
    fn values_as_slice(&self) -> &[Self::ValueType] {
        &[] // Packed representation does not expose values directly
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
