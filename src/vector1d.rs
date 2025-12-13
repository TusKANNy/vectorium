use crate::{ComponentType, DenseComponent, ValueType};

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

pub trait MutableVector1D: Vector1D {
    fn values_as_mut_slice(&mut self) -> &mut [Self::ValueType];
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseVector1D<V, AV>
where
    V: ValueType,
    AV: AsRef<[V]>,
{
    components: (),
    values: AV,
    phantom: std::marker::PhantomData<V>,
}

impl<V, AV> DenseVector1D<V, AV>
where
    V: ValueType,
    AV: AsRef<[V]>,
{
    pub fn new(values: AV) -> Self {
        Self {
            components: (),
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

    #[inline(always)]
    fn len(&self) -> usize {
        self.values.as_ref().len()
    }

    #[inline(always)]
    fn values_as_slice(&self) -> &[Self::ValueType] {
        self.values.as_ref()
    }

    #[inline(always)]
    fn components_as_slice(&self) -> &[Self::ComponentType] {
        &[]
    }
}

impl<V, AV> MutableVector1D for DenseVector1D<V, AV>
where
    V: ValueType,
    AV: AsRef<[V]> + AsMut<[V]>,
{
    #[inline(always)]
    fn values_as_mut_slice(&mut self) -> &mut [Self::ValueType] {
        self.values.as_mut()
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
    d: usize, // dimensionality of the vector space
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
    pub fn new(components: AC, values: AV, d: usize) -> Self {
        assert!(
            components.as_ref().len() == values.as_ref().len(),
            "Components and values must have the same length"
        );

        SparseVector1D {
            components,
            values,
            d,
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
        let v = SparseVector1D::new(comps.clone(), vals.clone(), 4);
        assert_eq!(v.len(), 2);
        assert_eq!(v.components_as_slice(), comps.as_slice());
        assert_eq!(v.values_as_slice(), vals.as_slice());
    }
}
