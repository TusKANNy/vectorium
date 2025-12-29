use crate::SpaceUsage;

mod sealed {
    pub trait Sealed {}
}

/// Implemented only for the crate-provided packed vector view.
///
/// This is a sealed trait: external types cannot implement it. It is used to
/// enforce that `PackedDataset` can only expose `PackedVector` as encoded vectors.
pub trait PackedEncoded<'a, T>: sealed::Sealed + Send {
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

impl<T> sealed::Sealed for PackedVector<T, &[T]> where T: SpaceUsage + Copy {}

impl<'a, T> PackedEncoded<'a, T> for PackedVector<T, &'a [T]>
where
    T: SpaceUsage + Copy + Send + Sync,
{
    #[inline]
    fn from_slice(slice: &'a [T]) -> Self {
        PackedVector::new(slice)
    }
}
