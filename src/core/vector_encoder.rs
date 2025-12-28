use crate::SpaceUsage;
use crate::numeric_markers::DenseComponent;
use crate::{ComponentType, ValueType};
use crate::{DenseVector1D, SparseVector1D, Vector1D, distances::Distance};

/// Marker trait for types that are valid query vectors for a given quantizer `Q`.
///
/// This allows `VectorEncoder::query_evaluator` to remain on the base trait while
/// still enforcing that dense/sparse quantizers only accept the corresponding
/// concrete vector representations.
pub trait QueryVectorFor<Q: VectorEncoder>:
    Vector1D<Value = Q::QueryValueType, Component = Q::QueryComponentType>
{
}

impl<Q, AV> QueryVectorFor<Q> for DenseVector1D<Q::QueryValueType, AV>
where
    Q: VectorEncoder<QueryComponentType = DenseComponent>,
    AV: AsRef<[Q::QueryValueType]>,
{
}

impl<Q, AC, AV> QueryVectorFor<Q>
    for SparseVector1D<Q::QueryComponentType, Q::QueryValueType, AC, AV>
where
    Q: SparseQuantizer,
    AC: AsRef<[Q::QueryComponentType]>,
    AV: AsRef<[Q::QueryValueType]>,
{
}

impl<Q, T> QueryVectorFor<Q> for &T
where
    Q: VectorEncoder,
    T: QueryVectorFor<Q> + ?Sized,
{
}

/// A query evaluator computes distances between a query and encoded vectors.
pub trait QueryEvaluator<Q: VectorEncoder>: Sized {
    fn compute_distance(&self, vector: Q::EncodedVector<'_>) -> Q::Distance;
}

pub trait VectorEncoder: Sized {
    type Distance: Distance;

    type QueryValueType: ValueType;
    type QueryComponentType: ComponentType;
    type InputValueType: ValueType;
    type InputComponentType: ComponentType;
    type OutputValueType: ValueType;
    type OutputComponentType: ComponentType;

    /// The query evaluator type for this quantizer and distance.
    ///
    /// The evaluator may borrow the query vector, hence it is lifetime-parameterized.
    type Evaluator<'a>: QueryEvaluator<Self>
    where
        Self: 'a;

    /// The encoded representation of dataset vectors produced by this quantizer.
    ///
    /// This is intentionally a *concrete* type chosen by the quantizer (e.g.
    /// `DenseVector1D<.., &'a [..]>`, `SparseVector1D<..>`, `&'a [u64]`, ...).
    ///
    /// Datasets that store a specific representation can constrain this type.
    type EncodedVector<'a>;

    /// Create a new quantizer for input vectors of the given dimensionality `input_dim` and number of components in the quantized vector `output_dim`.
    fn new(input_dim: usize, output_dim: usize) -> Self;

    /// Train the quantizer on a (sub)set of input vectors.
    ///
    /// Some quantizers (e.g. k-means/PQ-like) need a training step before they
    /// can be used effectively. This method exists to make that step explicit
    /// and type-safe: the quantizer receives an iterator over raw input vectors.
    ///
    /// NOTE: `train` accepts a looser `Vector1D` bound because some quantizers
    /// do not need the exact input component/value types. This avoids forcing
    /// callers to convert datasets into intermediate formats purely to satisfy
    /// strict type bounds.
    ///
    /// Default implementation is a no-op for quantizers that do not require training.
    #[inline]
    fn train<InputVector>(&mut self, _training_data: impl Iterator<Item = InputVector>)
    where
        InputVector: Vector1D,
    {
    }

    /// Get a query evaluator for the given distance type
    fn query_evaluator<'a, QueryVector>(&'a self, query: &'a QueryVector) -> Self::Evaluator<'a>
    where
        QueryVector: QueryVectorFor<Self> + ?Sized;

    /// Dimensionality of the encoded vector space.
    ///
    /// For dense encoders, this is typically the number of values produced per vector.
    ///
    /// Note: if the encoded representation is *packed* and has variable length
    /// in memory (e.g. `&[u64]` with per-vector offsets), `output_dim()` is **not**
    /// the packed blob length. It must remain the logical dimensionality of the
    /// post-quantization space used for distance evaluation; the packed length is
    /// storage-specific and determined by the dataset offsets.
    fn output_dim(&self) -> usize;

    /// Dimensionality of the original (input) vector space.
    ///
    /// For sparse vectors, this is the maximum possible component index + 1.
    fn input_dim(&self) -> usize;
}

/// A quantizer whose encoded representation is a packed slice of fixed-width elements.
///
/// This is meant to be used together with a `PackedDataset` (variable-length by offsets).
pub trait PackedQuantizer: VectorEncoder + SpaceUsage {
    /// Element type stored in the dataset backing array (e.g. `u64` for word-packed encodings).
    type EncodingType: SpaceUsage + Copy + Send + Sync + 'static;
}

pub trait DenseQuantizer:
    VectorEncoder<
        QueryComponentType = DenseComponent,
        InputComponentType = DenseComponent,
        OutputComponentType = DenseComponent,
    > + SpaceUsage
{
    /// Encode input vectors into quantized output vectors
    fn extend_with_encode<ValueContainer>(
        &self,
        input_vector: DenseVector1D<Self::InputValueType, impl AsRef<[Self::InputValueType]>>,
        values: &mut ValueContainer,
    ) where
        ValueContainer: Extend<Self::OutputValueType>;

    /// Quantize an input dense vector into owned output values.
    ///
    /// # Example
    /// ```
    /// use vectorium::{DenseQuantizer, DenseVector1D, DotProduct, PlainDenseQuantizer};
    /// use vectorium::{Vector1D, VectorEncoder};
    ///
    /// let quantizer = PlainDenseQuantizer::<f32, DotProduct>::new(3);
    /// let v = DenseVector1D::new(vec![1.0_f32, 0.0, 2.0]);
    /// let q = quantizer.quantize_vector(v);
    /// assert_eq!(q.values_as_slice().len(), 3);
    /// ```
    #[inline]
    fn quantize_vector<AV>(
        &self,
        input_vector: DenseVector1D<Self::InputValueType, AV>,
    ) -> DenseVector1D<Self::OutputValueType, Vec<Self::OutputValueType>>
    where
        AV: AsRef<[Self::InputValueType]>,
    {
        let values = input_vector.values_as_slice();
        let mut out_values = Vec::with_capacity(values.len());

        self.extend_with_encode(DenseVector1D::new(values), &mut out_values);

        DenseVector1D::new(out_values)
    }
}

pub trait SparseQuantizer: VectorEncoder + SpaceUsage {
    /// Encode input vectors into quantized output vectors
    fn extend_with_encode<ValueContainer, ComponentContainer>(
        &self,
        input_vector: SparseVector1D<
            Self::InputComponentType,
            Self::InputValueType,
            impl AsRef<[Self::InputComponentType]>,
            impl AsRef<[Self::InputValueType]>,
        >,
        components: &mut ComponentContainer,
        values: &mut ValueContainer,
    ) where
        ValueContainer: Extend<Self::OutputValueType>,
        ComponentContainer: Extend<Self::OutputComponentType>;

    /// Quantize an input sparse vector into owned output components/values.
    ///
    /// # Example
    /// ```
    /// use vectorium::{DotProduct, PlainSparseQuantizer, SparseQuantizer, SparseVector1D};
    /// use vectorium::{Vector1D, VectorEncoder};
    ///
    /// let quantizer = PlainSparseQuantizer::<u16, f32, DotProduct>::new(5, 5);
    /// let v = SparseVector1D::new(vec![1_u16, 3], vec![1.0, 2.0]);
    /// let q = quantizer.quantize_vector(v);
    /// assert_eq!(q.components_as_slice(), &[1_u16, 3]);
    /// assert_eq!(q.values_as_slice(), &[1.0, 2.0]);
    /// ```
    #[inline]
    fn quantize_vector<AC, AV>(
        &self,
        input_vector: SparseVector1D<Self::InputComponentType, Self::InputValueType, AC, AV>,
    ) -> SparseVector1D<
        Self::OutputComponentType,
        Self::OutputValueType,
        Vec<Self::OutputComponentType>,
        Vec<Self::OutputValueType>,
    >
    where
        AC: AsRef<[Self::InputComponentType]>,
        AV: AsRef<[Self::InputValueType]>,
    {
        let components = input_vector.components_as_slice();
        let values = input_vector.values_as_slice();

        let mut out_components = Vec::with_capacity(components.len());
        let mut out_values = Vec::with_capacity(values.len());

        self.extend_with_encode(
            SparseVector1D::new(components, values),
            &mut out_components,
            &mut out_values,
        );

        SparseVector1D::new(out_components, out_values)
    }
}
