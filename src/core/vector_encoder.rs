use crate::core::sealed;
use crate::numeric_markers::DenseComponent;
use crate::{ComponentType, ValueType};
use crate::{DenseVector1D, SparseVector1D, Vector1D, distances::Distance};

/// A query evaluator computes distances between a query and encoded vectors.
pub trait QueryEvaluator<V>: Sized {
    type Distance: Distance;
    fn compute_distance(&self, vector: V) -> Self::Distance;
}

/// Core encoder trait.
///
/// Note: in this crate, a "quantizer" is just a specific kind of encoder. The
/// API uses "encoder" consistently.
pub trait VectorEncoder: sealed::Sealed + Sized {
    type Distance: Distance;
    type QueryValueType: ValueType;
    type QueryComponentType: ComponentType;
    type InputValueType: ValueType;
    type InputComponentType: ComponentType;
    type OutputValueType: ValueType;
    type OutputComponentType: ComponentType;

    /// Canonical input vector view used by this encoder.
    type InputVectorType<'a>: Vector1D<
            Component = Self::InputComponentType,
            Value = Self::InputValueType,
        > + 'a
    where
        Self: 'a;

    /// Canonical encoded vector view used by this encoder.
    ///
    /// Note: packed encoders may use a non-`Vector1D` view.
    type EncodedVectorType<'a>: 'a
    where
        Self: 'a;

    /// Canonical query vector view used by this encoder.
    type QueryVectorType<'a>: Vector1D<
            Component = Self::QueryComponentType,
            Value = Self::QueryValueType,
        > + 'a
    where
        Self: 'a;

    /// The query evaluator type for this encoder and distance.
    ///
    /// The evaluator may borrow the query vector, hence it is lifetime-parameterized.
    type Evaluator<'a>: QueryEvaluator<Self::EncodedVectorType<'a>, Distance = Self::Distance>
    where
        Self: 'a;

    /// Create a new encoder for input vectors of the given dimensionality
    /// `input_dim` and number of components in the encoded vector `output_dim`.
    fn new(input_dim: usize, output_dim: usize) -> Self;

    /// Train the encoder on a (sub)set of input vectors.
    ///
    /// Some encoders (e.g. k-means/PQ-like) need a training step before they
    /// can be used effectively. This method exists to make that step explicit
    /// and type-safe: the encoder receives an iterator over raw input vectors.
    ///
    /// NOTE: `train` accepts a looser `Vector1D` bound because some encoders
    /// do not need the exact input component/value types. This avoids forcing
    /// callers to convert datasets into intermediate formats purely to satisfy
    /// strict type bounds.
    ///
    /// Default implementation is a no-op for encoders that do not require training.
    #[inline]
    fn train<TrainVector>(&mut self, _training_data: impl Iterator<Item = TrainVector>)
    where
        TrainVector: Vector1D,
    {
    }

    /// Get a query evaluator for the given distance type
    fn query_evaluator<'a>(&'a self, query: Self::QueryVectorType<'a>) -> Self::Evaluator<'a>;

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

/// Bridge trait for cases where a dataset's encoded vector can be reused as a query without
/// re-encoding or allocation. This exists because Rust cannot express equality constraints
/// between associated types (e.g., `EncodedVectorType == QueryVectorType`) in `where` clauses,
/// even when they are effectively the same for certain encoders (like sparse f32 quantizers).
/// Implementations must be explicit and are typically valid only for specific encoders (e.g.,
/// `OutValue = f32`), not for quantized outputs that require conversion.
pub trait QueryFromEncoded: VectorEncoder {
    fn query_from_encoded<'a, V>(
        &self,
        encoded: &'a V,
    ) -> Self::QueryVectorType<'a>
    where
        V: Vector1D<Component = Self::OutputComponentType, Value = Self::OutputValueType> + ?Sized;
}

/// A packed encoder whose encoded representation is a packed slice of fixed-width elements.
///
/// This is meant to be used together with a `PackedDataset` (variable-length by offsets).
pub trait PackedVectorEncoder: VectorEncoder {
    /// Element type stored in the dataset backing array (e.g. `u64` for word-packed encodings).
    type EncodingType: Copy + Send + Sync + 'static;

    /// Encode input sparse vectors into packed output words.
    fn extend_with_encode<AC, AV>(
        &self,
        input_vector: SparseVector1D<Self::InputComponentType, Self::InputValueType, AC, AV>,
        data: &mut Vec<Self::EncodingType>,
    ) where
        AC: AsRef<[Self::InputComponentType]>,
        AV: AsRef<[Self::InputValueType]>;
}

pub trait DenseVectorEncoder:
    VectorEncoder<
        QueryComponentType = DenseComponent,
        InputComponentType = DenseComponent,
        OutputComponentType = DenseComponent,
    >
{
    /// Wrap a slice of encoded values as an encoded vector view.
    fn encoded_from_slice<'a>(
        &self,
        values: &'a [<Self as VectorEncoder>::OutputValueType],
    ) -> Self::EncodedVectorType<'a>
    where
        Self::EncodedVectorType<'a>: Vector1D<
            Component = DenseComponent,
            Value = <Self as VectorEncoder>::OutputValueType,
        >;

    /// Encode input vectors into output vectors.
    ///
    /// Implementations must append exactly `self.output_dim()` values to `values`.
    fn extend_with_encode<ValueContainer>(
        &self,
        input_vector: DenseVector1D<Self::InputValueType, impl AsRef<[Self::InputValueType]>>,
        values: &mut ValueContainer,
    ) where
        ValueContainer: Extend<Self::OutputValueType>;

    /// Encode an input dense vector into owned output values.
    ///
    /// # Example
    /// ```
    /// use vectorium::{DenseVectorEncoder, DenseVector1D, DotProduct, PlainDenseQuantizer};
    /// use vectorium::{Vector1D, VectorEncoder};
    ///
    /// let encoder = PlainDenseQuantizer::<f32, DotProduct>::new(3);
    /// let v = DenseVector1D::new(vec![1.0_f32, 0.0, 2.0]);
    /// let q = encoder.encode_vector(v);
    /// assert_eq!(q.values_as_slice().len(), 3);
    /// ```
    #[inline]
    fn encode_vector<AV>(
        &self,
        input_vector: DenseVector1D<Self::InputValueType, AV>,
    ) -> DenseVector1D<Self::OutputValueType, Vec<Self::OutputValueType>>
    where
        AV: AsRef<[Self::InputValueType]>,
    {
        let values = input_vector.values_as_slice();
        let mut out_values = Vec::with_capacity(values.len());
        let before_len = out_values.len();

        self.extend_with_encode(DenseVector1D::new(values), &mut out_values);
        assert_eq!(
            out_values.len(),
            before_len + self.output_dim(),
            "DenseVectorEncoder::extend_with_encode must append exactly output_dim() values"
        );

        DenseVector1D::new(out_values)
    }
}

pub trait SparseVectorEncoder: VectorEncoder
{
    /// Encoded vector view used by sparse datasets.
    ///
    /// This is `VectorEncoder::EncodedVectorType` for sparse encoders.
    ///
    /// Note: packed encoders should not implement this trait.

    /// Wrap component and value slices as an encoded vector view.
    fn encoded_from_slices<'a>(
        &self,
        components: &'a [<Self as VectorEncoder>::OutputComponentType],
        values: &'a [<Self as VectorEncoder>::OutputValueType],
    ) -> Self::EncodedVectorType<'a>
    where
        Self::EncodedVectorType<'a>: Vector1D<
            Component = <Self as VectorEncoder>::OutputComponentType,
            Value = <Self as VectorEncoder>::OutputValueType,
        >;

    /// Encode input vectors into output vectors
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

    /// Encode an input sparse vector into owned output components/values.
    ///
    /// # Example
    /// ```
    /// use vectorium::{DotProduct, PlainSparseQuantizer, SparseVectorEncoder, SparseVector1D};
    /// use vectorium::{Vector1D, VectorEncoder};
    ///
    /// let encoder = PlainSparseQuantizer::<u16, f32, DotProduct>::new(5, 5);
    /// let v = SparseVector1D::new(vec![1_u16, 3], vec![1.0, 2.0]);
    /// let q = encoder.encode_vector(v);
    /// assert_eq!(q.components_as_slice(), &[1_u16, 3]);
    /// assert_eq!(q.values_as_slice(), &[1.0, 2.0]);
    /// ```
    #[inline]
    fn encode_vector<AC, AV>(
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
