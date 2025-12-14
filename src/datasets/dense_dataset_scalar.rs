//! Conversion implementations for DenseDataset using ScalarDenseQuantizer.
//! Provides methods to convert datasets between different scalar value types.

use crate::datasets::Dataset;
use crate::quantizers::Quantizer;
use crate::{DenseDataset, DenseVector1D, ScalarDenseQuantizer};
use crate::{Float, ScalarDenseSupportedDistance, ValueType};

/// Convert a DenseDataset with ScalarDenseQuantizer<SrcIn, SrcOut, D> to
/// DenseDataset<ScalarDenseQuantizer<SrcOut, Out, D>, Vec<Out>>.
///
/// The source dataset's output type (SrcOut) must match the target quantizer's
/// input type (In).
impl<In, Out, D> DenseDataset<ScalarDenseQuantizer<In, Out, D>, Vec<Out>>
where
    In: ValueType + Float,
    Out: ValueType + Float,
    D: ScalarDenseSupportedDistance,
{
    /// Convert a source dataset into this dataset type.
    ///
    /// Iterates over the source dataset vector by vector, encodes each vector
    /// with ScalarDenseQuantizer<In, Out, D> into a preallocated Vec.
    pub fn convert<SrcIn, SrcData>(
        source: &DenseDataset<ScalarDenseQuantizer<SrcIn, In, D>, SrcData>,
    ) -> Self
    where
        SrcIn: ValueType + Float,
        SrcData: AsRef<[In]>,
    {
        let (n_vecs, d) = source.shape();
        let quantizer: ScalarDenseQuantizer<In, Out, D> = ScalarDenseQuantizer::new(d);
        let m = quantizer.m();

        // Preallocate output buffer
        let mut output_data: Vec<Out> = vec![Out::default(); n_vecs * m];

        // Iterate vector by vector and encode
        for (i, src_vec) in source.iter().enumerate() {
            let start = i * m;
            let end = start + m;
            let mut out_slice = DenseVector1D::new(&mut output_data[start..end]);

            quantizer.encode(src_vec, &mut out_slice);
        }

        DenseDataset::from_raw(output_data, n_vecs, d, quantizer)
    }
}
