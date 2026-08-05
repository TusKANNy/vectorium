//! Binary dense encoder: one bit per component with symmetric dot-product distance.
//!
//! Each component is centered by subtracting a per-component mean (learned over the
//! whole dataset) and reduced to its sign: 1 bit. Bits are packed 64 per `u64`, so a
//! `d`-dimensional vector becomes a stream of `d / 64` `u64` words.
//!
//! Distance is *symmetric*: the query is binarized with the **same** per-component means,
//! then scored with bitwise ops. Interpreting each stored bit as `+1` (set) or `-1` (clear),
//! the dot product of two sign vectors of length `d` is
//!
//! ```text
//! u · v = d - 2 * popcount(xor(u, v))
//! ```
//!
//! because dimensions where the bits agree contribute `+1` and dimensions where they differ
//! (counted by `popcount(xor(..))`) contribute `-1`. The score is surfaced through the
//! existing [`DotProduct`] distance (larger is better).
//!
//! For now only dimensions that are a multiple of 64 are supported (no bit-padding).
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::core::distances::DotProduct;
use crate::core::vector::{DenseVectorOwned, DenseVectorView};
use crate::core::vector_encoder::{DenseVectorEncoder, QueryEvaluator, VectorEncoder};
use crate::encoders::rabitq_common::{hamming, hamming_batch6};
use crate::{Dataset, PlainDenseDataset, SpaceUsage};

/// Number of bits packed into a single `u64` word.
const WORD_BITS: usize = 64;

/// A binary quantizer that stores the sign of each mean-centered component as one bit.
///
/// The per-component `means` are learned from a dataset via [`BinaryQuantizer::train`]
/// (see the module-level docs). Distance is hard-wired to [`DotProduct`] because the
/// popcount score is inherently a sign-vector dot product.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryQuantizer {
    /// Input dimensionality (asserted to be a multiple of 64).
    d: usize,
    /// Per-component means used to center vectors before taking the sign; length `d`.
    means: Box<[f32]>,
}

impl BinaryQuantizer {
    /// Learn the per-component means over `dataset` and build a quantizer.
    ///
    /// Performs a single pass accumulating each component, then divides by the number of
    /// vectors. Panics unless the dataset dimension is a multiple of 64.
    pub fn train(dataset: &PlainDenseDataset<f32, DotProduct>) -> Self {
        let d = dataset.input_dim();
        assert!(
            d.is_multiple_of(WORD_BITS),
            "BinaryQuantizer requires dim % 64 == 0, got {d}"
        );

        let mut means = vec![0.0f32; d];
        for vector in dataset.iter() {
            for (m, &v) in means.iter_mut().zip(vector.values()) {
                *m += v;
            }
        }

        let n = dataset.len();
        if n > 0 {
            let inv = 1.0 / n as f32;
            for m in means.iter_mut() {
                *m *= inv;
            }
        }

        Self {
            d,
            means: means.into_boxed_slice(),
        }
    }

    /// Number of `u64` words needed to store one encoded vector.
    #[inline]
    fn num_words(&self) -> usize {
        self.d / WORD_BITS
    }

    /// Pack word `w` (64 components starting at `w * 64`) of a centered vector.
    ///
    /// Bit `i` is set when `values[w*64 + i] >= means[w*64 + i]` (i.e. the centered
    /// component is non-negative). `values` must have length `d`.
    #[inline]
    fn pack_word(&self, values: &[f32], w: usize) -> u64 {
        let base = w * WORD_BITS;
        let mut word = 0u64;
        for i in 0..WORD_BITS {
            if values[base + i] >= self.means[base + i] {
                word |= 1u64 << i;
            }
        }
        word
    }

    /// Binarize a full `f32` vector into `num_words()` packed `u64` words.
    #[inline]
    fn pack(&self, values: &[f32]) -> Vec<u64> {
        (0..self.num_words())
            .map(|w| self.pack_word(values, w))
            .collect()
    }
}

/// Dot product of two packed binary vectors as `d - 2 * popcount(xor)`, wrapped in [`DotProduct`].
///
/// The XOR+popcount itself comes from `hamming`, shared with the RaBitQ encoders: this quantizer
/// is the plain-BQ **baseline** those are measured against, so it has to reach the same scan kernel
/// — a scalar popcount loop here would make the baseline look slow for reasons that have nothing to
/// do with the coding scheme.
#[inline]
fn binary_dot(d: usize, a: &[u64], b: &[u64]) -> DotProduct {
    DotProduct((d as i64 - 2 * hamming(a, b) as i64) as f32)
}

/// Evaluator holding an owned packed query; scores packed `u64` vectors via xor+popcount.
#[derive(Debug, Clone)]
pub struct BinaryQueryEvaluator<'e> {
    _encoder: PhantomData<&'e BinaryQuantizer>,
    /// Packed query words (owned so the evaluator does not borrow the query).
    query_words: Vec<u64>,
    /// Dimensionality `d` (number of bits), needed for the `d - 2*popcount` score.
    d: usize,
}

impl<'e, 'v> QueryEvaluator<DenseVectorView<'v, u64>> for BinaryQueryEvaluator<'e> {
    type Distance = DotProduct;

    #[inline]
    fn compute_distance(&self, vector: DenseVectorView<'v, u64>) -> DotProduct {
        binary_dot(self.d, &self.query_words, vector.values())
    }

    /// Fused six-candidate scan (`hamming_batch6`): the query is loaded once per chunk and
    /// interleaved against six documents' independent popcount accumulators. Same reason as
    /// `binary_dot` — the baseline gets the same batch kernel the RaBitQ scan uses. The combine
    /// matches [`Self::compute_distance`] operation for operation, so the scores are bit-identical.
    #[inline]
    fn compute_distances_batch6(&self, vectors: [DenseVectorView<'v, u64>; 6]) -> [DotProduct; 6] {
        let codes: [&[u64]; 6] = vectors.map(|v| v.values());
        hamming_batch6(&self.query_words, codes)
            .map(|h| DotProduct((self.d as i64 - 2 * h as i64) as f32))
    }
}

impl DenseVectorEncoder for BinaryQuantizer {
    type InputValueType = f32;
    type OutputValueType = u64;

    /// Decode a packed vector into `±1` `f32` values (set bit → `+1.0`, clear bit → `-1.0`).
    ///
    /// Lossy, as expected for binary quantization, but required by the trait.
    fn decode_vector<'a>(&self, encoded: DenseVectorView<'a, u64>) -> DenseVectorOwned<f32> {
        let mut values = Vec::with_capacity(self.d);
        for &word in encoded.values() {
            for i in 0..WORD_BITS {
                let set = (word >> i) & 1 == 1;
                values.push(if set { 1.0f32 } else { -1.0f32 });
            }
        }
        DenseVectorOwned::new(values)
    }

    #[inline]
    fn push_encoded<'a, OutputContainer>(
        &self,
        input: DenseVectorView<'a, f32>,
        output: &mut OutputContainer,
    ) where
        OutputContainer: Extend<u64>,
    {
        assert_eq!(
            input.len(),
            self.d,
            "Input vector length must equal encoder input dimension."
        );
        let values = input.values();
        output.extend((0..self.num_words()).map(|w| self.pack_word(values, w)));
    }
}

impl VectorEncoder for BinaryQuantizer {
    type Distance = DotProduct;
    type InputVector<'a> = DenseVectorView<'a, f32>;
    type QueryVector<'q> = DenseVectorView<'q, f32>;
    type EncodedVector<'a> = DenseVectorView<'a, u64>;

    type Evaluator<'e>
        = BinaryQueryEvaluator<'e>
    where
        Self: 'e;

    /// Build an evaluator by binarizing the `f32` query with the stored means.
    #[inline]
    fn query_evaluator<'e>(&'e self, query: Self::QueryVector<'_>) -> Self::Evaluator<'e> {
        assert_eq!(
            query.len(),
            self.d,
            "Query vector length must equal encoder input dimension."
        );
        BinaryQueryEvaluator {
            _encoder: PhantomData,
            query_words: self.pack(query.values()),
            d: self.d,
        }
    }

    /// Build an evaluator from an already-encoded dataset vector (copy its packed words).
    #[inline]
    fn vector_evaluator<'e, 'v>(&'e self, vector: Self::EncodedVector<'v>) -> Self::Evaluator<'e> {
        BinaryQueryEvaluator {
            _encoder: PhantomData,
            query_words: vector.values().to_vec(),
            d: self.d,
        }
    }

    fn input_dim(&self) -> usize {
        self.d
    }

    fn output_dim(&self) -> usize {
        self.num_words()
    }

    /// Score two stored packed vectors directly, without building an evaluator.
    #[inline]
    fn compute_distance_between(
        &self,
        v1: Self::EncodedVector<'_>,
        v2: Self::EncodedVector<'_>,
    ) -> Self::Distance {
        binary_dot(self.d, v1.values(), v2.values())
    }
}

impl SpaceUsage for BinaryQuantizer {
    fn space_usage_bytes(&self) -> usize {
        self.d.space_usage_bytes() + self.means.space_usage_bytes()
    }
}

use crate::dataset::ConvertFrom;

/// Train a [`BinaryQuantizer`] over the dataset and encode every vector in parallel.
impl ConvertFrom<PlainDenseDataset<f32, DotProduct>> for crate::DenseDataset<BinaryQuantizer> {
    fn convert_from(dataset: PlainDenseDataset<f32, DotProduct>) -> Self {
        let encoder = BinaryQuantizer::train(&dataset);
        crate::DenseDataset::<BinaryQuantizer>::from_flat_par(
            encoder,
            dataset.values(),
            dataset.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::ConvertInto;
    use crate::{
        DatasetGrowable, DenseDataset, FlatIndex, Index, IndexSerializer,
        PlainDenseDatasetGrowable, PlainDenseQuantizer,
    };

    /// Build a `PlainDenseDataset<f32, DotProduct>` from a list of equal-length vectors.
    fn plain_dataset(vectors: &[Vec<f32>]) -> PlainDenseDataset<f32, DotProduct> {
        let d = vectors[0].len();
        let encoder = PlainDenseQuantizer::<f32, DotProduct>::new(d);
        let mut growable = PlainDenseDatasetGrowable::new(encoder);
        for v in vectors {
            growable.push(DenseVectorView::new(v));
        }
        growable.into()
    }

    #[test]
    fn train_computes_per_component_means() {
        let zeros = vec![0.0f32; 64];
        let twos = vec![2.0f32; 64];
        let dataset = plain_dataset(&[zeros, twos]);

        let enc = BinaryQuantizer::train(&dataset);
        assert_eq!(enc.means.len(), 64);
        assert!(enc.means.iter().all(|&m| (m - 1.0).abs() < 1e-6));
    }

    #[test]
    fn encode_and_decode_round_trip_signs() {
        // means = 1.0 everywhere. twos -> all bits set; zeros -> all bits clear.
        let dataset = plain_dataset(&[vec![0.0f32; 64], vec![2.0f32; 64]]);
        let enc = BinaryQuantizer::train(&dataset);

        let all_set = enc.encode_vector(DenseVectorView::new(&vec![2.0f32; 64]));
        assert_eq!(all_set.values(), &[u64::MAX]);
        let decoded = enc.decode_vector(all_set.as_view());
        assert!(decoded.values().iter().all(|&v| v == 1.0));

        let all_clear = enc.encode_vector(DenseVectorView::new(&vec![0.0f32; 64]));
        assert_eq!(all_clear.values(), &[0u64]);
        let decoded = enc.decode_vector(all_clear.as_view());
        assert!(decoded.values().iter().all(|&v| v == -1.0));
    }

    #[test]
    fn packing_uses_little_endian_bit_and_word_order() {
        // means = 0: bit set iff component >= 0. d = 128 -> two words.
        let enc = BinaryQuantizer {
            d: 128,
            means: vec![0.0f32; 128].into_boxed_slice(),
        };
        let mut values = vec![-1.0f32; 128];
        values[0] = 1.0; // bit 0 of word 0
        values[64] = 1.0; // bit 0 of word 1
        let encoded = enc.encode_vector(DenseVectorView::new(&values));
        assert_eq!(encoded.values(), &[1u64, 1u64]);
    }

    #[test]
    fn distance_matches_hand_computed_cases() {
        let enc = BinaryQuantizer {
            d: 64,
            means: vec![0.0f32; 64].into_boxed_slice(),
        };
        let all_set = DenseVectorView::new(&[u64::MAX]);
        let all_clear = DenseVectorView::new(&[0u64]);
        let one_off = DenseVectorView::new(&[u64::MAX ^ 1]); // one differing bit

        // identical -> +d
        assert_eq!(
            enc.compute_distance_between(all_set, all_set),
            DotProduct(64.0)
        );
        // bitwise opposite -> -d
        assert_eq!(
            enc.compute_distance_between(all_set, all_clear),
            DotProduct(-64.0)
        );
        // one differing bit -> d - 2
        assert_eq!(
            enc.compute_distance_between(all_set, one_off),
            DotProduct(62.0)
        );
    }

    #[test]
    fn symmetric_paths_agree() {
        let dataset = plain_dataset(&[vec![0.0f32; 64], vec![2.0f32; 64]]);
        let enc = BinaryQuantizer::train(&dataset);

        let query = vec![2.0f32; 64];
        let query_f32 = DenseVectorView::new(&query);
        let encoded = enc.encode_vector(query_f32); // same vector, binarized

        let via_query = enc
            .query_evaluator(query_f32)
            .compute_distance(encoded.as_view());
        let via_vector = enc
            .vector_evaluator(encoded.as_view())
            .compute_distance(encoded.as_view());
        let direct = enc.compute_distance_between(encoded.as_view(), encoded.as_view());

        assert_eq!(via_query, DotProduct(64.0));
        assert_eq!(via_query, via_vector);
        assert_eq!(via_query, direct);
    }

    #[test]
    fn search_finds_self_as_nearest() {
        // Three well-separated 64-d vectors; each query should retrieve itself first.
        let mut a = vec![-1.0f32; 64];
        a[..20].fill(1.0);
        let mut b = vec![-1.0f32; 64];
        b[20..44].fill(1.0);
        let mut c = vec![-1.0f32; 64];
        c[44..].fill(1.0);
        let dataset = plain_dataset(&[a.clone(), b.clone(), c.clone()]);
        let bin: DenseDataset<BinaryQuantizer> = dataset.convert_into();

        for (i, q) in [a, b, c].iter().enumerate() {
            let top = FlatIndex::from(&bin).search(DenseVectorView::new(q), 1, &());
            assert_eq!(top[0].vector as usize, i);
        }
    }

    #[test]
    fn batch6_matches_six_singles() {
        // d = 640 → 10 words: exercises the 8-wide chunk loop *and* the 2-word tail of the shared
        // batch kernel. The fused scan must be bit-identical to six separate calls.
        let vectors: Vec<Vec<f32>> = (0..6)
            .map(|k| {
                (0..640)
                    .map(|i| ((i * (k + 3)) as f32 * 0.13).sin())
                    .collect()
            })
            .collect();
        let dataset = plain_dataset(&vectors);
        let bin: DenseDataset<BinaryQuantizer> = dataset.convert_into();
        for q in &vectors {
            let evaluator = bin.encoder().query_evaluator(DenseVectorView::new(q));
            let views = std::array::from_fn(|k| bin.get(k as u64));
            let batch = evaluator.compute_distances_batch6(views);
            let singles = std::array::from_fn(|k| evaluator.compute_distance(bin.get(k as u64)));
            assert_eq!(batch, singles);
        }
    }

    #[test]
    fn dataset_serialization_round_trip() {
        let dataset = plain_dataset(&[vec![1.0f32; 64], vec![-1.0f32; 64], vec![0.5f32; 64]]);
        let bin: DenseDataset<BinaryQuantizer> = dataset.convert_into();

        let mut path = std::env::temp_dir();
        path.push(format!("vectorium_binary_{}.bin", std::process::id()));
        let path = path.to_str().unwrap().to_string();

        bin.save_index(&path).unwrap();
        let loaded = DenseDataset::<BinaryQuantizer>::load_index(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(bin, loaded);
    }

    #[test]
    #[should_panic(expected = "dim % 64 == 0")]
    fn train_rejects_non_multiple_of_64() {
        let dataset = plain_dataset(&[vec![0.0f32; 10]]);
        let _ = BinaryQuantizer::train(&dataset);
    }
}
