# Vectorium

Vectorium is a Rust library for storing, compressing, and searching dense and sparse embeddings.
Search here is done witha brute-force parallel scan of the entire dataset. Consider [Seismic](https://github.com/TusKANNy/seismic) and [kANNolo](https://github.com/TusKANNy/kannolo) for faster indexing solutions.

## Quick start

### Dense datasets (growable and immutable)

```rust
use vectorium::{
    Dataset, DenseDataset, DenseVector1D, DotProduct, GrowableDataset, PlainDenseDatasetGrowable,
    DenseVectorEncoder, PlainDenseQuantizer, Vector1D, VectorEncoder,
};

let encoder = PlainDenseQuantizer::<f32, DotProduct>::new(3);
let mut dataset = PlainDenseDatasetGrowable::new(encoder);

dataset.push(DenseVector1D::new(vec![1.0, 0.0, 2.0]));
dataset.push(DenseVector1D::new(vec![0.5, 1.5, 0.0]));

let v = dataset.get(0);
assert_eq!(v.values_as_slice(), &[1.0, 0.0, 2.0]);

let frozen: DenseDataset<_> = dataset.into();
let v = frozen.get(1);
assert_eq!(v.values_as_slice(), &[0.5, 1.5, 0.0]);
```

### Sparse datasets (growable and immutable)

```rust
use vectorium::{
    Dataset, DotProduct, GrowableDataset, PlainSparseDataset, PlainSparseDatasetGrowable,
    PlainSparseQuantizer, SparseVector1D, SparseVectorEncoder, Vector1D, VectorEncoder,
};

let encoder = PlainSparseQuantizer::<u16, f32, DotProduct>::new(5, 5);
let mut dataset = PlainSparseDatasetGrowable::new(encoder);

dataset.push(SparseVector1D::new(vec![1_u16, 3], vec![1.0, 2.0]));
dataset.push(SparseVector1D::new(vec![0_u16, 4], vec![0.5, 3.5]));

let frozen: PlainSparseDataset<u16, f32, DotProduct> = dataset.into();
let v = frozen.get(0);
assert_eq!(v.components_as_slice(), &[1_u16, 3]);
assert_eq!(v.values_as_slice(), &[1.0, 2.0]);
```

## Notes

- `Dataset::range_from_id` returns the slice range for a vector; use that range with `get_by_range`/`prefetch`.
- The growable dataset types (`*Growable`) can be converted into immutable datasets with `into()`.
- `Plain*Quantizer` types are concrete implementations of the `VectorEncoder` trait and the relevant `*VectorEncoder` specialization.
