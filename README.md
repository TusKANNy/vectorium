# Vectorium

Vectorium is a Rust library for storing, compressing, and searching dense and sparse embeddings.
Search here is done witha brute-force parallel scan of the entire dataset. Consider [Seismic](https://github.com/TusKANNy/seismic) and [kANNolo](https://github.com/TusKANNy/kannolo) for faster indexing solutions.

## Quick start

### Dense datasets (growable and immutable)

```rust
use vectorium::{
    Dataset, DenseDataset, DenseVectorView, DotProduct, DatasetGrowable, PlainDenseDatasetGrowable,
    DenseVectorEncoder, PlainDenseQuantizer, VectorEncoder,
};

let encoder = PlainDenseQuantizer::<f32, DotProduct>::new(3);
let mut dataset = PlainDenseDatasetGrowable::new(encoder);
let v0 = vec![1.0, 0.0, 2.0];
let v1 = vec![0.5, 1.5, 0.0];

dataset.push(DenseVectorView::new(v0.as_slice()));
dataset.push(DenseVectorView::new(v1.as_slice()));

let v = dataset.get(0);
assert_eq!(v.values(), &[1.0, 0.0, 2.0]);

let frozen: DenseDataset<_> = dataset.into();
let v = frozen.get(1);
assert_eq!(v.values(), &[0.5, 1.5, 0.0]);
```

### Sparse datasets (growable and immutable)

```rust
use vectorium::{
    Dataset, DotProduct, DatasetGrowable, PlainSparseDataset, PlainSparseDatasetGrowable,
    PlainSparseQuantizer, SparseVectorView, SparseVectorEncoder, VectorEncoder,
};

let encoder = PlainSparseQuantizer::<u16, f32, DotProduct>::new(5, 5);
let mut dataset = PlainSparseDatasetGrowable::new(encoder);

dataset.push(SparseVectorView::new(&[1_u16, 3], &[1.0, 2.0]));
dataset.push(SparseVectorView::new(&[0_u16, 4], &[0.5, 3.5]));

let frozen: PlainSparseDataset<u16, f32, DotProduct> = dataset.into();
let v = frozen.get(0);
assert_eq!(v.components(), &[1_u16, 3]);
assert_eq!(v.values(), &[1.0, 2.0]);
```

## Notes

- `Dataset::range_from_id` returns the slice range for a vector; use that range with `get_by_range`/`prefetch`.
- The growable dataset types (`*Growable`) can be converted into immutable datasets with `into()`.
- `Plain*Quantizer` types are concrete implementations of the `VectorEncoder` trait and the relevant `*VectorEncoder` specialization.
- `DenseData` and `SparseData` are marker traits for dataset categories whose encoders meet richer contracts (`DenseVectorEncoder` and the new `SparseDataEncoder`, respectively). That keeps any helper written for “dense” or “sparse” data honest: it can rely on the layout/query/decoding helpers the encoder exposes rather than guessing at the storage format.
- `SparseDataEncoder` is the minimal shared sparse-input trait that exposes the input/query sparse views, component/value types, and `decode_vector`, so code that only needs the common sparse behavior can stay agnostic about whether the encoder produces component/value pairs or a packed blob.
