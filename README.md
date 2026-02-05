# Vectorium

Vectorium is a Rust library for storing, accessing, and compressing dense and sparse embedding datasets.
The main goal is to provide a unified dataset/encoder interface that can be shared by indexing/search crates such as
[Seismic](https://github.com/TusKANNy/seismic) and [kANNolo](https://github.com/TusKANNy/kannolo).

If you are new to KNN: *exhaustive* KNN searches score every vector in the dataset and return the top‑k closest results.
That is accurate but slow at scale. ANN indexes (HNSW, IVF, Seismic, etc.) trade a bit of accuracy for speed by building extra data structures (e.g., proximity graphs, inverted indexes) on top of the same dataset/encoder primitives.

Vectorium includes an exhaustive search API (`Dataset::search`) and a binary executable for ground-truth computation on CPU. For state‑of‑the‑art ANN indexing, use these tools: [Seismic](https://github.com/TusKANNy/seismic) and [kANNolo](https://github.com/TusKANNy/kannolo).

## Command-line tool
Vectorium ships a binary `compute_groundtruth` (feature `cli`) under `src/bin/` to exhaustive top‑k for a set of queries, writes a TSV file.

### Build and run

This repository currently targets nightly Rust (`rust-toolchain.toml` pins it). For performance‑oriented builds, prefer
`--release` and (optionally) native CPU tuning:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --features cli
```

After building, you will find binary `compute_groundtruth` in `target/release/`.


You can also run via Cargo directly:

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --features cli --bin compute_groundtruth -- --help
```

> Note: `compute_groundtruth` depends on optional CLI dependencies and requires `--features cli`.


This tool computes exhaustive top‑k neighbors for each query and writes the results as a TSV file. It is designed for
research/benchmarking workflows where you need exact results (ground truth) or a baseline to compare against an ANN
index. It parallelizes across queries using Rayon, which is typically the right granularity for CPU ground‑truth runs.

#### Inputs and formats

- `--dataset-type dense` (default): expects `.npy` inputs (dataset + queries). The reader loads `.npy` as `f32` and can
  optionally convert the *dataset* storage via `--value-type`.
- `--dataset-type sparse`: expects Seismic binary format inputs (dataset + queries) via `read_seismic_format`.

#### Output format

Each output line is:

```text
query_id<TAB>doc_id<TAB>rank<TAB>score
```

- `query_id`: 0‑based query index.
- `doc_id`: a `VectorId` (0‑based vector index in the dataset).
- `rank`: `1..=k`.
- `score`:
  - for `--distance euclidean`: squared Euclidean distance (smaller is better);
  - for `--distance dotproduct`: dot product score (larger is better).

#### Examples

Dense vectors stored in `.npy`, Euclidean, PQ:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --features cli
./target/release/compute_groundtruth \
  -i <path>/dataset.npy \
  -q <path>/queries.npy \
  -o sift_gt_l2.tsv \
  --dataset-type dense \
  --value-type f32 \
  --distance euclidean \
  --encoder pq \
  --pq-subspaces 8
```

Dense vectors stored in `.npy`, dot product, plain (no PQ):

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --features cli
./target/release/compute_groundtruth \
  -i <path>/dataset.npy \
  -q <path>/queries.npy \
  -o sift_true_gt_ip.tsv \
  --dataset-type dense \
  --value-type f32 \
  --distance dotproduct \
  --encoder plain
```

Sparse dataset (Seismic binary), dot product, DotVByte compression:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --features cli
./target/release/compute_groundtruth \
  -i <path>/dataset.seismic \
  -q <path>/queries.seismic \
  -o sparse_gt_ip.tsv \
  --dataset-type sparse \
  --component-type u16 \
  --value-type f32 \
  --distance dotproduct \
  --encoder dotvbyte
```

#### Parameters and common combinations

- `-i, --input-file <PATH>`: dataset path (`.npy` when `--dataset-type dense`, Seismic binary when `sparse`).
- `-q, --query-file <PATH>`: query path (same format as the dataset type).
- `-o, --output-path <PATH>`: output TSV path.
- `-k, --k <N>`: number of neighbors to output per query (default: 10).
- `-d, --distance <euclidean|dotproduct>`: scoring metric (default: `euclidean`).
- `-v, --value-type <f32|f16|bf16|fixedu8|fixedu16>`: dataset storage value type (default: `f32`).
  - Dense queries are always `f32`.
  - Sparse queries are always `f32` (dataset values may be quantized).
- `--dataset-type <dense|sparse>`: dataset format (default: `dense`).
- `--component-type <u16|u32>`: sparse component type (default: `u32`).
- `--encoder <plain|pq|dotvbyte>`: encoder (default: `plain`).
  - `pq`: dense‑only, requires `--value-type f32`, and uses `--pq-subspaces`.
  - `dotvbyte`: sparse‑only, requires `--component-type u16` and `--distance dotproduct`.
- `--pq-subspaces <M>`: number of PQ subspaces (`0` auto‑selects a supported value that divides the dataset dimension).
  - Product Quantization splits vectors into `M` equal‑sized chunks, so `dim % M == 0` is required.
  - Supported values are currently `{128, 96, 64, 32, 16, 8, 4}`.


## Rust library

The library is organized around two core concepts:

1. A `Dataset` owns encoded vectors and an encoder.
2. A `VectorEncoder` defines the encoded representation and how to evaluate distances via a `QueryEvaluator`.

Indexes typically store a dataset plus extra search structures. During search they repeatedly need distances between a
query and *candidate vectors* inside the dataset. Vectorium is designed to make that path explicit and efficient: you can
build a query evaluator once, then score many candidate vectors without decoding them first.

### 1) Dense datasets (growable and immutable)

```rust
use vectorium::{Dataset, DenseDataset, DenseVectorView, DotProduct, DatasetGrowable, PlainDenseDatasetGrowable, PlainDenseQuantizer};

let encoder = PlainDenseQuantizer::<f32, DotProduct>::new(3);
let mut dataset = PlainDenseDatasetGrowable::new(encoder);

dataset.push(DenseVectorView::new(&[1.0, 0.0, 2.0]));
dataset.push(DenseVectorView::new(&[0.5, 1.5, 0.0]));

let frozen: DenseDataset<_> = dataset.into();
assert_eq!(frozen.len(), 2);
assert_eq!(frozen.get(0).values(), &[1.0, 0.0, 2.0]);
```

### 2) Sparse datasets (component/value pairs)

```rust
use vectorium::{Dataset, DotProduct, DatasetGrowable, PlainSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView};

let encoder = PlainSparseQuantizer::<u16, f32, DotProduct>::new(5, 5);
let mut dataset = PlainSparseDatasetGrowable::new(encoder);

dataset.push(SparseVectorView::new(&[1_u16, 3], &[1.0, 2.0]));
dataset.push(SparseVectorView::new(&[0_u16, 4], &[0.5, 3.5]));

let frozen: PlainSparseDataset<u16, f32, DotProduct> = dataset.into();
assert_eq!(frozen.len(), 2);
assert_eq!(frozen.get(0).components(), &[1_u16, 3]);
```

### 3) Distance computation with a query evaluator (index-style)

The typical pattern for an index is: build the evaluator once, then score many candidates.

```rust
use vectorium::{Dataset, DatasetGrowable, DenseDataset, DenseVectorView, DotProduct, PlainDenseDatasetGrowable, PlainDenseQuantizer, QueryEvaluator, VectorEncoder, VectorId};

let encoder = PlainDenseQuantizer::<f32, DotProduct>::new(3);
let mut growable = PlainDenseDatasetGrowable::new(encoder);
growable.push(DenseVectorView::new(&[1.0, 0.0, 2.0]));
growable.push(DenseVectorView::new(&[0.5, 1.5, 0.0]));
let dataset: DenseDataset<_> = growable.into();

let query = DenseVectorView::new(&[0.2, 0.1, 0.7]);
let evaluator = dataset.encoder().query_evaluator(query);

let candidate: VectorId = 0;
let score = evaluator.compute_distance(dataset.get(candidate));
let _ = score;
```

### 4) Exhaustive search (top‑k baseline)

If you do not have an ANN index yet, `Dataset::search(query, k)` provides an exhaustive top‑k baseline:

```rust
use vectorium::{Dataset, DatasetGrowable, DenseDataset, DenseVectorView, DotProduct, PlainDenseDatasetGrowable, PlainDenseQuantizer};

let encoder = PlainDenseQuantizer::<f32, DotProduct>::new(3);
let mut growable = PlainDenseDatasetGrowable::new(encoder);
growable.push(DenseVectorView::new(&[1.0, 0.0, 2.0]));
growable.push(DenseVectorView::new(&[0.5, 1.5, 0.0]));
let dataset: DenseDataset<_> = growable.into();

let query = DenseVectorView::new(&[0.2, 0.1, 0.7]);
let top2 = dataset.search(query, 2);
assert_eq!(top2.len(), 2);
```

### 5) Parallel ground-truth computation in Rust (query-level parallelism)

The easiest way to use multiple CPU cores is to parallelize across queries. This is the strategy used by
`compute_groundtruth`.

```rust
use rayon::prelude::*;
use vectorium::{Dataset, DatasetGrowable, DenseVectorView, DotProduct, PlainDenseDatasetGrowable, PlainDenseQuantizer};

let encoder = PlainDenseQuantizer::<f32, DotProduct>::new(3);
let mut growable = PlainDenseDatasetGrowable::new(encoder);
growable.push(DenseVectorView::new(&[1.0, 0.0, 2.0]));
growable.push(DenseVectorView::new(&[0.5, 1.5, 0.0]));

let queries = vec![vec![0.2_f32, 0.1, 0.7], vec![1.0_f32, 0.0, 0.0]];
let results: Vec<_> = queries
    .par_iter()
    .map(|q| growable.search(DenseVectorView::new(q.as_slice()), 1))
    .collect();

assert_eq!(results.len(), 2);
```

### 6) Range-based access and prefetch

Datasets expose `range_from_id`/`id_from_range` so callers can keep lightweight handles to the underlying storage ranges.
This is mainly useful for sparse/packed layouts, where range lookups can be a cache miss and prefetching can help.

```rust
use vectorium::{Dataset, DatasetGrowable, DenseDataset, DenseVectorView, DotProduct, PlainDenseDatasetGrowable, PlainDenseQuantizer};

let encoder = PlainDenseQuantizer::<f32, DotProduct>::new(3);
let mut growable = PlainDenseDatasetGrowable::new(encoder);
growable.push(DenseVectorView::new(&[1.0, 0.0, 2.0]));
let dataset: DenseDataset<_> = growable.into();

let id = 0;
let range = dataset.range_from_id(id);
dataset.prefetch_with_range(range.clone());
let view = dataset.get_with_range(range);
assert_eq!(view.values(), &[1.0, 0.0, 2.0]);
```

## Design notes

- **Type safety by construction:** datasets are sealed and are tied to a specific encoder type, so mixing dense/sparse/packed
  representations is prevented at compile time.
- **Evaluator-driven search:** encoders build a `QueryEvaluator` from the query once; the evaluator can then score many dataset
  vectors. This matches how most ANN indexes are structured internally.
- **Distance ordering:** distances implement `Ord`. `DotProduct` uses reversed ordering (larger is better), while
  `SquaredEuclideanDistance` uses the natural ordering (smaller is better). Distance values must not be NaN.
- **Range-based access:** `range_from_id`/`get_with_range` and `prefetch_with_range` are meant for index implementations that keep
  storage ranges around (especially for sparse/packed datasets) and want predictable, low-overhead access.

## License

MIT. See `LICENSE-MIT`.
