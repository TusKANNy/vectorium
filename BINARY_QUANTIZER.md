# Binary Quantizer

A 1-bit-per-component encoder for **dense** vectors, with a symmetric popcount
dot product. Compresses each `f32` component to a single sign bit (32× smaller
than `f32`), packs 64 components per `u64`, and scores with bitwise ops.

- Encoder: `src/encoders/binary.rs` — `BinaryQuantizer`, `BinaryQueryEvaluator`
- Re-exports / alias: `src/lib.rs` — `BinaryQuantizer`, `BinaryQueryEvaluator`,
  `pub type BinaryDenseDataset = DenseDataset<BinaryQuantizer>`
- Evaluation CLI: `src/bin/evaluate_binary.rs` (feature `cli`)
- Committed in `b30cc70`.

## How it works

**Training.** `BinaryQuantizer::train(&PlainDenseDataset<f32, DotProduct>)` makes
one pass over the dataset accumulating the **per-component mean**. The means are
the only learned parameter.

**Encoding.** Each component is centered (subtract its mean) and reduced to its
sign: bit `j` is set iff `x[j] - mean[j] >= 0`. Component `j` maps to bit `j & 63`
of word `j >> 6` (little-endian bit and word order). A `d`-dim vector becomes
`d / 64` `u64` words. Storage reuses `DenseDataset` with `OutputValueType = u64`
(stride = `output_dim()` = `d / 64`) — **no new dataset type**.

**Constraint.** `d % 64 == 0` (asserted in `train`); no bit padding for now.

**Symmetric distance.** The query is binarized with the **same** means, then
scored against a stored vector with:

```
u · v = d − 2 · popcount(xor(u, v))
```

This is the dot product of the two ±1 sign vectors (bit `1` → `+1`, bit `0` → `−1`):
agreeing dimensions contribute `+1`, differing contribute `−1`. The result is
surfaced through the existing `DotProduct` distance (larger = better), so it plugs
into `FlatIndex` and the CLIs with no new distance plumbing.
`compute_distance_between` is overridden for allocation-free stored-vs-stored
scoring.

## Construction

Follows the PQ pattern — means need whole-dataset knowledge, so there is **no**
`from_means` constructor and no growable-push path. Build via `ConvertFrom`:

```rust
let plain: PlainDenseDataset<f32, DotProduct> = /* ... */;
let binary: DenseDataset<BinaryQuantizer> = plain.convert_into(); // trains + encodes in parallel
```

## `compute_groundtruth` integration

`--encoder binary` is wired into `src/bin/compute_groundtruth.rs` for dense input.
Requires `--value-type f32` and `--distance dotproduct`; rejects euclidean and
sparse with clear errors.

## `evaluate_binary` — recall + speed harness

`src/bin/evaluate_binary.rs` (feature `cli`) measures both effectiveness and
efficiency of the binary encoder against a pre-computed exact `groundtruth.tsv`.

**Effectiveness.** For each oversampling value `k'`, reports
`recall@k = |binary top-k' ∩ true top-k| / |true top-k|`, macro-averaged over
queries present in the ground truth. This is the ceiling a perfect exact reranker
over the top-`k'` binary candidates could reach — it shows how much oversampling
buys before a rerank stage is worth building. (Metric is the intersection ceiling;
no `f32` vectors are kept resident.)

**Efficiency.** All queries are searched in parallel (Rayon); reports search wall
time, throughput (QPS), and thread count.

**Ground-truth convention.** Each `doc_id` in the TSV is compared **directly**
against the search-result `VectorId` (row index) — no external id mapping. The GT
format is `query_id \t doc_id \t rank \t score`, one row per (query, neighbor),
ranks `1..=k`, with `query_id` the 0-based row index into the query `.npy`.

### CLI

| Flag | Meaning | Default |
|------|---------|---------|
| `-i, --input-file` | documents `.npy` (dense, f32) | required |
| `-q, --query-file` | queries `.npy` (dense, f32) | required |
| `-g, --groundtruth` | exact-search TSV | required |
| `-n, --num-queries` | evaluate first N queries | all |
| `--oversample` | comma-separated `k'` list (each `>= k`) | `10,100,200` |
| `-k, --k` | true-NN cutoff for recall@k | `10` |

`k'` values are validated `>= k` and clamped to the document count.

### Example

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --features cli --bin evaluate_binary

./target/release/evaluate_binary \
  -i /path/to/documents.npy \
  -q /path/to/queries.npy \
  -g /path/to/groundtruth.tsv \
  -n 1000 --oversample 10,100,200
```

Expected output shape:

```
Recall@10 (macro-averaged over N queries):
        k'   recall@10
        10      0.xxxx
       100      0.xxxx
       200      0.xxxx
```

Recall is monotonically non-decreasing in `k'` and reaches ~1.0 as `k'` approaches
the document count (a full candidate set must contain all true NN).

### Caveat — GT id space

The raw-VectorId comparison is correct only if the ground truth was generated with
`doc_id = row index`. If a GT file instead stores **external** ids (e.g. from a
separate `doc_ids.npy` map), recall will be wrong — typically implausibly low even
at large `k'`. That mismatch is the first thing to check on a surprising result;
an optional `--doc-ids` mapping flag could be added if needed.

## Testing

`src/encoders/binary.rs` has 8 unit tests (mean training, encode/decode sign
round-trip, little-endian bit/word packing, hand-computed distances, symmetric-path
agreement, self-retrieval search, serde round-trip, `d % 64` panic).

```bash
cargo test binary                 # unit tests
cargo test --features cli         # includes CLI-gated code
```
