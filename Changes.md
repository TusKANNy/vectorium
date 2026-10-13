# Changes on branch `compression`

Running log of changes that matter to crates depending on vectorium (kANNolo, Seismic).
Baseline: `main` at `3fedb2f`. Newest entries go at the top of each section.

## What kANNolo and Seismic must support

Every dataset type below must be buildable from the dependent crate. Each one is built from a
plain sparse dataset with `ConvertFrom` / `.into()` (in Seismic: `convert_dataset_into::<T>()`).

| Dataset type | Components | Values | Conversion status |
|---|---|---|---|
| `BlockedSparseDataset` (jump_dot for queries with < 33 components) | `u16` | `f16`, uncompressed | available (`1569f03`) |
| `DotPacking8U16ScalarU8Dataset` | `u16` | 8-bit per-component scalar | available (0bcdd01) |
| `DotPacking8U16F16Dataset` | `u16` | `f16` | available (0bcdd01) |
| `DotPacking8U16Scalar4BitDataset` | `u16` | 4-bit per-component scalar | available (0bcdd01) |
| `BridgedDotPacking8U32ScalarU8Dataset` | `u32` | 8-bit per-component scalar | available (0bcdd01) |
| `DotPacking8U16FixedU8Dataset` | `u16` | `FixedU8Q` | available (0bcdd01), but FixedU8 is scheduled for removal: do not adopt |

How to use it when updating a dependent crate: read **Breaking changes** first (things that
stop compiling or change results), then **Renames** if the crate used `feature/quantization`,
then **Additions** for what can replace removed pieces.

---

## Breaking changes (relative to `main`)

### Planned, not yet done

| Change | Who is hit | Migration |
|---|---|---|
| **DotVByte removed** (`DotVByteFixedU8Encoder`, `DotVByteFixedU8QueryEvaluator`, the `encoders::dotvbyte_fixedu8` module, `--encoder dotvbyte` in `compute_groundtruth`). Plan: `plans/Plan_Remove_DotVByte.md`. | **Seismic**: `src/bin/build_inverted_index.rs:20,303` and `src/pylib/dotvbyte.rs` (`SeismicIndexDotVByte`) use `PackedSparseDataset<DotVByteFixedU8Encoder>` and `convert_dataset_into::<…>()`. | Replace the type with a DotPacking8 dataset alias, e.g. `DotPacking8U16ScalarU8Dataset` (the conversion trains inside, as DotVByte's did; available since 0bcdd01). |
| **FixedU8 / FixedU16 removed** (`FixedU8Q`, `FixedU16Q` and their users; FixedU8 replaced by ScalarU8 where needed). Not planned in detail yet. | **Seismic**: `src/lib.rs:36` re-exports `FixedU8Q, FixedU16Q`; `src/bin/build_inverted_index.rs:271-280` builds `ScalarSparseDataset<…, FixedU16Q|FixedU8Q, …>`. | To be defined with the plan. |

### Done

| Commit | Change | Who is hit | Migration |
|---|---|---|---|
| `0bcdd01` | `DotPacking8Encoder::train` takes any sparse dataset (`&SparseDatasetGeneric<E, S>` with `E::OutputComponentType = C`, any `ValueType` values, growable or immutable) instead of `&PlainSparseDataset<C, f32, D>`. Same result for the same data. `ComponentMapping::by_decreasing_frequency(dim, components_iter)` takes an iterator of component slices instead of a dataset. | Code that let `train` infer the dataset type, e.g. `let d = growable.into(); enc.train(&d);`, no longer compiles ("type annotations needed"). | Train on the growable dataset directly (`enc.train(&growable)`), or annotate the type. |
| `0db4183` | Component reordering uses **decreasing document frequency** instead of graph bisection; the `rgb` (enhanced-graph-bisection) git dependency is gone. The internal helper `permute_components_with_bisection` became `permute_components_by_frequency` (`pub(crate)`, not visible outside). | Anyone building a DotVByte (or DotPacking8) dataset: the component permutation, hence the encoded bytes and the index size, differ from `main`. Scores do not change (the permutation is applied to queries too). Serialized indexes from `main` still load (the mapping is stored). | None needed. Drop any `rgb`-related patch/feature in the dependent crate. |
| `0db4183` | The plain-sparse → `PackedSparseDataset<DotVByteFixedU8Encoder>` conversion trains the permutation on **all** vectors instead of the first 5% (min 50,000). | Build time of DotVByte conversions (one counting pass over all components; no longer a sample). | None. |
| `7c72295` | `KMeans` assignment and PQ tests use `total_cmp` instead of `partial_cmp().unwrap()`: a NaN distance no longer panics, it sorts last/first per IEEE total order. | Only behaviour on NaN distances. | None. |

---

## Renames for code written against `feature/quantization`

The DotPacking8 encoders on `compression` are type aliases of one generic
`DotPacking8Encoder<C, V>` (`C` = `u16`/`u32` component type, `V` = value codec).
Public names follow `DotPacking8<ComponentType><Quantization>`:

| `feature/quantization` | `compression` |
|---|---|
| `DotPacking8FixedU8Encoder` / `DotPacking8Fixedu8QueryEvaluator` | `DotPacking8U16FixedU8Encoder` / `DotPacking8U16FixedU8QueryEvaluator` |
| `DotPacking8ScalarU8Encoder` / `DotPacking8Scalaru8QueryEvaluator` | `DotPacking8U16ScalarU8Encoder` / `DotPacking8U16ScalarU8QueryEvaluator` |
| `DotPacking8f16Encoder` / `DotPacking8f16QueryEvaluator` | `DotPacking8U16F16Encoder` / `DotPacking8U16F16QueryEvaluator` |
| `DotPacking8Scalar4BitEncoder` / `DotPacking8Scalar4BitQueryEvaluator` | `DotPacking8U16Scalar4BitEncoder` / `DotPacking8U16Scalar4BitQueryEvaluator` |
| `OptimisticDotPacking8U32ScalarU8Encoder` / `…QueryEvaluator` | `BridgedDotPacking8U32ScalarU8Encoder` / `BridgedDotPacking8U32ScalarU8QueryEvaluator` |
| `BlockedSparseEncoder` path "v1" (`compute_distance_v1`) | `compute_distance_jump_dot` (private; public API unchanged) |

API differences besides the names:

- `new(input_dim)` is unchanged.
- `train(&PlainSparseDataset<C, f32, D>)` is generic over the distance `D` (was fixed to
  `SquaredEuclideanDistance`) and trains on **all** vectors (was a 5% sample, min 50,000).
- `train_components(iter)` is gone; `train` does both the permutation and the value
  quantizer. To set a permutation by hand: `set_component_mapping(ComponentMapping::from_permutation(..))`.
- `component_mapping()` / `inverse_component_mapping()` return `Option<&[C]>` as before.
- The quantizer types (`FixedU8Quantizer`, `ScalarU8Quantizer`, `Scalar4BitQuantizer`,
  `Centroid4BitQuantizer`, …) and the view/iterator types (`DotPacking8View`,
  `DotPacking8Iter`, `DotPacking8f16Iter`, …) are no longer public. Value codecs are
  `FixedU8Values`, `ScalarU8Values`, `F16Values`, `Scalar4BitValues` (sealed: no external
  implementations).
- Not ported (exist only on `feature/quantization`): `DotPacking8Centroid4BitEncoder`,
  `CDotPacking8*`, `DotPackingDp*`, `CDotPackingDp*`, `DotPackingSparse*`,
  `DotVByteScalarU8Encoder`, `DotVByteU32*`, `OptimisticDotVByte*`.
- **Serialized DotPacking8 encoders/datasets from `feature/quantization` do not load** on
  `compression` (different struct layout). Rebuild them. Encoded vector bytes are frozen by
  golden tests from commit `699f8b7` on.
- The u32 Bridged encoder: a stored value code of 0 marks a bridge entry; a real component is
  never quantized to code 0 (see `dotpacking8_u32_scalaru8.rs`).

---

## Additions (relative to `main`)

| Commit | Addition |
|---|---|
| `0bcdd01` | `From` / `ConvertFrom<SparseDataset<…>>` for every DotPacking8 dataset (one generic impl over `DotPacking8Encoder<C, V>`). Trains component order and value steps on the input, widens values of any float type to `f32` per vector, and makes no copy of the input. Produces exactly the bytes of `train` + `push`. |
| `1569f03` | `BlockedSparseEncoder`, `BlockedSparseQueryEvaluator`, `BlockedSparseDataset[Growable]`, `DataBlock`, `BLOCK_SIZE`, `SPARSE_QUERY_THRESHOLD` (= 33). Uncompressed `u16`/`f16` sparse vectors in 64-byte blocks of 16 entries. Queries with fewer than 33 components use **jump_dot** (skip every block whose last component is below the current query component); longer queries are densified. `From`/`ConvertFrom<SparseDataset<…, u16, …>>` for `PackedSparseDataset<BlockedSparseEncoder>`. Requires x86_64 with AVX2/F16C/FMA (not checked at run time). |
| `23ded88` … `d3c38f0` | DotPacking8 family: `DotPacking8U16{FixedU8,ScalarU8,F16,Scalar4Bit}Encoder`, `BridgedDotPacking8U32ScalarU8Encoder`, matching `…QueryEvaluator` and `…Dataset[Growable]` aliases. `DotProduct` only. Encode with `PackedSparseDatasetGrowable::new(encoder)` + `push`, after `encoder.train(&plain_dataset)`. No `From<SparseDataset>` conversion yet. |
| `67a3540`, `3f994c5` | `utils::train_sparse_scalar_quantizer`, `utils::train_sparse_scalar_quantizer_with_levels` (per-component scalar quantizer steps), `utils::load128_and_broadcast_to_256` (`unsafe`, SIMD helper). |
| `12211af` | `compute_groundtruth --encoder blocked` (u16 components, dot product): builds a `BlockedSparseDataset` and reports size, time and recall. |
| `4ee61ff`, `3f994c5` | `compute_groundtruth`: `--encoder dotpacking8`, `--component-type u16\|u32`, `--dotpacking8-values fixedu8\|scalaru8\|f16\|scalar4bit`, `--no-train` (fixedu8/f16 only), `--groundtruth <tsv>` (reports recall@k). Invalid combinations exit with code 2. |

## Performance-relevant changes

| Commit | Change |
|---|---|
| `7f76345` | `FlatIndex::search` (used by `Dataset::search`) marks its per-vector closure `#[inline(always)]`. Measured: no slowdown on plain sparse/dense, fixes a +4.3% regression on DotPacking8 F16. |
| — | No nightly feature was added or removed relative to `main`. |
