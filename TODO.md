# TODO

- `decode_vector(...)` should not return `f32`. It should return the encoder `OutputValueType` (decoded but still in the encoder output value domain).
- Conversion/decoding to `f32` should be performed only by `VectorEncoder` (e.g. inside `vector_evaluator(...)`) using a shared helper function, reused by `decode_vector(...)`.

- **Assess whether `ValueFor<S>: ToPrimitive` / `vectorium::FromF32` can be replaced with a single conversion trait (e.g., `ScoreValue: Quantizable`)** so the code doesn't need to refer to two distinct traits depending on direction (pruning vs summarization).

- 