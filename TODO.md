# TODO

- `decode_vector(...)` should not return `f32`. It should return the encoder `OutputValueType` (decoded but still in the encoder output value domain).
- Conversion/decoding to `f32` should be performed only by `VectorEncoder` (e.g. inside `vector_evaluator(...)`) using a shared helper function, reused by `decode_vector(...)`.

