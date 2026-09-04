# Weights (SPEC 7)

## Container (v1)

safetensors-compatible JSON header (little-endian length-prefixed) plus an
extension section:

- MX block layouts (kind, block size, padded last dim),
- per-tensor BLAKE3,
- embedded model schema (TOML) and schema hash,
- tokenizer/patcher hash,
- format version,
- ed25519-signed manifest over the sorted tensor table.

## Rules

- Loader never executes code from files; parser fuzzed (T7).
- The CPU backend reads tensors directly from the mapped/pread file
  (ADR-0008) without copying when dtype permits.
- `rhizome convert` produces deploy formats with calibration and a
  conformance report (perplexity delta, per-layer error) that gates release.
- `rhizome inspect` prints schema, shapes, formats, checksums.
- Checkpoints (training) share the tensor codec with extra sections
  (optimizer state, cursors, RNG streams, router biases, halting-scheduler
  state, EMA).
