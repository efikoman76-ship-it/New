# Determinism (R9 / SPEC 6.8)

Deterministic mode (`deterministic = true`, default in Test configs and
serving) guarantees:

1. Bit-identical outputs across runs on the same hardware class, regardless
   of thread count.
2. Batch invariance: a sequence's outputs do not depend on which other
   sequences share its batch.

## Mechanisms

- All randomness flows through Philox-4x32-10 keyed by
  (seed, stream_id, step, index) — never through global state or thread
  scheduling (R11).
- Reductions use fixed tree shapes and loop orders written in the kernels;
  the thread pool splits work by output ranges, never by atomic reduction.
- Decode kernels (mla_decode split-K, grouped GEMV) tile per sequence; a
  sequence's accumulation order is a function of its own length only.
- Top-k ties break to the lower index everywhere (router, PKM, sampling).
- Expert dispatch ordering is (expert id, token id).
- Halt threshold τ is fixed (0.5) in deterministic mode.
- Speculative decoding is forced off with greedy sampling (asserted).

## Kernels that required determinism-specific variants

Recorded here as they land (M3+): sequential-vs-parallel scan carry order,
softmax max-subtraction tree shape, split-K partial ordering.

## Tests

- `determinism` suite: run the same plan 3× with different thread counts and
  shuffled batch composition; assert bit-identical outputs (T2).
