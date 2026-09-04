# ADR-0007: Test-L byte-path dimensions are scaled down

- Status: accepted
- Date: 2026-09-04
- Milestone: M0

## Context

SPEC 3.10 defines Test-L (d=256, 2 strata, 8 experts, byte path on) for
nightly CPU runs but does not fix byte-path widths for it. Production
byte-path widths (d_local 512–1024, n-gram tables 3×256K, PKM slots 2^20)
would make nightly CPU ablations cost hours and ~1 GB of RAM for a test
tier whose trunk is 0.04B parameters.

## Decision

Test-L uses d_local = 128, ngram_buckets = 4096, patch window 64, patcher
(2 blocks, d=128), and PKM (8192 slots, 64×32 keys). The architecture and
all code paths are identical; only dimensions differ. Production tiers keep
the specified dimensions.

## Alternatives considered

- Full production byte-path dims in Test-L: rejected on nightly wall-time
  (R5's 30-minute PR budget shapes nightly expectations too).
- No byte path in Test-L: rejected (T4/nightly requires byte-path ablation
  coverage).

## Consequences

- Byte-path parity/ablation tests run nightly on CPU within budget;
  numerical behavior at production widths is covered by unit tests with
  synthetic shapes.
