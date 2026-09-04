# ADR-0004: Dependency-free benchmark harness with committed baselines

- Status: accepted
- Date: 2026-09-04
- Milestone: M0

## Context

T9 requires PR benchmarks against committed baselines. criterion is a
third-party crate (ADR-0001).

## Decision

`benches/` (`rhizome-bench`) implements a timing harness: per-kernel warmup,
N-iteration loops with `Instant`, min/median/CI reporting, JSON output, and
comparison against `benches/baselines/<runner-class>.json`. The runner class
is detected from CPU brand/family where available. PR CI posts the report as
an artifact (informational); the hard >10% regression gate applies only on
self-hosted runners, matching T9.

## Alternatives considered

- criterion vendored: rejected (ADR-0001).
- `#[bench]` nightly: rejected (nightly requirement).

## Consequences

- Statistical rigor is simpler than criterion's (no outlier classification);
  documented in benches/README.md; baselines are per runner-class.
