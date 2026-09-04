# ADR-0003: Structured deterministic fuzzing instead of cargo-fuzz/libFuzzer

- Status: accepted
- Date: 2026-09-04
- Milestone: M0

## Context

T7 requires fuzzing (60 s/target on PRs, 30 min nightly). cargo-fuzz requires
a pinned nightly toolchain and libFuzzer, which conflicts with the
dependency-free policy (ADR-0001) and complicates hermetic CI.

## Decision

`fuzz/` is a normal binary crate (`rhizome-fuzz`) exposing one entry point per
target (shard parser, weights loader, schema parser, grammar compiler, HTTP
request decoding, tokenizer decode). Each target drives the parser with
structured inputs generated from Philox streams (bit-flip, truncation, splice,
and grammar-aware mutations), with an internal depth/size cap. A crash is a
panic, abort, OOM blowup, or a hang beyond the per-input budget. PR CI runs
every target for 60 s; nightly for 30 min. Seeds are pinned so failures
reproduce with `rhizome-fuzz <target> --seed <s> --seconds <n>`.

## Alternatives considered

- cargo-fuzz with vendored libfuzzer-sys: rejected (dependency + nightly).
- No fuzzing: rejected (T7 is mandatory).

## Consequences

- Coverage is mutation-structure-driven rather than coverage-guided; the
  harness compensates with grammar-aware generators.
- Fuzz runs on stable, so it can also run as part of the normal test matrix
  in smoke form.
