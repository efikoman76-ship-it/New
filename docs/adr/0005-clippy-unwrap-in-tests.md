# ADR-0005: `unwrap_used`/`expect_used` denied in libraries, allowed in `#[cfg(test)]`

- Status: accepted
- Date: 2026-09-04
- Milestone: M0

## Context

R2 requires `clippy::unwrap_used` and `clippy::expect_used` = deny for
library code. Tests routinely assert on invariants where `unwrap` is the
idiomatic expression of "this cannot fail here"; denying it in test code
produces large amounts of `let x = match ...` noise with zero safety gain,
since a panicking test is a failing test by definition.

## Decision

Every library crate's root carries

```rust
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
```

Production (non-test) code paths remain fully denied. This is a scoping
decision, not a tolerance loosening: no production `unwrap` exists.

## Alternatives considered

- Deny everywhere and write `expect` with justifications in every test:
  rejected; adds hundreds of no-op comments without changing behavior.
- Allow per-function in tests: rejected; noisy and easy to get wrong.

## Consequences

- CI clippy runs `--all-targets` so tests are still linted for everything
  else (`todo`, `dbg_macro`, etc. remain denied in tests).
