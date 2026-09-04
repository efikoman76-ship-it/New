# Contributing to RHIZOME

- Priority order: correctness > reproducibility > memory safety >
  performance > portability.
- `SPEC.md` is the source of truth; deviations need an ADR
  (`docs/adr/NNNN-*.md`, template in 0000). Never loosen a tolerance,
  delete a test, or add a retry without an ADR (R12).
- CI must be green before review: `cargo fmt --all`, `cargo clippy
  --workspace --all-targets -- -D warnings`, `cargo test --workspace`,
  `cargo xtask spec-check`.
- No `unsafe` outside R6 crates; every unsafe block carries `// SAFETY:`.
- No `todo!`/`unimplemented!`/`dbg!`/commented-out code; tests needing
  unavailable hardware call `skip_if_unavailable!(reason)`.
- Conventional commits; keep PRs ≲2K-line diffs where feasible; update
  CHANGELOG.md and docs/milestones.md in the same PR as the milestone.
- Branch protection on `main` requires ci.yml where repository settings
  permit; if protection cannot be enabled by automation, the manual policy
  is: maintainers merge only from a green PR whose head is
  `arena/01a06e66-new` or a descendant (ADR-0011).
