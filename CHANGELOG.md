# Changelog

All notable changes to RHIZOME. Format: Keep a Changelog; versioning: semver.

## [0.1.0] — 2026-09-04 (M0 bootstrap)

### Added
- Cargo workspace with 16 `rhizome-*` crates + xtask + fuzz + benches
  (zero external dependencies; ADR-0001), pinned toolchain 1.97.0.
- `rhizome-core`: typed errors; dtypes incl. bf16/fp8-e4m3/MXFP4/MXINT4
  bit-exact conversions; shape math; Philox-4x32-10 RNG + distributions
  (AS241 quantiles); TOML subset parser/writer; RFC 8259 JSON; BLAKE3 +
  SHA-256/512; tensor storage + MX (de)quantization; versioned config
  schema (model/train/serve) with validation; exact parameter accounting,
  active-parameter and FLOPs estimates, deploy-memory model.
- Configs: test-s/test-m/test-l/edge/standard/flagship + phases presets.
- `rhizome params` CLI subcommand (table/markdown/json).
- `cargo xtask spec-check` (SPEC.md ↔ `rhizome params` sync, R7).
- SPEC.md, docs tree (kernels.md tolerance table, mup.md, determinism.md,
  training.md, serving.md, data.md, weights.md, milestones.md, ADRs
  0001–0012), LICENSE (Apache-2.0), deny.toml.
- CI: ci.yml (fmt/clippy/tests on ubuntu+macos+windows, doc, deny,
  spec-sync, fuzz-smoke, cuda-compile, metal-compile, bench-report),
  nightly.yml, gpu.yml, release.yml (Windows-only artifacts, ADR-0010),
  failure reporting via API comments (ADR-0012).
