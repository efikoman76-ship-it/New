# ADR-0001: Zero-dependency workspace

- Status: accepted
- Date: 2026-09-04
- Milestone: M0

## Context

The build prompt (R1) names third-party crates indirectly (`serde`, `thiserror`,
`axum`, `tonic`, `cudarc`, `criterion`, `cargo-fuzz`, `proptest`). The
engineering environment for this build has no access to `static.rust-lang.org`
or `crates.io` (registry blocked at the network layer), and R5 requires CI to
be hermetic with no network during test steps. Vendoring crates.io dependencies
into git is possible in principle but is explicitly avoided: it would add tens
of megabytes of third-party code of unaudited provenance to the repository and
make the hermetic story dependent on vendor-tree maintenance.

## Decision

The workspace has **zero external crate dependencies**. Every concern the
prompt assigns to a third-party crate is implemented in-repo:

| Concern | Replacement |
|---|---|
| serde (TOML) | `rhizome-core::toml` (subset parser, Section: crates/core/src/toml.rs) |
| serde (JSON) | `rhizome-core::json` (RFC 8259 parser/writer) |
| thiserror | hand-written `RhizomeError` enum + `Display`/`Error` impls |
| axum | hand-rolled HTTP/1.1 + SSE server on `std::net` (ADR-0009) |
| tonic | see ADR-0009 |
| rand | Philox-4x32-10 (R11) in `rhizome-core::philox`/`rng` |
| blake3/sha2 | `rhizome-core::hash` (official test vectors embedded) |
| criterion | `benches/` harness with committed baselines (ADR-0004) |
| cargo-fuzz | structured deterministic harnesses (ADR-0003) |
| proptest | Philox-seeded property-test helpers in-crate |
| cudarc | narrow FFI in `rhizome-kernels-cuda` (feature `cuda`) |

Feature flags (`cuda`, `metal`, `nccl`, `rdma`, `cubecl`, `serve-grpc`,
`xcheck`) keep their names and semantics; enabling them requires the
corresponding native toolchain, which CI compile-checks where available.
`cubecl` specifically cannot be vendored, so `rhizome-kernels-cubecl` provides
the portable long-tail kernels itself under the same feature contract.

## Alternatives considered

- Vendor all crates.io dependencies. Rejected: provenance, size, and the
  hermetic requirement is better served by owning the (small) surface we need.
- Implement only in CI with network. Rejected: violates R5 (tests must not
  need network).

## Consequences

- Cargo.lock contains only workspace members; `cargo-deny` remains a guard.
- Every hand-rolled component carries authoritative test vectors generated
  from reference implementations (Python cross-checks were used offline to
  produce them; the vectors themselves are committed).
- Upgrading to real dependencies later requires an ADR superseding this one.
