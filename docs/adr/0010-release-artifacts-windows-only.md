# ADR-0010: Release workflow builds Windows binaries only

- Status: accepted
- Date: 2026-09-04
- Milestone: M0 (release.yml)

## Context

The repository owner's build instruction states: "if you decide to make a
build in GitHub workflow only build for windows." Standard cross-platform
release matrices (linux x86_64/aarch64, macOS arm64) are therefore not
wanted for artifact builds, regardless of the default release plan in the
build prompt.

## Decision

`release.yml` produces signed release artifacts for
`x86_64-pc-windows-msvc` only (the `rhizome` CLI: train/serve/convert/eval
and friends). Tests still run on ubuntu/macos/windows in ci.yml — the
restriction applies to release *artifact builds*, not test execution.

## Consequences

- Release artifacts run on Windows hosts and WSL.
- Linux/macOS production users build from source (`cargo build --release`),
  which CI compile-checks on every PR anyway.
- Signed manifests (ed25519) and the conformance report attach to the
  Windows artifacts.
