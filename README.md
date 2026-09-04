# RHIZOME

A production-grade neural language-model architecture and its complete
training and inference stack, written in dependency-free Rust.

- Architecture: 5:1 hybrid of Gated Delta-Rule Local blocks and Multi-head
  Latent Attention Global blocks, MoE (group-limited routing) with PKM
  memory blocks, and a weight-shared adaptive-depth **Think Core**.
- Training: static execution plans, Muon + AdamW, WSD schedule, μP,
  ZeRO-3 / expert / pipeline / sequence parallelism over a pure-Rust TCP
  collective (CI-testable), bit-exact checkpoints.
- Serving: continuous batching, chunked prefill, paged latent KV + state
  slabs, prefix cache, speculative decoding, grammar-constrained sampling,
  deterministic mode, HTTP/SSE OpenAI-compatible API.
- Quantization: bit-exact FP8/MXFP4/MXINT4 emulation on CPU; deploy
  conversion with conformance gating.

See `SPEC.md` (source of truth), `docs/` (ADRs, kernels, milestones), and
`configs/` (test/edge/standard/flagship tiers). CI: `.github/workflows/`.

Quick start:

```sh
cargo run --bin rhizome -- params --all --format table
cargo test --workspace
cargo xtask spec-check
```
