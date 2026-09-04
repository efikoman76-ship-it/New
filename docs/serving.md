# Serving (SPEC 6)

See SPEC.md Section 6 for the normative text. This file collects operational
detail as the engine lands (M7/M8):

- Endpoint reference (HTTP/SSE; ADR-0009).
- Scheduler snapshot struct and replay tooling.
- Prefix-cache snapshot policy (node boundaries + every 4096 tokens).
- Admission control accounting (pages + slabs + Core slabs + Core cache).
- Deterministic mode guarantees (docs/determinism.md).
- `verify` gating: `serve` refuses traffic until conformance has passed on
  this host/hardware class for the current binary hash.
