# Milestones

A milestone is complete only when its acceptance criteria are verified by a
workflow run whose URL is recorded here (R8). Work lands on
`arena/01a06e66-new` (ADR-0011).

## M0 — Bootstrap

**Status: in progress**

Deliverables: workspace, lints, pinned toolchain, deny, xtask, LICENSE,
SPEC.md, ADR template, configs, workflows green on a compiling skeleton;
`rhizome params` computing params/FLOPs/memory for all configs; spec-sync
job passing.

Acceptance: ci.yml green (fmt-clippy, test-ubuntu, test-macos, test-windows,
doc, deny, spec-sync, fuzz-smoke, cuda-compile, metal-compile, bench-report)
and `rhizome params --all --format=markdown` equals the SPEC.md table.

- Workflow run: _(recorded when the first full green run completes)_

## M1 — core + ops + plan

f64 reference for every op; gradient checks; plan (forward/backward
derivation, arena, validation; no fusion); T3.

- Workflow run: — 

## M2 — Dense BPE transformer

MLA + dense FFN end-to-end CPU training and eval-harness basics; T4 copy +
PCFG; bit-exact checkpoint resume; μP coordinate check.

- Workflow run: —

## M3 — Gated Delta Local blocks

5:1 hybrid, chunk scan/step/multistep kernels (scalar + SIMD); T2 parity;
T4 MQAR ablation.

- Workflow run: —

## M4 — MoE

Grouped GEMM, group-limited routing, loss-free balancing, dropless; EP over
`tcp`; T4 load test; T5 EP.

- Workflow run: —

## M5 — Think Core

Injection, three blocks, double-buffered slabs, halting scheduler, scheduled
halting, both read modes; T4 depth monotonicity + calibration; ablation
report in docs/ablations/.

- Workflow run: —

## M6 — Distributed training

Muon, ZeRO-3, PP with Core stage balancing, SP, WSD, EMA; T5 complete.

- Workflow run: —

## M7 — Serve v1

Scheduler, paged latent cache, state slabs, chunked prefill, Core
micro-steps, HTTP API, metrics, deterministic + batch-invariant mode,
warmup/readiness; T6 core subset.

- Workflow run: —

## M8 — Serve v2

Prefix cache with snapshots, spec decode (MTP heads), constrained decoding,
hot swap, admission control, gRPC contract; T6 complete.

- Workflow run: —

## M9 — Quantization

FP8 training (CPU emulation), MX formats, QAT, convert + conformance
report, mmap MXINT4 CPU inference; Edge config end-to-end.

- Workflow run: —

## M10 — PKM + byte-latent path

PKM layers + memory-edit API + locality eval; byte front/back end, patcher,
byte-level spec decode and grammars; ablation vs BPE.

- Workflow run: —

## M11 — CUDA backend

Hot kernels, CUDA graphs, NCCL/RDMA; compile-check on every PR; T10 where
hardware exists.

- Workflow run: —

## M12 — Metal + CubeCL + RL + release

Metal/CubeCL compile-checked backends, GRPO loop with served rollouts,
disaggregated serving, release pipeline (Windows artifacts, ADR-0010), full
docs.

- Workflow run: —
