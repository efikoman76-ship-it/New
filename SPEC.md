# RHIZOME — Specification (living document)

RHIZOME is a production-grade neural language model architecture plus its
complete training and inference stack, written in Rust. This file is the
source of truth for the implementation (R7). Deviations from the original
build prompt are recorded as ADRs in `docs/adr/`.

Priority order when goals conflict:
**correctness > reproducibility > memory safety > performance > portability.**

Status legend: each section lists the milestone that lands it
(`docs/milestones.md`).

## 1. Glossary

| Term | Meaning |
|---|---|
| block | one residual unit = mixer (optional) + FFN, with pre-norm and sandwich norm |
| stratum | 5 Local blocks followed by 1 Global block |
| Local block | Gated Delta-Rule mixer + FFN |
| Global block | Multi-head Latent Attention (MLA) + FFN |
| FFN | MoE (shared + routed experts), PKM in designated blocks, or dense SwiGLU where stated |
| Think Core | a weight-shared 3-block unit executed 1..L_max times per token |
| prelude / coda | trunk strata before / after the Think Core |
| token | a BPE token (BPE path) or a byte patch (byte-latent path) |
| segment | one document inside a packed sequence; state resets at segments |
| plan | a static, serialized execution graph for one (phase, shape) |
| op | one member of the closed op vocabulary (Section 4) |
| Test config | tiny configs (d = 64..256) used in CI |

## 2. Non-negotiable rules

- **R1** Production code is Rust (pinned stable toolchain, `rust-toolchain.toml`).
  Hot GPU kernels may be CUDA C++/PTX or MSL behind narrow FFI. Python only
  under `tools/xcheck/` (CI-only). See ADR-0001 for the dependency-free policy.
- **R2** No `todo!()`, `unimplemented!()`, `panic!("not implemented")`, `dbg!`,
  commented-out code, or `#[ignore]` on main. Workspace lints deny
  `clippy::todo`, `clippy::unimplemented`, `clippy::dbg_macro`;
  `clippy::unwrap_used`/`clippy::expect_used` are denied in library crates
  (tests allow them via `cfg_attr`, ADR-0005). Unavailable-hardware tests call
  `skip_if_unavailable!(reason)` and are recorded, never ignored.
- **R3** Every op has (a) an f64 CPU reference, (b) ≥1 optimized backend,
  (c) forward parity vs (a). Differentiable ops additionally have (d)
  hand-written backwards per backend and (e) f64 finite-difference gradient
  checks, plus (f) property tests where meaningful. Non-differentiable ops
  document their straight-through/no-gradient adjoint.
- **R4** Every backend passes the same conformance suite; backends chosen at
  runtime by probing; unsupported (op, dtype, device) is a hard error at plan
  time. No silent numerical fallbacks.
- **R5** CI is hermetic: no network in test steps; all test data synthesized
  or vendored; `build.rs` never downloads. PR wall-time target ≤ 30 min.
- **R6** `unsafe` only in `rhizome-kernels-cpu`, `rhizome-kernels-cuda`,
  `rhizome-kernels-metal`, `rhizome-collective` (FFI), and arena/slab
  allocators in `rhizome-runtime`; each block carries `// SAFETY:`. All other
  crates `#![forbid(unsafe_code)]`. `unsafe_op_in_unsafe_fn = deny` workspace-wide.
- **R7** This file is the source of truth. Deviations are ADRs. CI checks the
  config table below equals `rhizome params --all --format=markdown`.
- **R8** Milestones complete only with an acceptance workflow URL recorded in
  `docs/milestones.md`.
- **R9** Deterministic mode is bit-identical across runs on the same hardware
  class AND batch-invariant. Both are asserted in tests.
- **R10** Library crates `#![deny(missing_docs)]`, typed errors, no panics on
  user input, `Send + Sync` public types, cancellation-safe async APIs.
- **R11** All randomness (init, sampling, data order, L sampling, halting
  thresholds) uses Philox-4x32-10 keyed by (seed, stream id, step, index);
  results independent of thread count and rank layout.
- **R12** Never loosen a tolerance, delete a test, or add a retry to go green
  without an ADR showing the original expectation was wrong.

## 3. Model architecture (normative)

### 3.1 Top-level

```
Input → Front end (3.7)
  → Prelude: strata 1..P
  → Think Core (3.5): 3 weight-shared blocks × L iterations
  → Coda: strata P+1..N
  → Back end (3.7) + MTP heads (3.8)
```

Residual stream in bf16 (fp32 in reference/CPU-f32 mode). Pre-norm RMSNorm
(eps 1e-6) on every branch input; sandwich norm: RMSNorm on each branch
output before the residual add. No positional encoding in Local blocks;
decoupled RoPE in attention.

### 3.2 Local block: Gated Delta-Rule mixer

From `x ∈ R^d`: `q, k, v ∈ R^{H×d_h}`, gate `g ∈ R^{H×d_h}`, per-head scalars
`α_logit, β_logit`.

1. Causal depthwise conv1d (kernel 4) on q, k, v then SiLU. Conv keeps a
   3-sample state per channel at inference; reset at segment starts.
2. `q ← q/‖q‖₂ · √d_h`; `k ← k/‖k‖₂` (per head).
3. `β = σ(β_logit) ∈ (0,1)`; `α = exp(−softplus(a_h) · softplus(α_logit))`,
   `a_h` learned per head, all in fp32.
4. State `S ∈ R^{d_h×d_h}` per head, fp32:
   `S_t = α_t · S_{t−1} · (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`; `o_t = S_t q_t`.
5. `o ←` per-head RMSNorm over `d_h` (learned scale), `o ← o ⊙ SiLU(g)`,
   then out-projection `W_o` (depth-scaled init).
6. Training kernel: chunkwise-parallel scan (chunk 64) with the WY
   representation for intra-chunk terms and a sequential inter-chunk carry;
   segment-aware (`α := 0`, conv state cleared at segment starts).
7. Inference kernels: `gated_delta_step` (rank-1 update per token) and
   `gated_delta_multistep` (≤ 8 positions, returning the state after every
   position; used by speculative decoding).
8. Then FFN (3.4).

### 3.3 Global block: Multi-head Latent Attention

- `x → c_kv ∈ R^512` (RMSNorm) and `k_rope ∈ R^64`. Cache per token per
  layer: `[c_kv 512, k_rope 64]` (576 values).
- `x → c_q` (RMSNorm) → per head: `q_nope ∈ R^128` (RMSNorm per head) and
  `q_rope ∈ R^64`. No per-head norm on K (keeps absorption exact).
- Prefill: up-project K (nope 128 + rope 64) and V (128) from `c_kv`;
  flash-style tiled kernel over the latent cache, causal, block-diagonal by
  segment, fp32 softmax. Decode: absorb `W_UK` into the query and `W_UV` into
  `W_O`; split-K over pages (page = 64 tokens).
- RoPE base 1e6 on the 64 rope dims, per-segment positions; YaRN scaling for
  context extension. Optional attention logit softcap (config; default off).
- Optional FP8 e4m3 latent cache with per-token scale (Flagship default on;
  reference off).
- Then FFN (3.4).

### 3.4 FFN: MoE with shared experts (dropless)

- `E_s` shared experts (always active) + `E_r` routed experts, top-k. All
  experts SwiGLU with hidden `h_e` (shared same shape as routed).
- Router (fp32): affinities `s_e = σ(clamp(x·w_e, −30, 30))`. Selection score
  `= s_e + b_e` (`b_e` used ONLY for selection). Group-limited routing:
  experts split into G groups; group score = sum of its top-3 selection
  scores; take top-2 groups; take top-k experts within them (ties → lower
  index). Gate values = selected `s_e` normalized to sum 1.
- Loss-free balancing: after each optimizer step,
  `b_e ← b_e − γ·sign(load_e − mean_load)`, `γ = 1e-3` linearly decayed to 0
  over the last 10% of pretraining; `b_e` checkpointed.
- Sequence-wise balance loss (weight 1e-4): per sequence,
  `f_e = (E_r/(k·T))·#{tokens routed to e}`, `P_e = mean_t s_e(t)`,
  `loss = Σ_e f_e·P_e`.
- Dropless: grouped GEMM with variable group sizes; no capacity factor.
- Expert parallel: EP world size `W_ep ∈ {1,2,4,8}`; groups mapped to ranks
  contiguously (`G/W_ep` groups per rank) so the top-2-group limit bounds each
  token's traffic to ≤ 2 ranks. Dispatch/combine all-to-all overlapped with
  the previous micro-batch's mixer compute (dual stream).

### 3.5 Think Core (adaptive per-token depth)

Placement: after stratum P. Parameters: one shared 3-block unit; adapter
`W_in ∈ R^{d×2d}`; iteration embeddings `e_iter[1..L_max]`; budget embeddings
`e_budget[1..L_max]`; halt head (RMSNorm → linear → σ).

Per token t with Core input `h_t⁰` (residual stream leaving the prelude):

```
Once: write MLA-compressed h_t⁰ into the Core-input cache.
r ← h_t⁰
For i = 1, 2, ...:
  r ← r + W_in·RMSNorm([r ; h_t⁰]) + e_iter[i] + e_budget[B]   (injection)
  Block C1 (Read): MLA attention, queries from r, keys/values from the
    READ SOURCE (config core_read_source):
      input (default): Core-input cache entries of tokens ≤ t.
      prev_iter (ABLATION-ONLY; off by default; needs L_max caches):
        iteration-(i−1) latents of tokens < t plus own r^{(i−1)}. When a
        token halts at j, its final latent is written into caches
        j+1..L_max (a 576-value write, no state copy).
    + MoE FFN.
  Block C2 (Local): Gated Delta mixer whose recurrent state lives in
    PER-ITERATION slab i. + MoE FFN.
  Block C3 (Dense): SwiGLU FFN, hidden 4·d.
  p_halt^{(i)} = halt_head(r).
Halt (inference): stop when p_halt^{(i)} > τ (default 0.5, fixed in
deterministic mode) or i == B. Output to coda = r at halt.
```

Per-iteration slab semantics (correctness-critical): each Core slab i (per
sequence, per head) is DOUBLE-BUFFERED with a per-sequence table `last_halt`
= iteration at which the previous token halted. For the current token at
iteration i the PRIOR state is `prior(i) = slab[min(i, last_halt)]` as of
after the previous token. Writes go to the alternate buffer of slab i, so
`slab[last_halt]`'s after-previous-token buffer stays intact while later
iterations read it. After the token halts at j: `last_halt = j` and flip the
write buffers of slabs 1..j. No state copies. Prefill applies the same rule
sequentially per token with batch compaction per iteration.

Training:

- Per optimizer step (same on all ranks; seeded by step), sample
  `L = round(TruncLogNormal(μ = ln 2, σ = 0.5)) ∈ [1, L_max]`; run exactly L
  iterations for all tokens; `B := L`.
- Truncated backprop through depth: gradients through the last `min(L,4)`
  iterations; earlier iterations run without gradient tracking.
- Halt-head targets (BCE; halt head detached from trunk):
  `conv_i = 1[‖r^{(i)} − r^{(i−1)}‖/‖r^{(i)}‖ < ε_c]` (ε_c = 0.05);
  `loss_i = 1[ℓ_t^{(i)} − ℓ_t^{(L)} < ε_l]` (ε_l = 0.02 nats) computed on a
  1/8 token subsample (configurable) by running the back end on those
  tokens' iterates; `target_i = loss_i where available, else conv_i`.
- Depth regularizer: `λ_d·E[depth]`, `E[depth] = Σ_i Π_{j<i}(1−p_halt^{(j)})`,
  gradient into the halt head only; λ_d default 1e-3.
- Scheduled halting: from 80% of pretraining onward and in all SFT/RL, with
  probability 0.5 per step the forward pass applies the halt policy with
  `τ ~ U(0.3, 0.7)` and the exact slab semantics above; other steps run fixed
  L. This is the ONLY forward path used by serving.
- Pipeline parallel: L fixed per step; the Core sits on a stage with fewer
  trunk blocks so stage time is balanced at `E[L]`; per-step imbalance is
  measured and logged.

### 3.6 Product-Key Memory (PKM) layers

In designated trunk blocks (config), routed experts are replaced by PKM;
shared experts remain. Multi-head PKM, 4 query heads: query `q ∈ R^512` per
head → `(q₁,q₂) ∈ R^256×R^256`; key tables `K₁,K₂ ∈ R^{1024×256}`; top-32 per
table → 1024 candidates (score = s₁+s₂) → top-32 → softmax weights → gather
rows of `V ∈ R^{1,048,576×d}` → weighted sum → per-head outputs summed →
output projection → residual. Values sharded by slot range across EP ranks:
all-to-all (slot ids, weights) → local gather → all-to-all back. Optimizer:
values via sparse (lazy) AdamW; keys via AdamW. Dead-key re-init if usage
< 1e-6 over 10K steps. Log slot-usage entropy.

Memory-edit API (`rhizome edit-memory`): `locate --probe` (gradient
attribution over slots), `patch`, `validate --probes` (collateral change on
held-out probes < threshold), `commit` (new signed version).

### 3.7 Front end / back end

(a) **BPE path** (always maintained; default for Edge and CI): 128K vocab
with byte fallback; untied unembedding (config); next-token CE + z-loss 1e-4.

(b) **Byte-latent path** (enabled per config after ablation gate M10):

- Patcher: separate frozen 150M Gated-Delta byte LM (dense FFN). Boundary
  before byte i if `H(next byte) > θ_g` or `H_i − H_{i−1} > δ`; min patch 1,
  max 64. Precomputed for training shards (shard header records patcher
  hash; mismatch → error); online at inference, O(1) per byte with carried
  recurrent state.
- Byte encoder: 4 Local blocks with DENSE SwiGLU FFN at width `d_local`;
  plus hashed n-gram embeddings (n = 3..8, 3 tables × 256K).
- Patch pooling: cross-attention from one learned query per patch to that
  patch's byte states → linear to `d_model`.
- Byte decoder: 6 Local blocks (dense FFN) at `d_local`, each followed by
  cross-attention to `[current patch trunk latent ; last 256 byte states]`;
  byte-level CE + z-loss.
- `d_local`: Edge 512, Standard 768, Flagship 1024 (Test-L scales down;
  ADR-0007).

### 3.8 Multi-token prediction heads

Two sequential heads. Head k = one Local block (dense FFN) over
`[trunk output ; embedding of ground-truth token/patch t+k]` → predicts token
t+k+1 (BPE: CE) or the trunk latent for patch t+k+1 (byte path: CE through
the shared byte decoder with teacher-forced bytes). Head 2 consumes head 1's
output. Loss weights `0.3·0.85^(k−1)`. Used as drafts at inference (6.5).

### 3.9 Numerics, init, parameterization

- μP over width for all trunk matrices (exact table in docs/mup.md: init
  variances, per-group LR multipliers, output logit multiplier). Depth-scaled
  init `1/√(2·N_blocks)` on mixer/FFN output projections. Muon params use a
  single global LR with RMS-matched update scaling
  `0.2·√max(d_out,d_in)`; AdamW params follow the μP LR table. A
  coordinate-check test (activation RMS stays O(1) across widths 64→512 at
  init and after 50 steps) is part of CI.
- Training GEMMs FP8 e4m3 (weights 128×128 block scales, activations 1×128),
  fp32 accumulate; the CPU path emulates FP8 bit-exactly. bf16 residual; fp32
  for recurrent state, softmax, router, norm stats, halt head, PKM scores,
  optimizer math, master weights.
- Deploy: MXFP4 (Standard/Flagship) or MXINT4 (Edge), 32-element blocks with
  e8m0 shared scale, for expert weights and mixer projections; bf16 for
  embeddings, router, norms, halt head, PKM keys; int8 per-row for PKM values;
  FP8 attention projections on Flagship.
- QAT: STE with learned per-block clipping during the final 10% of
  pretraining (inside the WSD decay).
- Global grad-norm clip 1.0; z-loss 1e-4.

### 3.10 Reference configurations

Regenerate with `rhizome params --all --format=markdown`; CI asserts this
table is current (spec-sync). Approximate entries marked `~` are targets the
generator must match within ±10% (flagship active-params and per-iteration
entries carry a documented ±25% tolerance: ADR-0006).

<!-- params:begin (generated by `rhizome params --all --format=markdown`; do not edit) -->
| Field | edge | flagship | standard | test-l | test-m | test-s |
|---|---|---|---|---|---|---|
| d_model | 1024 | 4096 | 2560 | 256 | 128 | 64 |
| trunk_strata (blocks) | 3 (18) | 7 (42) | 5 (30) | 2 (12) | 2 (12) | 1 (6) |
| core_after_stratum | 1 | 3 | 2 | 1 | 1 | 1 |
| l_max | 4 | 8 | 8 | 4 | 4 | 3 |
| n_heads (d_head) | 8 (128) | 32 (128) | 20 (128) | 8 (32) | 4 (32) | 2 (32) |
| experts routed/shared/topk | 32/1/4 | 64/2/6 | 64/2/4 | 8/1/2 | 8/1/2 | 4/1/2 |
| expert_hidden | 1024 | 1536 | 1536 | 256 | 192 | 128 |
| expert_groups | 4 | 8 | 8 | 4 | 2 | 2 |
| pkm_blocks | none | [12, 24, 36] | [15, 24] | [11] | none | none |
| front_end | BPE | both | both | both | BPE | BPE |
| approx_total_params | ~2.51B | ~70.81B | ~31.64B | ~0.04B | ~0.01B | ~0.00B |
| approx_active_l1 | ~0.43B | ~10.04B | ~3.27B | ~0.01B | ~0.00B | ~0.00B |
| per_core_iter | ~0.05B | ~0.64B | ~0.27B | ~0.00B | ~0.00B | ~0.00B |
| deploy | mxint4 | mxfp4+fp8attn | mxfp4 | f32 | f32 | f32 |
<!-- params:end -->

Also defined: Test-M (d=128, 2 strata, 8 experts) and Test-L (d=256, 2 strata,
8 experts, byte path on) for nightly.

## 4. The op vocabulary (closed set)

Each op: shape inference, f64 reference, adjoint rule (or explicit
non-differentiable marker), per-backend kernels, tolerance entry in
docs/kernels.md. (Lands with M1.)

- Elementwise/fusions: `add, mul, silu, sigmoid, swiglu, cast, residual_add,
  l2norm, softmax, quantize_mx, dequantize_mx, quantize_fp8_block, rope_apply,
  softcap`
- Norms: `rmsnorm, groupnorm_heads`
- Layout/indexing: `gather_rows, scatter_rows (add), compact (by mask),
  concat, split, transpose, copy`
- Linear algebra: `linear` (bf16/fp8/mx weights with block scales),
  `grouped_linear` (variable group sizes), `batched_matmul`
- Sequence: `causal_conv1d` (stateful), `gated_delta_scan_chunked`
  (segments), `gated_delta_step`, `gated_delta_multistep`, `mla_compress`,
  `mla_prefill`, `mla_decode` (paged, split-K, batch-invariant tiling in
  deterministic mode)
- Routing/memory: `router_topk_grouped` (non-diff selection; diff gates),
  `moe_permute, moe_unpermute, pkm_topk_product` (non-diff selection),
  `pkm_gather_sum, pkm_scatter_grad`
- Byte path: `hash_ngram_embed, patch_pool, patch_crossattn,
  entropy_boundary` (non-diff)
- Heads/loss: `embedding, unembed, cross_entropy_zloss` (fused), `halt_head,
  sampling` (temperature, top-k, top-p, min-p, repetition penalty, grammar
  mask; seeded; non-diff)
- Collectives (graph nodes): `all_reduce, all_gather, reduce_scatter,
  all_to_all, send, recv`
- Optimizer: `adamw_step, sparse_adamw_step, muon_newton_schulz` (5 iters,
  bf16 with fp32 accumulate), `ema_update`

Tolerances (docs/kernels.md, enforced by conformance): `|out − ref| ≤ atol +
rtol·|ref|` elementwise; defaults f32 CPU atol 1e-6 / rtol 1e-5; bf16/fp8/mx
backends atol `1e-2·RMS(ref)` / rtol 2e-2; per-op overrides require written
justification. Gradient checks: f64 central differences (h = 1e-5) on small
shapes, rel-err ≤ 1e-6, inputs sampled away from kinks.

## 5. Training system

### 5.1 Static planning

`rhizome-plan` builds the forward graph from the schema for (micro-batch
shape, L, phase, parallel layout), derives the backward graph via adjoint
rules, runs liveness analysis, assigns arena offsets, and chooses activation
checkpointing (recompute expert outputs; keep attention latents and chunk
states) via a planner trait (greedy first; DP/ILP later). Fusion passes:
norm+quant+GEMM prologue, GEMM+SwiGLU+GEMM epilogue, scan+gate+norm. Plans
are validated (no overlapping live buffers, dtype/device legality),
serialized, cached by content hash.

### 5.2 Parallelism

Data parallel with ZeRO-3 sharding of non-expert params (bucketed all-gathers
prefetched one block ahead); expert parallel (3.4); pipeline parallel over
strata with interleaved 1F1B and the Core on a balanced stage (3.5);
sequence parallel for context extension (Ulysses all-to-all for attention;
recurrent-state hand-off for Local blocks). All modes work on the `tcp`
collective so CI can test them with spawned processes.

### 5.3 Optimizer

Muon for all 2-D trunk/expert matrices (distributed: gather → Newton-Schulz →
re-shard, per matrix, per expert); AdamW for embeddings, norms, biases,
router, halt head, PKM keys; sparse AdamW for PKM values. WSD schedule
(warmup–stable–decay) with resumable stable phase; fp32 master weights and
sharded optimizer state; weight EMA for eval and QAT init.

### 5.4 Data

Shard format: header (magic, version, schema hash, tokenizer/patcher hash) +
frames + byte-offset index + optional patch-boundary index + BLAKE3 per
frame; mmap-able layout (ADR-0008); deterministic resumable sampling by
`hash(seed, epoch, rank, shard)`; packing with segment ids and per-segment
positions. Tools: BPE trainer, MinHash LSH near-dedup (0.8), quality
classifier (small Rust model), PII/secret scrubbing, language id, patch
precompute. Synthetic generators (CI): copy, reverse, MQAR, modular
arithmetic (depth 1–4), a PCFG language with computable entropy. Frame codec:
LZ4-block (ADR-0002).

### 5.5 Phases (configs/phases/)

P0 patcher LM; P1 pretrain at 4K; P2 context extension 32K → 256K (YaRN);
P3 QAT inside the WSD decay; P4 SFT with budget-conditioning labels; P5 RL:
GRPO-style (group size 8, clipped objective, KL to a frozen reference,
verifiable rewards: math exact-match, code unit tests in sandbox, JSON-schema
validity), `reward −= λ_c·think_steps`; rollouts via `rhizome-serve` in
deterministic mode (no train/serve skew).

### 5.6 Checkpointing / fault tolerance

Async sharded checkpoints (params, optimizer states, dataloader cursors, RNG
streams, router biases, halting-scheduler state, EMA) to local disk then
upload; content-addressed; topology-agnostic resharding on restore; elastic
DP world size; heartbeat failure detection with automatic relaunch. Resume
must be bit-exact (asserted in CI on Test-S).

### 5.7 Metrics

JSONL metrics per step (loss, lr, grad norm, expert loads, depth histogram,
halt calibration, throughput, memory); optional TensorBoard-compatible writer
in Rust.

## 6. Inference engine

### 6.1 Plans

Static plans per (phase ∈ {prefill, decode, core_step}, batch bucket ∈
{1,2,4,8,16,32,64,128}); CUDA-graph capture for decode; all buckets warmed
at startup before /readyz; zero heap allocation on the request hot path.
Persistent per-stratum megakernel decode is OPTIONAL (M11), feature-gated.

### 6.2 Memory

Paged latent KV (page = 64) with per-sequence page tables; fixed-size
recurrent state slabs per sequence; double-buffered Core slabs (3.5);
Core-input cache pages; one arena planned at startup. Admission control
accounts for pages + slabs + Core slabs + Core cache and enforces max
context. Prefix cache: radix tree over raw bytes (BPE tokens expanded to
bytes); nodes store latent pages and recurrent-state snapshots taken only at
node boundaries and every 4096 tokens, held in pinned host memory under an
LRU byte cap; on hit, restore the nearest snapshot and recompute the tail.

### 6.3 Scheduler

Continuous batching; chunked prefill (2048 tokens/patches) interleaved with
decode; micro-step order per step: prefill chunks → prelude decode for all
sequences → Core iteration 1 (all) → iterations 2..L_max (compacting halted
tokens; auto-switch to smaller buckets) → coda + back end. A global
controller lowers the Core cap when p99 inter-token latency exceeds the SLO
and restores it as load drops. Per-tenant budget ceilings, weighted fair
sharing, priority queues, request timeouts. Every scheduler decision is a
pure function of a snapshot struct (unit-testable; replayable from logs).

### 6.4 Byte path at inference

Byte decoder emits bytes; the recurrent patcher scores each byte; at a
boundary the trunk steps once; the decoder continues with the new patch
latent.

### 6.5 Speculative decoding (self-drafted)

MTP heads draft ≤ 3 tokens/patches. Verification: one pass where parallel
parts (attention, FFN, routing) process all draft positions as a batch and
Local blocks use `gated_delta_multistep` (per Core iteration as well); Core
halting runs per position; commit the recurrent states at the last accepted
position; byte path accepts at byte granularity. In deterministic mode,
greedy output with speculation on == off (asserted).

### 6.6 Constrained decoding

JSON Schema / regex / EBNF → byte-level automaton compiled once and cached by
hash; mask fused into the sampling kernel; exact for BPE (token→bytes
expansion) and byte paths.

### 6.7 API and operations

HTTP/SSE (hand-rolled HTTP/1.1 server; ADR-0009) and gRPC (feature; ADR-0009).
OpenAI-compatible chat/completions endpoints plus native endpoints exposing
think_budget, memory_edit, compute accounting (tokens and think_steps),
seeds, logprobs, stop sequences, n>1, JSON-schema output. Multi-model per
process; hot weight swap by arena flip between steps (same schema) or second
arena (new schema); canary by tenant %. Auth hook (trait; token-based
default), request size and rate limits, backpressure, graceful shutdown
(drain), /healthz, /readyz. `tracing`-style structured spans; Prometheus
metrics (TTFT, ITL, depth histogram, expert-load skew, cache hit rate,
bandwidth utilization, admissions/rejections). `serve` refuses traffic unless
`verify` has passed on this host/hardware class since the last binary change.

### 6.8 Deterministic mode

Fixed reduction order and tree shapes independent of thread count;
batch-invariant tiling/split-K for decode kernels; stable top-k tie-breaks
(lower index); expert dispatch ordered by (expert id, token id); Philox RNG
keyed per request seed; τ fixed. Documented in docs/determinism.md.

### 6.9 Distributed inference

EP up to 8-way + TP 2-way for attention/dense within GPU pairs; PKM sharded
by slot range; disaggregated prefill/decode streaming latent pages + state
snapshots over RDMA (feature) or TCP.

## 7. Weight format and tooling

Container: safetensors-compatible header + extension section (MX block
layouts, per-tensor BLAKE3, embedded model schema, tokenizer/patcher hash,
format version) + ed25519-signed manifest. Memory-mappable layout; the CPU
backend runs directly from the mapped file (ADR-0008). The loader never
executes code from files; the parser is fuzzed. `rhizome convert` produces
deploy formats with calibration and emits a conformance report (perplexity
delta, per-layer error) that gates release. `rhizome inspect` prints schema,
shapes, formats, checksums.

## 8. Evaluation harness (rhizome-eval)

Perplexity on shards; synthetic tasks (copy, MQAR, modular arithmetic, PCFG
entropy gap); needle / multi-needle; exact-match QA format; code-exec runner
(sandboxed subprocess, timeouts, no network); think-depth vs accuracy curves;
halt-head calibration (reliability diagram, ECE); memory-edit locality (edit
one fact, measure collateral change on 1K probes); all outputs JSON with
shared reporters used by `bench`.

## 9. Testing strategy

- **T1** Unit: every op vs f64 reference; gradient checks; proptest for shape
  math, quant round-trips, routing invariants (gates sum to 1, ≤ 2 groups,
  dropless), automaton correctness, shard format round-trips.
- **T2** Parity: sequential delta rule vs chunk scan (random segment
  boundaries); O(n²) attention vs `mla_prefill/decode` (incl. absorbed path);
  unfused vs fused; SIMD vs scalar; multistep vs repeated step; deterministic
  mode bit-identity across 3 runs and batch compositions.
- **T3** Planner: whole-loss finite-difference on Test-S in f64; no
  overlapping live buffers (instrumented allocator); recompute yields
  identical grads; plan hash stability.
- **T4** Convergence (ubuntu native, Test-S/M, ≤ 10 min, fixed seeds,
  generous margins; 2 extra seeds nightly): copy > 99%; MQAR (4 pairs, len
  64) > 95% with attention-removed ≥ 10 points worse; PCFG within 5% of
  generator entropy; modular arithmetic depth-3: loss at L=4 ≤ loss at L=1
  (tolerance) and halt-head ECE < 0.1; MoE: no expert below 20% of mean load
  after warmup; μP coordinate check across widths 64/128/256/512.
- **T5** Distributed (CI, 4 spawned processes, `tcp`): ZeRO-3, EP, PP, SP
  each produce grads equal to single-process within 1e-5 in fp32; checkpoint
  save/restore with a different world size resumes bit-exactly.
- **T6** Serving integration (CI, ephemeral ports): start server with
  Test-S; OpenAI-compatible request; streaming; constrained JSON validated;
  prefix-cache hit returns identical logits; spec-decode on/off greedy
  equality (deterministic mode); think_budget honored and accounted;
  admission rejection under memory cap; hot swap with zero dropped requests;
  graceful shutdown drains; /readyz gating.
- **T7** Fuzz: 60 s per target on PR; 30 min nightly (ADR-0003).
- **T8** Miri nightly on core, ops (reference paths), plan, tokenizer, data.
- **T9** Benchmarks: committed baselines per runner class; PR job posts an
  informational report; hard gate (>10% regression fails) only on
  self-hosted runners.
- **T10** GPU (self-hosted `gpu` label): full conformance; CUDA-graph decode
  parity vs CPU; deterministic + batch-invariant checks; roofline ≥ 70% HBM
  bandwidth for decode GEMV, ≥ 60% tensor-core peak for MXFP4 GEMM at
  M ≥ 4096. Skips gracefully (recorded) if no runner exists.

Milestones M0–M12 and their acceptance criteria: see `docs/milestones.md`.
