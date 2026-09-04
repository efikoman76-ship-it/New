# Kernel math, adjoints, and tolerances

Per-op normative entry: math, adjoint (or non-differentiable marker), and
conformance tolerance. Tolerances follow SPEC Section 4:
`|out − ref| ≤ atol + rtol·|ref|` elementwise; f64 central-difference
gradient checks (h = 1e-5) with rel-err ≤ 1e-6 on small shapes, inputs
sampled away from kinks. Per-op overrides require a written justification in
the Notes column (R12).

Class codes: `ew` elementwise/fusion, `nm` norm, `ly` layout, `la` linear
algebra, `sq` sequence, `rt` routing/memory, `by` byte path, `hd` heads/loss,
 `co` collective, `op` optimizer.

Differentiability (`diff`): `d` differentiable with hand-written backward;
`nd` non-differentiable — adjoint is explicitly straight-through (`st`) or
no-gradient (`ng`) as noted.

## Tolerance table

Baseline tolerances: f32 CPU atol 1e-6, rtol 1e-5. bf16/fp8/mx backends:
atol 1e-2·RMS(ref), rtol 2e-2. Gradient checks (f64): rel-err ≤ 1e-6.
Entries with overrides state them.

| op/name | class | diff | f32 atol | f32 rtol | lowprec atol | lowprec rtol | notes |
|---|---|---|---|---|---|---|---|
| op/add | ew | d | 1e-6 | 1e-5 | base | base | |
| op/mul | ew | d | 1e-6 | 1e-5 | base | base | |
| op/silu | ew | d | 1e-6 | 1e-5 | base | base | x·σ(x); kink-free |
| op/sigmoid | ew | d | 1e-6 | 1e-5 | base | base | |
| op/swiglu | ew | d | 1e-6 | 1e-5 | base | base | fused; parity vs unfused in T2 |
| op/cast | ew | d | 0 | 0 | 0 | 0 | bit-exact by definition |
| op/residual_add | ew | d | 1e-6 | 1e-5 | base | base | optionally bf16 accumulate off |
| op/l2norm | ew | d | 1e-6 | 1e-5 | 2e-2 | 2e-2 | per-head ‖·‖₂ |
| op/softmax | ew | d | 1e-6 | 1e-5 | 1e-3 | 1e-2 | fp32 math in all modes |
| op/quantize_mx | ew | nd(ng) | 0 | 0 | 0 | 0 | bit-exact grid round-trip tests |
| op/dequantize_mx | ew | nd(st) | 0 | 0 | 0 | 0 | STE backward to input |
| op/quantize_fp8_block | ew | nd(st) | 0 | 0 | 0 | 0 | CPU emulation bit-exact (SPEC 3.9) |
| op/rope_apply | ew | d | 1e-6 | 1e-5 | base | base | decoupled, base 1e6, YaRN |
| op/softcap | ew | d | 1e-6 | 1e-5 | base | base | c·tanh(x/c), default off |
| core/rmsnorm | nm | d | 1e-6 | 1e-5 | 1e-2 RMS | 2e-2 | eps 1e-6 |
| core/groupnorm_heads | nm | d | 1e-6 | 1e-5 | 1e-2 RMS | 2e-2 | |
| core/gather_rows | ly | d | 0 | 0 | 0 | 0 | indices exact |
| core/scatter_rows | ly | d | 0 | 0 | 0 | 0 | add-scatter |
| core/compact | ly | nd(st) | 0 | 0 | 0 | 0 | by mask |
| core/concat | ly | d | 0 | 0 | 0 | 0 | |
| core/split | ly | d | 0 | 0 | 0 | 0 | |
| core/transpose | ly | d | 0 | 0 | 0 | 0 | |
| core/copy | ly | d | 0 | 0 | 0 | 0 | |
| core/linear | la | d | 1e-5 | 1e-4 | base | base | fp32 accumulate always; fp8 path vs f64 ref in T2 |
| core/grouped_linear | la | d | 1e-5 | 1e-4 | base | base | variable group sizes |
| core/batched_matmul | la | d | 1e-5 | 1e-4 | base | base | |
| seq/causal_conv1d | sq | d | 1e-6 | 1e-5 | base | base | stateful; state parity asserted |
| seq/gated_delta_scan_chunked | sq | d | 1e-5 | 1e-3 | 1e-2 RMS | 2e-2 | vs sequential reference; chunk 64 |
| seq/gated_delta_step | sq | d | 1e-6 | 1e-5 | base | base | fp32 state |
| seq/gated_delta_multistep | sq | d | 1e-6 | 1e-5 | base | base | ≤8 steps, per-step states |
| seq/mla_compress | sq | d | 1e-6 | 1e-5 | base | base | c_kv/k_rope projection + norms |
| seq/mla_prefill | sq | d | 1e-5 | 1e-3 | 1e-2 RMS | 2e-2 | fp32 softmax; O(n²) reference |
| seq/mla_decode | sq | d | 1e-5 | 1e-3 | 1e-2 RMS | 2e-2 | absorbed path == prefill path parity |
| rt/router_topk_grouped | rt | nd(sel)+d(gates) | 0 | 0 | 0 | 0 | gates sum 1; ≤2 groups; ties→lower idx |
| rt/moe_permute | rt | nd(st) | 0 | 0 | 0 | 0 | |
| rt/moe_unpermute | rt | nd(st) | 0 | 0 | 0 | 0 | |
| rt/pkm_topk_product | rt | nd(sel) | 0 | 0 | 0 | 0 | top-32×top-32→1024→top-32 |
| rt/pkm_gather_sum | rt | d | 0 | 0 | 0 | 0 | exact gather in f32 |
| rt/pkm_scatter_grad | rt | d | 0 | 0 | 0 | 0 | scatter-add into value grads |
| by/hash_ngram_embed | by | d | 0 | 0 | 0 | 0 | exact table lookup |
| by/patch_pool | by | d | 1e-6 | 1e-5 | base | base | cross-attn from 1 query |
| by/patch_crossattn | by | d | 1e-6 | 1e-5 | 1e-2 RMS | 2e-2 | |
| by/entropy_boundary | by | nd(ng) | 0 | 0 | 0 | 0 | boundary decision only |
| hd/embedding | hd | d | 0 | 0 | 0 | 0 | row gather |
| hd/unembed | hd | d | 1e-6 | 1e-5 | base | base | |
| hd/cross_entropy_zloss | hd | d | 1e-6 | 1e-4 | base | base | fused CE + 1e-4 z-loss; logsumexp fp32 |
| hd/halt_head | hd | d | 1e-6 | 1e-5 | base | base | RMSNorm→linear→σ |
| hd/sampling | hd | nd(ng) | 0 | 0 | 0 | 0 | seeded Philox; determinism asserted |
| co/all_reduce | co | d | 1e-6 | 1e-5 | base | base | deterministic ring order |
| co/all_gather | co | d | 0 | 0 | 0 | 0 | |
| co/reduce_scatter | co | d | 1e-6 | 1e-5 | base | base | |
| co/all_to_all | co | d | 0 | 0 | 0 | 0 | ordered by (expert id, token id) |
| co/send | co | nd(ng) | 0 | 0 | 0 | 0 | |
| co/recv | co | nd(ng) | 0 | 0 | 0 | 0 | |
| op/adamw_step | op | nd(ng) | 1e-7 | 1e-6 | base | base | fp32 math |
| op/sparse_adamw_step | op | nd(ng) | 1e-7 | 1e-6 | base | base | lazy per-row state |
| op/muon_newton_schulz | op | nd(ng) | 1e-5 | 1e-4 | base | base | 5 iters, fp32 accumulate |
| op/ema_update | op | nd(ng) | 1e-7 | 1e-6 | base | base | |

`base` = the baseline tolerance for that precision class (SPEC Section 4).
Optimizer ops are "non-differentiable" in the autodiff sense (they consume
gradients); their numerics are still parity-checked against f64 references.

## Adjoint notes (completed as kernels land)

- Every differentiable op documents its vjp in the source adjacent to the
  forward, and the plan's backward derivation uses exactly those rules
  (T3 cross-checks the composed graph against finite differences on
  Test-S).
- `softmax`: standard `dL/dx = p ⊙ (g − ⟨g,p⟩)`; `rmsnorm`: full Jacobian
  including the mean term (not the approximate backward).
- `gated_delta_scan_chunked` backward: transpose of the WY-based forward
  linearization; the inter-chunk carry uses the transposed recurrence;
  verified against sequential-mode finite differences at T2 tolerance.
- `mla_decode` backward routes through the up-projected path (absorption is
  exact, SPEC 3.3).
- Straight-through ops (quantize, permute, compact) propagate the identity
  (or the gather-scatter permutation) and are tested by composed-graph
  finite differences (they contribute zero to second order).
