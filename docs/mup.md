# μP table and Muon scaling (SPEC 3.9)

Width scaling for all trunk matrices. `d` = d_model unless stated. The table
below is normative for init and optimizer LR multipliers; the coordinate
check (T4) asserts activation RMS stays O(1) across widths 64→512 at init
and after 50 steps.

## Init variances (fan-in style)

| Parameter | Init std | Notes |
|---|---|---|
| Embedding (in) | 1/√d | row-wise normal |
| Unembedding | 1/√d | untied |
| Mixer in-projection (q,k,v,gate) | μ_P = d^−1/2 | scaled per μP: d^−1/2·(1) |
| Mixer out-projection W_o | d_h^−1/2 · (2·N_blocks)^−1/2 | depth-scaled |
| MLA W_Q, W_DK, W_KR, W_UK, W_UV | d_in^−1/2 | |
| MLA W_O | d_in^−1/2 · (2·N_blocks)^−1/2 | depth-scaled |
| Expert SwiGLU (all 3 mats) | d^−1/2 | output mat additionally depth-scaled |
| Router w_e | d^−1/2 | zero bias b_e at init |
| Dense FFN (C3, MTP, byte path) | d^−1/2 | output mat depth-scaled |
| Core adapter W_in | (2d)^−1/2 | |
| Iteration/budget embeddings | d^−1/2 | |
| Halt head linear | d^−1/2, zero-init final | starts near p=0.5 |
| PKM keys K₁,K₂ | dims^−1/2 | |
| PKM values | d^−1/2 | |
| Norm scales | 1 (ones) | |
| a_h, α/β logits | 0 (→ α≈e^−softplus², β=0.5) | |

## LR multipliers (AdamW groups)

| Group | LR multiplier |
|---|---|
| Embeddings / unembedding rows | 1.0 (with row-wise scale d^0) |
| 2-D matrices handled by Muon | global Muon LR (below) |
| Norms, biases, router, halt head, PKM keys | 1.0 |
| PKM values (sparse AdamW) | 1.0 |

## Muon

- Update: G = Newton-Schulz(M, 5 iterations) on the momentum-accumulated
  gradient M (bf16 compute, fp32 accumulate); parameter step =
  `lr · 0.2 · √max(d_out, d_in) · G / ‖G‖_F-normalized` — concretely the
  RMS-matched scaling of the orthogonalized update.
- Single global LR (no per-matrix μP multipliers needed because the
  RMS-matching makes updates width-neutral).
- Distributed: gather shard → NS → re-shard per matrix (per expert).

## Output logit multiplier

Unembedding logits scaled by `d^−1/2` at the head (μP logit readout), i.e.
logits = (h/‖h‖_rms) · E^T · d^0.5→1.0 convention — see ops `unembed` where
the exact factor `d^−0.5` is applied to keep logit variance width-independent.
