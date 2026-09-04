# Training (SPEC 5)

## Stack

- Static plans per (shape, L, phase, parallel layout) — `rhizome-plan`.
- Optimizers: Muon (2-D matrices), AdamW (sparse for PKM values).
- WSD schedule with resumable stable phase; EMA weights for eval/QAT.
- Parallelism: DP + ZeRO-3, EP, PP (interleaved 1F1B, Core on a balanced
  stage), SP for context extension. All modes run on the `tcp` collective in
  CI (T5).
- Checkpoints: async, sharded, content-addressed, bit-exact resume.

## Phases (configs/phases/)

| Phase | File | Notes |
|---|---|---|
| P0 | p0-patcher.toml | patcher byte LM pretraining (byte path tiers) |
| P1 | p1-pretrain.toml | 4K context |
| P2 | p2-context.toml | 32K→256K YaRN extension |
| P3 | p3-qat.toml | QAT inside final 10% of WSD decay |
| P4 | p4-sft.toml | SFT with budget-conditioning labels |
| P5 | p5-rl.toml | GRPO with verifiable rewards |

## Metrics

JSONL per step; fields documented in crates/train/src/metrics.md as they
land (loss, lr, grad norm, expert loads, depth histogram, halt calibration,
throughput, memory).
