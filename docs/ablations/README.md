# Ablations

Reports land here as the milestones require them:

- M3: MQAR hybrid vs Local-only (T4 asserts ≥10-point gap).
- M5: Think-Core depth vs modular-arithmetic accuracy; halt-head
  calibration; `core_read_source` input vs prev_iter.
- M10: byte-latent vs BPE at fixed compute (Test-L nightly minimum,
  ADR-0007).

Each report records config, seeds, workflow run URL, and the direction of
the asserted effect (per the build prompt, when no GPU runner exists the
largest CPU-scale version is run and the limitation recorded).
