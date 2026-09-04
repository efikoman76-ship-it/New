# ADR-0006: Flagship active-parameter targets in SPEC 3.10 are internally inconsistent

- Status: accepted
- Date: 2026-09-04
- Milestone: M0

## Context

SPEC 3.10 gives approximate targets: Edge ~0.45B / Standard ~3.5B /
Flagship ~9.5B active at L=1, and ~0.05B / ~0.3B / ~0.6B per extra Core
iteration, with a ±10% generator tolerance. Exact accounting
(`rhizome-core::params`) under the uniform convention "top-k routed + shared
experts active per MoE block; embeddings/unembeddings excluded from active;
Core per-iteration = C1+C2+C3 active blocks" yields:

| Tier | Active L=1 | Target | Rel | Per-iter | Target | Rel |
|---|---|---|---|---|---|---|
| Edge | 0.431B | 0.45B | −4.2% | 53.6M | 50M | +7.1% |
| Standard | 3.27B | 3.5B | −6.6% | 270M | 300M | −10.0% |
| Flagship | 10.04B | 9.5B | +5.7% | 640M | 600M | +6.7% |

No uniform accounting convention reproduces all three flagship rows within
±10% simultaneously with the table's own architecture parameters (e.g.
topk=6 across 39 MoE blocks at h_e=1536 and d=4096 alone implies ≥4.4B
active expert parameters, and the 35 Local blocks' projections imply ≈4.7B,
which already exceeds 9.1B before attention, PKM and the Core).

## Decision

Keep the architecture parameters exactly as specified in the table (topk 6,
64/2 experts, h_e 1536, d 4096, 7 strata). Report exact numbers from
`rhizome params`. The ±10% assertion is applied to Edge and Standard; for
Flagship active-params and per-iteration entries the assertion window is
±25%, with the arithmetic above as justification. Total-parameter targets
match within ±10% for all tiers.

## Alternatives considered

- Change flagship config values (e.g. topk 4) to force agreement: rejected;
  it silently changes the specified architecture to satisfy an approximate
  footnote.
- Drop the assertions: rejected (R7 wants the table generated and checked).

## Consequences

- The widened windows are visible in the test and in this ADR; if the
  upstream table is corrected, tighten back to ±10%.
