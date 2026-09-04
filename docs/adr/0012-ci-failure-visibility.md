# ADR-0012: CI failures are posted to the commit/PR so they are readable without web log access

- Status: accepted
- Date: 2026-09-04
- Milestone: M0

## Context

The repository owner's instruction: workflow error logs cannot be read
directly (no web log access in the engineering environment), so CI must
"log the errors" in a machine-readable channel. GitHub Actions job logs are
only fetchable through URLs on domains that are not reachable here; the
GitHub REST/GraphQL APIs (api.github.com) are.

## Decision

Every CI job that can fail pipes its output through `tee` into a per-step
log file. An `on: failure` reporting step (`.github/actions/fail-report`)
uploads the full logs as artifacts (for the web UI) and additionally posts a
comment via the REST API — commit comment for push builds, PR comment for
pull requests — containing: failing step names, the last 300 lines of each
captured log, and direct job URLs. The agent reads these comments through
`gh api`, satisfying "make the workflows log the errors" for offline
troubleshooting.

## Alternatives considered

- Rely on the Actions web UI logs: not readable from the engineering
  environment.
- Check-runs with output annotations: capped at too few bytes to carry
  compiler traces.

## Consequences

- Failures are diagnosable from `gh api repos/.../comments` alone.
- Comments are truncated to keep under API limits; full logs remain as
  artifacts.
