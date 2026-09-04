# ADR-0011: All work lands on the session branch `arena/01a06e66-new`

- Status: accepted
- Date: 2026-09-04
- Milestone: M0

## Context

The build prompt's working method (Section 12) asks for milestone PRs merged
into `main` as CI goes green. This engineering session is pinned to the
branch `arena/01a06e66-new` (push access to any other branch is revoked by
the environment), so "merge to main on green" cannot be executed here.

## Decision

- All milestones are developed and pushed on `arena/01a06e66-new`.
- One pull request (`arena/01a06e66-new` → `main`) carries the whole build,
  with a milestone checklist in its description; every milestone is a commit
  group with conventional-commit messages and workflow run URLs recorded in
  `docs/milestones.md`.
- Branch protection on `main` requiring ci.yml is requested via the API
  where repository permissions permit; otherwise the manual policy is
  documented in CONTRIBUTING.md.

## Alternatives considered

- Pushing `main` directly: not permitted by the environment.

## Consequences

- `docs/milestones.md` references workflow runs on the session branch's PR,
  which satisfies R8 (verifiable runs) without merging.
- A maintainer merges the PR when adopting the work.
