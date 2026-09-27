# ADR-0011: No composite scalar reward; metrics computed post hoc

- Date: 2026-09-27
- Status: Accepted

## Context

A single scalar reward would encode a weighting of safety, cost, crew time and mission impact that no stakeholder has agreed to, and would invite reward hacking.

## Decision

The environment emits no composite scalar reward. `eval` computes a vector of named metrics post hoc from recorded episodes, with bootstrap confidence intervals, and reports them separately.

## Alternatives rejected

A weighted-sum reward with published weights; a lexicographic reward. Both fix trade-offs the benchmark should expose rather than resolve.

## Consequences

Reinforcement learning is not a V1 baseline (consistent with the CLAUDE.md rules). Baseline comparison is multi-objective, and the paper must present it that way (see `docs/PAPER_NOTES.md`).

## Status

Accepted 2026-09-27.
