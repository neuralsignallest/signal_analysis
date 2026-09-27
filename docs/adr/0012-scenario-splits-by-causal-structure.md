# ADR-0012: Scenario splits by causal structure; compound faults evaluation-only

- Date: 2026-09-27
- Status: Accepted

## Context

Random splits leak structure between training and evaluation scenarios and overstate generalization.

## Decision

Scenario splits are defined by causal structure (which fault mechanisms and couplings are active), not by random draw. Compound faults, where two mechanisms are active at once, appear only in the evaluation split.

## Alternatives rejected

Random or seed-based splits; including compound faults in development splits.

## Consequences

`scenarios` owns the split definitions and the rejection sampler must respect them. Any agent tuned on development splits meets compound faults for the first time at evaluation.

## Status

Accepted 2026-09-27.
