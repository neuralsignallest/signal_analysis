# ADR-0003: Belief-robust forward-simulation safety shield; Z3 optional, discrete constraints only

- Date: 2026-09-27
- Status: Accepted

## Context

Agents, including LLM baselines, will propose unsafe actions. The benchmark needs a shield that is principled, transparent and independent of the true sampled fault.

## Decision

The shield screens each proposed action by forward-simulating the plant from a set of belief particles and rejecting actions that violate safety invariants on any particle within a horizon, falling back to a fixed safing action. Z3 is an optional extra used only for discrete constraints (mutual exclusion, ordering, resource limits), never for the continuous dynamics.

## Alternatives rejected

Certifying the continuous dynamics with an SMT or reachability tool: out of reach at V1 fidelity. A shield that reads the scenario's sampled fault: trivially strong and useless as a benchmark component.

## Consequences

`shield` imports `plant` and never `scenarios`, enforced by import-linter. The shield's conservatism is a sensitivity axis. `z3-solver` lives behind the `smt` extra and the package must import cleanly without it.

## Status

Accepted 2026-09-27.
