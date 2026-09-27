# ADR-0006: Gymnasium wrapper only; PettingZoo adapter deferred

- Date: 2026-09-27
- Status: Accepted

## Context

External baselines expect a standard environment interface. The V1 problem has one decision-maker (ADR-0001).

## Decision

`api` exposes the native `EIOEnv` plus a Gymnasium-compatible wrapper. No PettingZoo (multi-agent) adapter in V1.

## Alternatives rejected

Shipping both adapters now: the PettingZoo one would wrap a problem we have decided not to pose yet.

## Consequences

The native env and the Gymnasium wrapper must agree on decision epochs, action schema and observation reports. A PettingZoo adapter will accompany any future Dec-POMDP ADR.

## Status

Accepted 2026-09-27.
