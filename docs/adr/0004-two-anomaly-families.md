# ADR-0004: Two anomaly families in V1 (CO2, depressurization) plus DC-bus coupling; thermal deferred

- Date: 2026-09-27
- Status: Accepted

## Context

Anomaly breadth trades against modeling depth and validation effort. The families chosen must be physically coupled enough to make diagnosis nontrivial.

## Decision

V1 anomalies are CO2 removal degradation and cabin depressurization, coupled through the DC bus (power availability affects sorbent and fan operation and vice versa). Thermal anomalies are deferred.

## Alternatives rejected

Adding thermal control in V1: a third coupled subsystem with its own sensors, balances and validation burden, at the cost of the schedule (ADR-0008). Uncoupled single-family scenarios: too easy to diagnose and not representative.

## Consequences

The plant contains gas, sorbent, leak and DC-bus models only. Each ships with conservation or balance tests. Thermal is an evaluation-only "unknown" family candidate for a later version and will need its own ADR.

## Status

Accepted 2026-09-27.
