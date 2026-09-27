# ADR-0005: Crew, robot and ground modeled as costed service resources; all parameters are sensitivity axes

- Date: 2026-09-27
- Status: Accepted

## Context

Humans and robots in the loop have latency, availability limits, error rates and costs. Fixing any of these as constants would bake one operational concept into the benchmark.

## Decision

Crew, robot and ground are service resources in `ops`: each request has a queueing model, a service time, a cost and a failure mode. Every parameter of those models is a named sensitivity axis that evaluation can sweep; none is a hidden constant.

## Alternatives rejected

Autonomous crew and robot agents (ADR-0001). Free or instantaneous service, which removes the value-of-information question the benchmark exists to pose.

## Consequences

Scenario templates carry resource parameters, `eval` can sweep them, and results are reported against them rather than at a single operating point.

## Status

Accepted 2026-09-27.
