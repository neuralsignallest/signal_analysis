# ADR-0008: Three gates: week 10 VOI go/no-go, week 18 shield, week 22 evaluation freeze

- Date: 2026-09-27
- Status: Accepted

## Context

A benchmark project can drift indefinitely in fidelity. Fixed review points force scope decisions to be made explicitly.

## Decision

Three gates. Week 10: a value-of-information (VOI) go/no-go, demonstrating that information-gathering actions change outcomes in the V1 scenarios. Week 18: the shield is complete and tested. Week 22: evaluation freeze, after which scenarios, splits, metrics and baselines change only by ADR.

## Alternatives rejected

Continuous integration of scope with no freeze; a single end-of-project review.

## Consequences

Each gate is a milestone issue with explicit done criteria. Failing the week 10 gate triggers a scope ADR, not a silent extension.

## Status

Accepted 2026-09-27.
