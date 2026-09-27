# ADR-0009: LLM baselines on a fixed 200-episode subset including an open-weight model

- Date: 2026-09-27
- Status: Accepted

## Context

LLM baselines are expensive per episode and their providers change models over time, which threatens reproducibility and cost control.

## Decision

LLM baselines run on a fixed, seeded 200-episode subset of the evaluation split. The baseline set includes at least one open-weight model so that results remain reproducible without any provider's continued cooperation.

## Alternatives rejected

Running LLM baselines on the full evaluation split (cost). Closed models only (irreproducible once a model is retired).

## Consequences

The 200-episode subset is defined by seed and recorded in `docs/RESOURCES.md`. Cost per model is logged in `docs/LOG.md`. Prompts and outputs are stored in the repository, never posted externally.

## Status

Accepted 2026-09-27.
