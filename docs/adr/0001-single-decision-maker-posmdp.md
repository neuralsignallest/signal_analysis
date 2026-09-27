# ADR-0001: Single decision-maker constrained POSMDP for V1; Dec-POMDP deferred

- Date: 2026-09-27
- Status: Accepted

## Context

The benchmark models a crew, a robot and a ground team acting on one spacecraft, which invites a decentralized multi-agent formulation. V1 must ship a solvable, evaluable problem on a fixed schedule.

## Decision

V1 is a single decision-maker constrained partially observable semi-Markov decision process (POSMDP): one policy chooses actions at decision epochs, and crew, robot and ground are resources it dispatches (see ADR-0005), not independent agents.

## Alternatives rejected

A Dec-POMDP with separate crew, robot and ground policies: more faithful, but it multiplies the baseline and evaluation surface and has no accepted V1 solver story.

## Consequences

The action schema, shield and baselines all assume one actor. A Dec-POMDP variant is deferred and will require a PettingZoo-style adapter (ADR-0006) and its own ADR.

## Status

Accepted 2026-09-27.
