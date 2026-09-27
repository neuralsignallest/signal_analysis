# ADR-0007: Abstract robot skills in V1; no embodiment

- Date: 2026-09-27
- Status: Accepted

## Context

A physically embodied robot (navigation, manipulation, perception) is a research program of its own and would dominate the codebase.

## Decision

The robot is a service resource (ADR-0005) that executes abstract skills such as inspect, reseat, isolate and report, each with a duration, success probability and cost. There is no kinematics, geometry or perception model.

## Alternatives rejected

Any embodiment layer, game engine or ROS bridge (also excluded by CLAUDE.md rules).

## Consequences

Robot fidelity is a sensitivity axis rather than a simulation. Skill definitions live in `ops` procedure graphs and scenario templates.

## Status

Accepted 2026-09-27.
