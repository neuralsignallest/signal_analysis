# ADR-0002: No JAX; NumPy RK4 at 1 s plant step; event queue for operations

- Date: 2026-09-27
- Status: Accepted

## Context

The plant is a small stiff-ish set of ODEs (gas, sorbent, DC bus) that must be cheap, deterministic and debuggable; operations are discrete and sparse in time.

## Decision

Integrate the plant with a fixed-step fourth-order Runge-Kutta scheme in NumPy at a 1 s step. Model operations (crew, robot, ground, procedures) with a discrete event queue layered on top of the plant clock. No JAX anywhere in V1.

## Alternatives rejected

JAX for vectorized or differentiable simulation: adds a heavy dependency and a second numerical stack for no V1 need. Adaptive-step SciPy integrators: harder to make step-for-step reproducible and to align with the event queue. Simulating operations inside the ODE state: bloats the state vector and obscures causality.

## Consequences

Plant code is plain NumPy with an explicit step, which makes conservation tests straightforward. Step size is a global constant and changing it is an RFC deviation. Vectorizing across episodes is done by process-level parallelism in `eval`, not array batching.

## Status

Accepted 2026-09-27.
