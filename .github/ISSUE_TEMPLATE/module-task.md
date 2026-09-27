---
name: Module task
about: A bounded implementation task for one module, sized for a single worker session.
title: "[module] short imperative title"
labels: task
assignees: ''
---

## Scope

<!-- Which subpackage(s), which functions or classes, and what is explicitly out of scope. -->

## Interface

<!-- Signatures, dataclasses or pydantic models, and units for every physical quantity
     (SI, unit suffix in the name: p_kpa, t_k, n_mol). Link the RFC section. -->

## Tests to pass

<!-- Test files and test names that must exist and pass. For plant/ changes, name the
     conservation or balance test. -->

## Done criteria

<!-- Observable conditions: tests green, lint-imports kept, LOG entry appended, ADR if deviating,
     verifier findings addressed. -->

## Forbidden imports

<!-- Imports this module must not make, per the layer rules in pyproject.toml. Call out
     anything the worker might be tempted to reach for (for example, shield -> scenarios). -->

## Model and effort

<!-- Which model runs the worker, expected effort (small / medium / large), and any budget. -->
