# EIO-CrewBench RFC

> **Status: placeholder.** The full RFC is being drafted as a Claude Doc and will be exported
> here verbatim once it stabilizes. Until then this file holds the module table, which is the
> binding reference for package layout and dependency direction. Any implementation that
> departs from the RFC requires a proposed ADR in `docs/adr/` in the same PR.

## Module table

| Subpackage  | Contents                                                                                  |
|-------------|-------------------------------------------------------------------------------------------|
| `scenarios` | templates, rejection sampler, seed streams, splits                                        |
| `plant`     | compartment gas model, sorbent beds, leaks, DC bus, RK4, fault parameter application      |
| `ops`       | event queue, crew and robot service calls, ground latency, procedure graphs               |
| `obs`       | sensor models, sensor faults, alarms, report assembly                                     |
| `api`       | EIOEnv, decision epochs, action schema, Gymnasium wrapper                                 |
| `shield`    | invariants, belief-robust forward simulation, fallback safing                             |
| `agents`    | baselines B1-B5, shared particle filter                                                   |
| `eval`      | episode runner, multiprocessing, metrics, bootstrap intervals, figures                    |
| `viz`       | static replay plots                                                                       |

## Dependency direction

Layers, top to bottom: `agents` > `eval` > `api` > (`obs`, `shield`, `ops`, `scenarios`) >
`plant`. Lower layers never import higher ones. `shield` may import `plant` but never
`scenarios`. `viz` may import `eval` only. `eio_crewbench.rng` is a leaf utility available to
every layer. These rules are enforced by import-linter contracts in `pyproject.toml` and by
`tests/test_import_contracts.py`.

## Decisions to date

See `docs/adr/` (ADRs 0001-0012, all Accepted 2026-09-27) for the V1 scope decisions.
