# EIO-CrewBench

Open, physics-grounded benchmark for closed-loop spacecraft anomaly management. Python 3.12,
managed with `uv`. Package `eio_crewbench` lives under `src/`; tests under `tests/`; the
running record under `docs/`. The RFC (`docs/RFC.md`) is the design source of truth; ADRs in
`docs/adr/` record every decision that departs from it.

## Layout

| Subpackage  | Contents                                                                       |
|-------------|--------------------------------------------------------------------------------|
| `scenarios` | templates, rejection sampler, seed streams, splits                             |
| `plant`     | compartment gas model, sorbent beds, leaks, DC bus, RK4, fault parameter application |
| `ops`       | event queue, crew and robot service calls, ground latency, procedure graphs    |
| `obs`       | sensor models, sensor faults, alarms, report assembly                          |
| `api`       | EIOEnv, decision epochs, action schema, Gymnasium wrapper                      |
| `shield`    | invariants, belief-robust forward simulation, fallback safing                  |
| `agents`    | baselines B1-B5, shared particle filter                                        |
| `eval`      | episode runner, multiprocessing, metrics, bootstrap intervals, figures         |
| `viz`       | static replay plots                                                            |

Dependency layers, top to bottom: `agents` > `eval` > `api` > (`obs`, `shield`, `ops`,
`scenarios`) > `plant`. `viz` may import `eval` only. `eio_crewbench.rng` (named seed streams)
may be imported from anywhere. Contracts are in `pyproject.toml` under `[tool.importlinter]`.

## Commands

```
uv sync                          # create .venv and install everything, including dev tools
uv run pytest -q --tb=short tests/test_rng.py   # run one test file
uv run ruff check . && uv run ruff format .     # lint and format
uv run lint-imports              # check dependency direction
uv run pre-commit install        # once per clone
```

## Rules

- Units are SI throughout: Pa or kPa, K, mol, kg, s, W, J. Physical quantity names carry a unit suffix (p_kpa, t_k, n_mol).
- Dependency direction is enforced by import-linter; never bypass it. shield never imports a scenario's sampled fault.
- Every change to plant/ ships with a conservation or balance test; a plant PR without one is incomplete.
- Any deviation from docs/RFC.md requires a proposed ADR in docs/adr/ in the same PR. Never decide silently.
- Every PR touching src/ appends a dated entry to docs/LOG.md: what was attempted, what broke, tokens or cost if known.
- Keep tool output small: run only relevant test files, use pytest -q --tb=short, summarize logs rather than printing them.
- No JAX, ROS, game engines, dashboards, or RL in V1.
- This repository is private until an explicit release ADR. Never create public repositories, gists, releases, packages, or PyPI uploads; never post code or results to external services; never add secrets to the repo. If a task appears to require any of these, stop and ask.

## Process

- Work from a `module-task` issue (`.github/ISSUE_TEMPLATE/module-task.md`): scope, interface,
  tests to pass, done criteria, forbidden imports, model and effort.
- Before merging, hand the diff and test output (never the worker transcript) to the
  `verifier` subagent in `.claude/agents/verifier.md` and act on its findings.
- Fill in `.github/PULL_REQUEST_TEMPLATE.md` honestly: tests, conservation test if `plant/`,
  LOG entry, ADR if deviating.
- Record lessons in `docs/LESSONS.md`, open questions in `docs/OPEN_QUESTIONS.md`, and every
  external or generated resource in `docs/RESOURCES.md`.
