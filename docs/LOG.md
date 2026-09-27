# LOG

Purpose: a dated, append-only record of every session that touches `src/`: what was attempted, what broke, and tokens or cost if known.

## Entry template

```
## YYYY-MM-DD  <short title>  (<branch or PR>)

- Attempted:
- Broke:
- Result:
- Tokens / cost:
```

---

## 2026-09-27  Repository scaffold  (claude/eio-crewbench-scaffold-jixilm)

- Attempted: scaffold the `eio_crewbench` package (nine subpackages with docstrings), `uv`
  project with `pyproject.toml` and `uv.lock`, ruff and pre-commit, GitHub Actions CI,
  import-linter layer contracts with a pytest wrapper, the `SeedStreams` determinism harness
  with in-process and cross-process tests, `CLAUDE.md`, the docs record (RFC placeholder,
  ADR template plus ADRs 0001-0012, LOG, LESSONS, RESOURCES, OPEN_QUESTIONS, PAPER_NOTES),
  the `verifier` subagent, and the issue and PR templates. No physics written.
- Broke: (1) `python -m importlinter.cli` exits 0 without running anything because the module
  has no `__main__` guard; the contract test now calls `importlinter.cli.lint_imports` and
  asserts the report text to guard against a vacuous pass. (2) The import-linter file cache
  keyed on mtime served stale results during mutation checks after files were copied back;
  CI and the test now run with `--no-cache`, and `.import_linter_cache/` is gitignored.
  (3) One test line exceeded 100 columns; fixed by `ruff format`.
- Result: 21 tests pass; 3 import-linter contracts kept, and each was shown to break under a
  deliberate violation (plant->agents, shield->scenarios, viz->plant, api->eval, eval->agents)
  and hold for permitted imports (viz->eval, obs->ops, shield->plant). Branch name follows the
  harness assignment rather than the requested `setup/scaffold`; default branch is `master`.
  Mid-layer siblings (obs, shield, ops, scenarios) are declared non-independent (`:`) so they
  may import one another; only shield->scenarios is forbidden.
- Tokens / cost: not measured this session.
