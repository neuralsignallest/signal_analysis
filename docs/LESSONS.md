# LESSONS

Purpose: durable lessons about tooling, modeling, and process that should change how the next session works; one entry per lesson, newest first.

## Entry template

```
## YYYY-MM-DD  <one-line lesson>

- Context:
- What we now do instead:
- Evidence (LOG entry, PR, or test):
```

---

## 2026-09-27  Assert on a tool's report text, not just its exit code

- Context: `python -m importlinter.cli` exited 0 while doing nothing, which would have made
  the dependency-direction test pass vacuously.
- What we now do instead: tests that wrap an external checker assert that the checker's
  report mentions the contracts it evaluated.
- Evidence: `tests/test_import_contracts.py`; LOG 2026-09-27.
