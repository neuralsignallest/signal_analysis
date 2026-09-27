## Summary

<!-- One or two sentences: what changed and why. Link the module-task issue. -->

## Checklist

- [ ] **Tests**: new or changed behavior is covered, and `uv run pytest -q --tb=short` passes.
- [ ] **Conservation test** (if `src/eio_crewbench/plant/` changed): a mass, energy, or balance test ships in this PR.
- [ ] **LOG entry** (if `src/` changed): a dated entry appended to `docs/LOG.md` with what was attempted, what broke, and tokens or cost if known.
- [ ] **ADR** (if deviating from `docs/RFC.md`): a proposed ADR added under `docs/adr/` in this PR.
- [ ] `uv run lint-imports` reports 0 broken contracts and `uv run ruff check .` is clean.

## Verifier findings

<!-- Paste the verifier subagent's findings list, or write "not run" and why. -->
