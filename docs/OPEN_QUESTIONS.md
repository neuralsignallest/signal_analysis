# OPEN_QUESTIONS

Purpose: questions that block or shape design and are not yet decided; each is closed by an ADR, an RFC edit, or an explicit "won't decide in V1".

## Entry template

```
## Q-NNNN  <question in one sentence>

- Raised: YYYY-MM-DD, by (session / PR)
- Why it matters:
- Options considered:
- Blocking: (module or gate)
- Status: open | closed by ADR-NNNN | closed by RFC edit | deferred past V1
```

---

## Q-0001  Should the mid-layer packages (obs, shield, ops, scenarios) be mutually independent?

- Raised: 2026-09-27, by the scaffold session
- Why it matters: the import-linter contract currently allows these four to import one
  another (only shield->scenarios is forbidden). Making them independent would prevent
  hidden cycles but may force awkward duplication of event or report types.
- Options considered: keep non-independent siblings (current); switch to `|` independent
  layers; split shared types into a lower package.
- Blocking: none yet; revisit when `obs` or `ops` first needs the other.
- Status: open
