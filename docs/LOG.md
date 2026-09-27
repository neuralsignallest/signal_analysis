# Development log

Append-only. Newest entry at the bottom.

## 2026-09-27 - API schemas, parameter and topology config

Branch: `claude/pydantic-schemas-config-8ghm93` (from `master`; the repo has no `main`).

### Added

- `eio_crewbench/api/schema.py`: Pydantic v2 contracts. `VehicleState`
  (per-compartment O2/CO2/N2 mole counts, fans, valves, hatches, sorbent
  beds, bus load, battery SOC), `Observation` (channel histories, alarm log,
  service calls, procedure state, comm state, shield rejections, timers),
  `Action` (discriminated union on `kind` over 15 command types), and
  `ScenarioTemplate` (family, root cause, component, severity band, sensor
  overlay, onset window, latency regime, crew setting, split). All models
  forbid extra fields.
- `eio_crewbench/config.py`: loaders for the two YAML files plus a `Param`
  model that rejects bare numbers and a `Topology` model with
  `to_graph()` -> `networkx.MultiGraph`.
- `config/params.yaml`: every engineering number as
  `{value, unit, source, confidence}`. Values are `TODO` wherever no
  defensible source is in hand (limits from NASA-STD-3001, bed and fan
  datasheets, power budget, sensor noise, service times).
- `config/topology.yaml`: habitation, node, lab, airlock as a hub-and-spoke
  around the node; three hatches and three IMV ducts as edges with
  `capacity_m3_per_s` and `transit_time_s`.
- `tests/`: JSON round trip for every exported model and every action
  variant; params field-completeness and no-bare-number checks; topology
  loads, is connected, and is crew-walkable via hatches alone.
- `pyproject.toml`, `.gitignore`.

### Decisions

- Compartment ids are a closed `StrEnum` (`CompartmentId`) shared by the
  schemas and the topology loader, so a `VehicleState` must carry all four
  compartments and an edge cannot name an unknown one.
- Action discriminator is a `kind` field with a `Literal` default on each
  class, so callers construct `Wait(duration_s=1)` without naming the kind.
- Topology numbers are placeholders scaled from ISS-class modules and say so
  in their `source` field. They are outside `params.yaml` on purpose: the
  four-field rule applies to `params.yaml` only, and topology keys carry
  their unit in the key name instead.
- No CLAUDE.md exists in this repo, so no repo conventions were applied
  beyond the task text. Worth adding one before the next slice.

### Open questions

- Reference vehicle: the `TODO` values and the placeholder volumes wait on
  choosing one. Fill `atmosphere.*` limits first; alarms derive from them.
- Whether `CommRegime` (live link state in `Observation`) and
  `LatencyRegime` (generator envelope in `ScenarioTemplate`) should collapse
  into one enum once the comm model is written.
- Whether `ScenarioTemplate.component` should be validated against a
  component registry derived from topology + params rather than a free
  identifier.
