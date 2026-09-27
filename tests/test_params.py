"""config/params.yaml: every entry has {value, unit, source, confidence}, no bare numbers."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from eio_crewbench.config import (
    PARAM_FIELDS,
    PARAMS_PATH,
    TODO,
    Confidence,
    Param,
    ParamGroup,
    load_params,
)


def _raw() -> dict:
    with open(PARAMS_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _entries(node: dict, path: str = "") -> Iterator[tuple[str, dict]]:
    """Yield (dotted path, mapping) for every mapping that has no mapping children."""
    child_maps = {k: v for k, v in node.items() if isinstance(v, dict)}
    if not child_maps:
        yield path, node
        return
    for key, child in child_maps.items():
        yield from _entries(child, f"{path}.{key}" if path else key)


def _scalars(node: object, path: str = "") -> Iterator[tuple[str, object]]:
    """Yield (dotted path, scalar) for every non-mapping leaf."""
    if isinstance(node, dict):
        for key, child in node.items():
            yield from _scalars(child, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for i, child in enumerate(node):
            yield from _scalars(child, f"{path}[{i}]")
    else:
        yield path, node


def test_params_file_exists() -> None:
    assert PARAMS_PATH.is_file()


def test_every_entry_has_exactly_the_four_fields() -> None:
    bad = {
        path: sorted(entry)
        for path, entry in _entries(_raw())
        if set(entry) != PARAM_FIELDS
    }
    assert not bad, f"entries without exactly {sorted(PARAM_FIELDS)}: {bad}"


def test_no_bare_numbers_outside_value_fields() -> None:
    offenders = [
        path
        for path, scalar in _scalars(_raw())
        if isinstance(scalar, (int, float))
        and not isinstance(scalar, bool)
        and not path.endswith(".value")
    ]
    assert not offenders, f"bare numbers found at: {offenders}"


def test_every_value_is_number_or_todo() -> None:
    bad = [
        path
        for path, entry in _entries(_raw())
        if not (
            entry["value"] == TODO
            or (isinstance(entry["value"], (int, float)) and not isinstance(entry["value"], bool))
        )
    ]
    assert not bad, f"value must be a number or 'TODO': {bad}"


def test_params_load_into_typed_model() -> None:
    params = load_params()
    flat = params.flatten()
    assert flat, "params.yaml has no entries"
    assert all(isinstance(p, Param) for p in flat.values())
    assert all(isinstance(p.confidence, Confidence) for p in flat.values())
    assert set(flat) == {path for path, _ in _entries(_raw())}


def test_known_and_todo_values_are_distinguished() -> None:
    flat = load_params().flatten()
    known = {k for k, p in flat.items() if not p.is_todo}
    todo = {k for k, p in flat.items() if p.is_todo}
    assert "constants.gas_constant" in known
    assert flat["constants.gas_constant"].number > 0
    assert todo, "expected at least one TODO while the reference vehicle is undecided"
    with pytest.raises(ValueError, match="TODO"):
        _ = flat[sorted(todo)[0]].number


def test_params_round_trip_through_json() -> None:
    params = load_params()
    back = ParamGroup.model_validate_json(params.model_dump_json())
    assert back == params


def test_loader_rejects_bare_number(tmp_path: Path) -> None:
    path = tmp_path / "params.yaml"
    path.write_text("sim:\n  timestep: 1.0\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="bare value"):
        load_params(path)


def test_loader_rejects_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "params.yaml"
    path.write_text(
        "sim:\n  timestep:\n    value: 1.0\n    unit: s\n    source: design choice\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_params(path)


def test_loader_rejects_bool_value() -> None:
    with pytest.raises(ValidationError, match="bool"):
        Param(value=True, unit="1", source="x", confidence=Confidence.HIGH)  # type: ignore[arg-type]
