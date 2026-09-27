"""Determinism tests for eio_crewbench.rng.SeedStreams."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from eio_crewbench.rng import STREAM_NAMES, SeedStreams

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "tests" / "helpers_rng_log.py"
N_DRAWS = 1_000


def _sequence(streams: SeedStreams, name: str, n: int = 64) -> np.ndarray:
    return streams.get(name).random(n)


def test_stream_names_are_exactly_the_six_specified() -> None:
    assert STREAM_NAMES == ("plant", "sensors", "crew", "robot", "ground", "scenario")
    assert tuple(SeedStreams(0)) == STREAM_NAMES


@pytest.mark.parametrize("name", STREAM_NAMES)
def test_same_seed_same_name_gives_identical_sequence(name: str) -> None:
    a, b = SeedStreams(17), SeedStreams(17)
    np.testing.assert_array_equal(_sequence(a, name), _sequence(b, name))


def test_different_names_give_different_sequences() -> None:
    streams = SeedStreams(17)
    sequences = {name: _sequence(streams, name) for name in STREAM_NAMES}
    names = list(STREAM_NAMES)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            assert not np.array_equal(sequences[left], sequences[right]), (left, right)


def test_different_seeds_give_different_sequences() -> None:
    assert not np.array_equal(
        _sequence(SeedStreams(17), "plant"), _sequence(SeedStreams(18), "plant")
    )


def test_stream_is_cached_and_stateful() -> None:
    streams = SeedStreams(17)
    rng = streams.get("plant")
    assert streams.get("plant") is rng
    first = rng.random(8)
    second = rng.random(8)
    assert not np.array_equal(first, second)


def test_consuming_one_stream_does_not_perturb_another() -> None:
    reference = _sequence(SeedStreams(17), "sensors")
    streams = SeedStreams(17)
    streams.get("plant").random(10_000)  # heavy use of an unrelated stream
    np.testing.assert_array_equal(_sequence(streams, "sensors"), reference)


def test_unknown_name_is_rejected() -> None:
    with pytest.raises(KeyError):
        SeedStreams(17).get("weather")


@pytest.mark.parametrize("bad_seed", [-1, 1.5, "17", True, None])
def test_invalid_seeds_are_rejected(bad_seed: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        SeedStreams(bad_seed)  # type: ignore[arg-type]


def test_numpy_integer_seed_is_accepted() -> None:
    np.testing.assert_array_equal(
        _sequence(SeedStreams(np.int64(17)), "plant"), _sequence(SeedStreams(17), "plant")
    )


def _run_helper(seed: int, out_path: Path) -> bytes:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT / "src"), env.get("PYTHONPATH")) if p
    )
    env["PYTHONHASHSEED"] = "random"  # the derivation must not depend on str hashing
    subprocess.run(
        [sys.executable, str(HELPER), str(seed), str(N_DRAWS), str(out_path)],
        check=True,
        env=env,
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return out_path.read_bytes()


def test_json_log_of_1000_draws_is_byte_identical_across_two_processes(tmp_path: Path) -> None:
    first = _run_helper(17, tmp_path / "run_1.json")
    second = _run_helper(17, tmp_path / "run_2.json")
    assert len(first) > 6 * 3 * N_DRAWS  # sanity: the log is not trivially empty
    assert first == second


def test_json_log_differs_for_a_different_seed(tmp_path: Path) -> None:
    assert _run_helper(17, tmp_path / "a.json") != _run_helper(18, tmp_path / "b.json")
