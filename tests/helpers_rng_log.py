"""Write a JSON log of draws from every SeedStreams stream.

Invoked as a script by ``tests/test_rng.py`` in a fresh interpreter so that cross-process
determinism is actually exercised::

    python tests/helpers_rng_log.py <seed> <n_draws> <out_path>

The output is written with sorted keys and fixed separators so two runs must be
byte-identical if and only if every draw is identical.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from eio_crewbench.rng import STREAM_NAMES, SeedStreams


def draw_log(seed: int, n_draws: int) -> dict[str, dict[str, list[float] | list[int]]]:
    """Return ``n_draws`` uniforms, normals and integers from each named stream."""
    streams = SeedStreams(seed)
    log: dict[str, dict[str, list[float] | list[int]]] = {}
    for name in STREAM_NAMES:
        rng = streams.get(name)
        log[name] = {
            "uniform": rng.random(n_draws).tolist(),
            "normal": rng.normal(size=n_draws).tolist(),
            "integers": rng.integers(0, 1_000_000, size=n_draws).tolist(),
        }
    return log


def main(argv: list[str]) -> int:
    seed, n_draws, out_path = int(argv[1]), int(argv[2]), Path(argv[3])
    payload = json.dumps(draw_log(seed, n_draws), sort_keys=True, separators=(",", ":"))
    out_path.write_bytes(payload.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
