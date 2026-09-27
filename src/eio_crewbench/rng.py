"""Deterministic, named random streams.

Every source of randomness in a run draws from a named stream so that changing one
component's consumption of random numbers (for example, adding a sensor noise term) does not
perturb the others. Streams are derived from a single integer seed and the stream name, so a
run is reproducible from ``(seed,)`` alone.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import numpy as np

__all__ = ["STREAM_NAMES", "SeedStreams"]

STREAM_NAMES: tuple[str, ...] = ("plant", "sensors", "crew", "robot", "ground", "scenario")
"""Names of the independent streams a :class:`SeedStreams` provides, in canonical order."""


def _stream_entropy(seed: int, name: str) -> list[int]:
    """Return the entropy words that identify stream ``name`` under ``seed``.

    The name is hashed with SHA-256 (not Python's salted ``hash``) so the derivation is
    stable across processes, platforms and interpreter sessions.
    """
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    name_words = [int.from_bytes(digest[i : i + 4], "little") for i in range(0, len(digest), 4)]
    return [seed, *name_words]


class SeedStreams:
    """Named, independent NumPy generators derived deterministically from one seed.

    >>> streams = SeedStreams(17)
    >>> plant_rng = streams.get("plant")
    >>> streams.get("plant") is plant_rng   # a stream is stateful and cached
    True

    Two ``SeedStreams`` built from the same seed yield identical draw sequences for each
    name; different names under the same seed yield different sequences.
    """

    def __init__(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
            raise TypeError(f"seed must be an int, got {type(seed).__name__}")
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")
        self._seed = int(seed)
        self._generators: dict[str, np.random.Generator] = {}

    @property
    def seed(self) -> int:
        """The root seed all streams derive from."""
        return self._seed

    @property
    def names(self) -> tuple[str, ...]:
        """The stream names this object can serve."""
        return STREAM_NAMES

    def get(self, name: str) -> np.random.Generator:
        """Return the generator for stream ``name``, creating it on first use.

        Raises:
            KeyError: if ``name`` is not one of :data:`STREAM_NAMES`.
        """
        if name not in STREAM_NAMES:
            raise KeyError(f"unknown stream {name!r}; expected one of {STREAM_NAMES}")
        generator = self._generators.get(name)
        if generator is None:
            sequence = np.random.SeedSequence(_stream_entropy(self._seed, name))
            generator = np.random.Generator(np.random.PCG64(sequence))
            self._generators[name] = generator
        return generator

    def __iter__(self) -> Iterator[str]:
        return iter(STREAM_NAMES)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(seed={self._seed})"
