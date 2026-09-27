"""EIO-CrewBench: an open, physics-grounded benchmark for closed-loop spacecraft anomaly management.

Subpackages are arranged in enforced dependency layers (see ``pyproject.toml``,
``[tool.importlinter]``): ``agents`` > ``eval`` > ``api`` > (``obs``, ``shield``, ``ops``,
``scenarios``) > ``plant``. ``viz`` may import ``eval`` only. ``rng`` is a leaf utility that
provides named, seed-derived random streams and may be imported from anywhere.
"""

__all__ = ["__version__"]

__version__ = "0.0.1"
