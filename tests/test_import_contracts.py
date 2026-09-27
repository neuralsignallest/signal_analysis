"""Run import-linter so CI fails on any dependency-direction violation.

The contracts live in ``pyproject.toml`` under ``[tool.importlinter]``. This test exists so
that a violation fails ``pytest`` as well as the dedicated ``lint-imports`` CI step. It runs
the linter in a fresh interpreter so the module graph is built from the source tree on disk,
not from whatever this test session has already imported.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"

_RUN_LINTER = (
    "import sys; from importlinter.cli import lint_imports; "
    "sys.exit(lint_imports(config_filename=sys.argv[1], no_cache=True, no_logo=True))"
)


def test_import_linter_contracts_hold() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _RUN_LINTER, str(PYPROJECT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    report = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"lint-imports failed\n{report}"
    # Guard against a vacuous pass: the linter must have actually evaluated contracts.
    assert "Contracts: " in result.stdout, report
    assert "0 broken" in result.stdout, report
    assert "kept" in result.stdout, report
