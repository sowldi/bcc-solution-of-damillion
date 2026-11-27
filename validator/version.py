"""Версия агента для паспорта прогона: версия пакета + коммит Git."""

from __future__ import annotations

import subprocess
from pathlib import Path

from validator import __version__

_PACKAGE_DIR = Path(__file__).resolve().parent


def agent_version(repo_dir: Path | None = None) -> str:
    """Например «0.1.0+a1b2c3d» или «0.1.0+a1b2c3d-dirty» (есть незакоммиченные правки)."""
    cwd = repo_dir if repo_dir is not None else _PACKAGE_DIR
    try:
        result = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--tags"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return f"{__version__} (без git)"
    described = result.stdout.strip()
    if result.returncode != 0 or not described:
        return f"{__version__} (без git)"
    return f"{__version__}+{described}"
