from __future__ import annotations

from pathlib import Path

import validator
from validator.version import agent_version


def test_version_starts_with_package_version() -> None:
    assert agent_version().startswith(validator.__version__)


def test_version_falls_back_without_git(tmp_path: Path) -> None:
    version = agent_version(repo_dir=tmp_path)

    assert version.startswith(validator.__version__)
    assert "без git" in version
