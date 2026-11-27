from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Переменные окружения, которые влияют на агента. В тестах они всегда очищаются,
# чтобы настоящий ключ пользователя никогда не участвовал в тестах.
AGENT_ENV_VARS = (
    "LLM_PROVIDER",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "LLM_TEMPERATURE",
    "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_RETRIES",
    "LLM_RPM",
    "LLM_MAX_OUTPUT_TOKENS",
    "LLM_MAX_REQUESTS",
    "LLM_BASE_URL",
)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Каждый тест работает в пустой папке и без переменных агента."""
    for name in AGENT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    yield


@pytest.fixture
def sample_md(tmp_path: Path) -> Path:
    path = tmp_path / "spec.md"
    path.write_text("# Спецификация\n\n* **R-1.** Система хранит заказ.\n", encoding="utf-8")
    return path


RunCli = Callable[..., subprocess.CompletedProcess[str]]


@pytest.fixture
def run_cli(tmp_path: Path) -> RunCli:
    """Запуск настоящей команды `python -m validator` в отдельном процессе."""

    def _run(
        *args: str, extra_env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        env = {k: v for k, v in os.environ.items() if k not in AGENT_ENV_VARS}
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        env["PYTHONIOENCODING"] = "utf-8"
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            [sys.executable, "-m", "validator", *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )

    return _run
