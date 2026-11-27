from __future__ import annotations

from pathlib import Path

import pytest

from validator.config import load_settings
from validator.errors import ConfigError

FAKE_KEY = "test-key-0123456789"


def test_defaults_without_env_file(tmp_path: Path) -> None:
    settings = load_settings(env_file=tmp_path / "missing.env")

    assert settings.provider == "gemini"
    assert settings.model == "gemini-3.6-flash"
    assert settings.temperature == 0.0
    assert settings.rpm > 0


def test_reads_values_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"LLM_PROVIDER=ollama\nOLLAMA_MODEL=qwen2.5:7b\nGEMINI_API_KEY={FAKE_KEY}\nLLM_RPM=5\n",
        encoding="utf-8",
    )

    settings = load_settings(env_file=env_file)

    assert settings.provider == "ollama"
    assert settings.model == "qwen2.5:7b"
    assert settings.rpm == 5


def test_process_environment_overrides_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_MODEL", "from-process")

    assert load_settings(env_file=env_file).model == "from-process"


def test_cli_overrides_win(tmp_path: Path) -> None:
    settings = load_settings(
        env_file=tmp_path / "missing.env", provider_override="ollama", model_override="m1"
    )

    assert settings.provider == "ollama"
    assert settings.model == "m1"


def test_api_key_is_never_shown_in_repr_or_str(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"GEMINI_API_KEY={FAKE_KEY}\n", encoding="utf-8")

    settings = load_settings(env_file=env_file)

    assert FAKE_KEY not in repr(settings)
    assert FAKE_KEY not in str(settings)
    assert settings.api_key is not None
    assert settings.api_key.reveal() == FAKE_KEY


def test_require_api_key_explains_where_to_put_it(tmp_path: Path) -> None:
    settings = load_settings(env_file=tmp_path / "missing.env")

    with pytest.raises(ConfigError) as exc:
        settings.require_api_key()

    message = str(exc.value)
    assert "GEMINI_API_KEY" in message
    assert ".env" in message


def test_empty_key_counts_as_missing(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=   \n", encoding="utf-8")

    assert load_settings(env_file=env_file).api_key is None


def test_ollama_needs_no_api_key(tmp_path: Path) -> None:
    settings = load_settings(env_file=tmp_path / "missing.env", provider_override="ollama")

    settings.require_api_key()  # не должно бросать


@pytest.mark.parametrize(
    ("line", "variable"),
    [
        ("LLM_PROVIDER=chatgpt", "LLM_PROVIDER"),
        ("LLM_RPM=0", "LLM_RPM"),
        ("LLM_RPM=abc", "LLM_RPM"),
        ("LLM_TEMPERATURE=3", "LLM_TEMPERATURE"),
        ("LLM_TIMEOUT_SECONDS=-1", "LLM_TIMEOUT_SECONDS"),
        ("LLM_MAX_RETRIES=-2", "LLM_MAX_RETRIES"),
    ],
)
def test_invalid_values_name_the_variable(tmp_path: Path, line: str, variable: str) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(line + "\n", encoding="utf-8")

    with pytest.raises(ConfigError) as exc:
        load_settings(env_file=env_file)

    assert variable in str(exc.value)


def test_model_required_for_provider_without_default(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as exc:
        load_settings(env_file=tmp_path / "missing.env", provider_override="openai")

    assert "OPENAI_MODEL" in str(exc.value)
