"""Загрузка настроек из .env и окружения.

Порядок приоритета: параметры командной строки > переменные окружения процесса > файл .env
> значения по умолчанию. Файл .env только читается (os.environ не изменяется).
Ключ API хранится в обёртке Secret и не попадает в repr/str/логи.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from validator.errors import ConfigError

PROVIDERS = ("gemini", "ollama", "openai", "anthropic")

# Модель по умолчанию есть только там, где её можно выбрать без риска устаревания.
# Для openai/anthropic модель обязана быть указана явно.
_DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-3.6-flash",
    "ollama": "qwen2.5:7b",
}
_MODEL_VARS = {p: f"{p.upper()}_MODEL" for p in PROVIDERS}
_KEY_VARS = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class Secret:
    """Строка-секрет, которая не раскрывается при печати."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "Secret('***')"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Secret) and other._value == self._value

    def __hash__(self) -> int:
        return hash(self._value)


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    api_key: Secret | None
    ollama_base_url: str
    temperature: float
    timeout_seconds: float
    max_retries: int
    rpm: int
    max_output_tokens: int = 4096  # лимит длины ответа модели
    max_requests: int = 10  # не больше запросов к модели за один прогон
    max_pair_requests: int = 10  # проверка пар требований (этап 7); 0 — выключена
    base_url: str | None = None  # свой адрес сервиса (прокси, совместимый API); иначе стандартный

    @property
    def key_variable(self) -> str | None:
        """Имя переменной с ключом для текущего провайдера (None, если ключ не нужен)."""
        return _KEY_VARS.get(self.provider)

    def require_api_key(self) -> Secret | None:
        """Вернуть ключ или объяснить, куда его положить. Ollama ключ не нужен."""
        variable = self.key_variable
        if variable is None:
            return None
        if self.api_key is None:
            raise ConfigError(
                f"Не задан {variable}. Скопируйте .env.example в .env в папке запуска "
                f"и впишите ключ в строку {variable}=... Без LLM документ можно проверить "
                "только детерминированными правилами: параметр --rules-only."
            )
        return self.api_key


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _parse_number(
    raw: str | None, variable: str, default: float, *, minimum: float, maximum: float
) -> float:
    if raw is None:
        return default
    try:
        value = float(raw.replace(",", "."))
    except ValueError:
        raise ConfigError(f"{variable}: ожидается число, получено «{raw}».") from None
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ConfigError(f"{variable}: значение {raw} вне диапазона {minimum:g}–{maximum:g}.")
    return value


def _parse_int(raw: str | None, variable: str, default: int, *, minimum: int, maximum: int) -> int:
    value = _parse_number(raw, variable, default, minimum=minimum, maximum=maximum)
    if not value.is_integer():
        raise ConfigError(f"{variable}: ожидается целое число, получено «{raw}».")
    return int(value)


def load_settings(
    *,
    env_file: Path | None = None,
    environ: Mapping[str, str] | None = None,
    provider_override: str | None = None,
    model_override: str | None = None,
) -> Settings:
    """Собрать настройки. env_file по умолчанию — .env в текущей папке."""
    env_path = env_file if env_file is not None else Path.cwd() / ".env"
    file_values: dict[str, str | None] = {}
    if env_path.is_file():
        file_values = dict(dotenv_values(env_path, encoding="utf-8-sig"))
    process_values = os.environ if environ is None else environ

    def get(name: str) -> str | None:
        return _clean(process_values.get(name)) or _clean(file_values.get(name))

    provider = (_clean(provider_override) or get("LLM_PROVIDER") or "gemini").lower()
    if provider not in PROVIDERS:
        raise ConfigError(
            f"LLM_PROVIDER: неизвестный провайдер «{provider}». Допустимо: {', '.join(PROVIDERS)}."
        )

    model_var = _MODEL_VARS[provider]
    model = _clean(model_override) or get(model_var) or _DEFAULT_MODELS.get(provider)
    if model is None:
        raise ConfigError(f"Для провайдера {provider} укажите модель в {model_var} (файл .env).")

    key_var = _KEY_VARS.get(provider)
    key_value = get(key_var) if key_var else None

    return Settings(
        provider=provider,
        model=model,
        api_key=Secret(key_value) if key_value else None,
        ollama_base_url=get("OLLAMA_BASE_URL") or "http://localhost:11434",
        temperature=_parse_number(
            get("LLM_TEMPERATURE"), "LLM_TEMPERATURE", 0.0, minimum=0.0, maximum=2.0
        ),
        timeout_seconds=_parse_number(
            get("LLM_TIMEOUT_SECONDS"), "LLM_TIMEOUT_SECONDS", 120.0, minimum=1.0, maximum=3600.0
        ),
        max_retries=_parse_int(get("LLM_MAX_RETRIES"), "LLM_MAX_RETRIES", 4, minimum=0, maximum=20),
        rpm=_parse_int(get("LLM_RPM"), "LLM_RPM", 10, minimum=1, maximum=10_000),
        max_output_tokens=_parse_int(
            get("LLM_MAX_OUTPUT_TOKENS"), "LLM_MAX_OUTPUT_TOKENS", 4096, minimum=256, maximum=65_536
        ),
        max_requests=_parse_int(
            get("LLM_MAX_REQUESTS"), "LLM_MAX_REQUESTS", 10, minimum=1, maximum=100
        ),
        max_pair_requests=_parse_int(
            get("LLM_MAX_PAIR_REQUESTS"), "LLM_MAX_PAIR_REQUESTS", 10, minimum=0, maximum=100
        ),
        base_url=get("LLM_BASE_URL"),
    )
