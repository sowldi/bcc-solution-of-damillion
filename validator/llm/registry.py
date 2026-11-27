"""Выбор провайдера по настройкам. Ключ проверяется до первого запроса; адрес сервиса —
только https (или локальный http), без логина и пароля в адресе."""

from __future__ import annotations

from urllib.parse import urlsplit

from validator.config import Settings
from validator.errors import ConfigError
from validator.llm.base import LLMProvider
from validator.llm.providers import (
    AnthropicProvider,
    GeminiProvider,
    HttpProvider,
    OllamaProvider,
    OpenAIProvider,
)
from validator.llm.transport import LOOPBACK_HOSTS, Transport, UrllibTransport

PROVIDER_CLASSES: dict[str, type[HttpProvider]] = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
}


def create_provider(settings: Settings, *, transport: Transport | None = None) -> LLMProvider:
    provider_class = PROVIDER_CLASSES.get(settings.provider)
    if provider_class is None:
        known = ", ".join(PROVIDER_CLASSES)
        raise ConfigError(
            f"LLM_PROVIDER: провайдер «{settings.provider}» не поддерживается; доступны: {known}."
        )
    key = settings.require_api_key()  # ConfigError с подсказкой, куда вписать ключ
    if settings.provider == "ollama":
        base_url, variable = settings.ollama_base_url, "OLLAMA_BASE_URL"
    else:
        base_url, variable = settings.base_url or provider_class.default_base_url, "LLM_BASE_URL"
    check_endpoint(base_url, variable=variable, sends_key=key is not None)
    return provider_class(
        model=settings.model,
        api_key=key,
        base_url=base_url,
        timeout=settings.timeout_seconds,
        transport=transport if transport is not None else UrllibTransport(),
    )


def check_endpoint(url: str, *, variable: str, sends_key: bool) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ConfigError(f"{variable}: некорректный адрес сервиса «{url}».")
    if parts.username or parts.password:
        raise ConfigError(f"{variable}: адрес не должен содержать логин и пароль.")
    local = parts.hostname.casefold() in LOOPBACK_HOSTS
    if sends_key and parts.scheme != "https" and not local:
        raise ConfigError(
            f"{variable}: ключ API передаётся только по https (или на локальный адрес), "
            f"а указан «{parts.scheme}://{parts.hostname}»."
        )
