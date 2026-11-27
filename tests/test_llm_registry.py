"""Выбор провайдера по настройкам: общий контракт, ключи, адреса — без сети."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.llm_helpers import FAKE_KEY
from validator.config import Settings, load_settings
from validator.errors import ConfigError
from validator.llm import LLMProvider, LLMRequest, create_provider
from validator.llm.fake import FakeProvider, FakeTransport, http_json
from validator.llm.registry import PROVIDER_CLASSES


def _settings(**environ: str) -> Settings:
    return load_settings(env_file=Path("missing.env"), environ=environ)


@pytest.mark.parametrize(
    ("name", "env"),
    [
        ("gemini", {"GEMINI_API_KEY": FAKE_KEY}),
        ("openai", {"OPENAI_API_KEY": FAKE_KEY, "OPENAI_MODEL": "m"}),
        ("anthropic", {"ANTHROPIC_API_KEY": FAKE_KEY, "ANTHROPIC_MODEL": "m"}),
        ("ollama", {}),
    ],
)
def test_every_configured_provider_has_an_adapter(name: str, env: dict[str, str]) -> None:
    provider = create_provider(_settings(LLM_PROVIDER=name, **env), transport=FakeTransport())

    assert isinstance(provider, LLMProvider)
    assert provider.name == name
    assert provider.model


def test_registry_matches_configuration() -> None:
    from validator.config import PROVIDERS

    assert set(PROVIDER_CLASSES) == set(PROVIDERS)


def test_missing_key_is_configuration_error_before_any_request() -> None:
    transport = FakeTransport()

    with pytest.raises(ConfigError, match="GEMINI_API_KEY") as caught:
        create_provider(_settings(LLM_PROVIDER="gemini"), transport=transport)

    assert "--rules-only" in str(caught.value)
    assert transport.calls == []


def test_unknown_provider_is_configuration_error() -> None:
    with pytest.raises(ConfigError, match="mistral"):
        _settings(LLM_PROVIDER="mistral")


def test_plain_http_endpoint_is_refused_for_keyed_provider() -> None:
    settings = _settings(
        LLM_PROVIDER="openai",
        OPENAI_API_KEY=FAKE_KEY,
        OPENAI_MODEL="m",
        LLM_BASE_URL="http://llm.example.test/v1",
    )

    with pytest.raises(ConfigError, match="https"):
        create_provider(settings, transport=FakeTransport())


def test_local_http_endpoint_is_allowed() -> None:
    settings = _settings(
        LLM_PROVIDER="openai",
        OPENAI_API_KEY=FAKE_KEY,
        OPENAI_MODEL="m",
        LLM_BASE_URL="http://127.0.0.1:8080/v1",
    )

    assert create_provider(settings, transport=FakeTransport()).endpoint.startswith("http://127")


@pytest.mark.parametrize(
    "url", ["ftp://llm.example.test", "https://user:pass@llm.example.test", "не адрес"]
)
def test_malformed_endpoint_is_refused(url: str) -> None:
    settings = _settings(LLM_PROVIDER="ollama", OLLAMA_BASE_URL=url)

    with pytest.raises(ConfigError):
        create_provider(settings, transport=FakeTransport())


def test_fake_provider_follows_the_same_contract() -> None:
    provider = FakeProvider('{"findings": []}')

    assert isinstance(provider, LLMProvider)
    response = provider.complete(LLMRequest(system="s", user="u", max_output_tokens=10))
    assert response.provider == "fake"


def test_new_provider_needs_only_the_contract() -> None:
    """Бизнес-логика работает с любым объектом, выполняющим контракт LLMProvider."""

    class EchoProvider:
        name = "echo"
        model = "echo-1"
        endpoint = "local"

        def complete(self, request: LLMRequest) -> object:
            return FakeProvider('{"findings": []}').complete(request)

    assert isinstance(EchoProvider(), LLMProvider)


def test_provider_uses_configured_timeout() -> None:
    transport = FakeTransport(http_json(200, {"message": {"content": "{}"}, "done_reason": "stop"}))
    provider = create_provider(
        _settings(LLM_PROVIDER="ollama", LLM_TIMEOUT_SECONDS="15"), transport=transport
    )

    provider.complete(LLMRequest(system="s", user="u", max_output_tokens=10))

    assert transport.calls[0].timeout == 15.0
