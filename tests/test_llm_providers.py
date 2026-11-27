"""Адаптеры провайдеров: форма запроса, разбор ответа, ошибки — без сети (FakeTransport)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.llm_helpers import FAKE_KEY
from validator.config import Settings, load_settings
from validator.errors import EXIT_CONFIG_ERROR, EXIT_PROVIDER_ERROR, ProviderError
from validator.llm import LLMRequest, create_provider
from validator.llm.errors import (
    ProviderAuthError,
    ProviderModelError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
)
from validator.llm.fake import FakeTransport, http_json
from validator.llm.transport import HttpResponse

PROVIDERS = ("gemini", "openai", "anthropic", "ollama")
REQUEST = LLMRequest(
    system="Системная инструкция.", user="Данные документа.", max_output_tokens=512
)


def _settings(provider: str, **env: str) -> Settings:
    environ = {
        "LLM_PROVIDER": provider,
        f"{provider.upper()}_MODEL": "test-model",
        **({} if provider == "ollama" else {f"{provider.upper()}_API_KEY": FAKE_KEY}),
        **env,
    }
    return load_settings(env_file=Path("missing.env"), environ=environ)


def _provider(provider: str, *responses: HttpResponse | Exception, **env: str) -> Any:
    transport = FakeTransport(*responses)
    return create_provider(_settings(provider, **env), transport=transport), transport


def _ok(provider: str, text: str = '{"findings": []}', **extra: Any) -> HttpResponse:
    bodies: dict[str, Any] = {
        "gemini": {
            "candidates": [
                {"content": {"parts": [{"text": text}], "role": "model"}, "finishReason": "STOP"}
            ]
        },
        "openai": {
            "choices": [
                {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
            ]
        },
        "anthropic": {
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
        },
        "ollama": {"message": {"role": "assistant", "content": text}, "done_reason": "stop"},
    }
    body = bodies[provider]
    body.update(extra)
    return http_json(200, body)


# --- форма запроса --------------------------------------------------------------------


def test_gemini_request_shape() -> None:
    provider, transport = _provider("gemini", _ok("gemini"))

    response = provider.complete(REQUEST)

    [call] = transport.calls
    assert call.url == (
        "https://generativelanguage.googleapis.com/v1beta/models/test-model:generateContent"
    )
    assert call.headers["x-goog-api-key"] == FAKE_KEY
    assert FAKE_KEY not in call.url
    assert call.payload["systemInstruction"] == {"parts": [{"text": "Системная инструкция."}]}
    assert call.payload["contents"] == [{"role": "user", "parts": [{"text": "Данные документа."}]}]
    assert call.payload["generationConfig"] == {
        "temperature": 0.0,
        "maxOutputTokens": 512,
        "responseMimeType": "application/json",
    }
    assert call.timeout == 120.0
    assert response.text == '{"findings": []}'
    assert (response.provider, response.model) == ("gemini", "test-model")


def test_openai_request_shape() -> None:
    provider, transport = _provider("openai", _ok("openai"))

    provider.complete(REQUEST)

    [call] = transport.calls
    assert call.url == "https://api.openai.com/v1/chat/completions"
    assert call.headers["Authorization"] == f"Bearer {FAKE_KEY}"
    assert call.payload["model"] == "test-model"
    assert call.payload["messages"] == [
        {"role": "system", "content": "Системная инструкция."},
        {"role": "user", "content": "Данные документа."},
    ]
    assert call.payload["max_completion_tokens"] == 512
    assert call.payload["response_format"] == {"type": "json_object"}


def test_anthropic_request_shape() -> None:
    provider, transport = _provider("anthropic", _ok("anthropic"))

    provider.complete(REQUEST)

    [call] = transport.calls
    assert call.url == "https://api.anthropic.com/v1/messages"
    assert call.headers["x-api-key"] == FAKE_KEY
    assert call.headers["anthropic-version"] == "2023-06-01"
    assert call.payload["system"] == "Системная инструкция."
    assert call.payload["messages"] == [{"role": "user", "content": "Данные документа."}]
    assert call.payload["max_tokens"] == 512


def test_ollama_request_shape_without_key() -> None:
    provider, transport = _provider(
        "ollama", _ok("ollama"), OLLAMA_BASE_URL="http://localhost:11434/"
    )

    provider.complete(REQUEST)

    [call] = transport.calls
    assert call.url == "http://localhost:11434/api/chat"
    assert not any("key" in name.casefold() or name == "Authorization" for name in call.headers)
    assert call.payload["stream"] is False
    assert call.payload["format"] == "json"
    assert call.payload["options"] == {"temperature": 0.0, "num_predict": 512}


def test_custom_endpoint_from_configuration() -> None:
    provider, transport = _provider(
        "openai", _ok("openai"), LLM_BASE_URL="https://llm-proxy.example.test/v1/"
    )

    provider.complete(REQUEST)

    assert transport.calls[0].url == "https://llm-proxy.example.test/v1/chat/completions"


def test_anthropic_joins_text_blocks_only() -> None:
    response = http_json(
        200,
        {
            "content": [
                {"type": "thinking", "thinking": "черновик"},
                {"type": "text", "text": '{"findings":'},
                {"type": "text", "text": " []}"},
            ],
            "stop_reason": "end_turn",
        },
    )
    provider, _ = _provider("anthropic", response)

    assert provider.complete(REQUEST).text == '{"findings": []}'


# --- некорректные ответы ------------------------------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_truncated_answer_is_an_error(name: str) -> None:
    truncated = {
        "gemini": _ok("gemini", candidates=[{"content": {"parts": [{"text": "{"}]},
                                              "finishReason": "MAX_TOKENS"}]),
        "openai": _ok("openai", choices=[{"message": {"content": "{"}, "finish_reason": "length"}]),
        "anthropic": _ok("anthropic", stop_reason="max_tokens"),
        "ollama": _ok("ollama", done_reason="length"),
    }[name]  # fmt: skip
    provider, _ = _provider(name, truncated)

    with pytest.raises(ProviderResponseError, match="обрезан"):
        provider.complete(REQUEST)


@pytest.mark.parametrize("name", PROVIDERS)
def test_empty_answer_is_an_error(name: str) -> None:
    provider, _ = _provider(name, _ok(name, "   "))

    with pytest.raises(ProviderResponseError, match="пуст"):
        provider.complete(REQUEST)


@pytest.mark.parametrize("name", PROVIDERS)
def test_unexpected_structure_is_an_error(name: str) -> None:
    provider, _ = _provider(name, http_json(200, {"unexpected": True}))

    with pytest.raises(ProviderResponseError, match="структур"):
        provider.complete(REQUEST)


@pytest.mark.parametrize("name", PROVIDERS)
def test_non_json_body_is_an_error(name: str) -> None:
    provider, _ = _provider(name, HttpResponse(200, b"<html>gateway</html>", {}))

    with pytest.raises(ProviderResponseError, match="JSON"):
        provider.complete(REQUEST)


def test_gemini_blocked_prompt_is_an_error() -> None:
    provider, _ = _provider("gemini", http_json(200, {"promptFeedback": {"blockReason": "SAFETY"}}))

    with pytest.raises(ProviderResponseError, match="SAFETY"):
        provider.complete(REQUEST)


# --- коды HTTP --------------------------------------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
@pytest.mark.parametrize("status", [401, 403])
def test_authorization_errors(name: str, status: int) -> None:
    provider, _ = _provider(name, http_json(status, {"error": {"message": "denied"}}))

    with pytest.raises(ProviderAuthError) as caught:
        provider.complete(REQUEST)

    assert not caught.value.transient
    assert caught.value.exit_code == EXIT_PROVIDER_ERROR
    assert FAKE_KEY not in str(caught.value)
    if name != "ollama":
        assert f"{name.upper()}_API_KEY" in str(caught.value)


def test_gemini_invalid_key_reported_as_400_is_authorization_error() -> None:
    body = {"error": {"code": 400, "message": "API key not valid. Please pass a valid API key."}}
    provider, _ = _provider("gemini", http_json(400, body))

    with pytest.raises(ProviderAuthError):
        provider.complete(REQUEST)


@pytest.mark.parametrize("name", PROVIDERS)
def test_unknown_model_is_configuration_error(name: str) -> None:
    provider, _ = _provider(name, http_json(404, {"error": {"message": "model not found"}}))

    with pytest.raises(ProviderModelError) as caught:
        provider.complete(REQUEST)

    assert caught.value.exit_code == EXIT_CONFIG_ERROR
    assert "test-model" in str(caught.value)
    assert f"{name.upper()}_MODEL" in str(caught.value)


@pytest.mark.parametrize("name", PROVIDERS)
def test_rate_limit_is_transient_and_keeps_retry_after(name: str) -> None:
    limited = HttpResponse(429, b'{"error": {"message": "slow down"}}', {"retry-after": "7"})
    provider, _ = _provider(name, limited)

    with pytest.raises(ProviderRateLimitError) as caught:
        provider.complete(REQUEST)

    assert caught.value.transient
    assert caught.value.retry_after == 7.0


@pytest.mark.parametrize("name", PROVIDERS)
@pytest.mark.parametrize("status", [500, 502, 503, 529])
def test_server_errors_are_transient(name: str, status: int) -> None:
    provider, _ = _provider(name, http_json(status, {"error": "overloaded"}))

    with pytest.raises(ProviderServerError) as caught:
        provider.complete(REQUEST)

    assert caught.value.transient
    assert str(status) in str(caught.value)


@pytest.mark.parametrize("status", [400, 413, 422])
def test_other_client_errors_are_not_retried(status: int) -> None:
    provider, _ = _provider("openai", http_json(status, {"error": {"message": "bad field"}}))

    with pytest.raises(ProviderRequestError) as caught:
        provider.complete(REQUEST)

    assert not caught.value.transient
    assert "bad field" in str(caught.value)


def test_redirect_is_refused() -> None:
    provider, _ = _provider("openai", HttpResponse(307, b"", {"location": "https://x.test"}))

    with pytest.raises(ProviderRequestError, match="перенаправ"):
        provider.complete(REQUEST)


def test_transport_failures_name_the_provider() -> None:
    provider, _ = _provider("gemini", ProviderTimeoutError("нет ответа за 120 с"))

    with pytest.raises(ProviderTimeoutError, match="gemini"):
        provider.complete(REQUEST)


def test_error_messages_never_contain_the_key() -> None:
    echo = {"error": {"message": f"key {FAKE_KEY} is invalid"}}
    for status in (400, 401, 404, 429, 500):
        provider, _ = _provider("openai", http_json(status, echo))
        with pytest.raises(ProviderError) as caught:
            provider.complete(REQUEST)
        assert FAKE_KEY not in str(caught.value)
        assert "***" in str(caught.value)


def test_provider_repr_hides_the_key() -> None:
    provider, _ = _provider("anthropic", _ok("anthropic"))

    assert FAKE_KEY not in repr(provider)
    assert FAKE_KEY not in str(vars(provider))
