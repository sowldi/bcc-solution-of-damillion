"""Адаптеры провайдеров поверх общего HTTP-транспорта: Gemini, OpenAI, Anthropic, Ollama.

Каждый адаптер только строит запрос своего API и извлекает текст ответа; коды HTTP,
обрезанный или пустой ответ, маскирование ключа — общие (HttpProvider). Ключ хранится в
обёртке Secret, раскрывается лишь при сборке заголовка и вырезается из любых сообщений.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, ClassVar
from urllib.parse import quote

from validator.config import Secret
from validator.errors import ProviderError
from validator.ingest.common import one_line
from validator.llm.base import LLMRequest, LLMResponse
from validator.llm.errors import (
    ProviderAuthError,
    ProviderModelError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
)
from validator.llm.transport import HttpResponse, Transport

MAX_RESPONSE_BYTES = 2_000_000
_DETAIL_LIMIT = 300

Headers = dict[str, str]
Payload = dict[str, Any]


class _Structure(Exception):
    """Ответ не той формы, которую описывает API провайдера."""


class HttpProvider:
    name: ClassVar[str]
    key_variable: ClassVar[str | None] = None
    model_variable: ClassVar[str]
    default_base_url: ClassVar[str] = ""

    def __init__(
        self,
        *,
        model: str,
        api_key: Secret | None,
        base_url: str,
        timeout: float,
        transport: Transport,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
    ) -> None:
        self._model = model
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport
        self._max_bytes = max_response_bytes

    @property
    def model(self) -> str:
        return self._model

    @property
    def endpoint(self) -> str:
        return self._base

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self._model!r}, endpoint={self._base!r})"

    # --- вызов ---------------------------------------------------------------------------

    def complete(self, request: LLMRequest) -> LLMResponse:
        url, headers, payload = self._build(request)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        all_headers = {"Content-Type": "application/json", **headers}
        try:
            response = self._transport.post(
                url, all_headers, body, timeout=self._timeout, max_bytes=self._max_bytes
            )
        except ProviderError as error:
            raise _renamed(error, f"{self.name}: {self._redact(str(error))}") from None
        if not 200 <= response.status < 300:
            raise self._status_error(response)
        try:
            data = response.json()
            text, finish, truncated = self._parse(data)
        except ProviderResponseError as error:
            raise ProviderResponseError(f"{self.name}: {error}") from None
        except (_Structure, KeyError, IndexError, TypeError, AttributeError):
            raise ProviderResponseError(
                f"{self.name}: неожиданная структура ответа — ответ не соответствует API."
            ) from None
        if truncated:
            raise ProviderResponseError(
                f"{self.name}: ответ модели обрезан по лимиту длины (LLM_MAX_OUTPUT_TOKENS) — "
                "неполный JSON не разбирается."
            )
        if not text.strip():
            raise ProviderResponseError(f"{self.name}: модель вернула пустой ответ.")
        return LLMResponse(text, self.name, self._model, finish)

    def _build(self, request: LLMRequest) -> tuple[str, Headers, Payload]:
        raise NotImplementedError

    def _parse(self, data: Any) -> tuple[str, str | None, bool]:
        """(текст, причина остановки, обрезан ли ответ)."""
        raise NotImplementedError

    def _key_value(self) -> str:
        return self._key.reveal() if self._key else ""

    # --- ошибки --------------------------------------------------------------------------

    def _redact(self, text: str) -> str:
        key = self._key_value()
        return text.replace(key, "***") if key else text

    def _is_auth_error(self, status: int, detail: str) -> bool:
        return status in (401, 403)

    def _status_error(self, response: HttpResponse) -> ProviderError:
        status = response.status
        detail = self._redact(_error_detail(response))
        suffix = f" Ответ сервиса: {detail}" if detail else ""
        prefix = f"{self.name} ({self._model}): HTTP {status}"
        if 300 <= status < 400:
            return ProviderRequestError(
                f"{prefix}: перенаправление не выполняется — проверьте адрес сервиса.{suffix}"
            )
        if self._is_auth_error(status, detail):
            where = f"Проверьте {self.key_variable}." if self.key_variable else "Нет доступа."
            return ProviderAuthError(f"{prefix}: ключ не принят или нет доступа. {where}{suffix}")
        if status == 404:
            return ProviderModelError(
                f"{prefix}: модель «{self._model}» не найдена или недоступна. "
                f"Проверьте {self.model_variable}.{suffix}"
            )
        if status == 408:
            return ProviderTimeoutError(f"{prefix}: сервис не дождался запроса.{suffix}")
        if status == 429:
            return ProviderRateLimitError(
                f"{prefix}: превышен лимит запросов (LLM_RPM, тариф провайдера).{suffix}",
                retry_after=_retry_after(response.headers),
            )
        if status >= 500:
            return ProviderServerError(f"{prefix}: сбой на стороне провайдера.{suffix}")
        return ProviderRequestError(f"{prefix}: запрос отклонён.{suffix}")


def _renamed(error: ProviderError, message: str) -> ProviderError:
    if isinstance(error, ProviderRateLimitError):
        return ProviderRateLimitError(message, retry_after=error.retry_after)
    return type(error)(message)


def _error_detail(response: HttpResponse) -> str:
    try:
        data = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        text = response.body.decode("utf-8", errors="replace")
    else:
        error = data.get("error", data) if isinstance(data, dict) else data
        if isinstance(error, dict):
            text = str(error.get("message") or error.get("type") or error)
        else:
            text = str(error)
    text = one_line(text)
    return text if len(text) <= _DETAIL_LIMIT else text[:_DETAIL_LIMIT] + "…"


def _retry_after(headers: Mapping[str, str]) -> float | None:
    value = headers.get("retry-after")
    try:
        seconds = float(value) if value is not None else None
    except ValueError:
        return None  # дата HTTP не поддерживается: используется обычная задержка
    return seconds if seconds is not None and seconds >= 0 else None


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise _Structure
    return value


def _list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise _Structure
    return value


def _dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _Structure
    return value


# --- адаптеры --------------------------------------------------------------------------


class GeminiProvider(HttpProvider):
    """Google Gemini API: POST {base}/models/{model}:generateContent."""

    name = "gemini"
    key_variable = "GEMINI_API_KEY"
    model_variable = "GEMINI_MODEL"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"
    _FILTERED = frozenset({"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"})

    def _build(self, request: LLMRequest) -> tuple[str, Headers, Payload]:
        url = f"{self._base}/models/{quote(self._model, safe='-._~')}:generateContent"
        payload = {
            "systemInstruction": {"parts": [{"text": request.system}]},
            "contents": [{"role": "user", "parts": [{"text": request.user}]}],
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_output_tokens,
                "responseMimeType": "application/json",
            },
        }
        return url, {"x-goog-api-key": self._key_value()}, payload

    def _is_auth_error(self, status: int, detail: str) -> bool:
        # неверный ключ Gemini возвращает как 400 INVALID_ARGUMENT «API key not valid»
        return status in (401, 403) or (status == 400 and "api key" in detail.casefold())

    def _parse(self, data: Any) -> tuple[str, str | None, bool]:
        body = _dict(data)
        candidates = body.get("candidates")
        if not candidates:
            blocked = _dict(body.get("promptFeedback") or {}).get("blockReason")
            if blocked:
                raise ProviderResponseError(f"запрос заблокирован фильтром провайдера ({blocked}).")
            raise _Structure
        candidate = _dict(_list(candidates)[0])
        finish = candidate.get("finishReason")
        if finish in self._FILTERED:
            raise ProviderResponseError(f"ответ остановлен фильтром провайдера ({finish}).")
        parts = _list(_dict(candidate.get("content") or {}).get("parts") or [])
        text = "".join(_text(_dict(part).get("text", "")) for part in parts)
        return text, finish, finish == "MAX_TOKENS"


class OpenAIProvider(HttpProvider):
    """OpenAI Chat Completions: POST {base}/chat/completions (и совместимые сервисы)."""

    name = "openai"
    key_variable = "OPENAI_API_KEY"
    model_variable = "OPENAI_MODEL"
    default_base_url = "https://api.openai.com/v1"

    def _build(self, request: LLMRequest) -> tuple[str, Headers, Payload]:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "temperature": request.temperature,
            "max_completion_tokens": request.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self._key_value()}"}
        return f"{self._base}/chat/completions", headers, payload

    def _parse(self, data: Any) -> tuple[str, str | None, bool]:
        choice = _dict(_list(_dict(data)["choices"])[0])
        finish = choice.get("finish_reason")
        if finish == "content_filter":
            raise ProviderResponseError("ответ остановлен фильтром провайдера (content_filter).")
        content = _dict(choice["message"]).get("content")
        return ("" if content is None else _text(content)), finish, finish == "length"


class AnthropicProvider(HttpProvider):
    """Anthropic Messages API: POST {base}/messages."""

    name = "anthropic"
    key_variable = "ANTHROPIC_API_KEY"
    model_variable = "ANTHROPIC_MODEL"
    default_base_url = "https://api.anthropic.com/v1"
    API_VERSION = "2023-06-01"

    def _build(self, request: LLMRequest) -> tuple[str, Headers, Payload]:
        payload = {
            "model": self._model,
            "max_tokens": request.max_output_tokens,
            "system": request.system,
            "messages": [{"role": "user", "content": request.user}],
            "temperature": request.temperature,
        }
        headers = {"x-api-key": self._key_value(), "anthropic-version": self.API_VERSION}
        return f"{self._base}/messages", headers, payload

    def _parse(self, data: Any) -> tuple[str, str | None, bool]:
        body = _dict(data)
        blocks = _list(body["content"])
        finish = body.get("stop_reason")
        if finish == "refusal":
            raise ProviderResponseError("модель отказалась отвечать (refusal).")
        text = "".join(
            _text(_dict(block).get("text", ""))
            for block in blocks
            if _dict(block).get("type") == "text"
        )
        return text, finish, finish == "max_tokens"


class OllamaProvider(HttpProvider):
    """Ollama (локальная модель): POST {base}/api/chat, ключ не нужен."""

    name = "ollama"
    model_variable = "OLLAMA_MODEL"
    default_base_url = "http://localhost:11434"

    def _build(self, request: LLMRequest) -> tuple[str, Headers, Payload]:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_output_tokens,
            },
        }
        return f"{self._base}/api/chat", {}, payload

    def _parse(self, data: Any) -> tuple[str, str | None, bool]:
        body = _dict(data)
        content = _text(_dict(body["message"]).get("content", ""))
        finish = body.get("done_reason")
        return content, finish, finish == "length"
