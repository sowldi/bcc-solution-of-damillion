"""Поддельные провайдер и транспорт для тестов: без сети и без ключей.

FakeProvider выполняет тот же контракт, что и настоящие провайдеры, и отдаёт ответы из
очереди; FakeTransport подменяет HTTP и записывает запросы для проверки их формы.
Фабрика create_provider их не создаёт: в настройках пользователя «fake» не бывает.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from validator.llm.base import LLMRequest, LLMResponse
from validator.llm.transport import HttpResponse

EMPTY_ANSWER = '{"findings": []}'


class FakeProvider:
    name = "fake"
    endpoint = "offline"

    def __init__(self, *replies: str | Exception, model: str = "fake-model") -> None:
        self._replies = list(replies)
        self.model = model
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        reply = self._replies.pop(0) if self._replies else EMPTY_ANSWER
        if isinstance(reply, Exception):
            raise reply
        return LLMResponse(reply, self.name, self.model, "stop")


@dataclass(frozen=True)
class RecordedCall:
    url: str
    headers: Mapping[str, str]
    payload: Any
    timeout: float
    max_bytes: int


def http_json(status: int, payload: Any, headers: Mapping[str, str] | None = None) -> HttpResponse:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return HttpResponse(status, body, dict(headers or {}))


class FakeTransport:
    def __init__(self, *responses: HttpResponse | Exception) -> None:
        self._responses = list(responses)
        self.calls: list[RecordedCall] = []

    def post(
        self, url: str, headers: Mapping[str, str], body: bytes, *, timeout: float, max_bytes: int
    ) -> HttpResponse:
        self.calls.append(RecordedCall(url, dict(headers), json.loads(body), timeout, max_bytes))
        if not self._responses:
            raise AssertionError("FakeTransport: ответы закончились")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response
