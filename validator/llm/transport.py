"""HTTP-транспорт для провайдеров на стандартной библиотеке (urllib) — без сторонних SDK.

- Ответ с любым HTTP-кодом возвращается как HttpResponse; разбор кода — дело провайдера.
- Перенаправления не выполняются: ключ из заголовка не уйдёт на другой адрес.
- Тело читается не больше max_bytes: огромный ответ не займёт память.
- Для локального адреса (localhost, 127.0.0.1, ::1) системный прокси не используется.
- Таймаут и сбой сети — ProviderTimeoutError / ProviderNetworkError (временные ошибки).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlsplit

from validator.llm.errors import (
    ProviderNetworkError,
    ProviderResponseError,
    ProviderTimeoutError,
)

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)  # имена в нижнем регистре

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ProviderResponseError("Ответ провайдера — не JSON.") from None


class Transport(Protocol):
    def post(
        self, url: str, headers: Mapping[str, str], body: bytes, *, timeout: float, max_bytes: int
    ) -> HttpResponse: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None  # urllib вернёт исходный ответ 3xx как HTTPError


class UrllibTransport:
    def __init__(self) -> None:
        self._direct = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)
        self._default = urllib.request.build_opener(_NoRedirect)

    def post(
        self, url: str, headers: Mapping[str, str], body: bytes, *, timeout: float, max_bytes: int
    ) -> HttpResponse:
        request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
        host = (urlsplit(url).hostname or "").casefold()
        opener = self._direct if host in LOOPBACK_HOSTS else self._default
        try:
            with opener.open(request, timeout=timeout) as reply:
                return _read(reply.status, reply, reply.headers, max_bytes)
        except urllib.error.HTTPError as error:
            with error:
                return _read(error.code, error, error.headers, max_bytes)
        except TimeoutError:
            raise ProviderTimeoutError(f"Нет ответа за {timeout:g} с.") from None
        except urllib.error.URLError as error:
            if isinstance(error.reason, TimeoutError):
                raise ProviderTimeoutError(f"Нет ответа за {timeout:g} с.") from None
            raise ProviderNetworkError(f"Сервис недоступен: {error.reason}.") from None
        except (ConnectionError, OSError) as error:
            raise ProviderNetworkError(f"Сбой соединения: {error.strerror or error}.") from None


def _read(status: int, stream: Any, headers: Any, max_bytes: int) -> HttpResponse:
    try:
        body = stream.read(max_bytes + 1)
    except TimeoutError:
        raise ProviderTimeoutError("Ответ не получен до конца за отведённое время.") from None
    if len(body) > max_bytes:
        raise ProviderResponseError(f"Ответ превышает лимит {max_bytes} байт.")
    names = {name.casefold(): value for name, value in (headers or {}).items()}
    return HttpResponse(status, body, names)
