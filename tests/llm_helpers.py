"""Помощники тестов LLM-слоя: локальный HTTP-сервер-заглушка и готовые ответы.

Сервер слушает только 127.0.0.1 и отвечает заранее заданными ответами — тесты не выходят
во внешнюю сеть и не используют настоящих ключей.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

FAKE_KEY = "sk-test-0000-SECRET-do-not-log"


@dataclass
class Reply:
    status: int = 200
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)
    delay: float = 0.0


@dataclass
class Received:
    path: str
    headers: dict[str, str]
    body: bytes


def json_reply(status: int, payload: Any, **headers: str) -> Reply:
    return Reply(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers)


def ollama_reply(content: str, done_reason: str = "stop") -> Reply:
    return json_reply(
        200,
        {
            "model": "test-model",
            "message": {"role": "assistant", "content": content},
            "done": True,
            "done_reason": done_reason,
        },
    )


def findings_json(*items: dict[str, Any]) -> str:
    return json.dumps({"findings": list(items)}, ensure_ascii=False)


def hypothesis(requirement_id: str, quote: str, **changes: Any) -> dict[str, Any]:
    """Элемент ответа модели в формате per-requirement review (промпт версии 2)."""
    item: dict[str, Any] = {
        "requirement_id": requirement_id,
        "category": "clarity",
        "issue": "Неоднозначный срок",
        "evidence_quote": quote,
        "reasoning": "Срок не задан в единицах времени.",
        "recommendation": "Укажите срок в минутах.",
        "confidence": 0.7,
    }
    item.update(changes)
    return item


def by_task(review: Reply, pairs: Reply) -> Callable[[bytes], Reply]:
    """Ответ по виду запроса: проверка пары (этап 7) или обзор одного требования (этап 6)."""

    def route(body: bytes) -> Reply:
        return pairs if b"relations" in body else review

    return route


class StubServer:
    """HTTP-сервер на 127.0.0.1: отдаёт ответы из очереди (последний повторяется) или,
    если задан route, — ответ по телу запроса."""

    def __init__(self, *replies: Reply, route: Callable[[bytes], Reply] | None = None) -> None:
        self.replies = list(replies)
        self.received: list[Received] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                stub.received.append(Received(self.path, dict(self.headers.items()), body))
                if route is not None:
                    reply = route(body)
                else:
                    reply = stub.replies.pop(0) if len(stub.replies) > 1 else stub.replies[0]
                if reply.delay:
                    time.sleep(reply.delay)
                self.send_response(reply.status)
                for name, value in reply.headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(reply.body)))
                self.end_headers()
                # Клиент ушёл по таймауту: на Windows это ConnectionAbortedError (WinError
                # 10053). Таймаут проверяется на стороне клиента — здесь только тишина в логе.
                with suppress(BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    self.wfile.write(reply.body)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host!s}:{port}"

    def __enter__(self) -> StubServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


@contextmanager
def stub_server(
    *replies: Reply, route: Callable[[bytes], Reply] | None = None
) -> Iterator[StubServer]:
    with StubServer(*replies, route=route) as server:
        yield server
