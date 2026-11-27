"""HTTP-транспорт LLM-слоя на стандартной библиотеке: только локальный сервер-заглушка."""

from __future__ import annotations

import socket

import pytest

from tests.llm_helpers import Reply, json_reply, stub_server
from validator.llm.errors import (
    ProviderNetworkError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from validator.llm.transport import UrllibTransport


def test_post_returns_status_body_and_headers() -> None:
    with stub_server(json_reply(200, {"ok": True}, **{"X-Test": "1"})) as server:
        response = UrllibTransport().post(
            f"{server.url}/v1/run",
            {"Content-Type": "application/json", "X-Key": "abc"},
            b'{"q": 1}',
            timeout=5,
            max_bytes=1000,
        )

    assert response.status == 200
    assert response.json() == {"ok": True}
    assert response.headers["x-test"] == "1"
    [received] = server.received
    assert received.path == "/v1/run"
    assert received.body == b'{"q": 1}'
    assert received.headers["X-Key"] == "abc"


def test_error_status_is_returned_not_raised() -> None:
    with stub_server(json_reply(503, {"error": "busy"})) as server:
        response = UrllibTransport().post(server.url, {}, b"{}", timeout=5, max_bytes=1000)

    assert response.status == 503
    assert response.json() == {"error": "busy"}


def test_response_larger_than_limit_is_refused() -> None:
    with (
        stub_server(Reply(200, b"x" * 5000)) as server,
        pytest.raises(ProviderResponseError, match="лимит"),
    ):
        UrllibTransport().post(server.url, {}, b"{}", timeout=5, max_bytes=1000)


def test_timeout_is_reported_as_timeout() -> None:
    with (
        stub_server(Reply(200, b"{}", delay=1.5)) as server,
        pytest.raises(ProviderTimeoutError, match=r"0\.3"),
    ):
        UrllibTransport().post(server.url, {}, b"{}", timeout=0.3, max_bytes=1000)


def test_unreachable_endpoint_is_network_error() -> None:
    with socket.socket() as probe:  # свободный порт, на котором никто не слушает
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    # Windows повторяет SYN около 2 с, прежде чем сообщить об отказе: таймаут с запасом
    with pytest.raises(ProviderNetworkError):
        UrllibTransport().post(f"http://127.0.0.1:{port}", {}, b"{}", timeout=20, max_bytes=100)


def test_redirect_is_not_followed() -> None:
    """Ключ не должен уйти на другой адрес через перенаправление."""
    with stub_server(Reply(307, b"", {"Location": "http://example.invalid/"})) as server:
        response = UrllibTransport().post(server.url, {}, b"{}", timeout=5, max_bytes=100)

    assert response.status == 307
    assert len(server.received) == 1


def test_invalid_json_body_is_response_error() -> None:
    with stub_server(Reply(200, b"<html>not json</html>")) as server:
        response = UrllibTransport().post(server.url, {}, b"{}", timeout=5, max_bytes=1000)

    with pytest.raises(ProviderResponseError, match="JSON"):
        response.json()
