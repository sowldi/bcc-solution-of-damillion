"""Повторы только для временных ошибок, с ограничением числа и задержки; лимит запросов."""

from __future__ import annotations

import pytest

from validator.errors import ProviderError
from validator.llm import LLMRequest, RateLimiter, complete_with_retry
from validator.llm.errors import (
    ProviderAuthError,
    ProviderModelError,
    ProviderNetworkError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
)
from validator.llm.fake import FakeProvider

REQUEST = LLMRequest(system="s", user="u", max_output_tokens=100)


class Sleeper:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def test_success_needs_no_retry() -> None:
    provider, sleep = FakeProvider('{"findings": []}'), Sleeper()

    response = complete_with_retry(provider, REQUEST, max_retries=3, sleep=sleep)

    assert response.text == '{"findings": []}'
    assert len(provider.requests) == 1
    assert sleep.delays == []


@pytest.mark.parametrize(
    "error",
    [ProviderTimeoutError("t"), ProviderNetworkError("n"), ProviderServerError("503")],
)
def test_transient_errors_are_retried_with_backoff(error: ProviderError) -> None:
    provider, sleep = FakeProvider(error, error, "ok"), Sleeper()

    response = complete_with_retry(provider, REQUEST, max_retries=3, sleep=sleep)

    assert response.text == "ok"
    assert len(provider.requests) == 3
    assert sleep.delays == [1.0, 2.0]


def test_rate_limit_honours_retry_after_within_cap() -> None:
    provider, sleep = (
        FakeProvider(
            ProviderRateLimitError("429", retry_after=7),
            ProviderRateLimitError("429", retry_after=3600),
            "ok",
        ),
        Sleeper(),
    )

    complete_with_retry(provider, REQUEST, max_retries=3, sleep=sleep, max_delay=30)

    assert sleep.delays == [7.0, 30.0]


@pytest.mark.parametrize(
    "error",
    [
        ProviderAuthError("401"),
        ProviderModelError("404"),
        ProviderRequestError("400"),
        ProviderResponseError("bad json"),
    ],
)
def test_permanent_errors_are_not_retried(error: ProviderError) -> None:
    provider, sleep = FakeProvider(error, "ok"), Sleeper()

    with pytest.raises(type(error)):
        complete_with_retry(provider, REQUEST, max_retries=5, sleep=sleep)

    assert len(provider.requests) == 1
    assert sleep.delays == []


def test_retries_are_bounded() -> None:
    provider, sleep = FakeProvider(*[ProviderServerError("500")] * 10), Sleeper()

    with pytest.raises(ProviderServerError, match="попыток: 3"):
        complete_with_retry(provider, REQUEST, max_retries=2, sleep=sleep)

    assert len(provider.requests) == 3
    assert sleep.delays == [1.0, 2.0]


def test_zero_retries_means_single_attempt() -> None:
    provider = FakeProvider(ProviderTimeoutError("t"), "ok")

    with pytest.raises(ProviderTimeoutError):
        complete_with_retry(provider, REQUEST, max_retries=0, sleep=Sleeper())

    assert len(provider.requests) == 1


def test_backoff_delay_is_capped() -> None:
    provider, sleep = FakeProvider(*[ProviderServerError("500")] * 8, "ok"), Sleeper()

    complete_with_retry(provider, REQUEST, max_retries=8, sleep=sleep, max_delay=10)

    assert max(sleep.delays) == 10.0


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_rate_limiter_spaces_requests() -> None:
    clock, sleep = Clock(), Sleeper()
    limiter = RateLimiter(rpm=30, clock=clock, sleep=sleep)  # не чаще раза в 2 с

    limiter.wait()
    limiter.wait()
    clock.now += 5
    limiter.wait()

    assert sleep.delays == [2.0]


def test_rate_limiter_applies_to_retries() -> None:
    clock, sleep = Clock(), Sleeper()
    limiter = RateLimiter(rpm=60, clock=clock, sleep=sleep)
    provider = FakeProvider(ProviderServerError("500"), "ok")

    complete_with_retry(provider, REQUEST, max_retries=1, sleep=sleep, limiter=limiter)

    assert len(provider.requests) == 2
    assert sleep.delays == [1.0, 1.0]  # пауза ретрая и интервал лимитера (часы стоят)
