"""Повтор временных ошибок и ограничение частоты запросов.

Повторяются только временные ошибки (transient): таймаут, сеть, 429, 5xx. Авторизация,
неверная модель, неверный запрос и непригодный ответ не повторяются — повтор их не исправит,
а только увеличит стоимость. Число повторов ограничено (LLM_MAX_RETRIES), пауза растёт
экспоненциально (1, 2, 4 … с) и не больше max_delay; для 429 учитывается Retry-After.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from validator.errors import ProviderError
from validator.llm.base import LLMProvider, LLMRequest, LLMResponse
from validator.llm.errors import ProviderRateLimitError

log = logging.getLogger("validator.llm")

Sleep = Callable[[float], None]
Clock = Callable[[], float]


class RateLimiter:
    """Не чаще rpm запросов в минуту: между началами запросов не меньше 60/rpm секунд."""

    def __init__(self, rpm: int, *, clock: Clock = time.monotonic, sleep: Sleep = time.sleep):
        self._interval = 60.0 / rpm
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            remaining = self._last + self._interval - self._clock()
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._clock()


def complete_with_retry(
    provider: LLMProvider,
    request: LLMRequest,
    *,
    max_retries: int,
    sleep: Sleep = time.sleep,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    limiter: RateLimiter | None = None,
) -> LLMResponse:
    attempts = max_retries + 1
    for attempt in range(1, attempts + 1):
        if limiter is not None:
            limiter.wait()
        try:
            return provider.complete(request)
        except ProviderError as error:
            if not error.transient:
                raise
            if attempt == attempts:
                if attempts == 1:
                    raise
                raise _with_attempts(error, attempts) from None
            delay = base_delay * 2 ** (attempt - 1)
            if isinstance(error, ProviderRateLimitError) and error.retry_after is not None:
                delay = error.retry_after
            delay = min(delay, max_delay)
            log.warning(
                "LLM %s: временная ошибка (%s), попытка %d из %d, пауза %.1f с",
                provider.name,
                type(error).__name__,
                attempt,
                attempts,
                delay,
            )
            sleep(delay)
    raise AssertionError("недостижимо")  # pragma: no cover


def _with_attempts(error: ProviderError, attempts: int) -> ProviderError:
    message = f"{error} (попыток: {attempts})"
    if isinstance(error, ProviderRateLimitError):
        return ProviderRateLimitError(message, retry_after=error.retry_after)
    return type(error)(message)
