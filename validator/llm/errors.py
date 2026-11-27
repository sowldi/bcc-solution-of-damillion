"""Ошибки LLM-слоя. Все — подклассы ProviderError (код выхода 4), кроме неизвестной модели:
это ошибка настройки (код 3). Сообщения не содержат ключей и заголовков запроса.

Временные (transient) ошибки можно повторить: таймаут, недоступность сети, 429, 5xx.
Авторизация, неверная модель, неверный запрос и непригодный ответ не повторяются.
"""

from __future__ import annotations

from validator.errors import EXIT_CONFIG_ERROR, ProviderError


class ProviderTimeoutError(ProviderError):
    """Провайдер не ответил за отведённое время."""

    transient = True


class ProviderNetworkError(ProviderError):
    """Нет соединения с провайдером (DNS, отказ в соединении, обрыв)."""

    transient = True


class ProviderRateLimitError(ProviderError):
    """HTTP 429: превышен лимит запросов. retry_after — пауза из заголовка Retry-After."""

    transient = True

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ProviderServerError(ProviderError):
    """HTTP 5xx: сбой на стороне провайдера."""

    transient = True


class ProviderAuthError(ProviderError):
    """HTTP 401/403: ключ не принят или нет доступа."""


class ProviderModelError(ProviderError):
    """Модель не найдена или недоступна (HTTP 404) — ошибка настройки."""

    exit_code = EXIT_CONFIG_ERROR


class ProviderRequestError(ProviderError):
    """Провайдер отклонил запрос (прочие 4xx, перенаправление)."""


class ProviderResponseError(ProviderError):
    """Ответ непригоден: пуст, обрезан, не JSON, не по схеме, превышает лимит."""
