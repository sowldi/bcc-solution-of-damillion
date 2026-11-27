"""Общий контракт LLM-провайдера.

Бизнес-логика (анализ требований) работает только с этим контрактом: новый провайдер —
это ещё один класс с методом complete(), без изменения анализа, CLI и отчёта.
Провайдер возвращает текст ответа модели; разбор и проверку схемы делает validator.llm.schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class LLMRequest:
    """system — инструкции агента; user — данные (текст документа), отдельным сообщением."""

    system: str
    user: str
    max_output_tokens: int
    temperature: float = 0.0


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    finish_reason: str | None = None


@runtime_checkable
class LLMProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def endpoint(self) -> str:
        """Адрес сервиса без секретов (для сообщений пользователю)."""
        ...

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Один вызов модели. Ошибки — подклассы ProviderError (validator.llm.errors)."""
        ...
