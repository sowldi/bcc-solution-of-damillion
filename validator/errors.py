"""Ошибки агента и коды выхода.

Каждая ожидаемая ошибка несёт понятное сообщение на русском и свой код выхода,
чтобы скрипты и CI могли отличать «плохой вход» от «плохой настройки» и сбоя модели.
"""

from __future__ import annotations

EXIT_OK = 0
EXIT_INTERNAL_ERROR = 1
EXIT_INPUT_ERROR = 2
EXIT_CONFIG_ERROR = 3
EXIT_PROVIDER_ERROR = 4


class ValidatorError(Exception):
    """Базовая ожидаемая ошибка агента."""

    exit_code: int = EXIT_INTERNAL_ERROR


class InputError(ValidatorError):
    """Проблема с входным документом или путём к отчёту."""

    exit_code = EXIT_INPUT_ERROR


class ConfigError(ValidatorError):
    """Неверная или неполная настройка (.env, параметры командной строки)."""

    exit_code = EXIT_CONFIG_ERROR


class ProviderError(ValidatorError):
    """Сбой обращения к LLM-провайдеру.

    transient — временная ошибка (таймаут, сеть, 429, 5xx): запрос можно повторить.
    """

    exit_code = EXIT_PROVIDER_ERROR
    transient: bool = False
