"""LLM-слой: провайдеры моделей, устойчивый вызов, строгая схема ответа, анализ требований.

Не зависит от CLI и отчёта. Только стандартная библиотека (urllib) — без SDK провайдеров.
Модель предлагает гипотезы; подтверждённым замечание делает только проверка доказательств.
"""

from validator.llm.analysis import LLMAnalysis, LLMOptions, analyze_with_llm
from validator.llm.base import LLMProvider, LLMRequest, LLMResponse
from validator.llm.registry import PROVIDER_CLASSES, create_provider
from validator.llm.retry import RateLimiter, complete_with_retry
from validator.llm.schema import Hypothesis, RejectedHypothesis, parse_hypotheses

__all__ = [
    "PROVIDER_CLASSES",
    "Hypothesis",
    "LLMAnalysis",
    "LLMOptions",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "RateLimiter",
    "RejectedHypothesis",
    "analyze_with_llm",
    "complete_with_retry",
    "create_provider",
    "parse_hypotheses",
]
