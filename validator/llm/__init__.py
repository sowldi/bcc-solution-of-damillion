"""LLM-слой: провайдеры моделей, устойчивый вызов, строгая схема ответа, обзор требований.

Не зависит от CLI и отчёта. Только стандартная библиотека (urllib) — без SDK провайдеров.
Модель предлагает гипотезы по каждому требованию отдельно (review) и о связях пар-кандидатов
Fact Graph (crosscheck); подтверждённым замечание делает только проверка доказательств,
а гипотеза LLM остаётся «unverified».
"""

from validator.llm.analysis import LLMAnalysis, LLMOptions, SkippedRequirement
from validator.llm.base import LLMProvider, LLMRequest, LLMResponse
from validator.llm.crosscheck import CrossCheckAnalysis, crosscheck_pairs
from validator.llm.registry import PROVIDER_CLASSES, create_provider
from validator.llm.retry import RateLimiter, complete_with_retry
from validator.llm.review import REVIEW_MAPPING, review_requirements
from validator.llm.schema import Hypothesis, RejectedHypothesis, parse_hypotheses, parse_relations

__all__ = [
    "PROVIDER_CLASSES",
    "REVIEW_MAPPING",
    "CrossCheckAnalysis",
    "Hypothesis",
    "LLMAnalysis",
    "LLMOptions",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "RateLimiter",
    "RejectedHypothesis",
    "SkippedRequirement",
    "complete_with_retry",
    "create_provider",
    "crosscheck_pairs",
    "parse_hypotheses",
    "parse_relations",
    "review_requirements",
]
