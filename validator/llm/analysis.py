"""Модели LLM-обзора: параметры прогона и результат.

Сам обзор (по одному запросу на атомарное требование) — validator.llm.review.
В результате только гипотезы со статусом «unverified» (Evidence Gate проверил цитату, но
не смысл), отклонённые элементы ответа с причинами и требования, которые модель не получила.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, cast, get_args

from validator.config import Settings
from validator.findings import Finding
from validator.llm.schema import RejectedHypothesis

SCHEMA = "validator.llm-analysis/2"

SkipReason = Literal["budget", "empty", "no_source"]
SKIP_REASONS: tuple[SkipReason, ...] = get_args(SkipReason)
SKIP_TITLES: dict[SkipReason, str] = {
    "budget": "лимит запросов LLM_MAX_REQUESTS",
    "empty": "пустой текст",
    "no_source": "некорректная привязка к источнику",
}


@dataclass(frozen=True)
class LLMOptions:
    max_output_tokens: int = 4096
    temperature: float = 0.0
    max_retries: int = 4
    max_requests: int = 10  # бюджет: запросов (= требований) за прогон
    max_pair_requests: int = 10  # бюджет проверки пар (этап 7): запросов (= пар); 0 — нет
    rpm: int | None = None  # None — без ограничения частоты

    @classmethod
    def from_settings(cls, settings: Settings) -> LLMOptions:
        return cls(
            max_output_tokens=settings.max_output_tokens,
            temperature=settings.temperature,
            max_retries=settings.max_retries,
            max_requests=settings.max_requests,
            max_pair_requests=settings.max_pair_requests,
            rpm=settings.rpm,
        )


@dataclass(frozen=True)
class SkippedRequirement:
    """Атомарное требование, не отправленное модели, и причина."""

    requirement_id: str
    code: SkipReason
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"requirement_id": self.requirement_id, "code": self.code, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: Any) -> SkippedRequirement:
        if not isinstance(data, dict):
            raise ValueError("skipped: ожидается объект")
        rid, code, reason = data.get("requirement_id"), data.get("code"), data.get("reason")
        if not isinstance(rid, str) or code not in SKIP_REASONS or not isinstance(reason, str):
            raise ValueError("skipped: неверные поля requirement_id/code/reason")
        return cls(rid, cast(SkipReason, code), reason)


@dataclass(frozen=True)
class LLMAnalysis:
    """Итог LLM-обзора. hypotheses — только неподтверждённые замечания модели."""

    provider: str
    model: str
    prompt_version: str
    requests: int
    requirements_sent: int
    requirements_total: int
    hypotheses: tuple[Finding, ...]
    rejected: tuple[RejectedHypothesis, ...] = ()
    document_sha256: str = ""  # документ, требования которого получила модель
    reviewed: tuple[str, ...] = ()  # ID требований, отправленных модели (каждое — один раз)
    skipped: tuple[SkippedRequirement, ...] = ()

    def __post_init__(self) -> None:
        for finding in self.hypotheses:
            if finding.origin != "llm" or finding.evidence_status != "unverified":
                raise ValueError(
                    f"Гипотеза LLM {finding.id} должна иметь origin=llm и статус unverified: "
                    "подтверждает замечания только проверка доказательств."
                )
            if finding.document_sha256 != self.document_sha256:
                raise ValueError(
                    f"Гипотеза LLM {finding.id} построена для другого документа "
                    "(SHA-256 не совпадает)."
                )
            if self.reviewed and not set(finding.requirement_ids) <= set(self.reviewed):
                raise ValueError(
                    f"Гипотеза LLM {finding.id} — к требованию, которое не проверялось."
                )
        if len(set(self.reviewed)) != len(self.reviewed):
            raise ValueError("Требование проверено моделью повторно.")
        if self.reviewed and len(self.reviewed) != self.requirements_sent:
            raise ValueError("requirements_sent не совпадает с числом проверенных требований.")

    @property
    def truncated(self) -> bool:
        """Часть требований не отправлена модели: сработал лимит LLM_MAX_REQUESTS."""
        return any(item.code == "budget" for item in self.skipped)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "document_sha256": self.document_sha256,
            "requests": self.requests,
            "requirements_sent": self.requirements_sent,
            "requirements_total": self.requirements_total,
            "reviewed": list(self.reviewed),
            "skipped": [item.to_dict() for item in self.skipped],
            "hypotheses": [finding.to_dict() for finding in self.hypotheses],
            "rejected": [item.to_dict() for item in self.rejected],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMAnalysis:
        if data.get("schema") != SCHEMA:
            raise ValueError(f"Неизвестная схема LLM-анализа: {data.get('schema')!r}")
        texts = [
            data.get(key) for key in ("provider", "model", "prompt_version", "document_sha256")
        ]
        numbers = [data.get(key) for key in ("requests", "requirements_sent", "requirements_total")]
        lists = [data.get(key) for key in ("reviewed", "skipped", "hypotheses", "rejected")]
        if (
            not all(isinstance(value, str) for value in texts)
            or not all(isinstance(n, int) and not isinstance(n, bool) for n in numbers)
            or not all(isinstance(value, list) for value in lists)
            or not all(isinstance(rid, str) for rid in cast(list[Any], lists[0]))
        ):
            raise ValueError("LLM-анализ: неверные типы полей")
        reviewed, skipped, hypotheses, rejected = (cast(list[Any], value) for value in lists)
        return cls(
            provider=str(texts[0]),
            model=str(texts[1]),
            prompt_version=str(texts[2]),
            requests=cast(int, numbers[0]),
            requirements_sent=cast(int, numbers[1]),
            requirements_total=cast(int, numbers[2]),
            hypotheses=tuple(Finding.from_dict(item) for item in hypotheses),
            rejected=tuple(RejectedHypothesis.from_dict(item) for item in rejected),
            document_sha256=str(texts[3]),
            reviewed=tuple(reviewed),
            skipped=tuple(SkippedRequirement.from_dict(item) for item in skipped),
        )

    @classmethod
    def from_json(cls, text: str) -> LLMAnalysis:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("JSON LLM-анализа должен быть объектом")
        return cls.from_dict(data)
