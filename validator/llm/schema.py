"""Строгая схема ответа модели на обзор одного требования (промпт версии 2).

Ответ целиком: один JSON-объект {"findings": [...]} без других полей (допускается обёртка
целиком в один блок ```json … ```). Нарушение — ProviderResponseError: ответ непригоден.
Элемент: ровно поля requirement_id, category, issue, evidence_quote, reasoning,
recommendation, confidence. Типы, допустимые значения и длины проверяются; неверный элемент
отклоняется с причинами, а не «чинится» (обрезкой, сменой регистра, подстановкой значения).
Важность, критерий, статус и происхождение модель не задаёт: такие поля — недопустимые.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from validator.findings import REVIEW_CATEGORIES, ReviewCategory, is_probability
from validator.llm.errors import ProviderResponseError

MAX_ITEMS = 10  # замечаний к одному требованию
# Текстовое поле → наибольшая длина (None — значение из перечня).
TEXT_FIELDS: dict[str, int | None] = {
    "requirement_id": 64,
    "category": None,
    "issue": 150,
    "evidence_quote": 500,
    "reasoning": 1000,
    "recommendation": 500,
}
FIELDS = (*TEXT_FIELDS, "confidence")
_FENCE = re.compile(r"\A\s*```(?:json)?[ \t]*\n(?P<body>.*)\n\s*```\s*\Z", re.DOTALL)


@dataclass(frozen=True)
class Hypothesis:
    """Гипотеза модели о дефекте одного требования — не замечание и не доказательство.

    index — номер элемента в списке findings ответа модели.
    """

    index: int
    requirement_id: str
    category: ReviewCategory
    issue: str
    evidence_quote: str
    reasoning: str
    recommendation: str
    confidence: float


@dataclass(frozen=True)
class RejectedHypothesis:
    """Элемент ответа модели, не принятый как гипотеза, и причины."""

    index: int
    requirement_id: str | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "requirement_id": self.requirement_id,
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: Any) -> RejectedHypothesis:
        if not isinstance(data, dict):
            raise ValueError("rejected: ожидается объект")
        index, rid, reasons = data.get("index"), data.get("requirement_id"), data.get("reasons")
        if (
            not isinstance(index, int)
            or not (rid is None or isinstance(rid, str))
            or not isinstance(reasons, list)
            or not all(isinstance(r, str) for r in reasons)
        ):
            raise ValueError("rejected: неверные поля index/requirement_id/reasons")
        return cls(index, rid, tuple(reasons))


@dataclass(frozen=True)
class ParsedAnswer:
    hypotheses: tuple[Hypothesis, ...]
    rejected: tuple[RejectedHypothesis, ...]


def parse_hypotheses(text: str) -> ParsedAnswer:
    if not text.strip():
        raise ProviderResponseError("Ответ модели пуст.")
    fenced = _FENCE.match(text)
    try:
        data = json.loads(fenced.group("body") if fenced else text)
    except json.JSONDecodeError:
        raise ProviderResponseError(
            'Ответ модели — не JSON по схеме {"findings": [...]}.'
        ) from None
    if not isinstance(data, dict):
        raise ProviderResponseError('Ответ модели должен быть JSON-объектом {"findings": [...]}.')
    if "findings" not in data:
        raise ProviderResponseError("В ответе модели нет поля findings.")
    unknown = sorted(set(data) - {"findings"})
    if unknown:
        raise ProviderResponseError(f"В ответе модели неожиданные поля: {', '.join(unknown)}.")
    items = data["findings"]
    if not isinstance(items, list):
        raise ProviderResponseError("Поле findings в ответе модели: ожидается список замечаний.")
    if len(items) > MAX_ITEMS:
        raise ProviderResponseError(
            f"Модель вернула {len(items)} замечаний к одному требованию — больше допустимых "
            f"{MAX_ITEMS}."
        )
    hypotheses: list[Hypothesis] = []
    rejected: list[RejectedHypothesis] = []
    for index, item in enumerate(items):
        reasons = _item_problems(item)
        if reasons:
            rid = item.get("requirement_id") if isinstance(item, dict) else None
            rejected.append(
                RejectedHypothesis(index, rid if isinstance(rid, str) else None, tuple(reasons))
            )
        else:
            hypotheses.append(
                Hypothesis(
                    index=index,
                    requirement_id=item["requirement_id"],
                    category=item["category"],
                    issue=item["issue"],
                    evidence_quote=item["evidence_quote"],
                    reasoning=item["reasoning"],
                    recommendation=item["recommendation"],
                    confidence=float(item["confidence"]),
                )
            )
    return ParsedAnswer(tuple(hypotheses), tuple(rejected))


def _item_problems(item: Any) -> list[str]:
    if not isinstance(item, dict):
        return ["Элемент findings должен быть объектом."]
    problems = [f"Недопустимое поле {key}." for key in sorted(set(item) - set(FIELDS))]
    for key, limit in TEXT_FIELDS.items():
        if key not in item:
            problems.append(f"Нет поля {key}.")
            continue
        value = item[key]
        if not isinstance(value, str):
            problems.append(f"Поле {key}: ожидается строка.")
        elif not value.strip():
            problems.append(f"Поле {key} пусто.")
        elif limit is not None and len(value) > limit:
            problems.append(f"Поле {key} длиннее {limit} символов.")
        elif key == "category" and value not in REVIEW_CATEGORIES:
            problems.append(f"Поле category: недопустимое значение «{value[:40]}».")
    if "confidence" not in item:
        problems.append("Нет поля confidence.")
    elif not is_probability(item["confidence"]):
        problems.append("Поле confidence: ожидается число от 0 до 1.")
    return problems
