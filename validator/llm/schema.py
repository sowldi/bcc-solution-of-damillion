"""Строгая схема ответа модели на обзор одного требования (промпт версии 2).

Ответ целиком: один JSON-объект {"findings": [...]} без других полей (допускается обёртка
целиком в один блок ```json … ```). Нарушение — ProviderResponseError: ответ непригоден.
Элемент: ровно поля requirement_id, category, issue, evidence_quote, reasoning,
recommendation, confidence. Типы, допустимые значения и длины проверяются; неверный элемент
отклоняется с причинами, а не «чинится» (обрезкой, сменой регистра, подстановкой значения).
Важность, критерий, статус и происхождение модель не задаёт: такие поля — недопустимые.

Проверка пары требований (этап 7, промпт pair-1) — так же строго: один объект
{"relations": [...]}, элемент — ровно поля source_id, target_id, relation, source_quote,
target_quote, explanation, recommendation, confidence; не больше MAX_RELATIONS связей.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from validator.findings import (
    RELATION_TYPES,
    REVIEW_CATEGORIES,
    RelationType,
    ReviewCategory,
    is_probability,
)
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
MAX_RELATIONS = 4  # связей в ответе на одну пару
RELATION_TEXT_FIELDS: dict[str, int | None] = {
    "source_id": 64,
    "target_id": 64,
    "relation": None,
    "source_quote": 500,
    "target_quote": 500,
    "explanation": 1000,
    "recommendation": 500,
}
_CHOICES: dict[str, tuple[str, ...]] = {"category": REVIEW_CATEGORIES, "relation": RELATION_TYPES}
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


@dataclass(frozen=True)
class RelationHypothesis:
    """Гипотеза модели о связи двух требований — не связь и не доказательство."""

    index: int
    source_id: str
    target_id: str
    relation: RelationType
    source_quote: str
    target_quote: str
    explanation: str
    recommendation: str
    confidence: float


@dataclass(frozen=True)
class ParsedRelations:
    relations: tuple[RelationHypothesis, ...]
    rejected: tuple[RejectedHypothesis, ...]


def parse_hypotheses(text: str) -> ParsedAnswer:
    items = _load_items(text, "findings", "замечаний", "к одному требованию", MAX_ITEMS)
    hypotheses: list[Hypothesis] = []
    rejected: list[RejectedHypothesis] = []
    for index, item in enumerate(items):
        reasons = _item_problems(item, "findings", TEXT_FIELDS)
        if reasons:
            rejected.append(_rejected(index, item, "requirement_id", reasons))
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


def parse_relations(text: str) -> ParsedRelations:
    items = _load_items(text, "relations", "связей", "для одной пары", MAX_RELATIONS)
    relations: list[RelationHypothesis] = []
    rejected: list[RejectedHypothesis] = []
    for index, item in enumerate(items):
        reasons = _item_problems(item, "relations", RELATION_TEXT_FIELDS)
        if reasons:
            rejected.append(_rejected(index, item, "source_id", reasons))
        else:
            relations.append(
                RelationHypothesis(
                    index=index,
                    source_id=item["source_id"],
                    target_id=item["target_id"],
                    relation=item["relation"],
                    source_quote=item["source_quote"],
                    target_quote=item["target_quote"],
                    explanation=item["explanation"],
                    recommendation=item["recommendation"],
                    confidence=float(item["confidence"]),
                )
            )
    return ParsedRelations(tuple(relations), tuple(rejected))


def _load_items(text: str, key: str, noun: str, scope: str, limit: int) -> list[Any]:
    """Ответ целиком — один JSON-объект {key: [...]} без других полей; иначе ответ непригоден."""
    if not text.strip():
        raise ProviderResponseError("Ответ модели пуст.")
    fenced = _FENCE.match(text)
    try:
        data = json.loads(fenced.group("body") if fenced else text)
    except json.JSONDecodeError:
        raise ProviderResponseError(
            f'Ответ модели — не JSON по схеме {{"{key}": [...]}}.'
        ) from None
    if not isinstance(data, dict):
        raise ProviderResponseError(f'Ответ модели должен быть JSON-объектом {{"{key}": [...]}}.')
    if key not in data:
        raise ProviderResponseError(f"В ответе модели нет поля {key}.")
    unknown = sorted(set(data) - {key})
    if unknown:
        raise ProviderResponseError(f"В ответе модели неожиданные поля: {', '.join(unknown)}.")
    items = data[key]
    if not isinstance(items, list):
        raise ProviderResponseError(f"Поле {key} в ответе модели: ожидается список {noun}.")
    if len(items) > limit:
        raise ProviderResponseError(
            f"Модель вернула {len(items)} {noun} {scope} — больше допустимых {limit}."
        )
    return items


def _rejected(index: int, item: Any, id_field: str, reasons: list[str]) -> RejectedHypothesis:
    rid = item.get(id_field) if isinstance(item, dict) else None
    return RejectedHypothesis(index, rid if isinstance(rid, str) else None, tuple(reasons))


def _item_problems(item: Any, key_name: str, text_fields: dict[str, int | None]) -> list[str]:
    if not isinstance(item, dict):
        return [f"Элемент {key_name} должен быть объектом."]
    allowed = (*text_fields, "confidence")
    problems = [f"Недопустимое поле {key}." for key in sorted(set(item) - set(allowed))]
    for key, limit in text_fields.items():
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
        elif key in _CHOICES and value not in _CHOICES[key]:
            problems.append(f"Поле {key}: недопустимое значение «{value[:40]}».")
    if "confidence" not in item:
        problems.append("Нет поля confidence.")
    elif not is_probability(item["confidence"]):
        problems.append("Поле confidence: ожидается число от 0 до 1.")
    return problems
