"""Строгая схема ответа модели.

Ответ целиком: один JSON-объект {"findings": [...]} без других полей (допускается обёртка
целиком в один блок ```json … ```). Нарушение — ProviderResponseError: ответ непригоден.
Каждый элемент проверяется отдельно: все поля обязательны, других полей нет, типы, допустимые
значения и длины — по схеме. Неверный элемент отклоняется с причинами, а не «чинится»
(обрезкой, сменой регистра, подстановкой значения).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from validator.criteria import CRITERIA
from validator.findings import RULE_CATEGORIES, SEVERITIES, RuleCategory, Severity
from validator.llm.errors import ProviderResponseError

MAX_ITEMS = 50
# Поле → наибольшая длина (None — значение из перечня).
FIELDS: dict[str, int | None] = {
    "requirement_id": 64,
    "category": None,
    "severity": None,
    "criterion": None,
    "title": 150,
    "explanation": 1000,
    "recommendation": 500,
    "quote": 500,
}
_ALLOWED: dict[str, frozenset[str]] = {
    "category": frozenset(RULE_CATEGORIES),
    "severity": frozenset(SEVERITIES),
    "criterion": frozenset(CRITERIA),
}
_FENCE = re.compile(r"\A\s*```(?:json)?[ \t]*\n(?P<body>.*)\n\s*```\s*\Z", re.DOTALL)


@dataclass(frozen=True)
class Hypothesis:
    """Гипотеза модели о дефекте — не замечание и не доказательство.

    index — номер элемента в списке findings ответа модели.
    """

    index: int
    requirement_id: str
    category: RuleCategory
    severity: Severity
    criterion: str
    title: str
    explanation: str
    recommendation: str
    quote: str


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
            f"Модель вернула {len(items)} замечаний — больше допустимых {MAX_ITEMS}."
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
            hypotheses.append(Hypothesis(index=index, **item))
    return ParsedAnswer(tuple(hypotheses), tuple(rejected))


def _item_problems(item: Any) -> list[str]:
    if not isinstance(item, dict):
        return ["Элемент findings должен быть объектом."]
    problems = [f"Недопустимое поле {key}." for key in sorted(set(item) - set(FIELDS))]
    for key, limit in FIELDS.items():
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
        elif key in _ALLOWED and value not in _ALLOWED[key]:
            problems.append(f"Поле {key}: недопустимое значение «{value[:40]}».")
    return problems
