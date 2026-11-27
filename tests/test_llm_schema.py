"""Строгая схема ответа модели: ответ целиком или неприемлем, или разбирается по элементам."""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.llm_helpers import findings_json, hypothesis
from validator.llm.errors import ProviderResponseError
from validator.llm.schema import MAX_ITEMS, parse_hypotheses


def test_valid_answer_is_parsed() -> None:
    parsed = parse_hypotheses(findings_json(hypothesis("Ф-1", "в течение дня")))

    [item] = parsed.hypotheses
    assert item.requirement_id == "Ф-1"
    assert item.evidence_quote == "в течение дня"
    assert item.category == "clarity"
    assert item.issue == "Неоднозначный срок"
    assert item.reasoning == "Срок не задан в единицах времени."
    assert item.confidence == 0.7
    assert parsed.rejected == ()


def test_no_findings_is_valid() -> None:
    parsed = parse_hypotheses('{"findings": []}')

    assert parsed.hypotheses == ()
    assert parsed.rejected == ()


def test_whole_answer_in_one_json_fence_is_accepted() -> None:
    text = "```json\n" + findings_json(hypothesis("Ф-1", "срок")) + "\n```"

    assert len(parse_hypotheses(text).hypotheses) == 1


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", "пуст"),
        ("   \n", "пуст"),
        ("Вот мои замечания: …", "JSON"),
        ('{"findings": [', "JSON"),
        ("[]", "объект"),
        ('{"items": []}', "findings"),
        ('{"findings": [], "verdict": "ok"}', "verdict"),
        ('{"findings": {}}', "список"),
        ('Ответ: {"findings": []}', "JSON"),
    ],
)
def test_unusable_answer_is_a_controlled_error(text: str, reason: str) -> None:
    with pytest.raises(ProviderResponseError, match=reason):
        parse_hypotheses(text)


def test_too_many_items_is_an_error() -> None:
    items = [hypothesis("Ф-1", "срок")] * (MAX_ITEMS + 1)

    with pytest.raises(ProviderResponseError, match=str(MAX_ITEMS)):
        parse_hypotheses(findings_json(*items))


def _rejected_reasons(item: Any) -> str:
    parsed = parse_hypotheses(json.dumps({"findings": [item]}, ensure_ascii=False))
    assert parsed.hypotheses == ()
    [rejected] = parsed.rejected
    return " | ".join(rejected.reasons)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"requirement_id": ""}, "requirement_id"),
        ({"evidence_quote": ""}, "evidence_quote"),
        ({"evidence_quote": 5}, "evidence_quote"),
        ({"category": "style"}, "category"),
        ({"category": "ambiguity"}, "category"),  # группа правил, а не категория обзора
        ({"issue": "т" * 151}, "issue"),
        ({"reasoning": "о" * 1001}, "reasoning"),
        ({"recommendation": "р" * 501}, "recommendation"),
        ({"evidence_quote": "ц" * 501}, "evidence_quote"),
        ({"confidence": 1.5}, "confidence"),
        ({"confidence": -0.1}, "confidence"),
        ({"confidence": "высокая"}, "confidence"),
        ({"confidence": True}, "confidence"),
        ({"severity": "major"}, "severity"),  # важность назначает система, не модель
        ({"evidence_status": "verified"}, "evidence_status"),
    ],
)
def test_invalid_item_is_rejected_with_reason(changes: dict[str, Any], reason: str) -> None:
    item = hypothesis("Ф-1", "срок")
    item.update(changes)

    assert reason in _rejected_reasons(item)


def test_missing_field_is_rejected() -> None:
    item = hypothesis("Ф-1", "срок")
    del item["reasoning"]

    assert "reasoning" in _rejected_reasons(item)


def test_item_that_is_not_an_object_is_rejected() -> None:
    assert "объект" in _rejected_reasons("строка вместо объекта")


def test_valid_items_survive_next_to_invalid_ones() -> None:
    good = hypothesis("Ф-1", "срок")
    bad = hypothesis("Ф-2", "срок", category="blocker")

    parsed = parse_hypotheses(findings_json(good, bad))

    assert [h.requirement_id for h in parsed.hypotheses] == ["Ф-1"]
    assert [r.index for r in parsed.rejected] == [1]
    assert parsed.rejected[0].requirement_id == "Ф-2"


def test_values_are_not_silently_repaired() -> None:
    """Неверное значение не «чинится» (обрезкой, приведением регистра) — элемент отклоняется."""
    parsed = parse_hypotheses(findings_json(hypothesis("Ф-1", "срок", category="CLARITY")))

    assert parsed.hypotheses == ()
