"""Stage 8: противоречия, критерии приёмки, полнота.

Подтверждённое противоречие — только с машинным контрпримером (ограничения одного
параметра в одном разделе не пересекаются). Всё остальное — потенциальное, с указанием,
что не проверено. Гипотезы LLM подтверждёнными не становятся. Документы — собственные
(домен «библиотека»); сети и ключей нет.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.graph_helpers import Built, load, relation_finding
from tests.lint_helpers import HEAD
from validator.evidence import check_finding, gate_findings
from validator.quality import (
    ACC_01,
    CTR_01,
    STAGE8_RULES,
    analyze_quality,
    compare_constraints,
    find_constraint,
    parse_constraint,
)


def _quality(tmp_path: Path, markdown: str):  # type: ignore[no-untyped-def]
    built = load(tmp_path, markdown)
    return analyze_quality(built.document, built.segmentation, built.lint, built.graph), built


def _items(markdown: str) -> str:
    return HEAD + markdown


# --- числовые ограничения -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("не более 3 дней", "≤ 3"),
        ("до 5 дней", "≤ 5"),
        ("не менее 2 часов", "≥ 2"),
        ("от 1 до 5 минут", "от 1 до 5"),
        ("1–5 минут", "от 1 до 5"),
        ("10 MB", "= 10"),
        ("свыше 1 000 ₸", "> 1000"),
        ("менее 0,5 секунды", "< 0.5"),
    ],
)
def test_constraint_values_are_parsed(text: str, expected: str) -> None:
    constraint = parse_constraint(text)

    assert constraint is not None
    assert constraint.describe() == expected


@pytest.mark.parametrize(
    "text",
    [
        "не позднее чем через 3 дня после оплаты",
        "быстро",
        "3 дня или 5 дней",
        "TBD",
        "",
    ],
)
def test_values_with_extra_words_are_not_guessed(text: str) -> None:
    assert parse_constraint(text) is None


@pytest.mark.parametrize(
    ("first", "second", "verdict"),
    [
        ("не более 3 дней", "5 дней", "disjoint"),
        ("не более 5 дней", "3 дня", "compatible"),
        ("3 дня", "72 часа", "compatible"),  # единицы пересчитываются
        ("не более 1 дня", "48 часов", "disjoint"),
        ("не более 2 MB", "2048 KB", "compatible"),
        ("5 попыток", "10 минут", "incomparable"),
        ("не более 2 месяцев", "90 дней", "incomparable"),  # месяц не равен 30 дням
        ("до 5 попыток", "7 попыток", "disjoint"),
        ("менее 3 дней", "3 дня", "disjoint"),  # граница исключена
        ("не более 3 дней", "3 дня", "compatible"),
    ],
)
def test_constraint_comparison(first: str, second: str, verdict: str) -> None:
    a, b = parse_constraint(first), parse_constraint(second)
    assert a is not None and b is not None

    assert compare_constraints(a, b).verdict == verdict


def test_free_text_search_refuses_negated_or_ambiguous_quotes() -> None:
    assert find_constraint("хранит бронь не более 3 дней") is not None
    assert find_constraint("не хранит бронь книги дольше 3 дней") is None  # отрицание глагола
    assert find_constraint("от 3 до 5 дней или 7 дней") is None  # два ограничения
    assert find_constraint("хранит бронь книги") is None


# --- противоречия ---------------------------------------------------------------------------


def test_disjoint_limits_of_one_parameter_are_a_confirmed_contradiction(tmp_path: Path) -> None:
    quality, _ = _quality(
        tmp_path,
        _items(
            "* Ф-1. Срок хранения брони: не более 3 дней.\n* Ф-2. Срок хранения брони — 5 дней.\n"
        ),
    )

    [item] = quality.confirmed
    assert (item.source, item.status, item.requirement_ids) == (
        "CTR-01",
        "confirmed",
        ("Ф-1", "Ф-2"),
    )
    assert item.numeric is not None and item.numeric.verdict == "disjoint"
    [finding] = [f for f in quality.findings if f.rule_id == "CTR-01"]
    assert finding.evidence_status == "verified"
    assert [e.requirement_id for e in finding.evidence] == ["Ф-1", "Ф-2"]
    assert "не пересекаются" in finding.explanation
    assert quality.rejected == ()


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Срок хранения брони: не более 5 дней.", "Срок хранения брони — 3 дня."),
        ("Срок хранения брони: 3 дня.", "Срок хранения брони — 72 часа."),
        ("Максимальный размер файла — 10 MB.", "Максимальный размер изображения — 20 MB."),
        (
            "Время ответа в рабочее время: 5 минут.",
            "Время ответа в нерабочее время: 30 минут.",
        ),
        ("Число попыток входа: 5 попыток.", "Число попыток входа: 10 минут."),
        ("Срок хранения брони: не позднее чем через 3 дня.", "Срок хранения брони: 5 дней."),
    ],
)
def test_no_contradiction_without_a_machine_counterexample(
    tmp_path: Path, first: str, second: str
) -> None:
    quality, _ = _quality(tmp_path, _items(f"* Ф-1. {first}\n* Ф-2. {second}\n"))

    assert quality.confirmed == ()
    assert [f.rule_id for f in quality.findings if f.rule_id == "CTR-01"] == []


def test_same_parameter_in_different_sections_is_only_potential(tmp_path: Path) -> None:
    quality, _ = _quality(
        tmp_path,
        _items("* Ф-1. Срок хранения: 3 дня.\n\n")
        + "## Нефункциональные требования\n\n* Н-1. Срок хранения: 5 дней.\n",
    )

    assert quality.confirmed == ()
    [item] = quality.potential
    assert (item.source, item.status) == ("CTR-02", "potential")
    assert "разных разделах" in item.explanation


@pytest.mark.parametrize(
    ("markdown", "reason"),
    [
        (
            "* Бронь книги:\n  * Срок: 3 дня.\n* Бронь зала:\n  * Срок: 5 дней.\n",
            "группах списка",
        ),
        ("Для брони книги:\n\n* Срок: 3 дня.\n\nДля брони зала:\n\n* Срок: 5 дней.\n", "группах"),
        (
            "* Ф-1. Читатель бронирует книгу.\n\nБронь книги:\n\n| Параметр | Значение |\n"
            "| --- | --- |\n| Срок | 3 дня |\n\nБронь зала:\n\n| Параметр | Значение |\n"
            "| --- | --- |\n| Срок | 5 дней |\n",
            "таблицах",
        ),
        ("* Версия API: 2.\n* Версия API: 3.\n", "без единицы"),
    ],
)
def test_sub_contexts_and_bare_numbers_are_only_potential(
    tmp_path: Path, markdown: str, reason: str
) -> None:
    quality, _ = _quality(tmp_path, _items(markdown))

    assert quality.confirmed == ()
    [item] = quality.potential
    assert item.source == "CTR-02"
    assert reason in item.explanation


def test_table_pairs_come_from_con03_and_are_not_duplicated(tmp_path: Path) -> None:
    table = (
        "\n## Параметры\n\n| Параметр | Значение |\n| --- | --- |\n"
        "| Срок хранения брони | не более 3 дней |\n| Срок хранения брони | 5 дней |\n"
        "| Лимит броней | до 5 броней |\n| Лимит броней | 3 брони |\n"
    )
    quality, _ = _quality(tmp_path, _items("* Ф-1. Читатель бронирует книгу.\n") + table)

    assert [f.rule_id for f in quality.findings if f.rule_id.startswith("CTR")] == []
    sources = {
        (item.source, item.status, item.numeric.verdict if item.numeric else None)
        for item in quality.contradictions
    }
    assert ("CON-03", "confirmed", "disjoint") in sources
    assert ("CON-03", "potential", "compatible") in sources  # «до 5» и «3» совместимы


def test_llm_conflict_hypotheses_stay_potential_with_numeric_note(tmp_path: Path) -> None:
    built = load(
        tmp_path,
        _items(
            "* Ф-1. Система хранит бронь книги не более 3 дней.\n"
            "* Ф-2. Система хранит бронь книги 5 дней после подтверждения.\n"
            "* Ф-3. Читатель получает напоминание о брони книги.\n"
        ),
    )
    disjoint = replace(
        relation_finding(built, ("Ф-1", "не более 3 дней"), ("Ф-2", "хранит бронь книги 5 дней")),
        evidence_status="unverified",
    )
    related = replace(
        relation_finding(
            built, ("Ф-1", "хранит бронь"), ("Ф-3", "напоминание о брони"), relation="related_to"
        ),
        evidence_status="unverified",
    )
    quality = analyze_quality(built.document, built.segmentation, built.lint, built.graph)

    merged = quality.with_hypotheses((disjoint, related))

    [item] = merged.potential
    assert (item.source, item.status) == ("LLM-PAIR", "potential")
    assert item.numeric is not None and item.numeric.verdict == "disjoint"
    assert merged.confirmed == ()


def test_stage8_rule_findings_pass_the_existing_gate_only_with_their_catalog(
    tmp_path: Path,
) -> None:
    quality, built = _quality(
        tmp_path,
        _items(
            "* Ф-1. Срок хранения брони: не более 3 дней.\n* Ф-2. Срок хранения брони — 5 дней.\n"
        ),
    )
    [finding] = quality.findings

    assert check_finding(finding, built.segmentation, built.document, rules=STAGE8_RULES) == ()
    no_catalog = gate_findings([finding], built.segmentation, built.document, rules=())
    assert no_catalog.verdicts[0].status == "rejected"
    tampered = replace(
        finding, evidence=(replace(finding.evidence[0], quote="5 дней"), *finding.evidence[1:])
    )
    assert check_finding(tampered, built.segmentation, built.document, rules=STAGE8_RULES)
    assert CTR_01.evidence_kind == "comparison"


def test_repeated_analysis_is_identical(tmp_path: Path) -> None:
    markdown = _items(
        "* Ф-1. Срок хранения брони: не более 3 дней.\n* Ф-2. Срок хранения брони — 5 дней.\n"
    )
    first, _ = _quality(tmp_path, markdown)
    second, _ = _quality(tmp_path, markdown)

    assert first.to_json() == second.to_json()


# --- критерии приёмки -------------------------------------------------------------------------

ACCEPTANCE = "\n## Критерии приёмки\n\n"


def test_complete_given_when_then_is_accepted(tmp_path: Path) -> None:
    quality, _ = _quality(
        tmp_path,
        _items("* Ф-1. Читатель бронирует книгу.\n")
        + ACCEPTANCE
        + "* Дано: читатель авторизован.\n* Когда читатель бронирует книгу.\n"
        + "* Тогда бронь появляется в личном кабинете.\n",
    )

    assert [f for f in quality.findings if f.rule_id == "ACC-01"] == []
    assert len(quality.acceptance) == 3


def test_scenario_without_expected_result_is_reported(tmp_path: Path) -> None:
    quality, _ = _quality(
        tmp_path,
        _items("* Ф-1. Читатель бронирует книгу.\n")
        + ACCEPTANCE
        + "* Дано: читатель авторизован.\n* Когда читатель бронирует книгу.\n\n"
        + "## Ограничения\n\n* О-1. Сервис работает круглосуточно.\n",
    )

    [finding] = [f for f in quality.findings if f.rule_id == "ACC-01"]
    assert finding.evidence_status == "verified"
    assert [e.quote for e in finding.evidence] == [
        "Дано: читатель авторизован.",
        "Когда читатель бронирует книгу.",
    ]
    assert ACC_01.title in finding.title


@pytest.mark.parametrize(
    ("text", "flagged"),
    [
        ("ПК-1. Дано читатель без абонемента, когда он бронирует книгу.", True),
        (
            "ПК-1. Дано читатель без абонемента, когда он бронирует книгу, тогда бронь отклонена.",
            False,
        ),
        ("ПК-1. Если оплата не прошла, система показывает ошибку.", False),
    ],
)
def test_inline_scenarios(tmp_path: Path, text: str, flagged: bool) -> None:
    quality, _ = _quality(
        tmp_path, _items("* Ф-1. Читатель бронирует книгу.\n") + ACCEPTANCE + f"* {text}\n"
    )

    assert any(f.rule_id == "ACC-01" for f in quality.findings) is flagged


def test_acceptance_links_to_requirements_are_traced(tmp_path: Path) -> None:
    quality, _ = _quality(
        tmp_path,
        _items("* Ф-1. Читатель бронирует книгу.\n* Ф-2. Читатель отменяет бронь.\n")
        + ACCEPTANCE
        + "* ПК-1. Бронь по Ф-1 появляется в личном кабинете за 5 секунд.\n",
    )

    assert quality.acceptance_links == ("Ф-1",)
    assert quality.unlinked_requirements == ("Ф-2",)


def test_document_without_acceptance_criteria(tmp_path: Path) -> None:
    quality, _ = _quality(tmp_path, _items("* Ф-1. Читатель бронирует книгу.\n"))

    assert quality.acceptance == ()
    assert quality.acceptance_links == ()


def test_shared_built_helper_is_used(tmp_path: Path) -> None:
    built: Built = load(tmp_path)
    quality = analyze_quality(built.document, built.segmentation, built.lint, built.graph)

    assert quality.document_sha256 == built.document.sha256


def test_model_hypothesis_on_a_confirmed_pair_is_not_duplicated(tmp_path: Path) -> None:
    built = load(
        tmp_path,
        _items(
            "* Ф-1. Срок хранения брони: не более 3 дней.\n* Ф-2. Срок хранения брони — 5 дней.\n"
        ),
    )
    hypothesis = replace(
        relation_finding(built, ("Ф-1", "не более 3 дней"), ("Ф-2", "5 дней")),
        evidence_status="unverified",
    )
    quality = analyze_quality(built.document, built.segmentation, built.lint, built.graph)

    merged = quality.with_hypotheses((hypothesis,))

    assert merged.potential == ()
    assert merged.hypotheses_confirmed == (hypothesis.id,)
    assert len(merged.confirmed) == 1
