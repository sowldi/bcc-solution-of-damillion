"""Каталог детерминированных правил: уникальные ID, полное описание, независимый запуск."""

from __future__ import annotations

import re
from pathlib import Path

from tests.lint_helpers import lint_markdown
from validator.criteria import CRITERIA
from validator.evidence import gate_findings
from validator.findings import EVIDENCE_KINDS, RULE_CATEGORIES, SEVERITIES
from validator.lint import CATALOG, CATALOG_VERSION, build_context, rule_by_id, run_rule

PREFIXES = {
    "testability": "TST",
    "ambiguity": "AMB",
    "completeness": "CMP",
    "consistency": "CON",
    "traceability": "TRC",
}

# Документ, на котором срабатывает большинство правил (собственные формулировки).
RICH = """# 1. Сервис бронирования книг

## 1.1 Термины

* АИБС — автоматизированная информационная библиотечная система.

## 1.2 Функциональные требования

* Ф-1. Каталог книг должен быть удобным, а поиск работает быстро.
* Ф-2. По возможности система продлевает бронь автоматически.
* Ф-3. При необходимости библиотекарь отменяет бронь.
* Ф-4. Читатель может забронировать несколько книг.
* Ф-5. Бронь хранится около 3 дней.
* Ф-6. Система своевременно уведомляет читателя.
* Ф-7. Срок выдачи: TBD.
* Ф-8. Каталог хранит автора, жанр, год и т.д.
* Ф-9. Продление выполняется согласно Ф-40.
* Ф-9. Формат отчёта описан в разделе 1.7.
* Читатель получает уведомление по почте.
* Ф-10. Читатель бронирует книгу через каталог.
* Ф-11. Читатель бронирует книгу через каталог.
* Бронь передаётся в ЕКБ.

## 1.3 Производительность

* Система должна выдерживать пиковую нагрузку.
* Система минимизирует время ожидания читателя.

## 1.4 Параметры

| Параметр | Значение |
| --- | --- |
| Срок брони | 3 дня |
| Штраф | — |
| Срок брони | 5 дней |

## 1.5 Согласование

| Роль | ФИО | Дата |
| --- | --- | --- |
| | | |
"""


def test_rule_ids_are_unique_and_well_formed() -> None:
    ids = [rule.rule_id for rule in CATALOG]

    assert len(ids) == len(set(ids))
    for rule in CATALOG:
        assert re.fullmatch(r"(TST|AMB|CMP|CON|TRC)-\d{2}", rule.rule_id), rule.rule_id
        assert rule.rule_id.startswith(PREFIXES[rule.category]), rule.rule_id


def test_every_rule_is_fully_described() -> None:
    for rule in CATALOG:
        assert re.fullmatch(r"\d+", rule.version), rule.rule_id
        assert rule.severity in SEVERITIES, rule.rule_id
        assert rule.category in RULE_CATEGORIES, rule.rule_id
        assert rule.criterion in CRITERIA, rule.rule_id
        assert rule.evidence_kind in EVIDENCE_KINDS, rule.rule_id
        for text in (rule.title, rule.description, rule.trigger, rule.rationale):
            assert text.strip(), rule.rule_id
        assert rule.recommendation.strip(), rule.rule_id
        assert rule.granularity in ("phrase", "fragment"), rule.rule_id
        assert callable(rule.check)


def test_catalog_covers_every_mandatory_category() -> None:
    assert {rule.category for rule in CATALOG} == set(RULE_CATEGORIES)


def test_deterministic_rules_do_not_claim_critical_severity() -> None:
    """Критичность — смысловая оценка; формальные признаки не дают на неё права."""
    assert all(rule.severity != "critical" for rule in CATALOG)


def test_catalog_has_a_version() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", CATALOG_VERSION)


def test_rule_lookup_by_id() -> None:
    assert rule_by_id("TST-01").rule_id == "TST-01"


def test_each_rule_runs_on_its_own(tmp_path: Path) -> None:
    linted = lint_markdown(tmp_path, RICH)
    context = build_context(linted.document, linted.segmentation)
    union: set[str] = set()

    for rule in CATALOG:
        found = run_rule(rule, context)
        assert all(finding.rule_id == rule.rule_id for finding in found), rule.rule_id
        gate = gate_findings(found, linted.segmentation, linted.document, rules=CATALOG)
        assert gate.rejected == (), (rule.rule_id, gate.rejected)
        union |= {finding.id for finding in gate.accepted}

    final = {finding.id for finding in linted.result.findings}
    assert final <= union
    # почти все правила каталога срабатывают на документе-образце
    assert len({f.rule_id for f in linted.result.findings}) >= 18


def test_rule_order_does_not_change_the_result(tmp_path: Path) -> None:
    forward = lint_markdown(tmp_path, RICH)
    backward = lint_markdown(tmp_path, RICH, rules=tuple(reversed(CATALOG)), name="b.md")

    assert [f.id for f in forward.result.findings] == [f.id for f in backward.result.findings]
