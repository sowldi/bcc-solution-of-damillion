"""Evidence Gate для связей между требованиями (этап 7): минимальное расширение этапа 5.

Связь — замечание с полем relation: ровно два разных атомарных требования, по точной
цитате из каждого (сначала источник, затем цель). Подтверждает связь правилом только явная
ссылка на ID; связь от модели остаётся «unverified». Замечания без relation проверяются
как раньше.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from tests.graph_helpers import Built, load, relation_finding
from validator.evidence import build_evidence, check_finding, gate_findings
from validator.findings import Finding
from validator.graph import XRF_01

DEPENDENCY = (("Ф-2", "Ф-1"), ("Ф-1", "Ф-1. Библиотекарь подтверждает бронь книги"))
CONFLICT = (("Ф-3", "не хранит бронь книги дольше 3 дней"), ("Ф-4", "хранит бронь книги 5 дней"))


@pytest.fixture
def built(tmp_path: Path) -> Built:
    return load(tmp_path)


def _reasons(finding: Finding, built: Built) -> str:
    return " ".join(check_finding(finding, built.segmentation, built.document, rules=(XRF_01,)))


def _status(finding: Finding, built: Built) -> str:
    gate = gate_findings([finding], built.segmentation, built.document, rules=(XRF_01,))
    return gate.verdicts[0].status


def test_dependency_by_explicit_reference_is_verified(built: Built) -> None:
    finding = relation_finding(built, *DEPENDENCY, relation="depends_on", origin="rule")

    gate = gate_findings([finding], built.segmentation, built.document, rules=(XRF_01,))

    assert [f.id for f in gate.accepted] == [finding.id]
    [verdict] = gate.verdicts
    assert verdict.requirement_ids == ("Ф-2", "Ф-1")
    assert [c.quote for c in verdict.checked] == [
        "Ф-1",
        "Ф-1. Библиотекарь подтверждает бронь книги",
    ]


def test_llm_relation_with_exact_quotes_stays_unverified(built: Built) -> None:
    finding = relation_finding(built, *CONFLICT)

    gate = gate_findings([finding], built.segmentation, built.document, rules=(XRF_01,))

    assert gate.accepted == ()
    assert [f.evidence_status for f in gate.unverified] == ["unverified"]
    assert gate.unverified[0].relation == "conflicts_with"


def test_both_quotes_must_be_bound_in_source_then_target_order(built: Built) -> None:
    finding = relation_finding(built, *CONFLICT)
    swapped = replace(finding, evidence=finding.evidence[::-1])

    assert "первая — из требования-источника" in _reasons(swapped, built)


def test_relation_needs_exactly_two_requirements(built: Built) -> None:
    finding = relation_finding(built, *CONFLICT)
    one = replace(finding, requirement_ids=("Ф-3",), evidence=finding.evidence[:1])
    extra = build_evidence(built.segmentation.by_id("Ф-5"), 0, 3, built.document)
    three = replace(
        finding,
        requirement_ids=("Ф-3", "Ф-4", "Ф-5"),
        evidence=(*finding.evidence, extra),
    )

    assert "ровно к двум требованиям" in _reasons(one, built)
    assert "ровно к двум требованиям" in _reasons(three, built)


def test_relation_of_a_requirement_with_itself_is_rejected(built: Built) -> None:
    finding = relation_finding(built, ("Ф-3", "не хранит бронь"), ("Ф-3", "дольше 3 дней"))

    assert "с самим собой" in _reasons(finding, built)


def test_quote_taken_from_the_other_requirement_is_rejected(built: Built) -> None:
    finding = relation_finding(built, *CONFLICT)
    _, target = finding.evidence
    foreign = replace(target, requirement_id="Ф-3")  # цитата Ф-4 выдана за цитату Ф-3

    reasons = _reasons(replace(finding, evidence=(foreign, target)), built)

    assert "не совпадает" in reasons or "ссылки на источник" in reasons


def test_unknown_requirement_id_is_rejected(built: Built) -> None:
    finding = relation_finding(built, *CONFLICT)
    source, target = finding.evidence

    reasons = _reasons(
        replace(
            finding,
            requirement_ids=("Ф-9", "Ф-4"),
            evidence=(replace(source, requirement_id="Ф-9"), target),
        ),
        built,
    )

    assert "Ф-9 не найдено" in reasons


@pytest.mark.parametrize("target", ["container", "heading"])
def test_relation_end_must_be_an_atomic_requirement(built: Built, target: str) -> None:
    fragment = (
        built.segmentation.by_id("Ф-6")  # составное: Ф-6/1 и Ф-6/2
        if target == "container"
        else next(f for f in built.segmentation.fragments if f.role == "heading")
    )
    finding = relation_finding(built, ("Ф-7", "Журнал хранится"), (fragment.id, fragment.text[:5]))

    assert "не атомарное требование" in _reasons(finding, built)


def test_relation_is_proven_by_comparison_of_two_quotes(built: Built) -> None:
    finding = replace(relation_finding(built, *CONFLICT), evidence_kind="quote")

    assert "comparison" in _reasons(finding, built)


def test_unknown_relation_type_is_rejected_not_crashing(built: Built) -> None:
    finding = replace(relation_finding(built, *CONFLICT), relation=cast(Any, "contradicts"))

    assert _status(finding, built) == "rejected"
    assert "relation" in _reasons(finding, built)


def test_rule_dependency_must_quote_the_target_id(built: Built) -> None:
    forged = relation_finding(
        built,
        ("Ф-2", "Читатель отменяет бронь"),
        ("Ф-1", "Библиотекарь подтверждает бронь"),
        relation="depends_on",
        origin="rule",
    )

    assert _status(forged, built) == "rejected"
    assert "явной ссылкой" in _reasons(forged, built)


@pytest.mark.parametrize("relation", ["conflicts_with", "related_to", "requires_condition"])
def test_rule_cannot_confirm_a_semantic_relation(built: Built, relation: str) -> None:
    finding = relation_finding(built, *DEPENDENCY, relation=cast(Any, relation), origin="rule")

    assert _status(finding, built) == "rejected"
    assert "depends_on" in _reasons(finding, built)


def test_llm_relation_relabeled_as_rule_is_rejected(built: Built) -> None:
    finding = replace(relation_finding(built, *CONFLICT), origin="rule")

    assert _status(finding, built) == "rejected"


def test_rule_relation_outside_the_catalog_is_rejected(built: Built) -> None:
    finding = relation_finding(built, *DEPENDENCY, relation="depends_on", origin="rule")

    gate = gate_findings([finding], built.segmentation, built.document, rules=())

    assert gate.verdicts[0].status == "rejected"
    assert "каталог" in " ".join(gate.verdicts[0].reasons)


def test_ambiguous_target_id_does_not_confirm_dependency(tmp_path: Path) -> None:
    built = load(
        tmp_path,
        "# Сервис\n\n## Требования\n\n"
        "* Ф-1. Библиотекарь выдаёт книгу читателю.\n"
        "* Ф-1. Библиотекарь принимает возврат книги.\n"
        "* Ф-2. Читатель продлевает выдачу согласно Ф-1.\n",
    )
    target = next(f for f in built.segmentation.atomic_requirements if f.doc_id == "Ф-1")
    finding = relation_finding(
        built, ("Ф-2", "Ф-1"), (target.id, target.text), relation="depends_on", origin="rule"
    )

    assert _status(finding, built) == "rejected"
    assert "неоднозначна" in _reasons(finding, built)


def test_relation_built_for_a_changed_document_is_rejected(tmp_path: Path) -> None:
    original = load(tmp_path)
    finding = relation_finding(original, *CONFLICT)
    changed = load(tmp_path, original.document.source.read_text(encoding="utf-8") + "\n")

    reasons = " ".join(check_finding(finding, changed.segmentation, changed.document))

    assert "другой версии документа" in reasons


def test_relation_cannot_carry_review_category(built: Built) -> None:
    finding = replace(relation_finding(built, *CONFLICT), review_category="clarity")

    assert "review_category" in _reasons(finding, built)


def test_relation_round_trips_and_plain_findings_keep_their_json(built: Built) -> None:
    finding = relation_finding(built, *CONFLICT)
    plain = replace(finding, relation=None, requirement_ids=("Ф-3", "Ф-4"))

    assert Finding.from_dict(finding.to_dict()) == finding
    assert finding.to_dict()["relation"] == "conflicts_with"
    assert "relation" not in plain.to_dict()  # JSON замечаний этапов 5–6 не меняется
    assert Finding.from_dict(plain.to_dict()) == plain


def test_from_dict_rejects_unknown_relation(built: Built) -> None:
    data = relation_finding(built, *CONFLICT).to_dict()
    data["relation"] = "contradicts"

    with pytest.raises(ValueError, match="relation"):
        Finding.from_dict(data)
