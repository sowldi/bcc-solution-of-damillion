"""Модель замечания (Finding) и доказательств, реестр критериев качества: структура и JSON."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tests.seg_helpers import segment_markdown
from validator.criteria import CRITERIA, criterion
from validator.findings import (
    EVIDENCE_KINDS,
    RULE_CATEGORIES,
    RULE_CATEGORY_TITLES,
    SEVERITIES,
    SEVERITY_TITLES,
    Evidence,
    Finding,
    RejectedFinding,
    severity_rank,
)
from validator.model import Location
from validator.segmentation import Segmentation, SourceRef
from validator.segmentation.model import source_ref_from_dict, source_ref_to_dict

LOCATION = Location(("Сервис", "Функциональные требования"), line_start=5, line_end=5)


def _evidence(requirement_id: str = "Ф-1", start: int = 6, end: int = 14) -> Evidence:
    return Evidence(
        requirement_id=requirement_id,
        start=start,
        end=end,
        quote="Читатель",
        sources=(SourceRef(3, start, end, LOCATION),),
    )


def _finding(**changes: Any) -> Finding:
    finding = Finding(
        id="TST-01:Ф-1@6",
        rule_id="TST-01",
        rule_version="1",
        severity="major",
        category="testability",
        criterion="VER",
        title="Субъективная оценка качества",
        explanation="«удобным» — субъективная оценка.",
        recommendation="Замените оценку измеримым критерием.",
        requirement_ids=("Ф-1",),
        evidence=(_evidence(),),
        evidence_kind="quote",
    )
    return replace(finding, **changes)


def test_finding_defaults_are_unverified_rule_findings() -> None:
    finding = _finding()

    assert finding.evidence_status == "unverified"
    assert finding.origin == "rule"


def test_finding_json_round_trip() -> None:
    finding = _finding(evidence_status="verified")

    data = json.loads(json.dumps(finding.to_dict(), ensure_ascii=False))

    assert Finding.from_dict(data) == finding


def test_multi_requirement_finding_keeps_every_participant() -> None:
    first, second = _evidence("Ф-1"), _evidence("Ф-1~2", 0, 8)
    absent = Evidence("F-0000aaaa", 0, 0, "", (SourceRef(9, 0, 0, Location(table=1, row=3)),))
    finding = _finding(
        id="CON-01:Ф-1@6+Ф-1~2@0",
        rule_id="CON-01",
        requirement_ids=("Ф-1", "Ф-1~2", "F-0000aaaa"),
        evidence=(first, second, absent),
        evidence_kind="comparison",
    )

    restored = Finding.from_dict(finding.to_dict())

    assert restored == finding
    assert [e.requirement_id for e in restored.evidence] == ["Ф-1", "Ф-1~2", "F-0000aaaa"]
    assert restored.evidence[2].sources[0].location.table == 1


def test_rejected_finding_round_trip() -> None:
    rejected = RejectedFinding(
        _finding(evidence_status="rejected"), ("цитата не совпадает с текстом требования",)
    )

    assert RejectedFinding.from_dict(rejected.to_dict()) == rejected


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("severity", "blocker"),
        ("category", "style"),
        ("evidence_kind", "opinion"),
        ("evidence_status", "maybe"),
        ("origin", "human"),
    ],
)
def test_from_dict_rejects_unknown_enumeration_values(key: str, value: str) -> None:
    data = _finding().to_dict()
    data[key] = value

    with pytest.raises(ValueError, match=key):
        Finding.from_dict(data)


@pytest.mark.parametrize(
    ("key", "value"),
    [("requirement_ids", "Ф-1"), ("evidence", {}), ("title", 5), ("id", None)],
)
def test_from_dict_rejects_wrong_types(key: str, value: object) -> None:
    data = _finding().to_dict()
    data[key] = value

    with pytest.raises(ValueError, match=key):
        Finding.from_dict(data)


def test_evidence_offsets_must_be_integers() -> None:
    data = _finding().to_dict()
    data["evidence"][0]["start"] = "6"

    with pytest.raises(ValueError, match="start"):
        Finding.from_dict(data)


def test_severity_scale_and_titles() -> None:
    assert SEVERITIES == ("critical", "major", "minor", "info")
    assert [severity_rank(s) for s in SEVERITIES] == [0, 1, 2, 3]
    assert SEVERITY_TITLES["major"] == "Существенное"
    assert set(SEVERITY_TITLES) == set(SEVERITIES)


def test_rule_categories_cover_mandatory_groups() -> None:
    assert RULE_CATEGORIES == (
        "testability",
        "ambiguity",
        "completeness",
        "consistency",
        "traceability",
    )
    assert set(RULE_CATEGORY_TITLES) == set(RULE_CATEGORIES)
    assert EVIDENCE_KINDS == ("quote", "absence", "comparison")


# --- критерии качества ---------------------------------------------------------------


def test_criteria_cover_iso_29148_and_ieee_830_clauses() -> None:
    clauses = {(c.standard, c.clause) for c in CRITERIA.values()}

    for letter in "abcdefghi":
        assert ("ISO/IEC/IEEE 29148:2018", f"5.2.5 {letter}") in clauses
    for letter in "abcde":
        assert ("ISO/IEC/IEEE 29148:2018", f"5.2.6 {letter}") in clauses
    assert ("IEEE 830-1998", "4.3.7") in clauses
    assert ("IEEE 830-1998", "4.3.8") in clauses


def test_criteria_are_fully_described() -> None:
    for code, item in CRITERIA.items():
        assert item.code == code
        assert item.title and item.description
        assert item.reference == f"{item.standard}, п. {item.clause}"


def test_unknown_criterion_is_an_error() -> None:
    assert criterion("VER").title == "Проверяемость"
    with pytest.raises(KeyError, match="XYZ"):
        criterion("XYZ")


# --- ссылки на источник: общий формат JSON с сегментацией ------------------------------


def test_source_ref_dict_round_trip_keeps_segmentation_format() -> None:
    ref = SourceRef(7, 2, 9, LOCATION)

    data = source_ref_to_dict(ref)

    assert list(data) == ["block", "start", "end", "location"]
    assert source_ref_from_dict(data) == ref


def test_segmentation_json_is_unchanged_by_shared_source_helpers(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        "# Сервис\n\n## Функциональные требования\n\n* Ф-1. Читатель бронирует книгу.\n",
    )

    data = seg.to_dict()

    assert data["fragments"][-1]["sources"] == [
        source_ref_to_dict(ref) for ref in seg.fragments[-1].sources
    ]
    assert Segmentation.from_json(seg.to_json()) == seg


@pytest.mark.parametrize("broken", [[], "x", {"block": 1}])
def test_source_ref_from_dict_rejects_malformed_data(broken: object) -> None:
    with pytest.raises(ValueError):
        source_ref_from_dict(broken)
