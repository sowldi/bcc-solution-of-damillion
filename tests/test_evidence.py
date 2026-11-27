"""Базовый Evidence Gate: замечание без проверяемого доказательства не проходит."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from tests.seg_helpers import unit
from validator.evidence import absence_evidence, build_evidence, check_finding, gate_findings
from validator.findings import Evidence, EvidenceKind, Finding, Severity
from validator.ingest import load_document
from validator.model import Document, Location
from validator.segmentation import Requirement, Segmentation, segment_document

SPEC = """# Сервис бронирования

## Функциональные требования

* Ф-1. Читатель бронирует книгу через каталог.

Библиотекарь подтверждает бронь

в течение рабочего дня.

| Роль | Дата |
| --- | --- |
| | |
"""

Pair = tuple[Segmentation, Document]


@pytest.fixture
def pair(tmp_path: Path) -> Pair:
    path = tmp_path / "spec.md"
    path.write_text(SPEC, encoding="utf-8")
    document = load_document(path)
    return segment_document(document), document


def _finding(
    document: Document, *spans: tuple[Requirement, int, int], kind: EvidenceKind = "quote"
) -> Finding:
    evidence = tuple(build_evidence(fragment, s, e, document) for fragment, s, e in spans)
    return Finding(
        id="TEST-01:" + "+".join(f"{f.id}@{s}" for f, s, _ in spans),
        rule_id="TEST-01",
        rule_version="1",
        severity="minor",
        category="ambiguity",
        criterion="UNA",
        title="Проверочное замечание",
        explanation="Пояснение.",
        recommendation="Рекомендация.",
        requirement_ids=tuple(dict.fromkeys(f.id for f, _, _ in spans)),
        evidence=evidence,
        evidence_kind=kind,
    )


def _reader(seg: Segmentation) -> tuple[Requirement, int, int]:
    fragment = seg.by_id("Ф-1")
    start = fragment.text.index("Читатель")
    return fragment, start, start + len("Читатель")


def _empty_row(seg: Segmentation) -> Requirement:
    return next(f for f in seg.fragments if f.text == "" and f.role != "table_header")


def _reasons(finding: Finding, pair: Pair) -> str:
    return " | ".join(check_finding(finding, *pair))


def test_valid_finding_passes(pair: Pair) -> None:
    seg, document = pair
    finding = _finding(document, _reader(seg))

    assert check_finding(finding, seg, document) == ()
    result = gate_findings([finding], seg, document)
    assert result.accepted == (replace(finding, evidence_status="verified"),)
    assert result.rejected == ()


def test_build_evidence_takes_exact_quote_and_sources(pair: Pair) -> None:
    seg, document = pair
    fragment, start, end = _reader(seg)

    evidence = build_evidence(fragment, start, end, document)

    assert evidence.quote == "Читатель"
    assert evidence.requirement_id == "Ф-1"
    [ref] = evidence.sources
    assert document.blocks[ref.block].text[ref.start : ref.end] == "Читатель"
    assert ref.location.line_start == 5


def test_quote_across_blocks_passes(pair: Pair) -> None:
    seg, document = pair
    fragment = unit(seg, "Библиотекарь подтверждает")
    start = fragment.text.index("бронь")
    end = fragment.text.index("рабочего")

    finding = _finding(document, (fragment, start, end))

    assert "\n" in finding.evidence[0].quote
    assert len(finding.evidence[0].sources) == 2
    assert check_finding(finding, seg, document) == ()


def test_tampered_quote_is_rejected(pair: Pair) -> None:
    seg, document = pair
    finding = _finding(document, _reader(seg))
    forged = replace(finding, evidence=(replace(finding.evidence[0], quote="Библиотекарь"),))

    assert "цитата" in _reasons(forged, pair)
    result = gate_findings([forged], seg, document)
    assert result.accepted == ()
    [rejected] = result.rejected
    assert rejected.finding.evidence_status == "rejected"
    assert any("цитата" in reason for reason in rejected.reasons)


def test_unknown_requirement_is_rejected(pair: Pair) -> None:
    seg, document = pair
    finding = _finding(document, _reader(seg))
    ghost = replace(
        finding,
        requirement_ids=("Ф-404",),
        evidence=(replace(finding.evidence[0], requirement_id="Ф-404"),),
    )

    assert "Ф-404" in _reasons(ghost, pair)
    assert "не найден" in _reasons(ghost, pair)


def test_shifted_span_with_old_quote_is_rejected(pair: Pair) -> None:
    seg, document = pair
    finding = _finding(document, _reader(seg))
    evidence = finding.evidence[0]
    shifted = replace(
        finding, evidence=(replace(evidence, start=evidence.start + 1, end=evidence.end + 1),)
    )

    assert "цитата" in _reasons(shifted, pair)


def test_span_outside_fragment_is_rejected(pair: Pair) -> None:
    seg, document = pair
    finding = _finding(document, _reader(seg))
    evidence = finding.evidence[0]
    outside = replace(finding, evidence=(replace(evidence, end=10_000),))

    assert "отрезок" in _reasons(outside, pair)


def test_wrong_source_reference_is_rejected(pair: Pair) -> None:
    seg, document = pair
    finding = _finding(document, _reader(seg))
    evidence = finding.evidence[0]
    [ref] = evidence.sources
    moved = replace(ref, location=Location(("Другой раздел",), line_start=99, line_end=99))
    forged = replace(finding, evidence=(replace(evidence, sources=(moved,)),))

    assert "источник" in _reasons(forged, pair)


def test_missing_sources_are_rejected(pair: Pair) -> None:
    seg, document = pair
    finding = _finding(document, _reader(seg))
    stripped = replace(finding, evidence=(replace(finding.evidence[0], sources=()),))

    assert "источник" in _reasons(stripped, pair)


def test_span_starting_on_block_separator_is_rejected(pair: Pair) -> None:
    seg, document = pair
    fragment = unit(seg, "Библиотекарь подтверждает")
    separator = fragment.text.index("\n")
    finding = _finding(document, (fragment, 0, 3))
    on_separator = replace(
        finding,
        evidence=(
            replace(
                finding.evidence[0],
                start=separator,
                end=separator + 2,
                quote=fragment.text[separator : separator + 2],
            ),
        ),
    )

    assert "разделител" in _reasons(on_separator, pair)


@pytest.mark.parametrize("field", ["id", "rule_id", "rule_version", "title", "explanation"])
def test_empty_required_field_is_rejected(pair: Pair, field: str) -> None:
    seg, document = pair
    finding = replace(_finding(document, _reader(seg)), **{field: "  "})

    assert field in _reasons(finding, pair)


def test_unknown_severity_or_criterion_is_rejected(pair: Pair) -> None:
    seg, document = pair
    finding = _finding(document, _reader(seg))

    assert "severity" in _reasons(replace(finding, severity=cast(Severity, "blocker")), pair)
    assert "criterion" in _reasons(replace(finding, criterion="XYZ"), pair)


def test_finding_without_evidence_is_rejected(pair: Pair) -> None:
    seg, document = pair
    finding = replace(_finding(document, _reader(seg)), evidence=())

    assert "доказательств" in _reasons(finding, pair)


def test_finding_without_requirements_is_rejected(pair: Pair) -> None:
    seg, document = pair
    finding = replace(_finding(document, _reader(seg)), requirement_ids=())

    assert "requirement_ids" in _reasons(finding, pair)


def test_evidence_for_unlisted_requirement_is_rejected(pair: Pair) -> None:
    seg, document = pair
    finding = _finding(document, _reader(seg))
    other = unit(seg, "Библиотекарь подтверждает")

    unlisted = replace(finding, requirement_ids=(other.id,))

    reasons = _reasons(unlisted, pair)
    assert "Ф-1" in reasons
    assert other.id in reasons


def test_every_participant_needs_its_own_evidence(pair: Pair) -> None:
    seg, document = pair
    finding = _finding(document, _reader(seg))
    other = unit(seg, "Библиотекарь подтверждает")

    partial = replace(finding, requirement_ids=("Ф-1", other.id))

    assert other.id in _reasons(partial, pair)


def test_repeated_requirement_id_is_rejected(pair: Pair) -> None:
    seg, document = pair
    finding = replace(_finding(document, _reader(seg)), requirement_ids=("Ф-1", "Ф-1"))

    assert "повтор" in _reasons(finding, pair)


def test_segmentation_of_another_document_is_rejected(pair: Pair) -> None:
    seg, document = pair
    finding = _finding(document, _reader(seg))
    other = replace(document, sha256="f" * 64)

    assert "документ" in " ".join(check_finding(finding, seg, other))


def test_absence_evidence_on_empty_row_passes(pair: Pair) -> None:
    seg, document = pair
    header = next(f for f in seg.fragments if f.role == "table_header")
    row = _empty_row(seg)
    header_evidence = build_evidence(header, 0, len(header.text), document)
    finding = replace(
        _finding(document, (header, 0, len(header.text)), kind="absence"),
        requirement_ids=(header.id, row.id),
        evidence=(header_evidence, absence_evidence(row)),
    )

    assert finding.evidence[1].quote == ""
    assert check_finding(finding, seg, document) == ()


def test_empty_quote_is_allowed_only_for_absence(pair: Pair) -> None:
    seg, document = pair
    row = _empty_row(seg)
    finding = replace(
        _finding(document, _reader(seg)),
        requirement_ids=(row.id,),
        evidence=(absence_evidence(row),),
    )

    assert "пуст" in _reasons(finding, pair)
    assert check_finding(replace(finding, evidence_kind="absence"), seg, document) == ()


def test_absence_on_non_empty_fragment_is_rejected(pair: Pair) -> None:
    seg, document = pair
    fragment = seg.by_id("Ф-1")
    fake = Evidence("Ф-1", 0, 0, "", fragment.sources)
    finding = replace(
        _finding(document, _reader(seg), kind="absence"),
        requirement_ids=("Ф-1",),
        evidence=(fake,),
    )

    assert "пуст" in _reasons(finding, pair)


def test_absence_evidence_requires_empty_fragment() -> None:
    fragment = Requirement(
        id="R-1",
        text="текст",
        normalized="текст",
        role="requirement",
        category="unspecified",
        sources=(),
        section_path=(),
    )

    with pytest.raises(ValueError, match="пуст"):
        absence_evidence(fragment)


def test_whitespace_only_quote_is_rejected(pair: Pair) -> None:
    seg, document = pair
    fragment = seg.by_id("Ф-1")
    space = fragment.text.index(" ")

    finding = _finding(document, (fragment, space, space + 1))

    assert "пуст" in _reasons(finding, pair)


def test_gate_keeps_order_and_marks_status(pair: Pair) -> None:
    seg, document = pair
    good = _finding(document, _reader(seg))
    bad = replace(good, id="TEST-01:bad", evidence=(replace(good.evidence[0], quote="x"),))
    second = replace(good, id="TEST-01:second")

    result = gate_findings([good, bad, second], seg, document)

    assert [f.id for f in result.accepted] == [good.id, second.id]
    assert all(f.evidence_status == "verified" for f in result.accepted)
    assert [r.finding.id for r in result.rejected] == ["TEST-01:bad"]


def test_build_evidence_refuses_bad_span(pair: Pair) -> None:
    seg, document = pair

    with pytest.raises(ValueError):
        build_evidence(seg.by_id("Ф-1"), 5, 2, document)
