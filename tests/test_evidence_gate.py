"""Evidence Gate (этап 5): строгая проверка доказательств и защита от обхода.

Документы и формулировки — собственные (домен «библиотека»); сети и ключей нет.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from validator.errors import EXIT_INPUT_ERROR, InputError
from validator.evidence import (
    EvidenceGateError,
    TrustedRule,
    absence_evidence,
    build_evidence,
    check_finding,
    ensure_source_unchanged,
    gate_findings,
)
from validator.findings import AbsenceClaim, Evidence, EvidenceKind, Finding
from validator.ingest import load_document
from validator.model import Document, Location
from validator.segmentation import Requirement, Segmentation, segment_document

SPEC = """# Сервис бронирования

## Функциональные требования

* Ф-1. Читатель бронирует книгу через каталог.
* Ф-2. Читатель не продлевает бронь чаще одного раза.
* Ф-3. Библиотекарь отменяет бронь читателя.
* Ф-4. Продление выполняется согласно Ф-9.
* Читатель получает уведомление о брони.

## Отчёты

## Согласование

| Роль | Дата |
| --- | --- |
| | |

## Параметры

| Параметр | Значение |
| --- | --- |
| Срок брони | 3 дня |
| Штраф | — |
"""

RULE = TrustedRule(
    rule_id="TEST-01",
    version="1",
    severity="minor",
    category="ambiguity",
    criterion="UNA",
    evidence_kind="quote",
    title="Проверочное замечание",
)
ABSENCE_RULE = replace(RULE, rule_id="TEST-02", evidence_kind="absence", title="Отсутствие")
RULES = (RULE, ABSENCE_RULE)

Pair = tuple[Segmentation, Document]


def _load(tmp_path: Path, text: str = SPEC, name: str = "spec.md") -> Pair:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    document = load_document(path)
    return segment_document(document), document


@pytest.fixture
def pair(tmp_path: Path) -> Pair:
    return _load(tmp_path)


def _span(seg: Segmentation, fragment_id: str, quote: str) -> tuple[Requirement, int, int]:
    fragment = seg.by_id(fragment_id)
    start = fragment.text.index(quote)
    return fragment, start, start + len(quote)


def _finding(
    document: Document,
    *spans: tuple[Requirement, int, int],
    kind: EvidenceKind = "quote",
    absence: AbsenceClaim | None = None,
    **changes: Any,
) -> Finding:
    evidence = tuple(
        absence_evidence(f)
        if (s, e) == (0, 0) and not f.text
        else build_evidence(f, s, e, document)
        for f, s, e in spans
    )
    rule = ABSENCE_RULE if kind == "absence" else RULE
    finding = Finding(
        id=f"{rule.rule_id}:" + "+".join(f"{f.id}@{s}" for f, s, _ in spans),
        rule_id=rule.rule_id,
        rule_version=rule.version,
        severity=rule.severity,
        category=rule.category,
        criterion=rule.criterion,
        title=rule.title,
        explanation="Пояснение.",
        recommendation="Рекомендация.",
        requirement_ids=tuple(dict.fromkeys(f.id for f, _, _ in spans)),
        evidence=evidence,
        evidence_kind=kind,
        document_sha256=document.sha256,
        absence=absence,
    )
    return replace(finding, **changes)


def _reasons(finding: Finding, pair: Pair) -> str:
    seg, document = pair
    return " | ".join(check_finding(finding, seg, document, rules=RULES))


def _status(finding: Finding, pair: Pair) -> str:
    result = gate_findings([finding], *pair, rules=RULES)
    [verdict] = result.verdicts
    return verdict.status


# --- итог проверки ------------------------------------------------------------------------


def test_valid_rule_finding_is_verified_with_full_verdict(pair: Pair) -> None:
    seg, document = pair
    finding = _finding(document, _span(seg, "Ф-1", "через каталог"))

    result = gate_findings([finding], seg, document, rules=RULES)

    assert result.accepted == (replace(finding, evidence_status="verified"),)
    assert result.document_sha256 == document.sha256
    assert result.document_name == "spec.md"
    [verdict] = result.verdicts
    assert (verdict.finding_id, verdict.origin, verdict.status) == (finding.id, "rule", "verified")
    assert verdict.reasons == ()
    assert verdict.requirement_ids == ("Ф-1",)
    [checked] = verdict.checked
    assert (checked.requirement_id, checked.quote, checked.basis) == (
        "Ф-1",
        "через каталог",
        "text",
    )
    assert "строка 5" in checked.places[0]


def test_gate_ignores_status_claimed_by_input(pair: Pair) -> None:
    seg, document = pair
    claimed = _finding(document, _span(seg, "Ф-1", "каталог"), evidence_status="verified")
    broken = replace(claimed, evidence=(replace(claimed.evidence[0], quote="другое"),))

    assert _status(broken, pair) == "rejected"


# --- 5.1 привязка к документу ---------------------------------------------------------------


def test_finding_must_be_bound_to_the_document(pair: Pair) -> None:
    seg, document = pair
    finding = _finding(document, _span(seg, "Ф-1", "каталог"))

    assert "документ" in _reasons(replace(finding, document_sha256=""), pair)
    assert "другой" in _reasons(replace(finding, document_sha256="f" * 64), pair)


def test_evidence_from_another_document_is_rejected(tmp_path: Path) -> None:
    seg_a, doc_a = _load(tmp_path, SPEC, "a.md")
    seg_b, doc_b = _load(tmp_path, SPEC.replace("Штраф | —", "Штраф | 100 ₸"), "b.md")
    finding = _finding(doc_a, _span(seg_a, "Ф-1", "каталог"))

    # текст по смещениям в документе B тот же, но замечание построено для A
    assert seg_b.by_id("Ф-1").text == seg_a.by_id("Ф-1").text
    assert "другой" in " ".join(check_finding(finding, seg_b, doc_b, rules=RULES))


def test_document_changed_after_finding_requires_new_check(tmp_path: Path) -> None:
    seg, document = _load(tmp_path)
    finding = _finding(document, _span(seg, "Ф-1", "каталог"))
    (tmp_path / "spec.md").write_text(SPEC + "\nДобавлен абзац.\n", encoding="utf-8")
    new_document = load_document(tmp_path / "spec.md")

    assert _status(finding, (segment_document(new_document), new_document)) == "rejected"


def test_source_file_changed_during_run_is_detected(tmp_path: Path) -> None:
    _, document = _load(tmp_path)
    ensure_source_unchanged(document)  # файл не менялся

    (tmp_path / "spec.md").write_text(SPEC + "\nправка\n", encoding="utf-8")

    with pytest.raises(InputError, match="изменён") as caught:
        ensure_source_unchanged(document)
    assert caught.value.exit_code == EXIT_INPUT_ERROR


def test_segmentation_inconsistent_with_document_stops_the_gate(pair: Pair) -> None:
    seg, document = pair
    finding = _finding(document, _span(seg, "Ф-1", "каталог"))
    fragment = seg.by_id("Ф-3")
    forged = Segmentation(
        seg.document_sha256,
        tuple(replace(f, text=f.text + "!") if f.id == fragment.id else f for f in seg.fragments),
        seg.warnings,
    )

    with pytest.raises(EvidenceGateError, match="Ф-3"):
        gate_findings([finding], forged, document, rules=RULES)


def test_tampered_normalized_text_is_an_integrity_error(pair: Pair) -> None:
    """M4: нормализованный текст тоже должен соответствовать документу."""
    seg, document = pair
    forged = Segmentation(
        seg.document_sha256,
        tuple(
            replace(f, normalized="Читатель продлевает бронь") if f.id == "Ф-2" else f
            for f in seg.fragments
        ),
        seg.warnings,
    )

    with pytest.raises(EvidenceGateError, match="normalized"):
        gate_findings([], forged, document, rules=RULES)


# --- 5.2 связь с требованием ------------------------------------------------------------------


def test_quote_from_another_requirement_is_rejected(pair: Pair) -> None:
    seg, document = pair
    finding = _finding(document, _span(seg, "Ф-2", "бронь"))
    evidence = finding.evidence[0]
    relabeled = replace(
        finding,
        requirement_ids=("Ф-3",),
        evidence=(replace(evidence, requirement_id="Ф-3"),),
    )

    assert _status(relabeled, pair) == "rejected"


def test_same_text_elsewhere_does_not_prove_the_finding(pair: Pair) -> None:
    """«бронь» есть и в Ф-2, и в Ф-3: подходящий фрагмент в другом месте не доказательство."""
    seg, document = pair
    own = _finding(document, _span(seg, "Ф-3", "бронь"))
    foreign_sources = build_evidence(*_span(seg, "Ф-2", "бронь"), document).sources
    forged = replace(own, evidence=(replace(own.evidence[0], sources=foreign_sources),))

    assert "источник" in _reasons(forged, pair)


def test_unknown_requirement_id_is_rejected(pair: Pair) -> None:
    seg, document = pair
    finding = _finding(document, _span(seg, "Ф-1", "каталог"))
    ghost = replace(
        finding,
        requirement_ids=("Ф-77",),
        evidence=(replace(finding.evidence[0], requirement_id="Ф-77"),),
    )

    assert "Ф-77" in _reasons(ghost, pair)


# --- 5.3 точная цитата -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "quote", ["через каталог.", "через катало", "через  каталог", "Через каталог"]
)
def test_quote_with_extra_or_missing_characters_is_rejected(pair: Pair, quote: str) -> None:
    seg, document = pair
    finding = _finding(document, _span(seg, "Ф-1", "через каталог"))

    forged = replace(finding, evidence=(replace(finding.evidence[0], quote=quote),))

    assert "цитата" in _reasons(forged, pair)


@pytest.mark.parametrize(("start_delta", "end_delta"), [(0, 1), (-1, 0), (1, 1), (0, -3)])
def test_changed_range_is_rejected(pair: Pair, start_delta: int, end_delta: int) -> None:
    seg, document = pair
    finding = _finding(document, _span(seg, "Ф-1", "через каталог"))
    evidence = finding.evidence[0]

    moved = replace(evidence, start=evidence.start + start_delta, end=evidence.end + end_delta)

    assert _status(replace(finding, evidence=(moved,)), pair) == "rejected"


def test_paraphrased_quote_is_rejected(pair: Pair) -> None:
    seg, document = pair
    finding = _finding(document, _span(seg, "Ф-1", "через каталог"))

    paraphrase = replace(finding, evidence=(replace(finding.evidence[0], quote="в каталоге"),))

    assert "цитата" in _reasons(paraphrase, pair)


def test_quote_cut_in_the_middle_of_a_word_is_rejected(pair: Pair) -> None:
    seg, document = pair
    fragment = seg.by_id("Ф-1")
    start = fragment.text.index("бронирует")

    head = _finding(document, (fragment, start, start + 5))  # «брони» от «бронирует»
    tail = _finding(document, (fragment, start + 3, start + 9))  # «нирует»

    assert "посреди слова" in _reasons(head, pair)
    assert "посреди слова" in _reasons(tail, pair)


def test_quote_that_drops_a_negation_is_rejected(pair: Pair) -> None:
    seg, document = pair

    without_negation = _finding(document, _span(seg, "Ф-2", "продлевает бронь"))
    with_negation = _finding(document, _span(seg, "Ф-2", "не продлевает бронь"))

    assert "отрицани" in _reasons(without_negation, pair)
    assert _reasons(with_negation, pair) == ""


def test_whitespace_only_or_empty_quote_for_text_evidence_is_rejected(pair: Pair) -> None:
    seg, document = pair
    fragment = seg.by_id("Ф-1")
    space = fragment.text.index(" ")
    row = next(f for f in seg.fragments if f.text == "" and f.role != "table_header")

    blank = _finding(document, (fragment, space, space + 1))
    empty = _finding(document, (row, 0, 0))

    assert "пуст" in _reasons(blank, pair)
    assert "пуст" in _reasons(empty, pair)


def test_quote_matching_only_the_normalized_text_is_rejected(tmp_path: Path) -> None:
    """Нормализация (кавычки, пробелы) не допускается в цитате: только точный текст."""
    seg, document = _load(
        tmp_path, "# С\n\n## Функциональные требования\n\n* Ф-1. Кнопка «Продлить» активна.\n"
    )
    finding = _finding(document, _span(seg, "Ф-1", "«Продлить»"))

    normalized = replace(finding, evidence=(replace(finding.evidence[0], quote='"Продлить"'),))

    assert _status(normalized, (seg, document)) == "rejected"
    assert _status(finding, (seg, document)) == "verified"


# --- 5.4 SourceRef ------------------------------------------------------------------------------


def test_reference_to_missing_block_is_rejected(pair: Pair) -> None:
    seg, document = pair
    finding = _finding(document, _span(seg, "Ф-1", "каталог"))
    [ref] = finding.evidence[0].sources

    forged = replace(
        finding, evidence=(replace(finding.evidence[0], sources=(replace(ref, block=999),)),)
    )

    assert "источник" in _reasons(forged, pair)


def test_reference_beyond_block_text_is_rejected(pair: Pair) -> None:
    seg, document = pair
    finding = _finding(document, _span(seg, "Ф-1", "каталог"))
    [ref] = finding.evidence[0].sources

    forged = replace(
        finding,
        evidence=(replace(finding.evidence[0], sources=(replace(ref, end=ref.end + 500),)),),
    )

    assert "источник" in _reasons(forged, pair)


def test_evidence_that_cannot_be_located_in_the_file_is_rejected(tmp_path: Path) -> None:
    """Блок без строки, страницы, абзаца и таблицы: место в исходном файле не доказуемо."""
    path = tmp_path / "spec.md"
    path.write_text("# С\n\n## Функциональные требования\n\n* Ф-1. Читатель бронирует.\n")
    document = load_document(path)
    blocks = tuple(replace(b, location=Location(b.location.section_path)) for b in document.blocks)
    nowhere = replace(document, blocks=blocks)
    seg = segment_document(nowhere)
    finding = _finding(nowhere, _span(seg, "Ф-1", "бронирует"))

    assert "место" in " ".join(check_finding(finding, seg, nowhere, rules=RULES))


# --- 7. доказательство отсутствия ---------------------------------------------------------------


def _table_header(seg: Segmentation, text: str) -> Requirement:
    return next(f for f in seg.fragments if f.role == "table_header" and f.text == text)


def _empty_table_finding(seg: Segmentation, document: Document, **claim: str) -> Finding:
    header = _table_header(seg, "Роль | Дата")
    row = next(f for f in seg.fragments if f.text == "" and f.role != "table_header")
    target = claim.get("target", str(header.location.table))
    return _finding(
        document,
        (header, 0, len(header.text)),
        (row, 0, 0),
        kind="absence",
        absence=AbsenceClaim("table", target, claim.get("missing", "data")),
    )


def test_absence_in_a_bounded_table_is_verified(pair: Pair) -> None:
    seg, document = pair
    finding = _empty_table_finding(seg, document)

    result = gate_findings([finding], seg, document, rules=RULES)

    [verdict] = result.verdicts
    assert verdict.status == "verified", verdict.reasons
    assert [c.basis for c in verdict.checked] == ["structure", "absence"]


def test_absence_without_bounded_scope_is_rejected(pair: Pair) -> None:
    seg, document = pair
    finding = replace(_empty_table_finding(seg, document), absence=None)

    assert "област" in _reasons(finding, pair)


def test_absence_claim_on_presence_evidence_is_rejected(pair: Pair) -> None:
    seg, document = pair
    finding = _finding(
        document,
        _span(seg, "Ф-1", "каталог"),
        absence=AbsenceClaim("requirement", "Ф-1", "number"),
    )

    assert "absence" in _reasons(finding, pair)


def test_absence_claim_outside_its_evidence_is_rejected(pair: Pair) -> None:
    seg, document = pair
    other_table = str(_table_header(seg, "Параметр | Значение").location.table)

    finding = _empty_table_finding(seg, document, target=other_table)

    assert _status(finding, pair) == "rejected"


@pytest.mark.parametrize(
    "claim",
    [
        AbsenceClaim("table", "99", "data"),
        AbsenceClaim("table", "один", "data"),
        AbsenceClaim("table", "1", "text"),  # недопустимое сочетание
        AbsenceClaim(cast(Any, "document"), "", "text"),  # «этого нет в документе»
    ],
)
def test_unbounded_or_unknown_absence_scope_is_rejected(pair: Pair, claim: AbsenceClaim) -> None:
    seg, document = pair
    finding = replace(_empty_table_finding(seg, document), absence=claim)

    assert _status(finding, pair) == "rejected"


def test_false_absence_claim_is_rejected(pair: Pair) -> None:
    seg, document = pair
    header = _table_header(seg, "Параметр | Значение")
    finding = _finding(
        document,
        (header, 0, len(header.text)),
        kind="absence",
        absence=AbsenceClaim("table", str(header.location.table), "data"),
    )

    assert "не подтверждается" in _reasons(finding, pair)


@pytest.mark.parametrize(
    ("fragment_id", "missing", "expected"),
    [
        ("Ф-1", "number", "verified"),
        ("Ф-1", "document_id", "rejected"),
        ("BARE", "document_id", "verified"),
    ],
)
def test_requirement_scoped_absence(
    pair: Pair, fragment_id: str, missing: str, expected: str
) -> None:
    seg, document = pair
    fragment = (
        next(f for f in seg.fragments if f.text.startswith("Читатель получает"))
        if fragment_id == "BARE"
        else seg.by_id(fragment_id)
    )
    finding = _finding(
        document,
        (fragment, 0, len(fragment.text)),
        kind="absence",
        absence=AbsenceClaim("requirement", fragment.id, missing),
    )

    assert _status(finding, pair) == expected


def test_value_absence_needs_an_empty_value(pair: Pair) -> None:
    seg, document = pair
    filled = next(f for f in seg.fragments if f.text == "Срок брони | 3 дня")
    dash = next(f for f in seg.fragments if f.text == "Штраф | —")

    def claim(fragment: Requirement) -> Finding:
        return _finding(
            document,
            (fragment, 0, len(fragment.text)),
            kind="absence",
            absence=AbsenceClaim("requirement", fragment.id, "value"),
        )

    assert _status(claim(dash), pair) == "verified"
    assert _status(claim(filled), pair) == "rejected"


def test_requirement_number_absence_is_rechecked(pair: Pair) -> None:
    seg, document = pair
    row = next(f for f in seg.fragments if f.text == "Срок брони | 3 дня")
    finding = _finding(
        document,
        (row, 0, len(row.text)),
        kind="absence",
        absence=AbsenceClaim("requirement", row.id, "number"),
    )

    assert _status(finding, pair) == "rejected"


def test_empty_section_and_missing_id_claims(pair: Pair) -> None:
    seg, document = pair
    heading = next(f for f in seg.fragments if f.text == "Отчёты")
    full_heading = next(f for f in seg.fragments if f.text == "Согласование")
    empty_section = _finding(
        document,
        (heading, 0, len(heading.text)),
        kind="absence",
        absence=AbsenceClaim("section", "Сервис бронирования › Отчёты", "text"),
    )
    filled_section = _finding(
        document,
        (full_heading, 0, len(full_heading.text)),
        kind="absence",
        absence=AbsenceClaim("section", "Сервис бронирования › Согласование", "text"),
    )
    missing_id = _finding(
        document,
        _span(seg, "Ф-4", "Ф-9"),
        kind="absence",
        absence=AbsenceClaim("id_registry", "Ф-9", "declaration"),
    )
    existing_id = replace(missing_id, absence=AbsenceClaim("id_registry", "Ф-1", "declaration"))

    assert _status(empty_section, pair) == "verified"
    assert _status(filled_section, pair) == "rejected"
    assert _status(missing_id, pair) == "verified"
    assert _status(existing_id, pair) == "rejected"


def test_heading_number_claim(tmp_path: Path) -> None:
    seg, document = _load(
        tmp_path,
        "# 1. Сервис\n\n## 1.1 Функциональные требования\n\n"
        "* Ф-1. Формат описан в разделе 1.4.\n* Ф-2. Поля описаны в разделе 1.1.\n",
    )

    def claim(fragment_id: str, quote: str, number: str) -> Finding:
        return _finding(
            document,
            _span(seg, fragment_id, quote),
            kind="absence",
            absence=AbsenceClaim("heading_registry", number, "heading"),
        )

    assert _status(claim("Ф-1", "разделе 1.4", "1.4"), (seg, document)) == "verified"
    assert _status(claim("Ф-2", "разделе 1.1", "1.1"), (seg, document)) == "rejected"


# --- происхождение и статус --------------------------------------------------------------------


def test_llm_finding_with_exact_quote_stays_unverified(pair: Pair) -> None:
    seg, document = pair
    hypothesis = _finding(
        document,
        _span(seg, "Ф-1", "через каталог"),
        origin="llm",
        rule_id="LLM",
        rule_version="prompt-1",
        evidence_status="verified",  # попытка «подтвердить» себя
    )

    result = gate_findings([hypothesis], seg, document, rules=RULES)

    assert result.accepted == ()
    [kept] = result.unverified
    assert kept.origin == "llm"
    assert kept.evidence_status == "unverified"
    assert result.verdicts[0].status == "unverified"


def test_llm_finding_relabeled_as_rule_is_rejected(pair: Pair) -> None:
    seg, document = pair
    disguised = _finding(
        document, _span(seg, "Ф-1", "каталог"), rule_id="LLM", rule_version="prompt-1"
    )

    assert "каталог" in _reasons(disguised, pair)
    assert _status(disguised, pair) == "rejected"


@pytest.mark.parametrize(
    "changes",
    [
        {"severity": "critical"},
        {"rule_version": "2"},
        {"criterion": "VER"},
        {"category": "testability"},
        {"title": "Другое название"},
    ],
)
def test_rule_finding_must_match_its_catalog_entry(pair: Pair, changes: dict[str, Any]) -> None:
    seg, document = pair
    finding = _finding(document, _span(seg, "Ф-1", "каталог"), **changes)

    assert _status(finding, pair) == "rejected"


def test_rule_findings_need_a_catalog_to_be_verified(pair: Pair) -> None:
    seg, document = pair
    finding = _finding(document, _span(seg, "Ф-1", "каталог"))

    result = gate_findings([finding], seg, document, rules=())

    assert result.accepted == ()
    assert result.verdicts[0].status == "rejected"


# --- fail closed --------------------------------------------------------------------------------


def test_invalid_field_types_are_rejected_not_crashing(pair: Pair) -> None:
    seg, document = pair
    finding = _finding(document, _span(seg, "Ф-1", "каталог"))
    bad_offsets = replace(finding, evidence=(replace(finding.evidence[0], start=cast(Any, "6")),))
    bad_ids = replace(finding, requirement_ids=cast(Any, "Ф-1"))
    not_evidence = replace(finding, evidence=cast(Any, ("строка вместо доказательства",)))

    for broken in (bad_offsets, bad_ids, not_evidence):
        assert _status(broken, pair) == "rejected"


def test_gate_never_turns_an_internal_error_into_success(
    pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    seg, document = pair
    finding = _finding(document, _span(seg, "Ф-1", "каталог"))

    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("сбой")

    monkeypatch.setattr("validator.evidence.span_sources", explode)
    result = gate_findings([finding], seg, document, rules=RULES)

    assert result.accepted == ()
    assert "сбой" in " ".join(result.rejected[0].reasons)


def test_rejected_evidence_is_reported_with_reasons(pair: Pair) -> None:
    seg, document = pair
    good = _finding(document, _span(seg, "Ф-1", "каталог"))
    bad = replace(good, id="TEST-01:bad", evidence=(replace(good.evidence[0], quote="x"),))

    result = gate_findings([good, bad], seg, document, rules=RULES)

    assert [v.status for v in result.verdicts] == ["verified", "rejected"]
    [rejected] = result.rejected
    assert rejected.finding.evidence_status == "rejected"
    assert result.verdicts[1].reasons == rejected.reasons


# --- JSON ---------------------------------------------------------------------------------


def test_bound_finding_with_absence_claim_round_trips(pair: Pair) -> None:
    seg, document = pair
    finding = _empty_table_finding(seg, document)

    restored = Finding.from_dict(finding.to_dict())

    assert restored == finding
    assert restored.document_sha256 == document.sha256
    assert restored.absence == finding.absence


@pytest.mark.parametrize(
    ("key", "value"),
    [("document_sha256", None), ("absence", "таблица"), ("absence", {"scope": "table"})],
)
def test_from_dict_validates_binding_and_absence(pair: Pair, key: str, value: object) -> None:
    seg, document = pair
    data = _empty_table_finding(seg, document).to_dict()
    data[key] = value

    with pytest.raises(ValueError):
        Finding.from_dict(data)


def test_evidence_kind_mismatch_with_rule_is_rejected(pair: Pair) -> None:
    seg, document = pair
    finding = _finding(document, _span(seg, "Ф-1", "каталог"), evidence_kind="comparison")

    assert _status(finding, pair) == "rejected"


def test_evidence_with_wrong_requirement_and_offsets_is_rejected(pair: Pair) -> None:
    seg, document = pair
    finding = _finding(document, _span(seg, "Ф-1", "каталог"))
    stranger = Evidence("Ф-3", 0, 3, "Ф-3", finding.evidence[0].sources)

    forged = replace(finding, evidence=(*finding.evidence, stranger))

    assert "Ф-3" in _reasons(forged, pair)


def test_rule_finding_cannot_carry_llm_review_fields(pair: Pair) -> None:
    seg, document = pair
    finding = _finding(document, _span(seg, "Ф-1", "каталог"))

    assert "LLM" in _reasons(replace(finding, review_category="clarity"), pair)
    assert "LLM" in _reasons(replace(finding, confidence=0.9), pair)


@pytest.mark.parametrize("confidence", [1.5, -1.0, "0.5", True])
def test_llm_confidence_must_be_a_probability(pair: Pair, confidence: object) -> None:
    seg, document = pair
    hypothesis = _finding(
        document,
        _span(seg, "Ф-1", "каталог"),
        origin="llm",
        rule_id="LLM",
        rule_version="prompt-2",
        review_category="clarity",
        confidence=cast(Any, confidence),
    )

    assert _status(hypothesis, pair) == "rejected"
