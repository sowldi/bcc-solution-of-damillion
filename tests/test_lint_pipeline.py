"""Конвейер lint: детерминированность, отсутствие дублей, Evidence Gate, JSON, крайние случаи."""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import pytest

from tests.lint_helpers import HEAD, lint_markdown
from validator.lint import CATALOG, LintResult, Rule, run_lint
from validator.lint.base import Hit, LintContext, Span
from validator.model import Document
from validator.segmentation import segment_document

AMBIGUOUS = HEAD + (
    "* Ф-1. Система должна быстро и удобно показывать несколько книг.\n"
    "* Ф-2. По возможности система продлевает бронь и т.д.\n"
    "* Ф-3. Срок брони: TBD.\n"
)


def test_correct_requirements_have_no_findings(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path,
        HEAD + "* Ф-1. Система сохраняет бронь не позднее 2 секунд после подтверждения.\n"
        "* Ф-2. Читатель отменяет бронь до выдачи книги.\n",
    )

    assert linted.result.findings == ()
    assert linted.result.rejected == ()


def test_ambiguous_requirement_gets_several_findings(tmp_path: Path) -> None:
    linted = lint_markdown(tmp_path, AMBIGUOUS)

    assert linted.rule_ids == ["TST-01", "TST-01", "AMB-01", "TST-02", "CMP-02", "CMP-01"]
    assert all(f.evidence_status == "verified" for f in linted.result.findings)


def test_document_without_requirements(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path, "# Заметки\n\nДокумент содержит справочную информацию о библиотеке.\n"
    )

    assert linted.segmentation.requirements == ()
    assert linted.result.findings == ()


def test_empty_document_is_handled() -> None:
    document = Document(source=Path("empty.md"), format="markdown", sha256="0" * 64, blocks=())
    segmentation = segment_document(document)

    result = run_lint(document, segmentation)

    assert result.findings == ()
    assert result.rejected == ()
    assert result.document_sha256 == "0" * 64


def test_lint_is_deterministic(tmp_path: Path) -> None:
    first = lint_markdown(tmp_path, AMBIGUOUS, name="a.md")
    second = lint_markdown(tmp_path, AMBIGUOUS, name="b.md")

    assert first.result.to_json() == second.result.to_json()


def test_finding_ids_are_unique_and_stable(tmp_path: Path) -> None:
    linted = lint_markdown(tmp_path, AMBIGUOUS)

    ids = [f.id for f in linted.result.findings]
    assert len(ids) == len(set(ids))
    assert "TST-02:Ф-2@5" in ids  # правило:фрагмент@начало цитаты


def test_no_overlapping_findings_on_one_requirement(tmp_path: Path) -> None:
    linted = lint_markdown(tmp_path, AMBIGUOUS)

    spans: dict[str, list[tuple[int, int]]] = {}
    for finding in linted.result.findings:
        if len(finding.evidence) == 1:
            evidence = finding.evidence[0]
            spans.setdefault(evidence.requirement_id, []).append((evidence.start, evidence.end))
    for found in spans.values():
        found.sort()
        for (_, end), (start, _) in pairwise(found):
            assert end <= start


def test_whole_requirement_findings_do_not_hide_phrase_findings(tmp_path: Path) -> None:
    """Цитата «нет ID» охватывает всё требование, но не поглощает замечание к слову в нём."""
    linted = lint_markdown(
        tmp_path,
        HEAD + "* Ф-1. Читатель бронирует книгу.\n* Ф-2. Читатель отменяет бронь.\n"
        "* Бронь передаётся в ЕКБ.\n",
    )

    assert sorted(linted.rule_ids) == ["AMB-05", "TRC-01"]
    assert linted.result.suppressed == 0


def test_result_json_round_trip(tmp_path: Path) -> None:
    linted = lint_markdown(tmp_path, AMBIGUOUS)

    restored = LintResult.from_json(linted.result.to_json())

    assert restored == linted.result
    assert restored.catalog_version == linted.result.catalog_version
    assert [rule_id for rule_id, _ in restored.rules] == [rule.rule_id for rule in CATALOG]


def _broken_rule(make_hit: object) -> Rule:
    base = CATALOG[0]
    return replace(base, rule_id="TST-99", check=make_hit)  # type: ignore[arg-type]


def test_rule_with_unverifiable_evidence_is_rejected_not_reported(tmp_path: Path) -> None:
    def beyond_text(context: LintContext) -> list[Hit]:
        fragment = context.targets[0]
        return [Hit((Span(fragment, 0, len(fragment.text) + 5),), "Отрезок за концом текста.")]

    linted = lint_markdown(tmp_path, AMBIGUOUS, rules=(*CATALOG, _broken_rule(beyond_text)))

    assert all(f.rule_id != "TST-99" for f in linted.result.findings)
    [rejected] = linted.result.rejected
    assert rejected.finding.rule_id == "TST-99"
    assert rejected.finding.evidence_status == "rejected"
    assert any("отрезок" in reason for reason in rejected.reasons)


def test_duplicate_hits_of_one_rule_are_reported_once(tmp_path: Path) -> None:
    def twice(context: LintContext) -> list[Hit]:
        fragment = context.targets[0]
        hit = Hit((Span(fragment, 0, 4),), "Повтор.")
        return [hit, hit]

    linted = lint_markdown(
        tmp_path, HEAD + "* Ф-1. Читатель бронирует книгу.\n", rules=(_broken_rule(twice),)
    )

    assert [f.rule_id for f in linted.result.findings] == ["TST-99"]


def test_result_refuses_unverified_findings(tmp_path: Path) -> None:
    linted = lint_markdown(tmp_path, AMBIGUOUS)
    unverified = replace(linted.result.findings[0], evidence_status="unverified")

    with pytest.raises(ValueError, match="verified"):
        replace(linted.result, findings=(unverified,))


def test_every_finding_is_bound_and_absence_is_bounded(tmp_path: Path) -> None:
    from tests.test_lint_catalog import RICH

    linted = lint_markdown(tmp_path, RICH)

    assert linted.result.rejected == ()
    for finding in linted.result.findings:
        assert finding.document_sha256 == linted.document.sha256
        if finding.evidence_kind == "absence":
            assert finding.absence is not None, finding.id
        else:
            assert finding.absence is None, finding.id


def test_result_refuses_findings_of_another_document(tmp_path: Path) -> None:
    linted = lint_markdown(tmp_path, AMBIGUOUS)
    foreign = replace(linted.result.findings[0], document_sha256="f" * 64)

    with pytest.raises(ValueError, match="документ"):
        replace(linted.result, findings=(foreign,))
