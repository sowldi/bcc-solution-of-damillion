"""Базовый Evidence Gate: замечание проходит, только если его доказательства проверяются
по документу.

Проверяется (для каждого доказательства):
- требование (фрагмент сегментации) существует и указано в requirement_ids;
- отрезок start–end лежит внутри текста фрагмента, цитата — ровно fragment.text[start:end]
  и содержит не только пробелы;
- ссылки на источник совпадают с вычисленными span_sources по документу (блок, смещения,
  место) — то есть цитату можно найти в исходном файле;
- пустая цитата допустима только в замечании об отсутствии содержания («absence») и только
  для фрагмента, у которого в документе действительно нет текста.
Для замечания в целом: обязательные поля заполнены, важность и критерий известны, у каждого
требования есть доказательство, сегментация построена по этому же документу (SHA-256).

Это базовая проверка этапа 3; полный Evidence Gate (этап 5) добавит проверку смысловых
утверждений LLM-гипотез.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace

from validator.criteria import CRITERIA
from validator.findings import (
    EVIDENCE_KINDS,
    RULE_CATEGORIES,
    SEVERITIES,
    Evidence,
    Finding,
    RejectedFinding,
)
from validator.model import Document
from validator.segmentation import Requirement, Segmentation, span_sources

_REQUIRED_TEXT = ("id", "rule_id", "rule_version", "title", "explanation", "recommendation")


@dataclass(frozen=True)
class GateResult:
    accepted: tuple[Finding, ...]  # статус «verified»
    rejected: tuple[RejectedFinding, ...]  # статус «rejected» и причины


def build_evidence(fragment: Requirement, start: int, end: int, document: Document) -> Evidence:
    """Доказательство-цитата fragment.text[start:end]. ValueError, если отрезок нельзя
    привязать к документу (см. span_sources)."""
    sources = span_sources(fragment, start, end, document)
    return Evidence(fragment.id, start, end, fragment.text[start:end], sources)


def absence_evidence(fragment: Requirement) -> Evidence:
    """Доказательство отсутствия содержания: пустой фрагмент и его места в документе."""
    if fragment.text:
        raise ValueError(
            f"Доказательство отсутствия допустимо только для пустого фрагмента; "
            f"у {fragment.id} есть текст."
        )
    return Evidence(fragment.id, 0, 0, "", fragment.sources)


def check_finding(
    finding: Finding, segmentation: Segmentation, document: Document
) -> tuple[str, ...]:
    """Причины, по которым замечание нельзя принять; пустой кортеж — замечание доказано."""
    problems = [
        f"Не заполнено обязательное поле {name}."
        for name in _REQUIRED_TEXT
        if not str(getattr(finding, name)).strip()
    ]
    if finding.severity not in SEVERITIES:
        problems.append(f"Недопустимая важность (severity): {finding.severity!r}.")
    if finding.category not in RULE_CATEGORIES:
        problems.append(f"Недопустимая категория (category): {finding.category!r}.")
    if finding.criterion not in CRITERIA:
        problems.append(f"Неизвестный критерий качества (criterion): {finding.criterion!r}.")
    if finding.evidence_kind not in EVIDENCE_KINDS:
        problems.append(f"Недопустимый вид доказательства: {finding.evidence_kind!r}.")
    if segmentation.document_sha256 != document.sha256:
        problems.append("Сегментация относится к другому документу (SHA-256 не совпадает).")
        return tuple(problems)
    if not finding.requirement_ids:
        problems.append("Не указаны требования (requirement_ids).")
    repeated = [rid for rid, count in Counter(finding.requirement_ids).items() if count > 1]
    if repeated:
        problems.append(f"ID требований повторяются: {', '.join(repeated)}.")
    if not finding.evidence:
        problems.append("Нет доказательств (evidence).")
    for evidence in finding.evidence:
        problems += _check_evidence(evidence, finding, segmentation, document)
    covered = {evidence.requirement_id for evidence in finding.evidence}
    problems += [
        f"У требования {rid} нет доказательства."
        for rid in dict.fromkeys(finding.requirement_ids)
        if rid not in covered
    ]
    return tuple(problems)


def gate_findings(
    findings: Iterable[Finding], segmentation: Segmentation, document: Document
) -> GateResult:
    """Разделить замечания на доказанные и отклонённые (порядок сохраняется)."""
    accepted: list[Finding] = []
    rejected: list[RejectedFinding] = []
    for finding in findings:
        problems = check_finding(finding, segmentation, document)
        if problems:
            rejected.append(RejectedFinding(replace(finding, evidence_status="rejected"), problems))
        else:
            accepted.append(replace(finding, evidence_status="verified"))
    return GateResult(tuple(accepted), tuple(rejected))


def _check_evidence(
    evidence: Evidence, finding: Finding, segmentation: Segmentation, document: Document
) -> list[str]:
    label = f"Доказательство для {evidence.requirement_id}"
    problems: list[str] = []
    if evidence.requirement_id not in finding.requirement_ids:
        problems.append(f"{label}: требование не указано в requirement_ids.")
    try:
        fragment = segmentation.by_id(evidence.requirement_id)
    except KeyError:
        return [*problems, f"{label}: требование {evidence.requirement_id} не найдено."]
    if evidence.start == evidence.end == 0 and not evidence.quote:
        return problems + _check_absence(evidence, finding, fragment, document, label)
    start, end = evidence.start, evidence.end
    if not 0 <= start < end <= len(fragment.text):
        return [
            *problems,
            f"{label}: отрезок {start}–{end} вне текста требования (длина {len(fragment.text)}).",
        ]
    actual = fragment.text[start:end]
    if actual != evidence.quote:
        problems.append(f"{label}: цитата не совпадает с текстом требования ({start}–{end}).")
    if not actual.strip():
        problems.append(f"{label}: цитата пуста — только пробельные символы.")
    try:
        expected = span_sources(fragment, start, end, document)
    except ValueError as exc:
        problems.append(f"{label}: {exc}")
    else:
        if expected != evidence.sources:
            problems.append(f"{label}: ссылки на источник не совпадают с документом.")
    return problems


def _check_absence(
    evidence: Evidence, finding: Finding, fragment: Requirement, document: Document, label: str
) -> list[str]:
    if finding.evidence_kind != "absence":
        return [f"{label}: пустая цитата допустима только в замечании об отсутствии содержания."]
    problems: list[str] = []
    if fragment.text:
        problems.append(f"{label}: доказательство отсутствия, но фрагмент не пуст.")
    blocks = document.blocks
    if not fragment.sources or not all(
        0 <= ref.block < len(blocks) and blocks[ref.block].text == "" for ref in fragment.sources
    ):
        problems.append(f"{label}: блоки документа этого фрагмента не пусты или не найдены.")
    if evidence.sources != fragment.sources:
        problems.append(f"{label}: ссылки на источник не совпадают с документом.")
    return problems
