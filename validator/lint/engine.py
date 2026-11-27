"""Движок lint: правила → замечания → Evidence Gate → без дублей, в порядке документа.

1. Каждое правило запускается отдельно на общем контексте (run_rule); совпадения правила
   превращаются в замечания с доказательствами-цитатами.
2. Все замечания проходят Evidence Gate; непроверяемые попадают в rejected с причинами
   и в отчёт как замечания не выходят.
3. Дубли удаляются: одинаковые доказательства у разных правил или пересекающиеся цитаты
   двух замечаний к фразам в одном требовании («быстрее» внутри «как можно быстрее») —
   остаётся одно замечание: более важное, при равной важности — с более длинной (более
   конкретной) цитатой, затем по ID правила. Замечание ко всему требованию (нет ID,
   оборванный текст) не поглощает замечание к слову внутри него.
4. Порядок — по месту в документе, затем по ID правила; результат детерминирован.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from validator.evidence import absence_evidence, build_evidence, gate_findings
from validator.findings import Evidence, Finding, RejectedFinding, Severity, severity_rank
from validator.lint.base import Hit, LintContext, Rule, Span
from validator.lint.catalog import CATALOG, CATALOG_VERSION
from validator.lint.context import build_context
from validator.model import Document
from validator.segmentation import Segmentation

SCHEMA = "validator.lint/1"


@dataclass(frozen=True)
class LintResult:
    """Итог детерминированной проверки. В findings — только доказанные замечания."""

    document_sha256: str
    catalog_version: str
    rules: tuple[tuple[str, str], ...]  # (rule_id, version) запущенных правил
    findings: tuple[Finding, ...]
    rejected: tuple[RejectedFinding, ...] = ()
    suppressed: int = 0  # удалено как дубли или пересечения

    def __post_init__(self) -> None:
        for finding in self.findings:
            if finding.evidence_status != "verified":
                raise ValueError(
                    f"Замечание {finding.id} не прошло проверку доказательств "
                    f"(статус {finding.evidence_status}, нужен verified)."
                )
        for rejected in self.rejected:
            if rejected.finding.evidence_status != "rejected":
                raise ValueError(f"Отклонённое замечание {rejected.finding.id} без статуса.")
        ids = [finding.id for finding in self.findings]
        if len(ids) != len(set(ids)):
            raise ValueError("ID замечаний не уникальны.")
        if self.suppressed < 0:
            raise ValueError("suppressed не может быть отрицательным.")

    def count(self, severity: Severity) -> int:
        return sum(1 for finding in self.findings if finding.severity == severity)

    # --- JSON ------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "document_sha256": self.document_sha256,
            "catalog_version": self.catalog_version,
            "rules": [{"rule_id": rule_id, "version": version} for rule_id, version in self.rules],
            "suppressed": self.suppressed,
            "findings": [finding.to_dict() for finding in self.findings],
            "rejected": [rejected.to_dict() for rejected in self.rejected],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LintResult:
        if data.get("schema") != SCHEMA:
            raise ValueError(f"Неизвестная схема результата lint: {data.get('schema')!r}")
        rules = data.get("rules")
        if not isinstance(rules, list) or not all(
            isinstance(r, dict)
            and isinstance(r.get("rule_id"), str)
            and isinstance(r.get("version"), str)
            for r in rules
        ):
            raise ValueError("Поле rules: ожидается список {rule_id, version}")
        suppressed = data.get("suppressed")
        if not isinstance(suppressed, int) or isinstance(suppressed, bool):
            raise ValueError("Поле suppressed: ожидается целое число")
        sha = data.get("document_sha256")
        version = data.get("catalog_version")
        if not isinstance(sha, str) or not isinstance(version, str):
            raise ValueError("Поля document_sha256 и catalog_version: ожидаются строки")
        return cls(
            document_sha256=sha,
            catalog_version=version,
            rules=tuple((r["rule_id"], r["version"]) for r in rules),
            findings=tuple(Finding.from_dict(item) for item in _items(data, "findings")),
            rejected=tuple(RejectedFinding.from_dict(item) for item in _items(data, "rejected")),
            suppressed=suppressed,
        )

    @classmethod
    def from_json(cls, text: str) -> LintResult:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("JSON результата lint должен быть объектом")
        return cls.from_dict(data)


def _items(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Поле {key}: ожидается список")
    return value


# --- запуск ---------------------------------------------------------------------------


def run_rule(rule: Rule, context: LintContext) -> tuple[Finding, ...]:
    """Замечания одного правила (ещё не проверенные Evidence Gate), без повторов."""
    found: dict[str, Finding] = {}
    for hit in rule.check(context):
        finding = _finding(rule, hit, context.document)
        found.setdefault(finding.id, finding)
    return tuple(found.values())


def run_lint(
    document: Document, segmentation: Segmentation, *, rules: Sequence[Rule] = CATALOG
) -> LintResult:
    context = build_context(document, segmentation)
    drafts = [finding for rule in rules for finding in run_rule(rule, context)]
    gate = gate_findings(drafts, segmentation, document)
    phrase_rules = {rule.rule_id for rule in rules if rule.granularity == "phrase"}
    findings, suppressed = _deduplicate(gate.accepted, phrase_rules)
    findings.sort(key=lambda finding: _document_order(finding, context))
    return LintResult(
        document_sha256=document.sha256,
        catalog_version=CATALOG_VERSION,
        rules=tuple((rule.rule_id, rule.version) for rule in rules),
        findings=tuple(findings),
        rejected=gate.rejected,
        suppressed=suppressed,
    )


def _finding(rule: Rule, hit: Hit, document: Document) -> Finding:
    finding_id = f"{rule.rule_id}:" + "+".join(f"{s.fragment.id}@{s.start}" for s in hit.spans)
    return Finding(
        id=finding_id,
        rule_id=rule.rule_id,
        rule_version=rule.version,
        severity=rule.severity,
        category=rule.category,
        criterion=rule.criterion,
        title=rule.title,
        explanation=hit.explanation,
        recommendation=rule.recommendation,
        requirement_ids=tuple(dict.fromkeys(span.fragment.id for span in hit.spans)),
        evidence=tuple(_evidence(span, document) for span in hit.spans),
        evidence_kind=rule.evidence_kind,
    )


def _evidence(span: Span, document: Document) -> Evidence:
    fragment = span.fragment
    if span.start == span.end == 0 and not fragment.text:
        return absence_evidence(fragment)
    try:
        return build_evidence(fragment, span.start, span.end, document)
    except ValueError:
        # Отрезок нельзя привязать к документу: замечание отклонит Evidence Gate
        # с точной причиной, а не исчезнет молча.
        quote = fragment.text[span.start : span.end]
        return Evidence(fragment.id, span.start, span.end, quote, ())


# --- дубли и порядок ---------------------------------------------------------------------


def _signature(finding: Finding) -> frozenset[tuple[str, int, int]]:
    return frozenset((e.requirement_id, e.start, e.end) for e in finding.evidence)


def _overlaps(a: Finding, b: Finding, phrase_rules: set[str]) -> bool:
    """Две цитаты одной фразы: обе — замечания к фразам, в одном требовании, пересекаются."""
    if not {a.rule_id, b.rule_id} <= phrase_rules:
        return False
    if len(a.evidence) != 1 or len(b.evidence) != 1:
        return False
    x, y = a.evidence[0], b.evidence[0]
    return x.requirement_id == y.requirement_id and x.start < y.end and y.start < x.end


def _deduplicate(findings: Iterable[Finding], phrase_rules: set[str]) -> tuple[list[Finding], int]:
    def priority(finding: Finding) -> tuple[int, int, str, str]:
        length = sum(e.end - e.start for e in finding.evidence)
        return severity_rank(finding.severity), -length, finding.rule_id, finding.id

    kept: list[Finding] = []
    suppressed = 0
    for finding in sorted(findings, key=priority):
        if any(
            finding.id == other.id
            or _signature(finding) == _signature(other)
            or _overlaps(finding, other, phrase_rules)
            for other in kept
        ):
            suppressed += 1
            continue
        kept.append(finding)
    return kept, suppressed


def _document_order(finding: Finding, context: LintContext) -> tuple[int, int, str, str]:
    first = finding.evidence[0]
    return context.order[first.requirement_id], first.start, finding.rule_id, finding.id
