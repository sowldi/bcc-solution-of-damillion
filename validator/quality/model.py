"""Результат этапа 8: противоречия (подтверждённые и потенциальные), критерии приёмки.

Подтверждённое противоречие — только с машинным контрпримером или по подтверждённому правилу
(CON-03 без числового разбора). Потенциальное — когда контекст или тождество объекта не
проверены (другой раздел, совместимые ограничения, гипотеза модели). Статус гипотезы модели
этап 8 не меняет: она остаётся «unverified».
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, Literal

from validator.findings import Evidence, Finding, RejectedFinding
from validator.quality.constraints import NumericCheck, compare_constraints, find_constraint

SCHEMA = "validator.quality/1"
ContradictionSource = Literal["CTR-01", "CTR-02", "CON-03", "LLM-PAIR"]
ContradictionStatus = Literal["confirmed", "potential"]


@dataclass(frozen=True)
class Contradiction:
    requirement_ids: tuple[str, str]
    source: ContradictionSource
    status: ContradictionStatus
    finding_id: str  # замечание или гипотеза, на которой основан вывод
    evidence: tuple[Evidence, ...]
    explanation: str
    numeric: NumericCheck | None = None  # машинное сравнение чисел (если удалось)

    def __post_init__(self) -> None:
        if self.source == "LLM-PAIR" and self.status != "potential":
            raise ValueError(
                f"{self.finding_id}: гипотеза модели не может быть подтверждённым противоречием."
            )
        if self.source == "CTR-02" and self.status != "potential":
            raise ValueError(f"{self.finding_id}: CTR-02 — только потенциальное противоречие.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_ids": list(self.requirement_ids),
            "source": self.source,
            "status": self.status,
            "finding_id": self.finding_id,
            "explanation": self.explanation,
            "numeric": self.numeric.to_dict() if self.numeric else None,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True)
class QualityResult:
    document_sha256: str
    findings: tuple[Finding, ...] = ()  # замечания правил этапа 8 («verified»)
    rejected: tuple[RejectedFinding, ...] = ()
    contradictions: tuple[Contradiction, ...] = ()
    acceptance: tuple[str, ...] = ()  # атомарные требования категории «критерий приёмки»
    acceptance_links: tuple[str, ...] = ()  # требования, на которые ссылаются критерии
    unlinked_requirements: tuple[str, ...] = ()  # требования без ссылки из критериев
    statements: int = 0  # разобранных утверждений «параметр — значение»
    hypotheses_checked: bool = False  # учтены гипотезы проверки пар моделью
    # Гипотезы модели о паре, где противоречие уже подтверждено правилом: не дублируются.
    hypotheses_confirmed: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for finding in self.findings:
            if finding.evidence_status != "verified" or finding.origin != "rule":
                raise ValueError(f"Замечание этапа 8 {finding.id} не подтверждено проверкой.")
            if finding.document_sha256 != self.document_sha256:
                raise ValueError(f"Замечание {finding.id} построено для другого документа.")

    @property
    def confirmed(self) -> tuple[Contradiction, ...]:
        return tuple(item for item in self.contradictions if item.status == "confirmed")

    @property
    def potential(self) -> tuple[Contradiction, ...]:
        return tuple(item for item in self.contradictions if item.status == "potential")

    def with_hypotheses(self, findings: Iterable[Finding]) -> QualityResult:
        """Гипотезы модели «conflicts_with» — потенциальные противоречия с числовой справкой."""
        added: list[Contradiction] = []
        matched: list[str] = []
        confirmed = {frozenset(item.requirement_ids) for item in self.confirmed}
        for finding in findings:
            if finding.relation != "conflicts_with" or finding.origin != "llm":
                continue
            if finding.evidence_status != "unverified":
                raise ValueError(f"Гипотеза {finding.id}: ожидается статус unverified.")
            if finding.document_sha256 != self.document_sha256:
                raise ValueError(f"Гипотеза {finding.id} построена для другого документа.")
            source, target = finding.requirement_ids
            if frozenset((source, target)) in confirmed:
                matched.append(finding.id)
                continue
            numeric = None
            a, b = (find_constraint(evidence.quote) for evidence in finding.evidence)
            if a is not None and b is not None:
                numeric = compare_constraints(a, b)
            added.append(
                Contradiction(
                    (source, target),
                    "LLM-PAIR",
                    "potential",
                    finding.id,
                    finding.evidence,
                    finding.explanation,
                    numeric,
                )
            )
        return replace(
            self,
            contradictions=self.contradictions + tuple(added),
            hypotheses_checked=True,
            hypotheses_confirmed=self.hypotheses_confirmed + tuple(matched),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "document_sha256": self.document_sha256,
            "statements": self.statements,
            "hypotheses_checked": self.hypotheses_checked,
            "hypotheses_confirmed": list(self.hypotheses_confirmed),
            "acceptance": list(self.acceptance),
            "acceptance_links": list(self.acceptance_links),
            "unlinked_requirements": list(self.unlinked_requirements),
            "findings": [finding.to_dict() for finding in self.findings],
            "rejected": [item.to_dict() for item in self.rejected],
            "contradictions": [item.to_dict() for item in self.contradictions],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
