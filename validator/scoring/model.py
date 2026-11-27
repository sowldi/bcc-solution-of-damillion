"""Модель результата этапа 9: оценка требований, сводка и итог.

Числовых «баллов» и процентов нет: статусы и вердикт — категории с явными правилами
(validator.scoring.assess), а сводка — счётчики «k из n».
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from validator.findings import RuleCategory, Severity
from validator.passport import Mode
from validator.segmentation import Category

SCHEMA = "validator.scorecard/1"

RequirementStatus = Literal[
    "confirmed_major",  # подтверждённый существенный дефект или подтверждённое противоречие
    "confirmed_minor",  # только подтверждённые незначительные дефекты
    "needs_review",  # подтверждённых нет, есть гипотезы или потенциальные противоречия
    "clean_full",  # замечаний нет; проверено правилами и моделью
    "clean_rules_only",  # замечаний нет; модель требование не проверяла
    "unchecked",  # требование вне графа (пустое, без источника) и замечаний нет
]
Verdict = Literal["not_recommended", "after_fixes", "no_blockers", "not_assessable"]
LlmReview = Literal["reviewed", "budget", "empty", "no_source", "not_used"]
PairCheck = Literal["checked", "not_checked", "no_candidate", "excluded", "not_used", "disabled"]
ReviewSource = Literal["potential-contradiction", "llm-review"]

STATUS_TITLES: dict[RequirementStatus, str] = {
    "confirmed_major": "Есть подтверждённые существенные дефекты",
    "confirmed_minor": "Есть подтверждённые незначительные дефекты",
    "needs_review": "Нужна проверка человеком",
    "clean_full": "Замечаний нет (правила и модель)",
    "clean_rules_only": "Замечаний нет (только правила)",
    "unchecked": "Не проверено",
}
VERDICT_TITLES: dict[Verdict, str] = {
    "not_recommended": "Начинать разработку и тестирование не рекомендуется",
    "after_fixes": "Можно начинать после исправления подтверждённых замечаний",
    "no_blockers": "Подтверждённых препятствий не найдено",
    "not_assessable": "Оценить нельзя: требования в документе не выделены",
}


@dataclass(frozen=True)
class DefectRef:
    """Подтверждённое замечание (verified, важность major или minor)."""

    finding_id: str
    rule_id: str
    title: str
    severity: Severity
    category: RuleCategory
    criterion: str
    requirement_ids: tuple[str, ...]
    contradiction: bool  # противоречие подтверждено (этап 8)

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "rule_id": self.rule_id,
            "title": self.title,
            "severity": self.severity,
            "category": self.category,
            "criterion": self.criterion,
            "requirement_ids": list(self.requirement_ids),
            "contradiction": self.contradiction,
        }


@dataclass(frozen=True)
class ReviewRef:
    """На проверку человеку: гипотеза модели или потенциальное противоречие."""

    item_id: str
    source: ReviewSource
    title: str
    requirement_ids: tuple[str, ...]
    severity: Severity  # предполагаемая (по категории или правилу), не подтверждённая

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "source": self.source,
            "title": self.title,
            "requirement_ids": list(self.requirement_ids),
            "severity": self.severity,
        }


@dataclass(frozen=True)
class RequirementAssessment:
    requirement_id: str
    doc_id: str | None
    category: Category
    status: RequirementStatus
    confirmed_major: tuple[str, ...]  # ID замечаний
    confirmed_minor: tuple[str, ...]
    contradictions_confirmed: tuple[str, ...]
    contradictions_potential: tuple[str, ...]
    hypotheses: tuple[str, ...]  # гипотезы LLM-обзора (раздел 5)
    llm_review: LlmReview
    pair_check: PairCheck

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "doc_id": self.doc_id,
            "category": self.category,
            "status": self.status,
            "confirmed_major": list(self.confirmed_major),
            "confirmed_minor": list(self.confirmed_minor),
            "contradictions_confirmed": list(self.contradictions_confirmed),
            "contradictions_potential": list(self.contradictions_potential),
            "hypotheses": list(self.hypotheses),
            "llm_review": self.llm_review,
            "pair_check": self.pair_check,
        }


@dataclass(frozen=True)
class Coverage:
    """Что и в каком объёме проверено."""

    requirements: int  # атомарных требований
    ambiguous: int  # неоднозначных фрагментов (правилами формулировок не проверялись)
    excluded: int  # требований вне графа
    llm_used: bool
    reviewed: int  # требований, проверенных моделью
    skipped: tuple[tuple[str, int], ...]  # причина → число требований без LLM-обзора
    pairs_disabled: bool
    pairs_candidates: int  # пар-кандидатов в отборе
    pairs_checked: int
    truncated_pairs: int  # исключено ограничением отбора
    outside_pairs: int  # пар вне отбора

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirements": self.requirements,
            "ambiguous": self.ambiguous,
            "excluded": self.excluded,
            "llm_used": self.llm_used,
            "reviewed": self.reviewed,
            "skipped": dict(self.skipped),
            "pairs_disabled": self.pairs_disabled,
            "pairs_candidates": self.pairs_candidates,
            "pairs_checked": self.pairs_checked,
            "truncated_pairs": self.truncated_pairs,
            "outside_pairs": self.outside_pairs,
        }


@dataclass(frozen=True)
class Scorecard:
    document_sha256: str
    mode: Mode
    verdict: Verdict
    verdict_reasons: tuple[str, ...]
    caveats: tuple[str, ...]  # ограничения вывода: что не проверено
    requirements: tuple[RequirementAssessment, ...]
    ambiguous: int
    defects: tuple[DefectRef, ...]  # подтверждённые, в порядке приоритета исправления
    review: tuple[ReviewRef, ...]  # на проверку человеку
    critical: tuple[str, ...]  # критичные по ТЗ: проверяемость, полнота, противоречия
    structural: tuple[str, ...]  # подтверждённые дефекты структуры, не отдельных требований
    info: tuple[str, ...]  # информационные замечания (поводы для проверки, не дефекты)
    rejected: tuple[tuple[str, int], ...]  # источник → отклонено проверкой доказательств
    relation_hypotheses: tuple[tuple[str, int], ...]  # тип связи → гипотез модели
    review_categories: tuple[tuple[str, int], ...]  # категория обзора → гипотез модели
    contradictions_confirmed: int
    contradictions_potential: int
    acceptance: int
    acceptance_scenarios: int  # сценариев без ожидаемого результата (ACC-01)
    acceptance_links: int
    coverage: Coverage

    @property
    def status_counts(self) -> tuple[tuple[RequirementStatus, int], ...]:
        return tuple(
            (status, sum(1 for r in self.requirements if r.status == status))
            for status in STATUS_TITLES
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "document_sha256": self.document_sha256,
            "mode": self.mode,
            "verdict": self.verdict,
            "verdict_reasons": list(self.verdict_reasons),
            "caveats": list(self.caveats),
            "status_counts": dict(self.status_counts),
            "ambiguous": self.ambiguous,
            "contradictions": {
                "confirmed": self.contradictions_confirmed,
                "potential": self.contradictions_potential,
            },
            "acceptance": {
                "criteria": self.acceptance,
                "scenarios_without_result": self.acceptance_scenarios,
                "linked_requirements": self.acceptance_links,
            },
            "critical": list(self.critical),
            "structural": list(self.structural),
            "info": list(self.info),
            "rejected": dict(self.rejected),
            "relation_hypotheses": dict(self.relation_hypotheses),
            "review_categories": dict(self.review_categories),
            "coverage": self.coverage.to_dict(),
            "defects": [item.to_dict() for item in self.defects],
            "review": [item.to_dict() for item in self.review],
            "requirements": [item.to_dict() for item in self.requirements],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
