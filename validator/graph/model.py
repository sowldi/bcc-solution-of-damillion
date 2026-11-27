"""Модель Fact Graph: узлы — атомарные требования, рёбра — типизированные связи.

Узел ссылается на требование сегментации (ID, места в документе через SourceRef); сам текст
не копируется — он в сегментации той же версии документа (SHA-256 графа).
Ребро — связь двух узлов с доказательством для обоих концов (по точной цитате из каждого),
происхождением (rule | llm) и статусом: «verified» — только у связей, подтверждённых
машинной проверкой, «unverified» — всегда у связей, предложенных моделью.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from validator.findings import (
    ORIGINS,
    RELATION_TYPES,
    SYMMETRIC_RELATIONS,
    Evidence,
    EvidenceStatus,
    Finding,
    Origin,
    RejectedFinding,
    RelationType,
)
from validator.segmentation import SourceRef
from validator.segmentation.model import source_ref_to_dict

SCHEMA = "validator.fact-graph/1"
# Ограничение отбора пар-кандидатов (этап 7.1): в 10 раз больше наибольшего бюджета модели
# (LLM_MAX_PAIR_REQUESTS <= 100); память отбора — O(n + MAX_CANDIDATES).
MAX_CANDIDATES = 1000

ExcludeCode = Literal["empty", "no_source"]
GapKind = Literal[
    "composite_target",  # ссылка на составное требование (несколько атомарных частей)
    "ambiguous_target",  # ID принадлежит нескольким требованиям
    "missing_target",  # требования с таким ID нет
    "excluded_target",  # цель ссылки вне графа (пустая или без источника)
    "conflict_outside_graph",  # участники CON-03 — не требования графа
]
EdgeKey = tuple[RelationType, str, str]


def edge_key(relation: RelationType, source_id: str, target_id: str) -> EdgeKey:
    """Ключ связи для поиска повторов: у симметричной связи направление не важно."""
    if relation in SYMMETRIC_RELATIONS and target_id < source_id:
        source_id, target_id = target_id, source_id
    return relation, source_id, target_id


@dataclass(frozen=True)
class GraphNode:
    requirement_id: str
    doc_id: str | None
    section_path: tuple[str, ...]
    sources: tuple[SourceRef, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "doc_id": self.doc_id,
            "section_path": list(self.section_path),
            "sources": [source_ref_to_dict(ref) for ref in self.sources],
        }


@dataclass(frozen=True)
class ExcludedRequirement:
    """Атомарное требование вне графа: пустое или без корректной привязки к источнику."""

    requirement_id: str
    code: ExcludeCode
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"requirement_id": self.requirement_id, "code": self.code, "reason": self.reason}


@dataclass(frozen=True)
class GraphGap:
    """Пропуск: связь есть в тексте, но к двум узлам графа не сведена."""

    kind: GapKind
    requirement_id: str
    target: str  # ID, на который ссылается требование, или ID замечания CON-03
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "requirement_id": self.requirement_id,
            "target": self.target,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class GraphEdge:
    source_id: str
    target_id: str
    relation: RelationType
    origin: Origin
    status: EvidenceStatus
    rule_id: str
    finding_id: str  # замечание, которое доказывает связь (XRF-01, CON-03, LLM-PAIR)
    evidence: tuple[Evidence, ...]  # (цитата источника, цитата цели)
    title: str = ""  # название правила или гипотезы
    explanation: str = ""
    recommendation: str = ""
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.relation not in RELATION_TYPES:
            raise ValueError(f"Недопустимый тип связи: {self.relation!r}.")
        if self.origin not in ORIGINS:
            raise ValueError(f"Недопустимое происхождение связи: {self.origin!r}.")
        if self.source_id == self.target_id:
            raise ValueError("Связь требования с самим собой не допускается.")
        ids = tuple(evidence.requirement_id for evidence in self.evidence)
        if ids != (self.source_id, self.target_id):
            raise ValueError(
                f"Связь {self.source_id} → {self.target_id}: нужны ровно две цитаты — "
                "источника и цели."
            )
        if self.origin == "rule" and self.status != "verified":
            raise ValueError(
                f"Связь правила {self.finding_id} в графе должна иметь статус verified: "
                "в граф попадают только связи, прошедшие машинную проверку."
            )
        if self.origin == "llm" and self.status != "unverified":
            raise ValueError(
                f"Связь модели {self.finding_id} в графе — только unverified: цитаты "
                "доказывают происхождение текста, но не саму связь."
            )

    @property
    def key(self) -> EdgeKey:
        return edge_key(self.relation, self.source_id, self.target_id)

    @classmethod
    def from_finding(cls, finding: Finding) -> GraphEdge:
        """Связь из замечания, прошедшего Evidence Gate."""
        if finding.relation is None:
            raise ValueError(f"Замечание {finding.id} не является связью (relation не задан).")
        if finding.evidence_status not in ("verified", "unverified"):
            raise ValueError(
                f"Замечание {finding.id} не прошло Evidence Gate "
                f"(статус {finding.evidence_status}): в граф не добавляется."
            )
        if len(finding.requirement_ids) != 2:
            raise ValueError(f"Связь {finding.id} должна относиться ровно к двум требованиям.")
        source_id, target_id = finding.requirement_ids
        return cls(
            source_id=source_id,
            target_id=target_id,
            relation=finding.relation,
            origin=finding.origin,
            status=finding.evidence_status,
            rule_id=finding.rule_id,
            finding_id=finding.id,
            evidence=finding.evidence,
            title=finding.title,
            explanation=finding.explanation,
            recommendation=finding.recommendation,
            confidence=finding.confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "origin": self.origin,
            "status": self.status,
            "rule_id": self.rule_id,
            "finding_id": self.finding_id,
            "title": self.title,
            "explanation": self.explanation,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "evidence": [evidence.to_dict() for evidence in self.evidence],
        }


@dataclass(frozen=True)
class CandidatePair:
    """Пара требований для проверки моделью: source — раньше в документе, чем target."""

    source_id: str
    target_id: str
    score: int  # общие основы слов (+1, если требования в одном разделе)
    shared: tuple[str, ...]  # общие значимые основы слов
    same_section: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "score": self.score,
            "shared": list(self.shared),
            "same_section": self.same_section,
        }


@dataclass(frozen=True)
class CandidateSelection:
    """Отбор пар-кандидатов: что отобрано, сколько оценено и что исключено и почему."""

    possible_pairs: int = 0  # n(n-1)/2 — перебор всех пар не выполняется
    evaluated_pairs: int = 0  # пары с хотя бы одним общим значимым словом (из индекса)
    pairs: tuple[CandidatePair, ...] = ()  # лучшие по оценке, не больше limit
    duplicates: tuple[tuple[str, str], ...] = ()  # дословные повторы (первые limit)
    linked: tuple[tuple[str, str], ...] = ()  # уже связаны подтверждённой связью
    common_stems: tuple[str, ...] = ()  # слишком общие слова — не учитывались
    min_shared: int = 2
    found: int = 0  # кандидатов до ограничения (точно, в том же проходе)
    limit: int = MAX_CANDIDATES  # ограничение отбора
    duplicate_count: int = 0  # дословных повторов всего

    def __post_init__(self) -> None:
        if len(self.pairs) > self.limit:
            raise ValueError("В отборе больше пар, чем позволяет ограничение отбора.")
        if len(self.pairs) > self.found:
            raise ValueError("В отборе больше пар, чем найдено кандидатов.")

    @property
    def truncated(self) -> bool:
        """Часть найденных кандидатов исключена ограничением отбора."""
        return self.found > len(self.pairs)

    @property
    def excluded_by_limit(self) -> int:
        return self.found - len(self.pairs)

    @property
    def outside_pairs(self) -> int:
        """Пары меньше чем с MIN_SHARED общими словами: в отбор не попали, моделью не
        проверяются."""
        return self.possible_pairs - self.found - self.duplicate_count - len(self.linked)

    def to_dict(self) -> dict[str, Any]:
        return {
            "possible_pairs": self.possible_pairs,
            "evaluated_pairs": self.evaluated_pairs,
            "min_shared": self.min_shared,
            "found": self.found,
            "limit": self.limit,
            "duplicate_count": self.duplicate_count,
            "common_stems": list(self.common_stems),
            "duplicates": [list(pair) for pair in self.duplicates],
            "linked": [list(pair) for pair in self.linked],
            "pairs": [pair.to_dict() for pair in self.pairs],
        }


@dataclass(frozen=True)
class FactGraph:
    document_sha256: str
    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()
    excluded: tuple[ExcludedRequirement, ...] = ()
    gaps: tuple[GraphGap, ...] = ()
    rejected: tuple[RejectedFinding, ...] = ()  # связи правил, отклонённые Evidence Gate
    candidates: CandidateSelection = field(default_factory=CandidateSelection)
    requirements_total: int = 0  # атомарных требований в сегментации

    def __post_init__(self) -> None:
        ids = [node.requirement_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("ID узлов графа не уникальны.")
        known = set(ids)
        keys = [edge.key for edge in self.edges]
        if len(keys) != len(set(keys)):
            raise ValueError("Связь в графе повторяется.")
        for edge in self.edges:
            if not {edge.source_id, edge.target_id} <= known:
                raise ValueError(
                    f"Связь {edge.source_id} → {edge.target_id}: конец связи не узел графа."
                )
        for pair in self.candidates.pairs:
            if not {pair.source_id, pair.target_id} <= known:
                raise ValueError(f"Пара {pair.source_id} — {pair.target_id}: не узлы графа.")

    def with_hypotheses(self, findings: Iterable[Finding]) -> FactGraph:
        """Граф с гипотезами модели — рёбрами со статусом «unverified»."""
        added: list[GraphEdge] = []
        for finding in findings:
            if finding.document_sha256 != self.document_sha256:
                raise ValueError(
                    f"Гипотеза {finding.id} построена для другого документа (SHA-256 не совпадает)."
                )
            edge = GraphEdge.from_finding(finding)
            if edge.origin != "llm":
                raise ValueError(f"{finding.id}: гипотеза связи должна иметь origin=llm.")
            added.append(edge)
        return replace(self, edges=self.edges + tuple(added))

    def count(self, relation: RelationType, origin: Origin) -> int:
        return sum(1 for e in self.edges if e.relation == relation and e.origin == origin)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "document_sha256": self.document_sha256,
            "requirements_total": self.requirements_total,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "excluded": [item.to_dict() for item in self.excluded],
            "gaps": [gap.to_dict() for gap in self.gaps],
            "rejected": [item.to_dict() for item in self.rejected],
            "candidates": self.candidates.to_dict(),
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
