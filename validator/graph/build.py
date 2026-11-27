"""Построение Fact Graph без модели: узлы и связи, подтверждённые машинной проверкой.

- Узлы — атомарные требования сегментации с текстом и корректной привязкой к источнику;
  остальные — вне графа с причиной (пустой текст, нет источника).
- depends_on (XRF-01): требование явно ссылается на ID другого требования («согласно Ф-1»).
  Связь строится, только если ID однозначно принадлежит одному атомарному требованию;
  доказательства — цитата ссылки в источнике и текст цели; связь проходит Evidence Gate.
  Ссылка на составное требование, на ID нескольких требований или на отсутствующий ID —
  пропуск с причиной, а не ребро.
- conflicts_with (CON-03): один параметр с разными значениями в одной таблице «параметр —
  значение» — из подтверждённого замечания CON-03 (пары строк с разными значениями).
  Тот же параметр в разных таблицах (разные контексты) — не противоречие.
Совпадение слов или чисел связи не создаёт: по нему только отбираются пары-кандидаты.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from validator.evidence import TrustedRule, build_evidence, gate_findings
from validator.findings import Evidence, Finding
from validator.graph.candidates import select_candidates
from validator.graph.model import (
    MAX_CANDIDATES,
    ExcludedRequirement,
    FactGraph,
    GraphEdge,
    GraphGap,
    GraphNode,
)
from validator.model import Document
from validator.segmentation import Requirement, Segmentation, span_sources
from validator.segmentation.text import content_key, is_document_id, locate_references

if TYPE_CHECKING:
    from validator.lint import LintResult

# Правило графа: вне каталога lint (детерминированный результат этапов 3–6 не меняется),
# но Evidence Gate сверяет связь с этой записью так же, как замечания правил.
XRF_01 = TrustedRule(
    rule_id="XRF-01",
    version="1",
    severity="info",
    category="traceability",
    criterion="TRC",
    evidence_kind="comparison",
    title="Явная ссылка на требование",
)
GRAPH_RULES = (XRF_01,)
CONFLICT_RULE = "CON-03"


def graph_nodes(
    segmentation: Segmentation, document: Document
) -> tuple[tuple[GraphNode, ...], tuple[ExcludedRequirement, ...]]:
    nodes: list[GraphNode] = []
    excluded: list[ExcludedRequirement] = []
    for fragment in segmentation.atomic_requirements:
        problem = _excluded(fragment, document)
        if problem is not None:
            excluded.append(problem)
            continue
        nodes.append(
            GraphNode(fragment.id, fragment.doc_id, fragment.section_path, fragment.sources)
        )
    return tuple(nodes), tuple(excluded)


def build_fact_graph(
    document: Document,
    segmentation: Segmentation,
    lint: LintResult | None = None,
    *,
    max_candidates: int = MAX_CANDIDATES,
) -> FactGraph:
    nodes, excluded = graph_nodes(segmentation, document)
    members = {node.requirement_id: segmentation.by_id(node.requirement_id) for node in nodes}
    drafts, gaps = _reference_links(document, segmentation, members)
    gate = gate_findings(drafts, segmentation, document, rules=GRAPH_RULES)
    edges = [GraphEdge.from_finding(finding) for finding in gate.accepted]
    if lint is not None:
        conflicts, outside = _table_conflicts(lint, document, members)
        edges += conflicts
        gaps += outside
    order = {node.requirement_id: index for index, node in enumerate(nodes)}
    edges.sort(key=lambda e: (order[e.source_id], order[e.target_id], e.relation))
    gaps.sort(key=lambda g: (order.get(g.requirement_id, len(order)), g.kind, g.target))
    linked = {frozenset((edge.source_id, edge.target_id)) for edge in edges}
    return FactGraph(
        document_sha256=document.sha256,
        nodes=nodes,
        edges=tuple(edges),
        excluded=excluded,
        gaps=tuple(gaps),
        rejected=gate.rejected,
        candidates=select_candidates(
            list(members.values()), document, linked, limit=max_candidates
        ),
        requirements_total=len(segmentation.atomic_requirements),
    )


def _excluded(fragment: Requirement, document: Document) -> ExcludedRequirement | None:
    if not fragment.text.strip():
        return ExcludedRequirement(fragment.id, "empty", "Текст требования пуст.")
    if not fragment.sources:
        return ExcludedRequirement(fragment.id, "no_source", "Нет ссылок на источник.")
    try:
        span_sources(fragment, 0, len(fragment.text), document)
    except ValueError as error:
        return ExcludedRequirement(fragment.id, "no_source", f"Некорректный источник: {error}")
    return None


# --- явные ссылки на требования (XRF-01) ---------------------------------------------------


def _reference_links(
    document: Document, segmentation: Segmentation, members: dict[str, Requirement]
) -> tuple[list[Finding], list[GraphGap]]:
    doc_ids = frozenset(f.doc_id for f in segmentation.fragments if f.doc_id)
    headings = tuple(block.text for block in document.blocks if block.kind == "heading")
    owners: dict[str, list[Requirement]] = {}
    for fragment in segmentation.atomic_requirements:
        if fragment.doc_id:
            owners.setdefault(fragment.doc_id, []).append(fragment)
    findings: list[Finding] = []
    gaps: list[GraphGap] = []
    for source in members.values():
        seen: set[str] = set()
        spans = locate_references(
            source.text, doc_ids=doc_ids, headings=headings, own_id=source.doc_id
        )
        for span in spans:
            kind, _, target = span.label.partition(":")
            if kind not in ("req", "unresolved") or not is_document_id(target):
                continue  # ссылки на разделы — не связи требований
            if target in seen:
                continue
            seen.add(target)
            targets = owners.get(target, []) if kind == "req" else []
            gap = _target_gap(source, target, targets, members)
            if gap is not None:
                gaps.append(gap)
                continue
            findings.append(_dependency(source, span.start, span.end, targets[0], document))
    return findings, gaps


def _target_gap(
    source: Requirement,
    target: str,
    targets: list[Requirement],
    members: dict[str, Requirement],
) -> GraphGap | None:
    ids = ", ".join(fragment.id for fragment in targets)
    if not targets:
        return GraphGap(
            "missing_target",
            source.id,
            target,
            f"атомарного требования с ID «{target}» в документе нет",
        )
    if len(targets) > 1:
        parents = {fragment.parent_id for fragment in targets}
        if len(parents) == 1 and None not in parents:
            return GraphGap(
                "composite_target",
                source.id,
                target,
                f"составное требование ({ids}) — с каким из атомарных требований связь, "
                "текст не указывает",
            )
        return GraphGap(
            "ambiguous_target",
            source.id,
            target,
            f"ID присвоен нескольким требованиям ({ids}) — ссылка неоднозначна",
        )
    if targets[0].id == source.id:
        return GraphGap("ambiguous_target", source.id, target, "ссылка требования на себя")
    if targets[0].id not in members:
        return GraphGap(
            "excluded_target",
            source.id,
            target,
            f"требование {targets[0].id} вне графа (пустое или без привязки к источнику)",
        )
    return None


def _dependency(
    source: Requirement, start: int, end: int, target: Requirement, document: Document
) -> Finding:
    evidence = (
        _evidence(source, start, end, document),
        _evidence(target, 0, len(target.text), document),
    )
    reference = source.text[start:end]
    return Finding(
        id=f"{XRF_01.rule_id}:{source.id}@{start}+{target.id}@0",
        rule_id=XRF_01.rule_id,
        rule_version=XRF_01.version,
        severity=XRF_01.severity,
        category=XRF_01.category,
        criterion=XRF_01.criterion,
        title=XRF_01.title,
        explanation=(
            f"Требование {source.id} явно ссылается на «{reference}»: его выполнение и "
            f"проверка зависят от требования {target.id}."
        ),
        recommendation=(
            f"При изменении {target.id} проверьте {source.id}; ссылку держите актуальной."
        ),
        requirement_ids=(source.id, target.id),
        evidence=evidence,
        evidence_kind="comparison",
        document_sha256=document.sha256,
        relation="depends_on",
    )


def _evidence(fragment: Requirement, start: int, end: int, document: Document) -> Evidence:
    try:
        return build_evidence(fragment, start, end, document)
    except ValueError:
        # Отрезок не привязывается к документу: связь отклонит Evidence Gate с причиной.
        return Evidence(fragment.id, start, end, fragment.text[start:end], ())


# --- разные значения одного параметра (CON-03) --------------------------------------------


def _table_conflicts(
    lint: LintResult, document: Document, members: dict[str, Requirement]
) -> tuple[list[GraphEdge], list[GraphGap]]:
    edges: list[GraphEdge] = []
    gaps: list[GraphGap] = []
    for finding in lint.findings:
        if finding.rule_id != CONFLICT_RULE or finding.document_sha256 != document.sha256:
            continue
        ids = finding.requirement_ids
        outside = [rid for rid in ids if rid not in members]
        if outside:
            gaps.append(
                GraphGap(
                    "conflict_outside_graph",
                    ids[0],
                    finding.id,
                    f"участники {', '.join(outside)} — не требования графа; замечание "
                    "остаётся в разделе 4",
                )
            )
            continue
        evidence = {item.requirement_id: item for item in finding.evidence}
        values = {rid: _value(members[rid], document) for rid in ids}
        for index, first in enumerate(ids):
            for second in ids[index + 1 :]:
                if values[first] == values[second]:
                    continue  # одинаковое значение — не противоречие
                edges.append(
                    GraphEdge(
                        source_id=first,
                        target_id=second,
                        relation="conflicts_with",
                        origin="rule",
                        status="verified",
                        rule_id=finding.rule_id,
                        finding_id=finding.id,
                        evidence=(evidence[first], evidence[second]),
                        title=finding.title,
                        explanation=finding.explanation,
                        recommendation=finding.recommendation,
                    )
                )
    return edges, gaps


def _value(fragment: Requirement, document: Document) -> str:
    block = document.blocks[fragment.sources[0].block]
    cell = block.cells[1] if len(block.cells) == 2 else block.text
    return content_key(cell)
