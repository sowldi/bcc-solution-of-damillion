"""Fact Graph (этап 7): узлы — атомарные требования, рёбра — типизированные связи.

Детерминированные связи подтверждаются машинной проверкой: явная ссылка на ID
(XRF-01, depends_on) и разные значения одного параметра в одной таблице (CON-03,
conflicts_with). Совпадение слов или чисел связи не создаёт — только пару-кандидата для
проверки моделью.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.graph_helpers import GRAPH_SPEC, load, relation_finding
from tests.lint_helpers import HEAD
from validator.graph import FactGraph, GraphEdge, graph_nodes
from validator.graph.candidates import content_stems

TABLE = (
    "## Параметры брони\n\n"
    "| Параметр | Значение |\n| --- | --- |\n"
    "| Срок хранения брони | 3 дня |\n"
    "| Срок хранения брони | 5 дней |\n"
)


def _pairs(graph: FactGraph) -> list[tuple[str, str]]:
    return [(pair.source_id, pair.target_id) for pair in graph.candidates.pairs]


# --- узлы --------------------------------------------------------------------------------


def test_nodes_are_atomic_requirements_bound_to_the_document(tmp_path: Path) -> None:
    built = load(tmp_path)
    graph = built.graph

    ids = [node.requirement_id for node in graph.nodes]
    assert ids == [f.id for f in built.segmentation.atomic_requirements]
    assert "Ф-6" not in ids and {"Ф-6/1", "Ф-6/2"} <= set(ids)  # составное — через части
    assert graph.document_sha256 == built.document.sha256
    node = graph.nodes[0]
    assert node.doc_id == "Ф-1"
    assert node.sources == built.segmentation.by_id("Ф-1").sources
    assert node.section_path == built.segmentation.by_id("Ф-1").section_path


def test_empty_and_unbound_requirements_are_excluded_with_reason(tmp_path: Path) -> None:
    built = load(tmp_path)
    fragments = list(built.segmentation.fragments)
    empty = next(i for i, f in enumerate(fragments) if f.id == "Ф-8")
    unbound = next(i for i, f in enumerate(fragments) if f.id == "Ф-5")
    fragments[empty] = replace(fragments[empty], text="  ")
    fragments[unbound] = replace(fragments[unbound], sources=())
    segmentation = replace(built.segmentation, fragments=tuple(fragments))

    nodes, excluded = graph_nodes(segmentation, built.document)

    assert {"Ф-5", "Ф-8"}.isdisjoint(node.requirement_id for node in nodes)
    assert [(item.requirement_id, item.code) for item in excluded] == [
        ("Ф-5", "no_source"),
        ("Ф-8", "empty"),
    ]


# --- детерминированные связи -------------------------------------------------------------


def test_explicit_reference_is_a_verified_dependency(tmp_path: Path) -> None:
    graph = load(tmp_path).graph

    [edge] = [edge for edge in graph.edges if edge.relation == "depends_on"]
    assert (edge.source_id, edge.target_id) == ("Ф-2", "Ф-1")
    assert (edge.origin, edge.status, edge.rule_id) == ("rule", "verified", "XRF-01")
    source, target = edge.evidence
    assert (source.requirement_id, source.quote) == ("Ф-2", "Ф-1")
    assert target.requirement_id == "Ф-1" and target.quote.startswith("Ф-1. Библиотекарь")
    assert graph.rejected == ()


def test_reference_to_a_composite_requirement_is_a_gap_not_an_edge(tmp_path: Path) -> None:
    graph = load(tmp_path).graph

    assert not any(edge.source_id == "Ф-7" for edge in graph.edges)
    [gap] = [gap for gap in graph.gaps if gap.requirement_id == "Ф-7"]
    assert gap.kind == "composite_target"
    assert "Ф-6/1" in gap.detail and "Ф-6/2" in gap.detail


def test_missing_and_ambiguous_targets_are_gaps(tmp_path: Path) -> None:
    graph = load(
        tmp_path,
        HEAD
        + "* Ф-1. Библиотекарь выдаёт книгу читателю.\n"
        + "* Ф-1. Библиотекарь принимает возврат книги.\n"
        + "* Ф-2. Читатель продлевает выдачу согласно Ф-1.\n"
        + "* Ф-3. Читатель оплачивает штраф согласно Ф-9.\n",
    ).graph

    assert graph.edges == ()
    kinds = {(gap.requirement_id, gap.kind) for gap in graph.gaps}
    assert ("Ф-2", "ambiguous_target") in kinds
    assert ("Ф-3", "missing_target") in kinds


def test_conflicting_values_in_one_table_are_a_verified_conflict(tmp_path: Path) -> None:
    built = load(tmp_path, HEAD + "* Ф-1. Библиотекарь выдаёт книгу читателю.\n\n" + TABLE)

    [con03] = [f for f in built.lint.findings if f.rule_id == "CON-03"]
    [edge] = [edge for edge in built.graph.edges if edge.relation == "conflicts_with"]
    assert (edge.origin, edge.status, edge.rule_id) == ("rule", "verified", "CON-03")
    assert edge.finding_id == con03.id
    assert (edge.source_id, edge.target_id) == con03.requirement_ids
    assert [e.quote for e in edge.evidence] == [e.quote for e in con03.evidence]
    # Уже подтверждённая связь не тратит бюджет модели: пара не кандидат.
    assert built.graph.candidates.linked == (con03.requirement_ids,)
    assert con03.requirement_ids not in _pairs(built.graph)


def test_same_parameter_in_different_contexts_is_not_a_conflict(tmp_path: Path) -> None:
    other = (
        "## Параметры абонемента\n\n"
        "| Параметр | Значение |\n| --- | --- |\n"
        "| Срок хранения брони | 5 дней |\n"
    )
    single = TABLE.replace("| Срок хранения брони | 5 дней |\n", "")
    graph = load(tmp_path, HEAD + "* Ф-1. Библиотекарь выдаёт книгу.\n\n" + single + other).graph

    assert not any(edge.relation == "conflicts_with" for edge in graph.edges)


def test_relations_are_not_created_from_shared_words(tmp_path: Path) -> None:
    graph = load(tmp_path).graph

    # Ф-3 и Ф-4 разделяют четыре значимых слова — это кандидат, но не ребро.
    assert ("Ф-3", "Ф-4") in _pairs(graph)
    assert not any({edge.source_id, edge.target_id} == {"Ф-3", "Ф-4"} for edge in graph.edges)
    assert {edge.relation for edge in graph.edges} == {"depends_on"}


# --- отбор пар-кандидатов ---------------------------------------------------------------


def test_candidates_are_ranked_by_shared_meaningful_stems(tmp_path: Path) -> None:
    graph = load(tmp_path).graph
    selection = graph.candidates

    first = selection.pairs[0]
    assert (first.source_id, first.target_id) == ("Ф-3", "Ф-4")
    assert {"брон", "книг", "хран"} <= set(first.shared)
    nodes = len(graph.nodes)
    assert selection.possible_pairs == nodes * (nodes - 1) // 2
    assert 0 < len(selection.pairs) <= selection.evaluated_pairs < selection.possible_pairs


def test_shared_number_alone_does_not_make_a_candidate(tmp_path: Path) -> None:
    graph = load(tmp_path).graph

    # У Ф-3 и Ф-5 общее только число 3 — ложный конфликт из-за числа не предлагается.
    assert ("Ф-3", "Ф-5") not in _pairs(graph)
    assert all(not stem.isdigit() for pair in graph.candidates.pairs for stem in pair.shared)


def test_stems_ignore_ids_numbers_negations_and_modal_words(tmp_path: Path) -> None:
    built = load(tmp_path)

    stems = content_stems(built.segmentation.by_id("Ф-3"), built.document)

    assert "брон" in stems and "хран" in stems
    assert not {"не", "3", "ф"} & stems
    modal = content_stems(built.segmentation.by_id("Ф-2"), built.document)
    assert "соглас" not in modal and not any(stem.startswith("ф-") for stem in modal)


def test_each_unordered_pair_is_a_candidate_once(tmp_path: Path) -> None:
    graph = load(tmp_path).graph
    pairs = _pairs(graph)
    order = {node.requirement_id: index for index, node in enumerate(graph.nodes)}

    assert len({frozenset(pair) for pair in pairs}) == len(pairs)
    assert all(order[a] < order[b] for a, b in pairs)


def test_identical_text_with_different_ids_is_not_a_candidate(tmp_path: Path) -> None:
    graph = load(
        tmp_path,
        HEAD
        + "* Ф-1. Читатель бронирует книгу в электронном каталоге.\n"
        + "* Ф-2. Читатель бронирует книгу в электронном каталоге.\n"
        + "* Ф-3. Читатель отменяет бронь книги в электронном каталоге.\n",
    ).graph

    assert ("Ф-1", "Ф-2") not in _pairs(graph)
    assert graph.candidates.duplicates == (("Ф-1", "Ф-2"),)
    assert {("Ф-1", "Ф-3"), ("Ф-2", "Ф-3")} <= set(_pairs(graph))


def test_too_common_stems_do_not_link_everything(tmp_path: Path) -> None:
    lines = [
        "* Ф-1. Система выдаёт книгу.",
        "* Ф-2. Система продлевает абонемент.",
        "* Ф-3. Система печатает квитанцию.",
        "* Ф-4. Система блокирует карточку.",
    ]
    graph = load(tmp_path, HEAD + "\n".join(lines) + "\n").graph

    assert "систем" in graph.candidates.common_stems
    assert graph.candidates.pairs == ()


# --- пустой граф, повторный прогон, инварианты --------------------------------------------


def test_document_without_requirements_gives_an_empty_graph(tmp_path: Path) -> None:
    graph = load(tmp_path, "# Сервис\n\n## Введение\n\nОбщее описание библиотеки.\n").graph

    assert (graph.nodes, graph.edges, graph.gaps, graph.candidates.pairs) == ((), (), (), ())
    assert graph.candidates.possible_pairs == 0


def test_repeated_build_is_identical(tmp_path: Path) -> None:
    first = load(tmp_path).graph
    second = load(tmp_path).graph

    assert first.to_json() == second.to_json()
    assert first.to_dict()["schema"] == "validator.fact-graph/1"


def test_graph_keeps_llm_edges_unverified(tmp_path: Path) -> None:
    built = load(tmp_path)
    finding = relation_finding(
        built,
        ("Ф-3", "не хранит бронь книги дольше 3 дней"),
        ("Ф-4", "хранит бронь книги 5 дней"),
    )
    llm_edge = GraphEdge.from_finding(replace(finding, evidence_status="unverified"))

    graph = built.graph.with_hypotheses((replace(finding, evidence_status="unverified"),))

    assert llm_edge in graph.edges
    assert (llm_edge.origin, llm_edge.status) == ("llm", "unverified")
    with pytest.raises(ValueError, match="unverified"):
        GraphEdge.from_finding(replace(finding, evidence_status="verified"))
    with pytest.raises(ValueError, match="verified"):
        replace(built.graph.edges[0], status="unverified")


def test_graph_rejects_edges_for_another_document(tmp_path: Path) -> None:
    built = load(tmp_path)
    finding = relation_finding(
        built,
        ("Ф-3", "не хранит бронь книги дольше 3 дней"),
        ("Ф-4", "хранит бронь книги 5 дней"),
    )
    foreign = replace(finding, evidence_status="unverified", document_sha256="0" * 64)

    with pytest.raises(ValueError, match="SHA-256"):
        built.graph.with_hypotheses((foreign,))


def test_graph_spec_has_expected_shape(tmp_path: Path) -> None:
    # Защита от случайного изменения общего примера тестов.
    assert GRAPH_SPEC.count("* Ф-") == 8
    assert len(load(tmp_path).graph.nodes) == 9
