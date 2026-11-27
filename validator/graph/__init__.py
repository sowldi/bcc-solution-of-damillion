"""Fact Graph (этап 7): атомарные требования и типизированные связи между ними.

Подтверждённые связи — только доказанные машинной проверкой (явная ссылка на ID,
разные значения параметра в одной таблице). Пары-кандидаты для проверки моделью
отбираются по общим значимым словам; сама проверка пар — validator.llm.crosscheck,
её связи остаются гипотезами («unverified»). Пакет не зависит от LLM-слоя.
"""

from validator.graph.build import GRAPH_RULES, XRF_01, build_fact_graph, graph_nodes
from validator.graph.candidates import (
    CandidateInput,
    content_words,
    lexemes,
    rank_candidates,
    select_candidates,
)
from validator.graph.model import (
    MAX_CANDIDATES,
    CandidatePair,
    CandidateSelection,
    ExcludedRequirement,
    FactGraph,
    GraphEdge,
    GraphGap,
    GraphNode,
    edge_key,
)

__all__ = [
    "GRAPH_RULES",
    "MAX_CANDIDATES",
    "XRF_01",
    "CandidateInput",
    "CandidatePair",
    "CandidateSelection",
    "ExcludedRequirement",
    "FactGraph",
    "GraphEdge",
    "GraphGap",
    "GraphNode",
    "build_fact_graph",
    "content_words",
    "edge_key",
    "graph_nodes",
    "lexemes",
    "rank_candidates",
    "select_candidates",
]
