"""Помощники тестов Fact Graph и проверки пар (этап 7): собственные документы, домен
«библиотека»; сети и ключей нет.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.lint_helpers import HEAD
from validator.evidence import build_evidence
from validator.findings import Finding, Origin, RelationType
from validator.graph import XRF_01, FactGraph, build_fact_graph
from validator.ingest import load_document
from validator.lint import LintResult, run_lint
from validator.model import Document
from validator.segmentation import Segmentation, segment_document

# Ф-2 явно ссылается на Ф-1; Ф-3 и Ф-4 — несовместимые сроки хранения брони (в Ф-3 —
# отрицание); у Ф-3 и Ф-5 общее только число 3; Ф-6 — составное требование (Ф-6/1, Ф-6/2),
# на которое ссылается Ф-7; Ф-8 ни с чем не связан.
GRAPH_SPEC = HEAD + (
    "* Ф-1. Библиотекарь подтверждает бронь книги в течение рабочего дня.\n"
    "* Ф-2. Читатель отменяет бронь книги до выдачи согласно Ф-1.\n"
    "* Ф-3. Система не хранит бронь книги дольше 3 дней.\n"
    "* Ф-4. Система хранит бронь книги 5 дней после подтверждения.\n"
    "* Ф-5. Читатель получает уведомление за 3 дня до окончания абонемента.\n"
    "* Ф-6. Система отправляет уведомление читателю. Система записывает событие в журнал.\n"
    "* Ф-7. Журнал хранится согласно Ф-6.\n"
    "* Ф-8. Отчёт по залам формируется ежемесячно.\n"
)


@dataclass(frozen=True)
class Built:
    graph: FactGraph
    document: Document
    segmentation: Segmentation
    lint: LintResult


def load(tmp_path: Path, markdown: str = GRAPH_SPEC, name: str = "spec.md") -> Built:
    path = tmp_path / name
    path.write_text(markdown, encoding="utf-8")
    document = load_document(path)
    segmentation = segment_document(document)
    lint = run_lint(document, segmentation)
    return Built(build_fact_graph(document, segmentation, lint), document, segmentation, lint)


def relation_finding(
    built: Built,
    source: tuple[str, str],
    target: tuple[str, str],
    *,
    relation: RelationType = "conflicts_with",
    origin: Origin = "llm",
) -> Finding:
    """Связь двух требований: (ID, цитата) источника и цели, цитаты — точные отрезки."""
    evidence = []
    for requirement_id, quote in (source, target):
        fragment = built.segmentation.by_id(requirement_id)
        start = fragment.text.index(quote)
        evidence.append(build_evidence(fragment, start, start + len(quote), built.document))
    rule = origin == "rule"
    return Finding(
        id=f"TEST:{source[0]}+{target[0]}:{relation}",
        rule_id=XRF_01.rule_id if rule else "LLM-PAIR",
        rule_version=XRF_01.version if rule else "prompt-pair-1",
        severity=XRF_01.severity if rule else "major",
        category=XRF_01.category if rule else "consistency",
        criterion=XRF_01.criterion if rule else "S-CNS",
        title=XRF_01.title if rule else "Возможное противоречие",
        explanation="Пояснение связи.",
        recommendation="Согласуйте требования.",
        requirement_ids=(source[0], target[0]),
        evidence=tuple(evidence),
        evidence_kind="comparison",
        origin=origin,
        document_sha256=built.document.sha256,
        relation=relation,
    )


def relations_json(*items: dict[str, Any]) -> str:
    return json.dumps({"relations": list(items)}, ensure_ascii=False)


def relation(
    source_id: str,
    source_quote: str,
    target_id: str,
    target_quote: str,
    **changes: Any,
) -> dict[str, Any]:
    """Элемент ответа модели на проверку пары (промпт pair-1)."""
    item: dict[str, Any] = {
        "source_id": source_id,
        "target_id": target_id,
        "relation": "conflicts_with",
        "source_quote": source_quote,
        "target_quote": target_quote,
        "explanation": "Сроки хранения брони несовместимы.",
        "recommendation": "Оставьте один срок хранения брони.",
        "confidence": 0.8,
    }
    item.update(changes)
    return item
