"""Проверка пар требований моделью (этап 7): связи — гипотезы, а не выводы.

1. Модели отправляются только пары-кандидаты Fact Graph (validator.graph.candidates), по
   одному запросу на пару, в порядке оценки; сверх бюджета LLM_MAX_PAIR_REQUESTS — не
   отправляются и перечисляются как непроверенные (0 — проверка пар выключена). Провайдер,
   повторы временных ошибок (LLM_MAX_RETRIES, Retry-After) и ограничитель частоты (LLM_RPM,
   общий с обзором этапа 6) — те же, что у обзора.
2. Модели передаются оба требования: ID, точный текст, раздел, родительский фрагмент и
   границы источника — как недоверенные данные.
3. Ответ — строгая схема {"relations": [...]}. Гипотеза принимается в работу, только если
   оба ID — именно проверяемая пара (ID по ответу модели не меняются), требования разные и
   каждая цитата один раз дословно найдена в тексте своего требования и содержательна.
   Симметричная связь (conflicts_with, related_to) хранится в порядке документа: A→B и B→A
   — одна связь, повтор отклоняется; связь, уже подтверждённая правилом, — тоже.
4. Все гипотезы проходят Evidence Gate (обе цитаты, SourceRef, SHA-256 документа, ровно два
   атомарных требования). Прошедшие — «unverified»: цитаты доказывают происхождение текста,
   но не связь. Отклонённые сохраняются с причинами.
5. Сбой провайдера или непригодный ответ прерывает проверку той же ошибкой с указанием пары
   и прогресса; частичные результаты не сохраняются (как у обзора этапа 6).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from validator.errors import ProviderError
from validator.evidence import EvidenceGateError, gate_findings
from validator.findings import (
    SYMMETRIC_RELATIONS,
    Evidence,
    Finding,
    RelationType,
    RuleCategory,
    Severity,
)
from validator.graph import FactGraph, edge_key
from validator.llm.analysis import LLMOptions
from validator.llm.base import LLMProvider, LLMRequest
from validator.llm.prompts import PAIR_PROMPT_VERSION, PAIR_SYSTEM_PROMPT, build_pair_message
from validator.llm.retry import Clock, RateLimiter, Sleep, complete_with_retry
from validator.llm.review import _failure_kind, _interrupted, _parent, locate_quote
from validator.llm.schema import RelationHypothesis, parse_relations
from validator.model import Document
from validator.segmentation import Requirement, Segmentation

log = logging.getLogger("validator.llm")

SCHEMA = "validator.llm-crosscheck/1"
RULE_ID = "LLM-PAIR"
# Тип связи → (группа замечаний, критерий качества, предполагаемая важность).
RELATION_MAPPING: dict[RelationType, tuple[RuleCategory, str, Severity]] = {
    "conflicts_with": ("consistency", "S-CNS", "major"),
    "depends_on": ("traceability", "TRC", "minor"),
    "requires_condition": ("completeness", "COM", "minor"),
    "related_to": ("consistency", "S-CNS", "info"),
}
# Название гипотезы задаёт система, а не модель: «возможное», а не установленное.
HYPOTHESIS_TITLES: dict[RelationType, str] = {
    "conflicts_with": "Возможное противоречие",
    "depends_on": "Возможная зависимость",
    "requires_condition": "Возможное условие выполнения",
    "related_to": "Возможная связь",
}

Pair = tuple[str, str]


@dataclass(frozen=True)
class RejectedRelation:
    """Элемент ответа модели по паре, не принятый как гипотеза связи, и причины."""

    pair: Pair
    index: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"pair": list(self.pair), "index": self.index, "reasons": list(self.reasons)}


@dataclass(frozen=True)
class CrossCheckAnalysis:
    """Итог проверки пар: какие пары модель получила, какие нет, гипотезы и отказы."""

    provider: str
    model: str
    prompt_version: str
    document_sha256: str
    budget: int  # LLM_MAX_PAIR_REQUESTS
    candidates_total: int
    checked: tuple[Pair, ...] = ()  # отправлены модели (каждая пара — один раз)
    skipped: tuple[Pair, ...] = ()  # кандидаты сверх бюджета
    covered: tuple[str, ...] = ()  # требования хотя бы в одной проверенной паре
    hypotheses: tuple[Finding, ...] = ()  # только «unverified»
    rejected: tuple[RejectedRelation, ...] = ()

    def __post_init__(self) -> None:
        pairs = [frozenset(pair) for pair in (*self.checked, *self.skipped)]
        if len(pairs) != len(set(pairs)):
            raise ValueError("Пара требований проверена или пропущена повторно.")
        checked = {frozenset(pair) for pair in self.checked}
        for finding in self.hypotheses:
            if finding.origin != "llm" or finding.evidence_status != "unverified":
                raise ValueError(
                    f"Гипотеза связи {finding.id} должна иметь origin=llm и статус unverified: "
                    "модель связь только предполагает."
                )
            if finding.relation is None:
                raise ValueError(f"Гипотеза {finding.id} не является связью (relation).")
            if finding.document_sha256 != self.document_sha256:
                raise ValueError(
                    f"Гипотеза связи {finding.id} построена для другого документа "
                    "(SHA-256 не совпадает)."
                )
            if frozenset(finding.requirement_ids) not in checked:
                raise ValueError(f"Гипотеза связи {finding.id} — к паре, которая не проверялась.")
        if set(self.covered) != {rid for pair in self.checked for rid in pair}:
            raise ValueError("covered не совпадает с требованиями проверенных пар.")
        if len(self.checked) > self.budget:
            raise ValueError("Проверено больше пар, чем позволяет LLM_MAX_PAIR_REQUESTS.")

    @property
    def requests(self) -> int:
        return len(self.checked)

    @property
    def disabled(self) -> bool:
        """Проверка пар выключена: LLM_MAX_PAIR_REQUESTS = 0."""
        return self.budget == 0

    @property
    def truncated(self) -> bool:
        """Часть кандидатов не проверена из-за лимита."""
        return bool(self.skipped) and not self.disabled

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "document_sha256": self.document_sha256,
            "budget": self.budget,
            "candidates_total": self.candidates_total,
            "checked": [list(pair) for pair in self.checked],
            "skipped": [list(pair) for pair in self.skipped],
            "covered": list(self.covered),
            "hypotheses": [finding.to_dict() for finding in self.hypotheses],
            "rejected": [item.to_dict() for item in self.rejected],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def crosscheck_pairs(
    document: Document,
    segmentation: Segmentation,
    graph: FactGraph,
    provider: LLMProvider,
    options: LLMOptions,
    *,
    sleep: Sleep = time.sleep,
    clock: Clock = time.monotonic,
    limiter: RateLimiter | None = None,
) -> CrossCheckAnalysis:
    if not graph.document_sha256 == segmentation.document_sha256 == document.sha256:
        raise EvidenceGateError(
            "Fact Graph или сегментация построены для другой версии документа — пары "
            "проверять нельзя."
        )
    candidates = graph.candidates.pairs
    budget = max(options.max_pair_requests, 0)
    selected = [(pair.source_id, pair.target_id) for pair in candidates[:budget]]
    skipped = tuple((pair.source_id, pair.target_id) for pair in candidates[budget:])
    if limiter is None and options.rpm and selected:
        limiter = RateLimiter(options.rpm, clock=clock, sleep=sleep)
    order = {node.requirement_id: index for index, node in enumerate(graph.nodes)}
    confirmed = {edge.key: edge for edge in graph.edges}
    known: dict[tuple[str, str, str], str] = {}
    drafts: list[tuple[Pair, int, Finding]] = []
    rejected: list[RejectedRelation] = []
    for position, pair in enumerate(selected):
        first, second = (segmentation.by_id(rid) for rid in pair)
        request = LLMRequest(
            system=PAIR_SYSTEM_PROMPT,
            user=build_pair_message(
                first, second, (_parent(first, segmentation), _parent(second, segmentation))
            ),
            max_output_tokens=options.max_output_tokens,
            temperature=options.temperature,
        )
        try:
            response = complete_with_retry(
                provider, request, max_retries=options.max_retries, sleep=sleep, limiter=limiter
            )
            parsed = parse_relations(response.text)
        except ProviderError as error:
            raise _interrupted(
                error,
                f"Проверка пар прервана на паре {pair[0]} — {pair[1]}. Тип сбоя: "
                f"{_failure_kind(error)}. До сбоя проверено {position} из {len(selected)} "
                f"запланированных пар (пар-кандидатов: {len(candidates)}); частичные "
                f"результаты не сохраняются. Причина ({provider.name}): {error}",
            ) from None
        rejected += [RejectedRelation(pair, item.index, item.reasons) for item in parsed.rejected]
        for hypothesis in parsed.relations:
            result = _to_finding(hypothesis, first, second, document, order)
            if isinstance(result, Finding):
                assert result.relation is not None
                key = edge_key(result.relation, *result.requirement_ids)
                if key in confirmed:
                    result = [
                        f"Связь уже подтверждена правилом {confirmed[key].rule_id}: гипотеза "
                        "модели её не дополняет."
                    ]
                elif key in known:
                    result = [f"Связь — повтор уже принятой {known[key]}."]
                else:
                    known[key] = result.id
            if isinstance(result, Finding):
                drafts.append((pair, hypothesis.index, result))
            else:
                rejected.append(RejectedRelation(pair, hypothesis.index, tuple(result)))
    hypotheses: tuple[Finding, ...] = ()
    if drafts:
        # Тот же Evidence Gate; каталога для модели нет — прошедшие получают «unverified».
        gate = gate_findings([draft for *_, draft in drafts], segmentation, document, rules=())
        hypotheses = gate.unverified
        where = {draft.id: (pair, index) for pair, index, draft in drafts}
        rejected += [
            RejectedRelation(*where[item.finding.id], item.reasons) for item in gate.rejected
        ]
    position_of = {pair: index for index, pair in enumerate(selected)}
    rejected.sort(key=lambda item: (position_of[item.pair], item.index))
    covered = {rid for pair in selected for rid in pair}
    log.info(
        "Проверка пар %s/%s: пар %d из %d кандидатов, гипотез %d, отклонено %d",
        provider.name,
        provider.model,
        len(selected),
        len(candidates),
        len(hypotheses),
        len(rejected),
    )
    return CrossCheckAnalysis(
        provider=provider.name,
        model=provider.model,
        prompt_version=PAIR_PROMPT_VERSION,
        document_sha256=document.sha256,
        budget=budget,
        candidates_total=len(candidates),
        checked=tuple(selected),
        skipped=skipped,
        covered=tuple(sorted(covered, key=order.__getitem__)),
        hypotheses=hypotheses,
        rejected=tuple(rejected),
    )


def _to_finding(
    hypothesis: RelationHypothesis,
    first: Requirement,
    second: Requirement,
    document: Document,
    order: dict[str, int],
) -> Finding | list[str]:
    """Гипотеза модели → черновик связи с цитатами из обоих требований пары."""
    pair = {first.id: first, second.id: second}
    if hypothesis.source_id not in pair or hypothesis.target_id not in pair:
        return [
            f"Ответ относится к паре {hypothesis.source_id} и {hypothesis.target_id}, а "
            f"проверялась пара {first.id} и {second.id}: ID требований по ответу модели "
            "не меняются."
        ]
    if hypothesis.source_id == hypothesis.target_id:
        return ["Связь требования с самим собой не допускается."]
    ends = [
        (pair[hypothesis.source_id], hypothesis.source_quote),
        (pair[hypothesis.target_id], hypothesis.target_quote),
    ]
    if hypothesis.relation in SYMMETRIC_RELATIONS and order[ends[0][0].id] > order[ends[1][0].id]:
        ends.reverse()  # направление симметричной связи не важно: храним в порядке документа
    evidence: list[Evidence] = []
    reasons: list[str] = []
    for fragment, quote in ends:
        located = locate_quote(fragment, quote, document)
        if isinstance(located, Evidence):
            evidence.append(located)
        else:
            reasons += [f"Цитата из {fragment.id}: {reason}" for reason in located]
    if reasons:
        return reasons
    (source, _), (target, _) = ends
    category, criterion, severity = RELATION_MAPPING[hypothesis.relation]
    return Finding(
        id=(
            f"{RULE_ID}:{source.id}@{evidence[0].start}+{target.id}@{evidence[1].start}:"
            f"{hypothesis.relation}"
        ),
        rule_id=RULE_ID,
        rule_version=f"prompt-{PAIR_PROMPT_VERSION}",
        severity=severity,
        category=category,
        criterion=criterion,
        title=HYPOTHESIS_TITLES[hypothesis.relation],
        explanation=hypothesis.explanation,
        recommendation=hypothesis.recommendation,
        requirement_ids=(source.id, target.id),
        evidence=tuple(evidence),
        evidence_kind="comparison",
        evidence_status="unverified",
        origin="llm",
        document_sha256=document.sha256,
        confidence=hypothesis.confidence,
        relation=hypothesis.relation,
    )
