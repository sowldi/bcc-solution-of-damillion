"""Проверка пар требований моделью (этап 7): только пары-кандидаты Fact Graph, по запросу
на пару, тот же провайдер, бюджет и устойчивый вызов, что у обзора этапа 6.

Гипотеза связи принимается в работу, только если относится именно к проверяемой паре и обе
цитаты точно найдены в своих требованиях; затем Evidence Gate. Статус — всегда
«unverified»: цитаты доказывают происхождение текста, но не саму связь.
Модель подменена FakeProvider — сети и ключей нет.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tests.graph_helpers import GRAPH_SPEC, Built, load, relation, relations_json
from validator.evidence import EvidenceGateError
from validator.graph import CandidatePair
from validator.llm import LLMOptions, RateLimiter, review_requirements
from validator.llm.crosscheck import RELATION_MAPPING, CrossCheckAnalysis, crosscheck_pairs
from validator.llm.errors import (
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
)
from validator.llm.fake import FakeProvider
from validator.llm.prompts import PAIR_PROMPT_VERSION, PAIR_SYSTEM_PROMPT
from validator.llm.schema import parse_relations

OPTIONS = LLMOptions(max_retries=0, max_pair_requests=10)
EMPTY = relations_json()
CONFLICT = relation(
    "Ф-3", "не хранит бронь книги дольше 3 дней", "Ф-4", "хранит бронь книги 5 дней"
)

Checked = tuple[CrossCheckAnalysis, FakeProvider, Built]


def _check(
    tmp_path: Path,
    *replies: str | Exception,
    options: LLMOptions = OPTIONS,
    markdown: str = GRAPH_SPEC,
    sleeps: list[float] | None = None,
    built: Built | None = None,
) -> Checked:
    built = built or load(tmp_path, markdown)
    provider = FakeProvider(*replies)
    pause = sleeps.append if sleeps is not None else (lambda _: None)
    analysis = crosscheck_pairs(
        built.document, built.segmentation, built.graph, provider, options, sleep=pause
    )
    return analysis, provider, built


def _payload(user: str) -> dict[str, Any]:
    data = json.loads(user[user.index("{") :])
    assert isinstance(data, dict)
    return data


def _reasons(analysis: CrossCheckAnalysis) -> str:
    return " | ".join(" ".join(item.reasons) for item in analysis.rejected)


def _candidates(built: Built) -> list[tuple[str, str]]:
    return [(pair.source_id, pair.target_id) for pair in built.graph.candidates.pairs]


# --- запросы ------------------------------------------------------------------------------


def test_each_candidate_pair_is_sent_once_with_both_requirements(tmp_path: Path) -> None:
    analysis, provider, built = _check(tmp_path)

    candidates = _candidates(built)
    assert analysis.checked == tuple(candidates)
    assert len(provider.requests) == len(candidates) == analysis.requests
    first = provider.requests[0]
    assert first.system == PAIR_SYSTEM_PROMPT
    data = _payload(first.user)
    ids = [item["id"] for item in data["requirements"]]
    assert ids == ["Ф-3", "Ф-4"]
    texts = [item["text"] for item in data["requirements"]]
    assert texts == [built.segmentation.by_id(rid).text for rid in ids]
    assert all(item["section"] and item["source"] for item in data["requirements"])
    assert "недоверенные данные" in first.user


def test_prompt_separates_relations_from_coincidences() -> None:
    prompt = PAIR_SYSTEM_PROMPT.casefold()

    for phrase in ("разные сценарии", "отрицани", "модальн", "одинаковые числа", "цитат"):
        assert phrase in prompt
    assert PAIR_PROMPT_VERSION == "pair-1"


# --- гипотезы -------------------------------------------------------------------------------


def test_candidate_conflict_becomes_an_unverified_hypothesis(tmp_path: Path) -> None:
    analysis, _, _ = _check(tmp_path, relations_json(CONFLICT))

    [finding] = analysis.hypotheses
    assert finding.relation == "conflicts_with"
    assert finding.requirement_ids == ("Ф-3", "Ф-4")
    assert (finding.origin, finding.evidence_status) == ("llm", "unverified")
    assert [e.quote for e in finding.evidence] == [
        CONFLICT["source_quote"],
        CONFLICT["target_quote"],
    ]
    assert finding.title == "Возможное противоречие"
    assert (finding.category, finding.criterion, finding.severity) == RELATION_MAPPING[
        "conflicts_with"
    ]
    assert finding.confidence == 0.8
    assert analysis.rejected == ()


def test_reverse_direction_of_symmetric_relation_is_one_hypothesis(tmp_path: Path) -> None:
    reverse = relation(
        "Ф-4", "хранит бронь книги 5 дней", "Ф-3", "не хранит бронь книги дольше 3 дней"
    )

    analysis, _, _ = _check(tmp_path, relations_json(reverse, CONFLICT))

    [finding] = analysis.hypotheses
    assert finding.requirement_ids == ("Ф-3", "Ф-4")  # порядок документа
    assert "повтор" in _reasons(analysis)


def test_directional_relation_keeps_its_direction(tmp_path: Path) -> None:
    condition = relation(
        "Ф-4",
        "после подтверждения",
        "Ф-1",
        "подтверждает бронь книги",
        relation="requires_condition",
    )

    analysis, _, built = _check(tmp_path, EMPTY, relations_json(condition))

    assert _candidates(built)[1] == ("Ф-1", "Ф-4")
    [finding] = analysis.hypotheses
    assert finding.requirement_ids == ("Ф-4", "Ф-1")
    assert finding.relation == "requires_condition"
    assert finding.title == "Возможное условие выполнения"


def test_quote_from_the_other_requirement_is_rejected(tmp_path: Path) -> None:
    foreign = relation("Ф-3", "хранит бронь книги 5 дней", "Ф-4", "хранит бронь книги 5 дней")

    analysis, _, _ = _check(tmp_path, relations_json(foreign))

    assert analysis.hypotheses == ()
    assert "Цитата из Ф-3" in _reasons(analysis)
    assert "не найдена" in _reasons(analysis)


@pytest.mark.parametrize(
    ("source_id", "target_id"), [("Ф-9", "Ф-4"), ("Ф-1", "Ф-4"), ("Ф-3", "Ф-3/1")]
)
def test_ids_outside_the_checked_pair_are_rejected(
    tmp_path: Path, source_id: str, target_id: str
) -> None:
    item = dict(CONFLICT, source_id=source_id, target_id=target_id)

    analysis, _, _ = _check(tmp_path, relations_json(item))

    assert analysis.hypotheses == ()
    assert "проверялась пара Ф-3 и Ф-4" in _reasons(analysis)


def test_relation_of_a_requirement_with_itself_is_rejected(tmp_path: Path) -> None:
    item = relation("Ф-3", "не хранит бронь", "Ф-3", "дольше 3 дней")

    analysis, _, _ = _check(tmp_path, relations_json(item))

    assert analysis.hypotheses == ()
    assert "с самим собой" in _reasons(analysis)


def test_quote_that_drops_a_negation_is_rejected(tmp_path: Path) -> None:
    item = relation("Ф-3", "хранит бронь книги дольше 3 дней", "Ф-4", "хранит бронь книги 5 дней")

    analysis, _, _ = _check(tmp_path, relations_json(item))

    assert analysis.hypotheses == ()
    assert "отрицание" in _reasons(analysis)


def test_quote_of_only_the_requirement_id_is_rejected(tmp_path: Path) -> None:
    item = relation("Ф-3", "Система не хранит", "Ф-4", "Ф-4")

    analysis, _, _ = _check(tmp_path, relations_json(item))

    assert analysis.hypotheses == ()
    assert "Цитата из Ф-4" in _reasons(analysis)
    assert "служебная" in _reasons(analysis)


def test_invalid_item_is_rejected_by_schema(tmp_path: Path) -> None:
    item = dict(CONFLICT, relation="contradicts", status="verified")

    analysis, _, _ = _check(tmp_path, relations_json(item))

    assert analysis.hypotheses == ()
    reasons = _reasons(analysis)
    assert "relation" in reasons and "status" in reasons


def test_relation_already_verified_by_a_rule_is_not_repeated(tmp_path: Path) -> None:
    built = load(tmp_path)
    linked = CandidatePair("Ф-1", "Ф-2", 2, ("брон", "книг"), True)
    graph = replace(built.graph, candidates=replace(built.graph.candidates, pairs=(linked,)))
    dependency = relation(
        "Ф-2", "Ф-1", "Ф-1", "Библиотекарь подтверждает бронь", relation="depends_on"
    )

    analysis, _, _ = _check(tmp_path, relations_json(dependency), built=replace(built, graph=graph))

    assert analysis.hypotheses == ()
    assert "уже подтверждена правилом XRF-01" in _reasons(analysis)


def test_hypotheses_join_the_graph_as_unverified_edges(tmp_path: Path) -> None:
    analysis, _, built = _check(tmp_path, relations_json(CONFLICT))

    graph = built.graph.with_hypotheses(analysis.hypotheses)

    [edge] = [edge for edge in graph.edges if edge.origin == "llm"]
    assert (edge.source_id, edge.target_id, edge.status) == ("Ф-3", "Ф-4", "unverified")
    assert edge.relation == "conflicts_with"


# --- бюджет, покрытие, сбои ------------------------------------------------------------------


def test_budget_limits_pairs_and_reports_coverage(tmp_path: Path) -> None:
    options = replace(OPTIONS, max_pair_requests=2)

    analysis, provider, built = _check(tmp_path, options=options)

    candidates = _candidates(built)
    assert len(provider.requests) == 2
    assert analysis.checked == tuple(candidates[:2])
    assert analysis.skipped == tuple(candidates[2:])
    assert analysis.truncated
    assert analysis.covered == ("Ф-1", "Ф-3", "Ф-4")


def test_zero_budget_disables_the_pair_check(tmp_path: Path) -> None:
    analysis, provider, built = _check(tmp_path, options=replace(OPTIONS, max_pair_requests=0))

    assert provider.requests == []
    assert analysis.disabled
    assert analysis.checked == ()
    assert analysis.skipped == tuple(_candidates(built))


def test_empty_graph_sends_nothing(tmp_path: Path) -> None:
    analysis, provider, _ = _check(tmp_path, markdown="# Сервис\n\nОбщее описание.\n")

    assert provider.requests == []
    assert (analysis.checked, analysis.skipped, analysis.hypotheses) == ((), (), ())


def test_provider_failure_names_the_pair_and_progress(tmp_path: Path) -> None:
    with pytest.raises(ProviderServerError) as caught:
        _check(tmp_path, EMPTY, ProviderServerError("HTTP 503"))

    message = str(caught.value)
    assert "Проверка пар прервана на паре Ф-1 — Ф-4" in message
    assert "сбой на стороне провайдера" in message
    assert "проверено 1 из" in message
    assert "частичные результаты не сохраняются" in message


def test_timeout_on_the_first_pair_is_reported_without_progress(tmp_path: Path) -> None:
    with pytest.raises(ProviderTimeoutError) as caught:
        _check(tmp_path, ProviderTimeoutError("нет ответа за 120 с"))

    assert "Тип сбоя: таймаут" in str(caught.value)
    assert "проверено 0 из" in str(caught.value)


def test_graph_of_another_document_version_is_refused(tmp_path: Path) -> None:
    old = load(tmp_path)
    changed = load(tmp_path, GRAPH_SPEC + "* Ф-9. Читатель оценивает книгу.\n")

    with pytest.raises(EvidenceGateError, match="другой версии документа"):
        crosscheck_pairs(changed.document, changed.segmentation, old.graph, FakeProvider(), OPTIONS)


@pytest.mark.parametrize(
    "reply", ["", "не JSON", '{"findings": []}', '{"relations": {}}', '{"relations": [], "x": 1}']
)
def test_empty_or_malformed_answer_stops_the_check(tmp_path: Path, reply: str) -> None:
    with pytest.raises(ProviderResponseError, match="Проверка пар прервана на паре Ф-3 — Ф-4"):
        _check(tmp_path, reply)


def test_rate_limit_waits_for_retry_after(tmp_path: Path) -> None:
    sleeps: list[float] = []
    options = replace(OPTIONS, max_retries=1, max_pair_requests=1)

    analysis, provider, _ = _check(
        tmp_path,
        ProviderRateLimitError("HTTP 429", retry_after=7.0),
        relations_json(CONFLICT),
        options=options,
        sleeps=sleeps,
    )

    assert sleeps == [7.0]
    assert len(provider.requests) == 2
    assert len(analysis.hypotheses) == 1


def test_review_and_pair_check_share_one_rate_limiter(tmp_path: Path) -> None:
    built = load(tmp_path)
    sleeps: list[float] = []
    limiter = RateLimiter(60, clock=lambda: 0.0, sleep=sleeps.append)  # 1 запрос в секунду
    options = LLMOptions(max_retries=0, max_requests=2, max_pair_requests=2)

    review_requirements(
        built.document, built.segmentation, FakeProvider(), options, limiter=limiter
    )
    crosscheck_pairs(
        built.document, built.segmentation, built.graph, FakeProvider(), options, limiter=limiter
    )

    assert sleeps == [1.0, 1.0, 1.0]  # 4 запроса подряд — 3 паузы, без «сброса» между фазами


def test_repeated_run_gives_the_same_result(tmp_path: Path) -> None:
    first, _, _ = _check(tmp_path, relations_json(CONFLICT))
    second, _, _ = _check(tmp_path, relations_json(CONFLICT))

    assert first.to_json() == second.to_json()


# --- инварианты результата -----------------------------------------------------------------


def test_analysis_holds_only_unverified_relations_of_checked_pairs(tmp_path: Path) -> None:
    analysis, _, _ = _check(tmp_path, relations_json(CONFLICT))
    [finding] = analysis.hypotheses

    for bad, message in [
        (replace(finding, evidence_status="verified"), "unverified"),
        (replace(finding, relation=None), "relation"),
        (replace(finding, document_sha256="0" * 64), "SHA-256"),
    ]:
        with pytest.raises(ValueError, match=message):
            replace(analysis, hypotheses=(bad,))
    with pytest.raises(ValueError, match="не проверялась"):
        replace(analysis, checked=(("Ф-1", "Ф-4"),))


# --- схема ответа ---------------------------------------------------------------------------


def test_schema_accepts_fenced_json_and_rejects_extra_fields() -> None:
    parsed = parse_relations("```json\n" + relations_json(CONFLICT, dict(CONFLICT, x=1)) + "\n```")

    assert [item.source_id for item in parsed.relations] == ["Ф-3"]
    [rejected] = parsed.rejected
    assert rejected.index == 1 and "Недопустимое поле x" in rejected.reasons[0]


def test_schema_limits_the_number_of_relations() -> None:
    with pytest.raises(ProviderResponseError, match="больше допустимых"):
        parse_relations(relations_json(*[CONFLICT] * 5))
