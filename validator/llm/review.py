"""LLM Review per requirement (этап 6): каждое атомарное требование проверяется отдельно.

1. Требования берутся из сегментации (атомарные: утверждения и неделимые фрагменты).
   Не отправляются: пустые, без корректной привязки к источнику и всё сверх бюджета
   LLM_MAX_REQUESTS — с причиной в результате. Одинаковый текст с другим ID проверяется
   отдельно (другой контекст). Каждое требование (ID) отправляется не больше одного раза;
   запросы идут последовательно,
   с ограничением частоты (LLM_RPM) и повтором только временных ошибок (LLM_MAX_RETRIES).
2. Модели передаётся минимум: ID, точный текст, границы источника (SourceRef и место) и
   контекст для понимания — раздел и родительский фрагмент. Межтребовательный анализ
   (противоречия, зависимости) — не задача этого этапа.
3. Ответ проверяется строгой схемой. Гипотеза принимается в работу, только если относится
   именно к проверяемому требованию (ID по ответу модели не меняется), её цитата один раз
   дословно встречается в его тексте и содержательна (не ID и не короткое служебное слово).
   При сбое провайдера обзор прерывается той же ошибкой с указанием требования и прогресса.
4. Все гипотезы проходят Evidence Gate этапа 5. Прошедшие получают статус «unverified»:
   точная цитата доказывает происхождение текста, но не вывод модели. Отклонённые
   сохраняются с причинами. Уверенность модели (confidence) на статус не влияет.
5. Категория модели (ReviewCategory) переводится системой в группу, критерий стандарта и
   важность (REVIEW_MAPPING): модель их не выбирает.
"""

from __future__ import annotations

import logging
import re
import time

from validator.errors import ProviderError
from validator.evidence import build_evidence, gate_findings
from validator.findings import Finding, ReviewCategory, RuleCategory, Severity
from validator.llm.analysis import LLMAnalysis, LLMOptions, SkippedRequirement
from validator.llm.base import LLMProvider, LLMRequest
from validator.llm.errors import ProviderRateLimitError
from validator.llm.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_message
from validator.llm.retry import Clock, RateLimiter, Sleep, complete_with_retry
from validator.llm.schema import Hypothesis, RejectedHypothesis, parse_hypotheses
from validator.model import Document
from validator.segmentation import Requirement, Segmentation, span_sources
from validator.segmentation.verify import body_start

log = logging.getLogger("validator.llm")

RULE_ID = "LLM"
# Содержательная цитата: хотя бы одно слово или число из трёх и более символов.
_MEANINGFUL = re.compile(r"[^\W_]{3,}")

# Категория обзора → (группа замечаний, критерий качества, предполагаемая важность).
REVIEW_MAPPING: dict[ReviewCategory, tuple[RuleCategory, str, Severity]] = {
    "clarity": ("ambiguity", "UNA", "minor"),
    "testability": ("testability", "VER", "major"),
    "completeness": ("completeness", "COM", "major"),
    "atomicity": ("traceability", "SIN", "minor"),
    "undefined_term": ("ambiguity", "S-CMP", "minor"),
    "subjectivity": ("testability", "VER", "major"),
    "missing_condition": ("completeness", "COM", "major"),
}


def review_requirements(
    document: Document,
    segmentation: Segmentation,
    provider: LLMProvider,
    options: LLMOptions,
    *,
    sleep: Sleep = time.sleep,
    clock: Clock = time.monotonic,
) -> LLMAnalysis:
    targets = segmentation.atomic_requirements
    limiter = RateLimiter(options.rpm, clock=clock, sleep=sleep) if options.rpm else None
    reviewed: list[str] = []
    skipped: list[SkippedRequirement] = []
    drafts: list[tuple[int, Finding]] = []
    rejected: list[RejectedHypothesis] = []
    offset = 0
    for fragment in targets:
        problem = _not_reviewable(fragment, document)
        if problem is not None:
            skipped.append(problem)
            continue
        # Одинаковый текст с другим ID — другое требование со своим контекстом (раздел,
        # родитель): оно проверяется отдельно, требования не объединяются (Stage 6.1).
        if len(reviewed) >= options.max_requests:
            skipped.append(
                SkippedRequirement(
                    fragment.id,
                    "budget",
                    f"Исчерпан лимит запросов LLM_MAX_REQUESTS ({options.max_requests}).",
                )
            )
            continue
        reviewed.append(fragment.id)
        parent = _parent(fragment, segmentation)
        request = LLMRequest(
            system=SYSTEM_PROMPT,
            user=build_user_message(fragment, parent),
            max_output_tokens=options.max_output_tokens,
            temperature=options.temperature,
        )
        try:
            response = complete_with_retry(
                provider, request, max_retries=options.max_retries, sleep=sleep, limiter=limiter
            )
            parsed = parse_hypotheses(response.text)
        except ProviderError as error:
            # Частичный сбой: тип ошибки (и код выхода) сохраняется, но сообщение говорит,
            # на каком требовании прервался обзор и сколько требований успели проверить.
            raise _interrupted(
                error,
                f"LLM-обзор прерван на требовании {fragment.id}: до сбоя проверено "
                f"{len(reviewed) - 1} из {len(targets)}; частичные результаты не сохраняются. "
                f"Причина ({provider.name}): {error}",
            ) from None
        rejected += [
            RejectedHypothesis(offset + item.index, item.requirement_id, item.reasons)
            for item in parsed.rejected
        ]
        known = {finding.id for _, finding in drafts}
        for hypothesis in parsed.hypotheses:
            index = offset + hypothesis.index
            result = _to_finding(hypothesis, fragment, document)
            if isinstance(result, Finding) and result.id in known:
                result = [f"Гипотеза — повтор уже принятой {result.id}."]
            if isinstance(result, Finding):
                known.add(result.id)
                drafts.append((index, result))
            else:
                rejected.append(RejectedHypothesis(index, hypothesis.requirement_id, tuple(result)))
        offset += len(parsed.hypotheses) + len(parsed.rejected)
    hypotheses: tuple[Finding, ...] = ()
    if drafts:
        # Тот же Evidence Gate, что и для правил; каталога для LLM нет, поэтому статус
        # прошедших проверку гипотез — «unverified», а не «verified».
        gate = gate_findings([finding for _, finding in drafts], segmentation, document, rules=())
        index_of = {finding.id: index for index, finding in drafts}
        hypotheses = gate.unverified
        rejected += [
            RejectedHypothesis(
                index_of[item.finding.id], item.finding.requirement_ids[0], item.reasons
            )
            for item in gate.rejected
        ]
    rejected.sort(key=lambda item: item.index)
    log.info(
        "LLM-обзор %s/%s: требований %d из %d, пропущено %d, гипотез %d, отклонено %d",
        provider.name,
        provider.model,
        len(reviewed),
        len(targets),
        len(skipped),
        len(hypotheses),
        len(rejected),
    )
    return LLMAnalysis(
        provider=provider.name,
        model=provider.model,
        prompt_version=PROMPT_VERSION,
        requests=len(reviewed),
        requirements_sent=len(reviewed),
        requirements_total=len(targets),
        hypotheses=hypotheses,
        rejected=tuple(rejected),
        document_sha256=document.sha256,
        reviewed=tuple(reviewed),
        skipped=tuple(skipped),
    )


def _interrupted(error: ProviderError, message: str) -> ProviderError:
    """Та же ошибка с контекстом частичного сбоя: тип и код выхода не меняются."""
    if isinstance(error, ProviderRateLimitError):
        return ProviderRateLimitError(message, retry_after=error.retry_after)
    return type(error)(message)


def _not_reviewable(fragment: Requirement, document: Document) -> SkippedRequirement | None:
    """Пустое или некорректно сформированное требование модели не отправляется."""
    if not fragment.text.strip():
        return SkippedRequirement(fragment.id, "empty", "Текст требования пуст.")
    if not fragment.sources:
        return SkippedRequirement(fragment.id, "no_source", "Нет ссылок на источник.")
    try:
        span_sources(fragment, 0, len(fragment.text), document)
    except ValueError as error:
        return SkippedRequirement(fragment.id, "no_source", f"Некорректный источник: {error}")
    return None


def _parent(fragment: Requirement, segmentation: Segmentation) -> Requirement | None:
    if fragment.parent_id is None:
        return None
    try:
        return segmentation.by_id(fragment.parent_id)
    except KeyError:
        return None


def _to_finding(
    hypothesis: Hypothesis, fragment: Requirement, document: Document
) -> Finding | list[str]:
    """Гипотеза → черновик замечания с цитатой из проверяемого требования."""
    if hypothesis.requirement_id != fragment.id:
        return [
            f"Ответ относится к требованию {hypothesis.requirement_id}, а проверялось "
            f"{fragment.id}: ID требования по ответу модели не меняется."
        ]
    quote = hypothesis.evidence_quote
    occurrences = fragment.text.count(quote)
    if occurrences == 0:
        return ["Гипотеза без доказательства: цитата не найдена дословно в тексте требования."]
    if occurrences > 1:
        return [
            f"Цитата встречается в требовании несколько раз ({occurrences}) — место "
            "доказательства неоднозначно."
        ]
    start = fragment.text.index(quote)
    # Цитата должна указывать на содержание требования: ID («Ф-1.»), номер пункта и
    # короткое служебное слово («по», «и») встречаются один раз, но проблему не показывают.
    content = fragment.text[max(start, body_start(fragment, document)) : start + len(quote)]
    if not _MEANINGFUL.search(content):
        return [
            "Цитата слишком короткая или служебная (только ID, номер пункта или короткое "
            "слово) — она не указывает на проблему в тексте требования."
        ]
    try:
        evidence = build_evidence(fragment, start, start + len(quote), document)
    except ValueError as error:
        return [str(error)]
    category, criterion, severity = REVIEW_MAPPING[hypothesis.category]
    return Finding(
        id=f"{RULE_ID}:{fragment.id}@{start}:{hypothesis.category}",
        rule_id=RULE_ID,
        rule_version=f"prompt-{PROMPT_VERSION}",
        severity=severity,
        category=category,
        criterion=criterion,
        title=hypothesis.issue,
        explanation=hypothesis.reasoning,
        recommendation=hypothesis.recommendation,
        requirement_ids=(fragment.id,),
        evidence=(evidence,),
        evidence_kind="quote",
        evidence_status="unverified",
        origin="llm",
        document_sha256=document.sha256,
        review_category=hypothesis.category,
        confidence=hypothesis.confidence,
    )
