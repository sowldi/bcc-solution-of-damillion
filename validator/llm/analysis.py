"""LLM-анализ требований: модель предлагает гипотезы, система их не подтверждает.

1. Атомарные требования отправляются пакетами (не больше batch_size требований и batch_chars
   символов в запросе, не больше max_requests запросов за прогон — стоимость ограничена
   и видна в отчёте).
2. Ответ проверяется строгой схемой (validator.llm.schema).
3. Гипотеза становится замечанием Finding(origin="llm") только если её цитата дословно
   найдена в тексте требования, которое было отправлено модели, и проходит базовую проверку
   доказательств (check_finding). Статус остаётся «unverified»: правильный JSON и найденная
   цитата не доказывают сам дефект. Подтверждение гипотез — задача Evidence Gate этапа 5.
4. Ошибки провайдера не скрываются: они поднимаются к вызывающему коду.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from validator.config import Settings
from validator.evidence import build_evidence, check_finding
from validator.findings import Finding
from validator.llm.base import LLMProvider, LLMRequest
from validator.llm.errors import ProviderResponseError
from validator.llm.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_message
from validator.llm.retry import Clock, RateLimiter, Sleep, complete_with_retry
from validator.llm.schema import Hypothesis, RejectedHypothesis, parse_hypotheses
from validator.model import Document
from validator.segmentation import Requirement, Segmentation

log = logging.getLogger("validator.llm")

SCHEMA = "validator.llm-analysis/1"
RULE_ID = "LLM"


@dataclass(frozen=True)
class LLMOptions:
    max_output_tokens: int = 4096
    temperature: float = 0.0
    max_retries: int = 4
    max_requests: int = 10
    rpm: int | None = None  # None — без ограничения частоты
    batch_size: int = 20
    batch_chars: int = 8000

    @classmethod
    def from_settings(cls, settings: Settings) -> LLMOptions:
        return cls(
            max_output_tokens=settings.max_output_tokens,
            temperature=settings.temperature,
            max_retries=settings.max_retries,
            max_requests=settings.max_requests,
            rpm=settings.rpm,
        )


@dataclass(frozen=True)
class LLMAnalysis:
    """Итог LLM-анализа. hypotheses — только неподтверждённые замечания модели."""

    provider: str
    model: str
    prompt_version: str
    requests: int
    requirements_sent: int
    requirements_total: int
    hypotheses: tuple[Finding, ...]
    rejected: tuple[RejectedHypothesis, ...] = ()

    def __post_init__(self) -> None:
        for finding in self.hypotheses:
            if finding.origin != "llm" or finding.evidence_status != "unverified":
                raise ValueError(
                    f"Гипотеза LLM {finding.id} должна иметь origin=llm и статус unverified: "
                    "подтверждает замечания только проверка доказательств."
                )

    @property
    def truncated(self) -> bool:
        """Не все требования отправлены модели (сработал лимит LLM_MAX_REQUESTS)."""
        return self.requirements_sent < self.requirements_total

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "requests": self.requests,
            "requirements_sent": self.requirements_sent,
            "requirements_total": self.requirements_total,
            "hypotheses": [finding.to_dict() for finding in self.hypotheses],
            "rejected": [item.to_dict() for item in self.rejected],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMAnalysis:
        if data.get("schema") != SCHEMA:
            raise ValueError(f"Неизвестная схема LLM-анализа: {data.get('schema')!r}")
        texts = [data.get(key) for key in ("provider", "model", "prompt_version")]
        numbers = [data.get(key) for key in ("requests", "requirements_sent", "requirements_total")]
        hypotheses, rejected = data.get("hypotheses"), data.get("rejected")
        if (
            not all(isinstance(value, str) for value in texts)
            or not all(isinstance(n, int) and not isinstance(n, bool) for n in numbers)
            or not isinstance(hypotheses, list)
            or not isinstance(rejected, list)
        ):
            raise ValueError("LLM-анализ: неверные типы полей")
        return cls(
            str(texts[0]),
            str(texts[1]),
            str(texts[2]),
            int(numbers[0]),  # type: ignore[arg-type]
            int(numbers[1]),  # type: ignore[arg-type]
            int(numbers[2]),  # type: ignore[arg-type]
            tuple(Finding.from_dict(item) for item in hypotheses),
            tuple(RejectedHypothesis.from_dict(item) for item in rejected),
        )

    @classmethod
    def from_json(cls, text: str) -> LLMAnalysis:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("JSON LLM-анализа должен быть объектом")
        return cls.from_dict(data)


def analyze_with_llm(
    document: Document,
    segmentation: Segmentation,
    provider: LLMProvider,
    options: LLMOptions,
    *,
    sleep: Sleep = time.sleep,
    clock: Clock = time.monotonic,
) -> LLMAnalysis:
    targets = segmentation.atomic_requirements
    batches = _batches(targets, options)[: options.max_requests]
    limiter = RateLimiter(options.rpm, clock=clock, sleep=sleep) if options.rpm else None
    hypotheses: dict[str, Finding] = {}
    rejected: list[RejectedHypothesis] = []
    offset = 0
    for batch in batches:
        request = LLMRequest(
            system=SYSTEM_PROMPT,
            user=build_user_message(batch),
            max_output_tokens=options.max_output_tokens,
            temperature=options.temperature,
        )
        response = complete_with_retry(
            provider, request, max_retries=options.max_retries, sleep=sleep, limiter=limiter
        )
        try:
            parsed = parse_hypotheses(response.text)
        except ProviderResponseError as error:
            raise ProviderResponseError(f"{provider.name}: {error}") from None
        sent = {fragment.id: fragment for fragment in batch}
        rejected += [
            RejectedHypothesis(offset + item.index, item.requirement_id, item.reasons)
            for item in parsed.rejected
        ]
        for hypothesis in parsed.hypotheses:
            index = offset + hypothesis.index
            result = _to_finding(hypothesis, sent, document, segmentation)
            if isinstance(result, Finding) and result.id in hypotheses:
                result = [f"Гипотеза — повтор уже принятой {result.id}."]
            if isinstance(result, Finding):
                hypotheses[result.id] = result
            else:
                rejected.append(RejectedHypothesis(index, hypothesis.requirement_id, tuple(result)))
        offset += len(parsed.hypotheses) + len(parsed.rejected)
    sent_count = sum(len(batch) for batch in batches)
    log.info(
        "LLM %s/%s: запросов %d, требований %d из %d, гипотез %d, отклонено %d",
        provider.name,
        provider.model,
        len(batches),
        sent_count,
        len(targets),
        len(hypotheses),
        len(rejected),
    )
    rejected.sort(key=lambda item: item.index)
    return LLMAnalysis(
        provider=provider.name,
        model=provider.model,
        prompt_version=PROMPT_VERSION,
        requests=len(batches),
        requirements_sent=sent_count,
        requirements_total=len(targets),
        hypotheses=tuple(hypotheses.values()),
        rejected=tuple(rejected),
    )


def _batches(targets: Sequence[Requirement], options: LLMOptions) -> list[list[Requirement]]:
    batches: list[list[Requirement]] = []
    current: list[Requirement] = []
    size = 0
    for fragment in targets:
        if current and (
            len(current) >= options.batch_size or size + len(fragment.text) > options.batch_chars
        ):
            batches.append(current)
            current, size = [], 0
        current.append(fragment)
        size += len(fragment.text)
    if current:
        batches.append(current)
    return batches


def _to_finding(
    hypothesis: Hypothesis,
    sent: dict[str, Requirement],
    document: Document,
    segmentation: Segmentation,
) -> Finding | list[str]:
    fragment = sent.get(hypothesis.requirement_id)
    if fragment is None:
        return [
            f"Требование {hypothesis.requirement_id} не передавалось модели в этом запросе "
            "(неизвестный ID)."
        ]
    start = fragment.text.find(hypothesis.quote)
    if start < 0:
        return ["Гипотеза без доказательства: цитата не найдена дословно в тексте требования."]
    try:
        evidence = build_evidence(fragment, start, start + len(hypothesis.quote), document)
    except ValueError as error:
        return [str(error)]
    finding = Finding(
        id=f"{RULE_ID}:{fragment.id}@{start}",
        rule_id=RULE_ID,
        rule_version=f"prompt-{PROMPT_VERSION}",
        severity=hypothesis.severity,
        category=hypothesis.category,
        criterion=hypothesis.criterion,
        title=hypothesis.title,
        explanation=hypothesis.explanation,
        recommendation=hypothesis.recommendation,
        requirement_ids=(fragment.id,),
        evidence=(evidence,),
        evidence_kind="quote",
        evidence_status="unverified",
        origin="llm",
    )
    problems = check_finding(finding, segmentation, document)
    return list(problems) if problems else finding
