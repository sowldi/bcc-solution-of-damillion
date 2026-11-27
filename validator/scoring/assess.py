"""Оценка требований (этап 9): сводит результаты этапов 3–8, ничего не проверяя заново.

Определения (README, раздел «Оценка, сводка и итог»):

- **Подтверждённый дефект** — замечание со статусом `verified` и важностью critical, major
  или minor: замечания правил (раздел 4) и правил этапа 8 CTR-01, ACC-01 (раздел 7).
  CTR-02 — не дефект, а потенциальное противоречие: доказаны цитаты, но не то, что значения
  относятся к одному объекту в одних условиях. Важность info (CON-02) — повод для проверки,
  не дефект. Рёбра Fact Graph — связи, не дефекты.
- **Подтверждённое противоречие** — `Contradiction` со статусом confirmed (CTR-01, CON-03 по
  классификации этапа 8). Замечание CON-03 остаётся подтверждённым замечанием правила, но
  противоречием считается, только если этап 8 его подтвердил.
- **На проверку человеку** — гипотезы LLM-обзора (unverified) и потенциальные противоречия
  (CTR-02, CON-03 с совместимыми значениями, гипотезы модели о противоречии). Они не входят в
  счётчики дефектов и не меняют вердикт; уверенность модели (confidence) не используется.
- **Не проверено** — требования вне графа, без LLM-обзора, без проверенной пары; пары вне
  отбора и исключённые ограничением; неоднозначные фрагменты.

Вердикт (категория, без баллов):

- требований нет → not_assessable;
- подтверждённое противоречие или подтверждённый существенный дефект проверяемости или
  полноты → not_recommended;
- иначе есть подтверждённые дефекты → after_fixes;
- иначе → no_blockers, всегда с оговорками о покрытии.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from validator.findings import (
    RELATION_TYPES,
    REVIEW_CATEGORIES,
    Finding,
    RuleCategory,
    Severity,
    severity_rank,
)
from validator.graph import FactGraph
from validator.passport import Mode
from validator.quality import Contradiction, QualityResult
from validator.scoring.model import (
    Coverage,
    DefectRef,
    LlmReview,
    PairCheck,
    RequirementAssessment,
    RequirementStatus,
    ReviewRef,
    Scorecard,
    Verdict,
)
from validator.segmentation import Category, Segmentation

if TYPE_CHECKING:  # оценка не загружает LLM-слой: в режиме --rules-only его нет
    from validator.lint import LintResult
    from validator.llm import CrossCheckAnalysis, LLMAnalysis

DEFECT_SEVERITIES: frozenset[Severity] = frozenset({"critical", "major", "minor"})
MAJOR_SEVERITIES: frozenset[Severity] = frozenset({"critical", "major"})
# Подтверждены цитаты, но не противоречие: в счётчики дефектов не входят.
POTENTIAL_RULES = frozenset({"CTR-02"})
# Критичные дефекты по ТЗ: непроверяемые и неполные формулировки (и подтверждённые
# противоречия). Существенный дефект этих категорий блокирует начало разработки.
CRITICAL_CATEGORIES: frozenset[RuleCategory] = frozenset({"testability", "completeness"})
CATEGORY_PRIORITY: tuple[RuleCategory, ...] = (
    "testability",
    "completeness",
    "consistency",
    "ambiguity",
    "traceability",
)
SKIP_ORDER = ("budget", "empty", "no_source")
SKIP_TITLES = {
    "budget": "лимит LLM_MAX_REQUESTS",
    "empty": "пустой текст",
    "no_source": "нет привязки к источнику",
}
POTENTIAL_TITLES = {
    "CTR-02": "Возможное противоречие: одноимённый параметр в разных контекстах",
    "CON-03": "Разные значения параметра в таблице: значения совместимы, противоречие не "
    "подтверждено",
    "LLM-PAIR": "Возможное противоречие по выводу модели",
    "CTR-01": "Возможное противоречие",
}
POTENTIAL_SEVERITY: dict[str, Severity] = {
    "CTR-02": "minor",
    "CON-03": "major",
    "LLM-PAIR": "major",
    "CTR-01": "major",
}
_LIST_IN_REASON = 5


def assess(
    segmentation: Segmentation,
    lint: LintResult,
    graph: FactGraph,
    quality: QualityResult,
    *,
    analysis: LLMAnalysis | None = None,
    crosscheck: CrossCheckAnalysis | None = None,
    mode: Mode = "standard",
) -> Scorecard:
    sha = segmentation.document_sha256
    if not lint.document_sha256 == graph.document_sha256 == quality.document_sha256 == sha:
        raise ValueError("Результаты этапов построены для разных версий документа.")
    order = {fragment.id: index for index, fragment in enumerate(segmentation.fragments)}
    atomic = segmentation.atomic_requirements
    atomic_ids = {fragment.id for fragment in atomic}
    confirmed_ids = {item.finding_id for item in quality.confirmed}
    findings = [*lint.findings, *quality.findings]

    def position(ids: Sequence[str], start: int = 0) -> tuple[int, int]:
        return min((order.get(rid, len(order)) for rid in ids), default=len(order)), start

    defects = sorted(
        (
            _defect(f, f.id in confirmed_ids)
            for f in findings
            if f.severity in DEFECT_SEVERITIES and f.rule_id not in POTENTIAL_RULES
        ),
        key=lambda d: (
            not d.contradiction,
            severity_rank(d.severity),
            _category_rank(d.category),
            -len(d.requirement_ids),
            position(d.requirement_ids, _start(findings, d.finding_id)),
            d.finding_id,
        ),
    )
    review = sorted(
        [_potential(item) for item in quality.potential]
        + [
            ReviewRef(f.id, "llm-review", f.title, f.requirement_ids, f.severity)
            for f in (analysis.hypotheses if analysis else ())
        ],
        key=lambda r: (
            r.source != "potential-contradiction",
            severity_rank(r.severity),
            position(r.requirement_ids),
            r.item_id,
        ),
    )
    excluded = {item.requirement_id for item in graph.excluded}
    candidates = {rid for p in graph.candidates.pairs for rid in (p.source_id, p.target_id)}
    requirements = tuple(
        _requirement(
            fragment.id,
            fragment.doc_id,
            fragment.category,
            defects,
            quality,
            analysis,
            crosscheck,
            excluded,
            candidates,
        )
        for fragment in atomic
    )
    coverage = _coverage(segmentation, graph, analysis, crosscheck)
    verdict, reasons = _verdict(defects, coverage)
    return Scorecard(
        document_sha256=sha,
        mode=mode,
        verdict=verdict,
        verdict_reasons=reasons,
        caveats=_caveats(mode, coverage, graph, quality, analysis),
        requirements=requirements,
        ambiguous=len(segmentation.ambiguous),
        defects=tuple(defects),
        review=tuple(review),
        critical=tuple(
            d.finding_id for d in defects if d.contradiction or d.category in CRITICAL_CATEGORIES
        ),
        structural=tuple(d.finding_id for d in defects if not set(d.requirement_ids) & atomic_ids),
        info=tuple(f.id for f in findings if f.severity == "info"),
        rejected=(
            ("rules", len(lint.rejected)),
            ("stage8-rules", len(quality.rejected)),
            ("graph-rules", len(graph.rejected)),
            ("llm-review", len(analysis.rejected) if analysis else 0),
            ("llm-pairs", len(crosscheck.rejected) if crosscheck else 0),
        ),
        relation_hypotheses=_counts(
            RELATION_TYPES, (f.relation for f in (crosscheck.hypotheses if crosscheck else ()))
        ),
        review_categories=_counts(
            REVIEW_CATEGORIES,
            (f.review_category for f in (analysis.hypotheses if analysis else ())),
        ),
        contradictions_confirmed=len(quality.confirmed),
        contradictions_potential=len(quality.potential),
        acceptance=len(quality.acceptance),
        acceptance_scenarios=sum(1 for f in quality.findings if f.rule_id == "ACC-01"),
        acceptance_links=len(quality.acceptance_links),
        coverage=coverage,
    )


def _defect(finding: Finding, contradiction: bool) -> DefectRef:
    return DefectRef(
        finding.id,
        finding.rule_id,
        finding.title,
        finding.severity,
        finding.category,
        finding.criterion,
        finding.requirement_ids,
        contradiction,
    )


def _start(findings: Sequence[Finding], finding_id: str) -> int:
    return next(f.evidence[0].start for f in findings if f.id == finding_id and f.evidence)


def _category_rank(category: RuleCategory) -> int:
    if category in CATEGORY_PRIORITY:
        return CATEGORY_PRIORITY.index(category)
    return len(CATEGORY_PRIORITY)


def _potential(item: Contradiction) -> ReviewRef:
    return ReviewRef(
        item.finding_id,
        "potential-contradiction",
        POTENTIAL_TITLES[item.source],
        item.requirement_ids,
        POTENTIAL_SEVERITY[item.source],
    )


def _counts(keys: Iterable[str], values: Iterable[str | None]) -> tuple[tuple[str, int], ...]:
    counter = Counter(value for value in values if value is not None)
    return tuple((key, counter[key]) for key in keys if counter[key])


def _requirement(
    rid: str,
    doc_id: str | None,
    category: Category,
    defects: Sequence[DefectRef],
    quality: QualityResult,
    analysis: LLMAnalysis | None,
    crosscheck: CrossCheckAnalysis | None,
    excluded: set[str],
    candidates: set[str],
) -> RequirementAssessment:
    mine = [d for d in defects if rid in d.requirement_ids]
    major = tuple(d.finding_id for d in mine if d.severity in MAJOR_SEVERITIES)
    minor = tuple(d.finding_id for d in mine if d.severity == "minor")
    confirmed = tuple(c.finding_id for c in quality.confirmed if rid in c.requirement_ids)
    potential = tuple(c.finding_id for c in quality.potential if rid in c.requirement_ids)
    hypotheses = tuple(
        f.id for f in (analysis.hypotheses if analysis else ()) if rid in f.requirement_ids
    )
    llm: LlmReview = "not_used"
    if analysis is not None:
        skipped = {item.requirement_id: item.code for item in analysis.skipped}
        llm = "reviewed" if rid in analysis.reviewed else skipped.get(rid, "budget")
    pairs: PairCheck
    if rid in excluded:
        pairs = "excluded"
    elif crosscheck is None:
        pairs = "not_used"
    elif crosscheck.disabled:
        pairs = "disabled"
    elif rid in crosscheck.covered:
        pairs = "checked"
    else:
        pairs = "not_checked" if rid in candidates else "no_candidate"
    status: RequirementStatus
    if major or confirmed:
        status = "confirmed_major"
    elif minor:
        status = "confirmed_minor"
    elif hypotheses or potential:
        status = "needs_review"
    elif rid in excluded:
        status = "unchecked"
    elif llm == "reviewed":
        status = "clean_full"
    else:
        status = "clean_rules_only"
    return RequirementAssessment(
        requirement_id=rid,
        doc_id=doc_id,
        category=category,
        status=status,
        confirmed_major=major,
        confirmed_minor=minor,
        contradictions_confirmed=confirmed,
        contradictions_potential=potential,
        hypotheses=hypotheses,
        llm_review=llm,
        pair_check=pairs,
    )


def _coverage(
    segmentation: Segmentation,
    graph: FactGraph,
    analysis: LLMAnalysis | None,
    crosscheck: CrossCheckAnalysis | None,
) -> Coverage:
    skipped: Counter[str] = Counter(item.code for item in (analysis.skipped if analysis else ()))
    return Coverage(
        requirements=len(segmentation.atomic_requirements),
        ambiguous=len(segmentation.ambiguous),
        excluded=len(graph.excluded),
        llm_used=analysis is not None,
        reviewed=len(analysis.reviewed) if analysis else 0,
        skipped=tuple((code, skipped[code]) for code in SKIP_ORDER if skipped[code]),
        pairs_disabled=crosscheck is not None and crosscheck.disabled,
        pairs_candidates=len(graph.candidates.pairs),
        pairs_checked=crosscheck.requests if crosscheck else 0,
        truncated_pairs=graph.candidates.excluded_by_limit,
        outside_pairs=graph.candidates.outside_pairs,
    )


def _listed(ids: Sequence[str]) -> str:
    shown = ", ".join(ids[:_LIST_IN_REASON])
    rest = len(ids) - _LIST_IN_REASON
    return shown + (f" и ещё {rest}" if rest > 0 else "")


def _verdict(defects: Sequence[DefectRef], coverage: Coverage) -> tuple[Verdict, tuple[str, ...]]:
    if coverage.requirements == 0:
        reasons = ["В документе не выделено ни одного атомарного требования."]
        if coverage.ambiguous:
            reasons.append(
                f"Неоднозначных фрагментов: {coverage.ambiguous} — сначала нужно подтвердить, "
                "что это требования."
            )
        return "not_assessable", tuple(reasons)
    contradictions = [d.finding_id for d in defects if d.contradiction]
    blocking = [
        d.finding_id
        for d in defects
        if not d.contradiction
        and d.severity in MAJOR_SEVERITIES
        and d.category in CRITICAL_CATEGORIES
    ]
    if contradictions or blocking:
        reasons = []
        if contradictions:
            reasons.append(
                f"Подтверждённых противоречий: {len(contradictions)} ({_listed(contradictions)})."
            )
        if blocking:
            reasons.append(
                "Подтверждённых существенных дефектов проверяемости и полноты: "
                f"{len(blocking)} ({_listed(blocking)})."
            )
        return "not_recommended", tuple(reasons)
    if defects:
        major = sum(1 for d in defects if d.severity in MAJOR_SEVERITIES)
        return "after_fixes", (
            f"Подтверждённых дефектов: {len(defects)}, из них существенных: {major}.",
            "Подтверждённых противоречий и существенных дефектов проверяемости и полноты "
            "не найдено.",
        )
    return "no_blockers", ("Подтверждённых дефектов не найдено.",)


def _caveats(
    mode: Mode,
    coverage: Coverage,
    graph: FactGraph,
    quality: QualityResult,
    analysis: LLMAnalysis | None,
) -> tuple[str, ...]:
    notes: list[str] = []
    if not coverage.llm_used:
        reason = " (режим --rules-only)" if mode == "rules-only" else ""
        notes.append(
            f"Модель не использовалась{reason}: смысловые дефекты требований и связи между "
            "ними не проверялись."
        )
    elif coverage.reviewed < coverage.requirements:
        skipped = ", ".join(f"{SKIP_TITLES[code]} — {count}" for code, count in coverage.skipped)
        notes.append(
            f"Модель проверила требований: {coverage.reviewed} из {coverage.requirements}"
            + (f"; не проверено: {skipped}." if skipped else ".")
        )
    if not coverage.llm_used and coverage.pairs_candidates:
        notes.append(
            f"Пар-кандидатов для проверки моделью: {coverage.pairs_candidates} — не проверялись."
        )
    if coverage.pairs_disabled:
        notes.append(
            "Проверка пар моделью отключена (LLM_MAX_PAIR_REQUESTS = 0): связи и противоречия "
            "между требованиями моделью не проверялись."
        )
    elif coverage.llm_used and coverage.pairs_checked < coverage.pairs_candidates:
        notes.append(
            f"Модель проверила пар-кандидатов: {coverage.pairs_checked} из "
            f"{coverage.pairs_candidates}."
        )
    if coverage.truncated_pairs:
        notes.append(
            f"Ограничение отбора исключило пар-кандидатов: {coverage.truncated_pairs}; моделью "
            "они не проверялись."
        )
    if coverage.outside_pairs:
        notes.append(
            f"Пар вне отбора (общих значимых слов меньше {graph.candidates.min_shared}): "
            f"{coverage.outside_pairs} — моделью не проверялись."
        )
    if quality.potential:
        notes.append(
            f"Потенциальных противоречий: {len(quality.potential)} — требуют проверки человеком."
        )
    if analysis is not None and analysis.hypotheses:
        notes.append(
            f"Гипотез модели о дефектах: {len(analysis.hypotheses)} — требуют проверки человеком."
        )
    if coverage.ambiguous:
        notes.append(
            f"Неоднозначных фрагментов: {coverage.ambiguous} — правилами формулировок и моделью "
            "не проверялись, пока не подтверждено, что это требования."
        )
    if coverage.excluded:
        notes.append(
            f"Требований вне графа (пустые или без привязки к источнику): {coverage.excluded}."
        )
    notes.append(
        "Итог основан только на подтверждённых замечаниях: гипотезы и непроверенные области его "
        "не меняют, но и не позволяют утверждать, что других дефектов в документе нет."
    )
    return tuple(notes)
