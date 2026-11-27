"""Этап 8 без модели: противоречия с машинным контрпримером и критерии приёмки.

Противоречия (CTR-01, CTR-02):
- утверждение «параметр — значение»: строка таблицы «параметр | значение» или фрагмент
  «Параметр: значение» / «Параметр — значение», где значение целиком разбирается как
  числовое ограничение (validator.quality.constraints); всё остальное не сравнивается;
- параметры сравниваются только при одинаковом названии (без учёта регистра, пробелов и
  кавычек): «размер файла» и «размер изображения», «время ответа в рабочее время» и «… в
  нерабочее время» — разные параметры;
- CTR-01 — подтверждённое противоречие: в одном контексте (раздел, группа списка, таблица)
  ограничения одной величины не пересекаются (контрпример: нет значения, удовлетворяющего
  обоим);
- CTR-02 — потенциальное: то же, но в разных контекстах — разделах, таблицах, группах
  списка — или для чисел без единицы (номер, версия): контекст может различаться;
- пары строк одной таблицы — задача CON-03 (этап 3): их этап 8 не дублирует, а
  классифицирует — не пересекаются → подтверждено, совместимы → потенциально.

Критерии приёмки (ACC-01): сценарий «Дано …» (Допустим, Пусть, Given) с шагами «Когда/Если/И»
без шага «Тогда/То/Ожидаемый результат» — по нему нельзя решить, пройдена ли приёмка.
Обычное условное требование («Если …, система …») сценарием не считается.

Связь критериев с требованиями — по явным ссылкам на ID из критериев приёмки.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from validator.evidence import TrustedRule, build_evidence, gate_findings
from validator.findings import Evidence, Finding
from validator.graph import FactGraph
from validator.model import Document
from validator.quality.constraints import Constraint, compare_constraints, parse_constraint
from validator.quality.model import Contradiction, QualityResult
from validator.segmentation import Requirement, Segmentation
from validator.segmentation.text import content_key
from validator.segmentation.verify import body_start

if TYPE_CHECKING:
    from validator.lint import LintResult

CTR_01 = TrustedRule(
    rule_id="CTR-01",
    version="1",
    severity="major",
    category="consistency",
    criterion="S-CNS",
    evidence_kind="comparison",
    title="Несовместимые ограничения одного параметра",
)
CTR_02 = TrustedRule(
    rule_id="CTR-02",
    version="1",
    severity="minor",
    category="consistency",
    criterion="S-CNS",
    evidence_kind="comparison",
    title="Несовместимые значения одноимённого параметра в разных контекстах",
)
ACC_01 = TrustedRule(
    rule_id="ACC-01",
    version="1",
    severity="major",
    category="testability",
    criterion="VER",
    evidence_kind="quote",
    title="Сценарий приёмки без ожидаемого результата",
)
STAGE8_RULES = (CTR_01, CTR_02, ACC_01)

_VALUE_HEADER = re.compile(r"значени|value|величин", re.IGNORECASE)
_STATEMENT = re.compile(r"(?P<key>[^:—–\n]{2,80}?)\s*(?::|\s[—–-]\s)\s*(?P<value>[^\n]+)")
_GIVEN = re.compile(r"(?:дано|допустим|пусть|предусловие|given)\b", re.IGNORECASE)
_STEP = re.compile(
    r"(?:когда|если|при|и|а также|но|тогда|то|ожидаемый результат|результат|when|and|but|then)\b",
    re.IGNORECASE,
)
_THEN = re.compile(r"(?:тогда|то|ожидаемый результат|результат|then)\b", re.IGNORECASE)
_INLINE_WHEN = re.compile(r"(?<!\w)(?:когда|если|when)(?!\w)", re.IGNORECASE)
_INLINE_THEN = re.compile(r"(?<!\w)(?:тогда|then|ожидаемый результат)(?!\w)", re.IGNORECASE)


@dataclass(frozen=True)
class _Statement:
    fragment: Requirement
    key: str  # нормализованное название параметра
    name: str  # название как в тексте
    value: str
    constraint: Constraint
    table: int | None  # номер таблицы для строки таблицы
    start: int  # начало цитаты (текст после ID или вся строка таблицы)
    end: int


def analyze_quality(
    document: Document,
    segmentation: Segmentation,
    lint: LintResult | None,
    graph: FactGraph,
) -> QualityResult:
    nodes = [segmentation.by_id(node.requirement_id) for node in graph.nodes]
    order = {fragment.id: index for index, fragment in enumerate(segmentation.fragments)}
    statements = [s for f in nodes if (s := _statement(f, document)) is not None]
    drafts = _contradiction_findings(statements, document)
    drafts += _scenario_findings(segmentation, document)
    gate = gate_findings(drafts, segmentation, document, rules=STAGE8_RULES)
    findings = tuple(sorted(gate.accepted, key=lambda f: (order[f.requirement_ids[0]], f.id)))
    contradictions = [_from_finding(f, statements) for f in findings if f.rule_id != "ACC-01"]
    contradictions += _table_conflicts(graph, document, segmentation)
    contradictions.sort(key=lambda c: (order[c.requirement_ids[0]], order[c.requirement_ids[1]]))
    acceptance = tuple(f.id for f in segmentation.atomic_requirements if f.category == "acceptance")
    links = _acceptance_links(segmentation, acceptance)
    unlinked = tuple(
        f.id for f in nodes if f.category != "acceptance" and f.id not in links and acceptance
    )
    return QualityResult(
        document_sha256=document.sha256,
        findings=findings,
        rejected=gate.rejected,
        contradictions=tuple(contradictions),
        acceptance=acceptance,
        acceptance_links=links,
        unlinked_requirements=unlinked if links else (),
        statements=len(statements),
    )


# --- утверждения «параметр — значение» --------------------------------------------------------


def _statement(fragment: Requirement, document: Document) -> _Statement | None:
    block = document.blocks[fragment.sources[0].block]
    if block.kind == "table_row":
        header = block.header
        if len(block.cells) != 2 or len(header) != 2 or not _VALUE_HEADER.search(header[1]):
            return None
        name, value = block.cells
        start, end, table = 0, len(fragment.text), block.location.table
    elif block.kind in ("paragraph", "list_item"):
        start = body_start(fragment, document)
        body = fragment.text[start:].rstrip()
        match = _STATEMENT.fullmatch(body)
        if match is None:
            return None
        name, value = match.group("key"), match.group("value")
        if len(name.split()) > 8 or not any(char.isalpha() for char in name):
            return None
        end, table = start + len(body), None
    else:
        return None
    constraint = parse_constraint(value)
    key = content_key(name)
    if constraint is None or not key:
        return None
    shown = value.strip().rstrip(".;").strip()
    return _Statement(fragment, key, name.strip(), shown, constraint, table, start, end)


def _contradiction_findings(statements: Sequence[_Statement], document: Document) -> list[Finding]:
    groups: dict[str, list[_Statement]] = {}
    for statement in statements:
        groups.setdefault(statement.key, []).append(statement)
    findings: list[Finding] = []
    for members in groups.values():
        for index, first in enumerate(members):
            for second in members[index + 1 :]:
                if first.table is not None and first.table == second.table:
                    continue  # строки одной таблицы — CON-03
                check = compare_constraints(first.constraint, second.constraint)
                if check.verdict != "disjoint":
                    continue
                reason = _context_difference(first, second)
                findings.append(_contradiction(first, second, reason, document))
    return findings


def _context_difference(first: _Statement, second: _Statement) -> str | None:
    """Чем различается контекст двух утверждений; None — контекст один (раздел, группа списка,
    таблица), и значения — величины с единицей измерения."""
    a, b = first.fragment, second.fragment
    if a.section_path != b.section_path:
        return "в разных разделах"
    if first.table != second.table:
        return "в разных таблицах или в таблице и тексте одного раздела"
    if a.parent_id != b.parent_id:
        return "в разных группах списка (у пунктов разные вводные фразы)"
    if not first.constraint.family or not second.constraint.family:
        return "без единицы измерения (это может быть номер или версия, а не величина)"
    return None


def _contradiction(
    first: _Statement, second: _Statement, difference: str | None, document: Document
) -> Finding:
    rule = CTR_01 if difference is None else CTR_02
    a, b = first.fragment.id, second.fragment.id
    if difference is None:
        explanation = (
            f"Параметр «{first.name}»: в {a} — «{first.value}», в {b} — «{second.value}». "
            "Ограничения не пересекаются: нет значения, которое удовлетворяет обоим, — "
            "требования одного раздела нельзя выполнить одновременно."
        )
        recommendation = (
            "Оставьте одно значение параметра или явно разведите требования по условиям "
            "(сценарий, роль, период)."
        )
    else:
        explanation = (
            f"Параметр «{first.name}» {difference}: в {a} — «{first.value}», в {b} — "
            f"«{second.value}»; ограничения не пересекаются. Если это один и тот же "
            "параметр в одних условиях — требования противоречат друг другу; если контексты "
            "разные, это не противоречие, но название параметра неоднозначно."
        )
        recommendation = "Уточните контекст каждого значения или согласуйте значения."
    return Finding(
        id=f"{rule.rule_id}:{a}@{first.start}+{b}@{second.start}",
        rule_id=rule.rule_id,
        rule_version=rule.version,
        severity=rule.severity,
        category=rule.category,
        criterion=rule.criterion,
        title=rule.title,
        explanation=explanation,
        recommendation=recommendation,
        requirement_ids=(a, b),
        evidence=(_evidence(first, document), _evidence(second, document)),
        evidence_kind="comparison",
        document_sha256=document.sha256,
    )


def _evidence(statement: _Statement, document: Document) -> Evidence:
    return _quote(statement.fragment, statement.start, statement.end, document)


def _quote(fragment: Requirement, start: int, end: int, document: Document) -> Evidence:
    try:
        return build_evidence(fragment, start, end, document)
    except ValueError:  # не привязывается к документу — Gate отклонит с причиной
        return Evidence(fragment.id, start, end, fragment.text[start:end], ())


def _from_finding(finding: Finding, statements: Sequence[_Statement]) -> Contradiction:
    by_id = {s.fragment.id: s for s in statements}
    first, second = (by_id[rid] for rid in finding.requirement_ids)
    return Contradiction(
        (finding.requirement_ids[0], finding.requirement_ids[1]),
        "CTR-01" if finding.rule_id == "CTR-01" else "CTR-02",
        "confirmed" if finding.rule_id == "CTR-01" else "potential",
        finding.id,
        finding.evidence,
        finding.explanation,
        compare_constraints(first.constraint, second.constraint),
    )


def _table_conflicts(
    graph: FactGraph, document: Document, segmentation: Segmentation
) -> list[Contradiction]:
    """Пары CON-03 из графа: числовой разбор решает, подтверждено противоречие или нет."""
    result: list[Contradiction] = []
    for edge in graph.edges:
        if edge.rule_id != "CON-03":
            continue
        values = []
        for rid in (edge.source_id, edge.target_id):
            block = document.blocks[segmentation.by_id(rid).sources[0].block]
            values.append(parse_constraint(block.cells[1]) if len(block.cells) == 2 else None)
        a, b = values
        numeric = compare_constraints(a, b) if a is not None and b is not None else None
        if numeric is not None and numeric.verdict == "compatible":
            status, note = (
                "potential",
                f"{edge.explanation} Как ограничения значения совместимы ({numeric.detail}) — "
                "возможно, это уточнение, а не противоречие.",
            )
        elif numeric is not None and numeric.verdict == "disjoint":
            status, note = "confirmed", f"{edge.explanation} Контрпример: {numeric.detail}."
        else:
            status, note = (
                "confirmed",
                f"{edge.explanation} Числовой контрпример не построен (значения не разобраны "
                "как сравнимые ограничения); противоречие следует из правила CON-03.",
            )
        result.append(
            Contradiction(
                (edge.source_id, edge.target_id),
                "CON-03",
                "confirmed" if status == "confirmed" else "potential",
                edge.finding_id,
                edge.evidence,
                note,
                numeric,
            )
        )
    return result


# --- сценарии приёмки -------------------------------------------------------------------------


def _scenario_findings(segmentation: Segmentation, document: Document) -> list[Finding]:
    containers = {f.parent_id for f in segmentation.fragments if f.parent_id}
    leaves = [
        f
        for f in segmentation.fragments
        if f.role in ("requirement", "ambiguous") and f.id not in containers and f.text.strip()
    ]
    findings: list[Finding] = []
    index = 0
    while index < len(leaves):
        first = leaves[index]
        body = _body(first, document)
        if not _GIVEN.match(body):
            index += 1
            continue
        steps = [first]
        index += 1
        while (
            index < len(leaves)
            and leaves[index].section_path == first.section_path
            and _STEP.match(_body(leaves[index], document))
            and not _GIVEN.match(_body(leaves[index], document))
        ):
            steps.append(leaves[index])
            index += 1
        texts = [_body(step, document) for step in steps]
        has_then = any(_THEN.match(t) or _INLINE_THEN.search(t) for t in texts)
        has_when = len(steps) > 1 or bool(_INLINE_WHEN.search(texts[0]))
        if has_when and not has_then:
            findings.append(_scenario(steps, document))
    return findings


def _body(fragment: Requirement, document: Document) -> str:
    return fragment.text[body_start(fragment, document) :].strip()


def _scenario(steps: Sequence[Requirement], document: Document) -> Finding:
    evidence = []
    for step in steps:
        start = body_start(step, document)
        evidence.append(_quote(step, start, len(step.text.rstrip()), document))
    ids = ", ".join(step.id for step in steps)
    return Finding(
        id=f"{ACC_01.rule_id}:" + "+".join(f"{e.requirement_id}@{e.start}" for e in evidence),
        rule_id=ACC_01.rule_id,
        rule_version=ACC_01.version,
        severity=ACC_01.severity,
        category=ACC_01.category,
        criterion=ACC_01.criterion,
        title=ACC_01.title,
        explanation=(
            f"Сценарий приёмки ({ids}) задаёт исходное условие и действие, но не задаёт "
            "ожидаемый результат (нет шага «Тогда»): по нему нельзя решить, пройдена ли приёмка."
        ),
        recommendation="Добавьте шаг «Тогда …» с наблюдаемым и проверяемым результатом.",
        requirement_ids=tuple(dict.fromkeys(step.id for step in steps)),
        evidence=tuple(evidence),
        evidence_kind="quote",
        document_sha256=document.sha256,
    )


def _acceptance_links(segmentation: Segmentation, acceptance: Sequence[str]) -> tuple[str, ...]:
    owners: dict[str, list[str]] = {}
    for fragment in segmentation.atomic_requirements:
        if fragment.doc_id and fragment.category != "acceptance":
            owners.setdefault(fragment.doc_id, []).append(fragment.id)
    linked: dict[str, None] = {}
    for rid in acceptance:
        for reference in segmentation.by_id(rid).references:
            if reference.startswith("req:"):
                for target in owners.get(reference.removeprefix("req:"), []):
                    linked[target] = None
    order = {f.id: i for i, f in enumerate(segmentation.fragments)}
    return tuple(sorted(linked, key=order.__getitem__))
