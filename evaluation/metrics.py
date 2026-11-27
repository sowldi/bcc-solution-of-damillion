"""Разметка корпуса и метрики детерминированного слоя.

Разметка документа — JSON `<имя>.labels.json` рядом с документом:

- `expected` — внедрённые дефекты: якорь требования, ожидаемое правило (`rule`), допустимые
  альтернативные правила (`alt`: перекрывающиеся правила, из которых движок оставляет одно),
  второе требование пары (`pair`) для правил сравнения;
- `traps` — ловушки: требование, на котором перечисленные правила срабатывать не должны;
- `contradictions` — ожидаемый статус противоречия этапа 8 (`confirmed` / `potential`);
- `semantic` — смысловые дефекты, которые правила по замыслу не ловят (для честного учёта
  пропусков);
- `related_pairs` — пары требований об одном объекте (для полноты отбора кандидатов);
- `verdict` — ожидаемый вердикт этапа 9.

Якорь — ID документа («Ф-3») или `quote:<фрагмент текста>`. Якорь, которого нет в документе,
— ошибка разметки (LabelError), а не пропуск валидатора.

Сопоставление по множествам: замечание верно (не FP), если совпадает хотя бы с одной меткой
(правило из `rule`/`alt`, требование якоря, требование пары); метка найдена (не FN), если её
закрывает хотя бы одно замечание. Так два разных правила, верно сработавшие на одном
требовании (TST-04 и TST-05 на «минимизируется»), не превращаются в ложное срабатывание.
Метрики считаются по замечаниям с важностью critical/major/minor (info — повод для
проверки, не дефект). TP засчитывается правилу сработавшего замечания (каждое верное
замечание — один TP), FN — правилу метки.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from validator.findings import Finding
from validator.segmentation import Segmentation


class LabelError(ValueError):
    """Разметка не соответствует документу или формату."""


@dataclass(frozen=True)
class Label:
    anchor: str
    rule: str
    alt: tuple[str, ...] = ()
    pair: str | None = None
    note: str = ""

    @property
    def rules(self) -> tuple[str, ...]:
        return (self.rule, *self.alt)


@dataclass(frozen=True)
class Trap:
    anchor: str
    rules: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True)
class ContradictionLabel:
    anchor: str
    status: str
    pair: str | None = None
    note: str = ""


@dataclass(frozen=True)
class SemanticLabel:
    anchor: str
    category: str
    note: str = ""


@dataclass(frozen=True)
class DocumentLabels:
    document: str
    split: str
    verdict: str | None
    expected: tuple[Label, ...] = ()
    traps: tuple[Trap, ...] = ()
    contradictions: tuple[ContradictionLabel, ...] = ()
    semantic: tuple[SemanticLabel, ...] = ()
    related_pairs: tuple[tuple[str, str], ...] = ()


_TOP = {
    "document",
    "split",
    "verdict",
    "expected",
    "traps",
    "contradictions",
    "semantic",
    "related_pairs",
}


def _fields(item: Any, allowed: set[str], required: set[str], where: str) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise LabelError(f"{where}: ожидается объект")
    unknown = set(item) - allowed
    missing = required - set(item)
    if unknown or missing:
        raise LabelError(f"{where}: лишние поля {sorted(unknown)}, нет полей {sorted(missing)}")
    return item


def load_labels(path: Path) -> DocumentLabels:
    data = json.loads(path.read_text(encoding="utf-8"))
    data = _fields(data, _TOP, {"document", "split"}, path.name)
    if data["split"] not in ("dev", "test"):
        raise LabelError(f"{path.name}: split — dev или test")
    expected = tuple(
        Label(e["anchor"], e["rule"], tuple(e.get("alt", ())), e.get("pair"), e.get("note", ""))
        for e in (
            _fields(x, {"anchor", "rule", "alt", "pair", "note"}, {"anchor", "rule"}, "expected")
            for x in data.get("expected", [])
        )
    )
    traps = tuple(
        Trap(t["anchor"], tuple(t["rules"]), t.get("note", ""))
        for t in (
            _fields(x, {"anchor", "rules", "note"}, {"anchor", "rules"}, "traps")
            for x in data.get("traps", [])
        )
    )
    contradictions = tuple(
        ContradictionLabel(c["anchor"], c["status"], c.get("pair"), c.get("note", ""))
        for c in (
            _fields(x, {"anchor", "status", "pair", "note"}, {"anchor", "status"}, "contradictions")
            for x in data.get("contradictions", [])
        )
    )
    for label in contradictions:
        if label.status not in ("confirmed", "potential"):
            raise LabelError(f"contradictions: недопустимый статус {label.status!r}")
    semantic = tuple(
        SemanticLabel(s["anchor"], s["category"], s.get("note", ""))
        for s in (
            _fields(x, {"anchor", "category", "note"}, {"anchor", "category"}, "semantic")
            for x in data.get("semantic", [])
        )
    )
    pairs = tuple((str(a), str(b)) for a, b in data.get("related_pairs", []))
    return DocumentLabels(
        data["document"],
        data["split"],
        data.get("verdict"),
        expected,
        traps,
        contradictions,
        semantic,
        pairs,
    )


def resolve(anchor: str, segmentation: Segmentation) -> frozenset[str]:
    """ID фрагментов, на которые указывает якорь; пустой результат — ошибка разметки."""
    if anchor.startswith("quote:"):
        quote = anchor.removeprefix("quote:")
        found = frozenset(f.id for f in segmentation.fragments if quote in f.text)
    else:
        found = frozenset(
            f.id
            for f in segmentation.fragments
            if f.doc_id == anchor or f.id == anchor or f.id.startswith((f"{anchor}/", f"{anchor}~"))
        )
    if not found:
        raise LabelError(f"Якорь «{anchor}» не найден в документе.")
    return found


def check_anchors(labels: DocumentLabels, segmentation: Segmentation) -> None:
    anchors = [label.anchor for label in labels.expected]
    anchors += [label.pair for label in labels.expected if label.pair]
    anchors += [trap.anchor for trap in labels.traps]
    anchors += [c.anchor for c in labels.contradictions] + [
        c.pair for c in labels.contradictions if c.pair
    ]
    anchors += [s.anchor for s in labels.semantic]
    anchors += [rid for pair in labels.related_pairs for rid in pair]
    for anchor in anchors:
        resolve(anchor, segmentation)


@dataclass
class RuleCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    def add(self, other: RuleCounts) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn


@dataclass
class MatchResult:
    per_rule: dict[str, RuleCounts] = field(default_factory=dict)
    true_positives: list[tuple[Label, Finding]] = field(default_factory=list)
    false_negatives: list[Label] = field(default_factory=list)
    false_positives: list[Finding] = field(default_factory=list)

    def counts(self, rule: str) -> RuleCounts:
        return self.per_rule.setdefault(rule, RuleCounts())


def defect_findings(findings: Iterable[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity != "info"]


def match(
    labels: Sequence[Label], findings: Sequence[Finding], segmentation: Segmentation
) -> MatchResult:
    result = MatchResult()
    resolved = [
        (
            label,
            resolve(label.anchor, segmentation),
            resolve(label.pair, segmentation) if label.pair else None,
        )
        for label in labels
    ]

    def fits(finding: Finding, anchor: frozenset[str], pair: frozenset[str] | None) -> bool:
        ids = set(finding.requirement_ids)
        return bool(anchor & ids) and (pair is None or bool(pair & ids))

    for label, anchor, pair in resolved:
        hits = [f for f in findings if f.rule_id in label.rules and fits(f, anchor, pair)]
        if hits:
            result.true_positives.append((label, hits[0]))
        else:
            result.false_negatives.append(label)
            result.counts(label.rule).fn += 1
    for finding in findings:
        if any(
            finding.rule_id in label.rules and fits(finding, anchor, pair)
            for label, anchor, pair in resolved
        ):
            result.counts(finding.rule_id).tp += 1
        else:
            result.false_positives.append(finding)
            result.counts(finding.rule_id).fp += 1
    return result


def trap_hits(trap: Trap, findings: Sequence[Finding], segmentation: Segmentation) -> list[str]:
    """ID замечаний, которые сработали на ловушке (должно быть пусто)."""
    anchor = resolve(trap.anchor, segmentation)
    return [f.id for f in findings if f.rule_id in trap.rules and anchor & set(f.requirement_ids)]


def ratio(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}"


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"
