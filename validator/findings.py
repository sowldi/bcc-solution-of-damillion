"""Замечание (Finding) и его доказательства.

Замечание без проверяемого доказательства не существует: в отчёт попадают только замечания,
прошедшие Evidence Gate (validator.evidence), со статусом «verified». Доказательство — точная
цитата fragment.text[start:end] фрагмента сегментации и ссылки на блоки документа (SourceRef),
которые можно перепроверить по исходному файлу.

Модель общая для детерминированных правил (origin="rule") и будущих LLM-гипотез
(origin="llm"); JSON-схема одна.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast, get_args

from validator.segmentation import SourceRef
from validator.segmentation.model import source_ref_from_dict, source_ref_to_dict

Severity = Literal["critical", "major", "minor", "info"]
RuleCategory = Literal["testability", "ambiguity", "completeness", "consistency", "traceability"]
EvidenceKind = Literal["quote", "absence", "comparison"]
EvidenceStatus = Literal["unverified", "verified", "rejected"]
Origin = Literal["rule", "llm"]
# Область, в которой утверждается отсутствие, и что именно отсутствует.
AbsenceScope = Literal["table", "section", "requirement", "id_registry", "heading_registry"]
ABSENCE_SCOPES: tuple[AbsenceScope, ...] = get_args(AbsenceScope)
ABSENCE_CLAIMS: frozenset[tuple[str, str]] = frozenset(
    {
        ("table", "data"),  # в таблице нет заполненных строк данных
        ("section", "text"),  # в разделе нет текста (кроме заголовков)
        ("requirement", "number"),  # в требовании нет числа
        ("requirement", "document_id"),  # у требования нет ID документа
        ("requirement", "value"),  # значение не заполнено (пусто или прочерк)
        ("id_registry", "declaration"),  # ID нигде в документе не объявлен
        ("heading_registry", "heading"),  # раздела с таким номером нет
    }
)

SEVERITIES: tuple[Severity, ...] = get_args(Severity)
RULE_CATEGORIES: tuple[RuleCategory, ...] = get_args(RuleCategory)
EVIDENCE_KINDS: tuple[EvidenceKind, ...] = get_args(EvidenceKind)
EVIDENCE_STATUSES: tuple[EvidenceStatus, ...] = get_args(EvidenceStatus)
ORIGINS: tuple[Origin, ...] = get_args(Origin)

SEVERITY_TITLES: dict[Severity, str] = {
    "critical": "Критичное",
    "major": "Существенное",
    "minor": "Незначительное",
    "info": "Информационное",
}
RULE_CATEGORY_TITLES: dict[RuleCategory, str] = {
    "testability": "Проверяемость",
    "ambiguity": "Однозначность",
    "completeness": "Полнота",
    "consistency": "Согласованность",
    "traceability": "Трассируемость",
}
EVIDENCE_KIND_TITLES: dict[EvidenceKind, str] = {
    "quote": "цитата с дефектом",
    "absence": "цитата места, где не хватает содержания",
    "comparison": "сопоставление нескольких цитат",
}


def severity_rank(severity: Severity) -> int:
    """0 — самое важное (critical)."""
    return SEVERITIES.index(severity)


@dataclass(frozen=True)
class Evidence:
    """Доказательство: quote == фрагмент.text[start:end], sources — те же символы в документе.

    Пустая цитата (start == end == 0) допустима только для замечаний вида «absence» и только
    для фрагмента, у которого в документе действительно нет текста (пустая строка таблицы).
    """

    requirement_id: str
    start: int
    end: int
    quote: str
    sources: tuple[SourceRef, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "start": self.start,
            "end": self.end,
            "quote": self.quote,
            "sources": [source_ref_to_dict(ref) for ref in self.sources],
        }

    @classmethod
    def from_dict(cls, data: Any) -> Evidence:
        if not isinstance(data, dict):
            raise ValueError("evidence: ожидается объект")
        return cls(
            requirement_id=_str(data, "requirement_id"),
            start=_int(data, "start"),
            end=_int(data, "end"),
            quote=_str(data, "quote"),
            sources=tuple(source_ref_from_dict(item) for item in _list(data, "sources")),
        )


@dataclass(frozen=True)
class AbsenceClaim:
    """Утверждение об отсутствии в ограниченной, проверяемой области документа.

    scope/target: таблица (номер), раздел (путь заголовков через « › »), требование (ID
    фрагмента), реестр ID документа (отсутствующий ID), нумерация разделов (номер раздела).
    missing — что именно отсутствует (допустимые сочетания — ABSENCE_CLAIMS). Evidence Gate
    перепроверяет утверждение по документу; «этого нет в документе» без области не принимается.
    """

    scope: AbsenceScope
    target: str
    missing: str

    def to_dict(self) -> dict[str, Any]:
        return {"scope": self.scope, "target": self.target, "missing": self.missing}

    @classmethod
    def from_dict(cls, data: Any) -> AbsenceClaim:
        if not isinstance(data, dict):
            raise ValueError("absence: ожидается объект")
        return cls(
            scope=cast(AbsenceScope, _choice(data, "scope", ABSENCE_SCOPES)),
            target=_str(data, "target"),
            missing=_str(data, "missing"),
        )


@dataclass(frozen=True)
class Finding:
    """Замечание к документу.

    id — стабильный и уникальный: «ПРАВИЛО:ФРАГМЕНТ@НАЧАЛО», для нескольких участников
    части соединены «+». requirement_ids — ID фрагментов сегментации, к которым относится
    замечание (требование, а для структурных правил — также заглушка, заголовок, строка
    таблицы); у каждого есть своё доказательство в evidence.
    document_sha256 — документ, по которому замечание построено: для другого файла или
    другой версии документа замечание не принимается. absence — область отсутствия
    (обязательна для evidence_kind="absence").
    """

    id: str
    rule_id: str
    rule_version: str
    severity: Severity
    category: RuleCategory
    criterion: str
    title: str
    explanation: str
    recommendation: str
    requirement_ids: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    evidence_kind: EvidenceKind
    evidence_status: EvidenceStatus = "unverified"
    origin: Origin = "rule"
    document_sha256: str = ""  # пусто — замечание не привязано к документу и не пройдёт Gate
    absence: AbsenceClaim | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "document_sha256": self.document_sha256,
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "origin": self.origin,
            "severity": self.severity,
            "category": self.category,
            "criterion": self.criterion,
            "title": self.title,
            "explanation": self.explanation,
            "recommendation": self.recommendation,
            "requirement_ids": list(self.requirement_ids),
            "evidence_kind": self.evidence_kind,
            "evidence_status": self.evidence_status,
            "absence": self.absence.to_dict() if self.absence else None,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, data: Any) -> Finding:
        if not isinstance(data, dict):
            raise ValueError("finding: ожидается объект")
        return cls(
            id=_str(data, "id"),
            rule_id=_str(data, "rule_id"),
            rule_version=_str(data, "rule_version"),
            severity=cast(Severity, _choice(data, "severity", SEVERITIES)),
            category=cast(RuleCategory, _choice(data, "category", RULE_CATEGORIES)),
            criterion=_str(data, "criterion"),
            title=_str(data, "title"),
            explanation=_str(data, "explanation"),
            recommendation=_str(data, "recommendation"),
            requirement_ids=_str_tuple(data, "requirement_ids"),
            evidence=tuple(Evidence.from_dict(item) for item in _list(data, "evidence")),
            evidence_kind=cast(EvidenceKind, _choice(data, "evidence_kind", EVIDENCE_KINDS)),
            evidence_status=cast(
                EvidenceStatus, _choice(data, "evidence_status", EVIDENCE_STATUSES)
            ),
            origin=cast(Origin, _choice(data, "origin", ORIGINS)),
            document_sha256=_str(data, "document_sha256"),
            absence=AbsenceClaim.from_dict(data["absence"])
            if data.get("absence") is not None
            else None,
        )


@dataclass(frozen=True)
class RejectedFinding:
    """Замечание, не прошедшее проверку доказательств, и причины отказа."""

    finding: Finding
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"finding": self.finding.to_dict(), "reasons": list(self.reasons)}

    @classmethod
    def from_dict(cls, data: Any) -> RejectedFinding:
        if not isinstance(data, dict):
            raise ValueError("rejected: ожидается объект")
        return cls(Finding.from_dict(data.get("finding")), _str_tuple(data, "reasons"))


# --- разбор JSON ------------------------------------------------------------------------


def _str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Поле {key}: ожидается строка")
    return value


def _int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Поле {key}: ожидается целое число")
    return value


def _list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Поле {key}: ожидается список")
    return value


def _str_tuple(data: dict[str, Any], key: str) -> tuple[str, ...]:
    values = _list(data, key)
    if not all(isinstance(value, str) for value in values):
        raise ValueError(f"Поле {key}: ожидается список строк")
    return tuple(values)


def _choice(data: dict[str, Any], key: str, allowed: tuple[str, ...]) -> str:
    value = data.get(key)
    if value not in allowed:
        raise ValueError(f"Недопустимое значение {key}: {value!r}")
    return str(value)
