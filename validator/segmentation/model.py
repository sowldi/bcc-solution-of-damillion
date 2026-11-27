"""Модель сегментации: требования и сопутствующие фрагменты с доказательной привязкой к источнику.

Документ делится на фрагменты (Requirement). Требованием считается фрагмент с role="requirement";
остальные роли (пояснение, пример, справочный текст, заголовок, шапка таблицы, код,
неоднозначный фрагмент) сохраняются, чтобы не терять текст и контекст для следующих этапов.

Грамматика ID (см. README, раздел «Сегментация требований»):
- «Ф-1» — идентификатор из документа, если он есть в начале фрагмента;
- «R-3f9a2c1b» / «F-3f9a2c1b» — 8 hex SHA-256 от нормализованного текста (R — требования и
  неоднозначные фрагменты, F — остальные);
- «X~2» — второй фрагмент с тем же базовым ID (дубль или повтор идентификатора);
- «X/1» — утверждение № 1 внутри фрагмента X (знак «/» встречается только здесь).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, cast, get_args

from validator.model import Location

Role = Literal[
    "requirement",
    "ambiguous",
    "explanation",
    "example",
    "reference",
    "heading",
    "table_header",
    "code",
]
Category = Literal[
    "functional",
    "business_rule",
    "non_functional",
    "performance",
    "security",
    "availability",
    "constraint",
    "interface",
    "api",
    "data",
    "integration",
    "acceptance",
    "goal",
    "unspecified",
]
Confidence = Literal["high", "medium", "low"]

ROLE_TITLES: dict[Role, str] = {
    "requirement": "требование",
    "ambiguous": "неоднозначный фрагмент",
    "explanation": "пояснение",
    "example": "пример",
    "reference": "справочный текст",
    "heading": "заголовок",
    "table_header": "шапка таблицы",
    "code": "код / технический фрагмент",
}
CATEGORY_TITLES: dict[Category, str] = {
    "functional": "функциональное",
    "business_rule": "бизнес-правило",
    "non_functional": "нефункциональное",
    "performance": "производительность",
    "security": "безопасность",
    "availability": "доступность",
    "constraint": "ограничение",
    "interface": "интерфейс",
    "api": "API",
    "data": "данные",
    "integration": "интеграция",
    "acceptance": "критерий приёмки",
    "goal": "цель",
    "unspecified": "не определён",
}
SCHEMA = "validator.segmentation/1"

_ROLES = frozenset(get_args(Role))
_CATEGORIES = frozenset(get_args(Category))
_CONFIDENCES = frozenset(get_args(Confidence))


@dataclass(frozen=True)
class SourceRef:
    """Доказательная ссылка: цитата — это ровно document.blocks[block].text[start:end].

    location — место цитаты; строки уточнены до цитаты, только если строки текста блока
    доказуемо совпадают со строками источника (иначе — место всего блока).
    """

    block: int
    start: int
    end: int
    location: Location


@dataclass(frozen=True)
class Requirement:
    """Фрагмент документа после сегментации.

    text — дословная цитата (у фрагмента из нескольких блоков цитаты блоков соединены «\\n»).
    normalized — тот же текст для анализа: без идентификатора/номера в начале, пробелы
    схлопнуты, кавычки унифицированы. Исходная цитата при этом не меняется.
    detection — имена сработавших правил: почему фрагмент получил такую роль и тип.
    """

    id: str
    text: str
    normalized: str
    role: Role
    category: Category
    sources: tuple[SourceRef, ...]
    section_path: tuple[str, ...]
    doc_id: str | None = None
    doc_number: str | None = None
    parent_id: str | None = None
    children: tuple[str, ...] = ()
    duplicate_of: str | None = None
    similar_to: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    detection: tuple[str, ...] = ()
    confidence: Confidence = "medium"
    warnings: tuple[str, ...] = ()

    @property
    def location(self) -> Location:
        """Место начала фрагмента (для отчёта)."""
        return self.sources[0].location if self.sources else Location(self.section_path)

    @property
    def is_statement(self) -> bool:
        """Утверждение, выделенное внутри фрагмента (ID вида «X/1»)."""
        return "/" in self.id

    @property
    def section(self) -> str | None:
        """Ближайший раздел (последний заголовок в цепочке)."""
        return self.section_path[-1] if self.section_path else None

    @property
    def parent_section(self) -> str | None:
        """Раздел уровнем выше ближайшего."""
        return self.section_path[-2] if len(self.section_path) > 1 else None


@dataclass(frozen=True)
class Segmentation:
    """Результат сегментации документа (фрагменты в порядке документа, утверждения — сразу
    после своего фрагмента)."""

    document_sha256: str
    fragments: tuple[Requirement, ...]
    warnings: tuple[str, ...] = ()
    _index: dict[str, Requirement] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        index: dict[str, Requirement] = {}
        for fragment in self.fragments:
            if fragment.id in index:
                raise ValueError(f"ID фрагмента не уникален: {fragment.id}")
            index[fragment.id] = fragment
        object.__setattr__(self, "_index", index)

    def by_id(self, fragment_id: str) -> Requirement:
        return self._index[fragment_id]

    @property
    def requirements(self) -> tuple[Requirement, ...]:
        """Все фрагменты-требования, включая составные и их утверждения."""
        return tuple(f for f in self.fragments if f.role == "requirement")

    @property
    def atomic_requirements(self) -> tuple[Requirement, ...]:
        """Требования без дочерних требований: утверждения и неделимые фрагменты."""
        containers = {
            f.parent_id for f in self.fragments if f.role == "requirement" and f.parent_id
        }
        return tuple(f for f in self.requirements if f.id not in containers)

    @property
    def ambiguous(self) -> tuple[Requirement, ...]:
        return tuple(f for f in self.fragments if f.role == "ambiguous")

    # --- JSON ------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "document_sha256": self.document_sha256,
            "warnings": list(self.warnings),
            "fragments": [_requirement_to_dict(f) for f in self.fragments],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Segmentation:
        if data.get("schema") != SCHEMA:
            raise ValueError(f"Неизвестная схема сегментации: {data.get('schema')!r}")
        return cls(
            document_sha256=_str(data, "document_sha256"),
            fragments=tuple(_requirement_from_dict(item) for item in _list(data, "fragments")),
            warnings=_str_tuple(data, "warnings"),
        )

    @classmethod
    def from_json(cls, text: str) -> Segmentation:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("JSON сегментации должен быть объектом")
        return cls.from_dict(data)


# --- сериализация ---------------------------------------------------------------------


def _location_to_dict(location: Location) -> dict[str, Any]:
    return {
        "section_path": list(location.section_path),
        "page": location.page,
        "line_start": location.line_start,
        "line_end": location.line_end,
        "paragraph": location.paragraph,
        "table": location.table,
        "row": location.row,
    }


def _location_from_dict(data: Any) -> Location:
    if not isinstance(data, dict):
        raise ValueError("location должен быть объектом")
    return Location(
        section_path=_str_tuple(data, "section_path"),
        page=_opt_int(data, "page"),
        line_start=_opt_int(data, "line_start"),
        line_end=_opt_int(data, "line_end"),
        paragraph=_opt_int(data, "paragraph"),
        table=_opt_int(data, "table"),
        row=_opt_int(data, "row"),
    )


def _requirement_to_dict(fragment: Requirement) -> dict[str, Any]:
    return {
        "id": fragment.id,
        "text": fragment.text,
        "normalized": fragment.normalized,
        "role": fragment.role,
        "category": fragment.category,
        "sources": [
            {
                "block": ref.block,
                "start": ref.start,
                "end": ref.end,
                "location": _location_to_dict(ref.location),
            }
            for ref in fragment.sources
        ],
        "section_path": list(fragment.section_path),
        "doc_id": fragment.doc_id,
        "doc_number": fragment.doc_number,
        "parent_id": fragment.parent_id,
        "children": list(fragment.children),
        "duplicate_of": fragment.duplicate_of,
        "similar_to": list(fragment.similar_to),
        "references": list(fragment.references),
        "detection": list(fragment.detection),
        "confidence": fragment.confidence,
        "warnings": list(fragment.warnings),
    }


def _requirement_from_dict(data: Any) -> Requirement:
    if not isinstance(data, dict):
        raise ValueError("фрагмент должен быть объектом")
    sources = []
    for item in _list(data, "sources"):
        if not isinstance(item, dict):
            raise ValueError("source должен быть объектом")
        sources.append(
            SourceRef(
                block=_int(item, "block"),
                start=_int(item, "start"),
                end=_int(item, "end"),
                location=_location_from_dict(item.get("location")),
            )
        )
    return Requirement(
        id=_str(data, "id"),
        text=_str(data, "text"),
        normalized=_str(data, "normalized"),
        role=cast(Role, _choice(data, "role", _ROLES)),
        category=cast(Category, _choice(data, "category", _CATEGORIES)),
        sources=tuple(sources),
        section_path=_str_tuple(data, "section_path"),
        doc_id=_opt_str(data, "doc_id"),
        doc_number=_opt_str(data, "doc_number"),
        parent_id=_opt_str(data, "parent_id"),
        children=_str_tuple(data, "children"),
        duplicate_of=_opt_str(data, "duplicate_of"),
        similar_to=_str_tuple(data, "similar_to"),
        references=_str_tuple(data, "references"),
        detection=_str_tuple(data, "detection"),
        confidence=cast(Confidence, _choice(data, "confidence", _CONFIDENCES)),
        warnings=_str_tuple(data, "warnings"),
    )


def _str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Поле {key}: ожидается строка")
    return value


def _opt_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"Поле {key}: ожидается строка или null")
    return value


def _int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Поле {key}: ожидается целое число")
    return value


def _opt_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    return _int(data, key)


def _list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Поле {key}: ожидается список")
    return value


def _str_tuple(data: dict[str, Any], key: str) -> tuple[str, ...]:
    values = _list(data, key)
    if not all(isinstance(v, str) for v in values):
        raise ValueError(f"Поле {key}: ожидается список строк")
    return tuple(values)


def _choice(data: dict[str, Any], key: str, allowed: frozenset[str]) -> str:
    value = data.get(key)
    if value not in allowed:
        raise ValueError(f"Недопустимое значение {key}: {value!r}")
    return str(value)
