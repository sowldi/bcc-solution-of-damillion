"""Сегментация документа: блоки этапа 1 → фрагменты (требования и сопутствующий текст).

Порядок работы:
1. каждый блок классифицируется (rules.classify); абзац со строчной буквы после фрагмента без
   конечной пунктуации в том же разделе — продолжение предыдущего фрагмента (несколько источников);
   значения JSON рядом со строками JSON — тоже код (JSON, набранный абзацем на строку);
2. связи: пункты списка после фрагмента, оканчивающегося «:», — части перечисления этого
   фрагмента; блок-пример или блок-пояснение — к ближайшему предыдущему требованию раздела;
   ID документа из заголовка («### FR-1 …») передаётся требованию под этим заголовком;
3. ID: идентификатор документа или содержательный хеш; повторы — суффикс «~N»;
4. фрагмент-требование (абзац или пункт), где несколько самостоятельных утверждений, делится
   на утверждения «X/1», «X/2»; их источники вычисляет публичная span_sources;
5. похожие (не дословные) требования связываются через similar_to;
6. результат проверяется verify_segmentation — при нарушении SegmentationError (это ошибка
   агента, а не документа, и молча не пропускается).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher

from validator.ingest.common import one_line
from validator.model import Block, BlockKind, Document
from validator.segmentation.model import (
    Category,
    Confidence,
    Requirement,
    Role,
    Segmentation,
    SourceRef,
)
from validator.segmentation.rules import (
    W_MODAL_OUTSIDE,
    W_MODAL_REFERENCE,
    W_SUBJECT_OUTSIDE,
    classify,
    looks_like_json_value,
)
from validator.segmentation.sources import source_location, span_sources
from validator.segmentation.text import (
    Prefix,
    Statement,
    content_key,
    find_references,
    is_document_id,
    normalize_text,
    parse_prefix,
    split_statements,
)
from validator.segmentation.verify import verify_segmentation


class SegmentationError(RuntimeError):
    """Внутреннее нарушение: сегментация не смогла доказать привязку к источнику."""


SIMILARITY_THRESHOLD = 0.9
_MIN_SIMILAR_CHARS = 20
_ANALYZED: frozenset[Role] = frozenset({"requirement", "ambiguous"})
_TEXTUAL: frozenset[Role] = frozenset(
    {"requirement", "ambiguous", "reference", "example", "explanation"}
)
_SENTENCE_END = tuple(".!?:;…")

W_NO_REQUIREMENTS = "Требования не обнаружены: ни один фрагмент не распознан как требование."


@dataclass
class _Piece:
    block: int
    length: int  # длина текста блока
    offset: int  # начало блока в тексте фрагмента


@dataclass
class _Unit:
    pieces: list[_Piece]
    text: str
    kind: BlockKind
    section_path: tuple[str, ...]
    prefix: Prefix
    role: Role
    category: Category
    confidence: Confidence
    detection: list[str]
    warnings: list[str]
    doc_id: str | None = None  # ID документа фрагмента: свой или унаследованный от заголовка
    id_base: str | None = None  # ID документа, который фрагмент несёт как свой ID; иначе хеш
    declared_id: str | None = None  # ID документа в тексте заголовка («FR-1 Создание заказа»)
    id: str = ""
    parent: _Unit | None = None
    duplicate_of: str | None = None
    statements: tuple[Statement, ...] = ()
    attached: list[_Unit] = field(default_factory=list)

    @property
    def body(self) -> str:
        return self.text[self.prefix.body_start :]


def segment_document(document: Document) -> Segmentation:
    units = _build_units(document.blocks)
    _mark_code_sequences(units)
    _link_enumerations(units)
    _link_notes(units)
    heading_warnings = _link_heading_ids(units)
    _assign_ids(units)
    _split_statements(units)
    fragments = _link_similar(_materialize(units, document))
    warnings = (*heading_warnings, *_global_warnings(fragments))
    segmentation = Segmentation(document.sha256, tuple(fragments), warnings)
    problems = verify_segmentation(segmentation, document)
    if problems:
        details = "; ".join(problems[:5])
        raise SegmentationError(f"Сегментация не прошла проверку источников: {details}")
    return segmentation


# --- 1. фрагменты из блоков -----------------------------------------------------------


def _prefix(block: Block) -> Prefix:
    if block.kind == "table_row":
        first = block.cells[0].strip() if block.cells else ""
        return Prefix(first if is_document_id(first) else None, None, 0)
    if block.kind == "heading":
        # ID в заголовке относится к требованию под ним (_link_heading_ids), а не к заголовку
        return Prefix(None, None, 0)
    return parse_prefix(block.text, block.kind)


def _build_units(blocks: tuple[Block, ...]) -> list[_Unit]:
    units: list[_Unit] = []
    for index, block in enumerate(blocks):
        if units and _continues(units[-1], block):
            unit = units[-1]
            unit.pieces.append(_Piece(index, len(block.text), len(unit.text) + 1))
            unit.text = f"{unit.text}\n{block.text}"
            if "continuation" not in unit.detection:
                unit.detection.append("continuation")
            continue
        prefix = _prefix(block)
        result = classify(block, prefix)
        declared = parse_prefix(block.text, block.kind).doc_id if block.kind == "heading" else None
        units.append(
            _Unit(
                pieces=[_Piece(index, len(block.text), 0)],
                text=block.text,
                kind=block.kind,
                section_path=block.location.section_path,
                prefix=prefix,
                role=result.role,
                category=result.category,
                confidence=result.confidence,
                detection=list(result.detection),
                warnings=list(result.warnings),
                doc_id=prefix.doc_id,
                id_base=prefix.doc_id,
                declared_id=declared,
            )
        )
    return units


def _mark_code_sequences(units: list[_Unit]) -> None:
    """Значение JSON («"push",», «1,») внутри серии строк кода того же раздела — тоже код.

    Вперёд: значение после строки кода (внутри открытой структуры). Назад: значение перед
    строкой кода — только с запятой в конце («"a",»), то есть элемент перечня. Метка вроде
    «404» перед примером ответа запятой не оканчивается и кодом не становится.
    """
    for ordered, needs_comma in ((units, False), (units[::-1], True)):
        previous: _Unit | None = None
        for unit in ordered:
            neighbour_is_code = (
                previous is not None
                and previous.role == "code"
                and previous.section_path == unit.section_path
            )
            if (
                neighbour_is_code
                and unit.role != "code"
                and unit.kind in ("paragraph", "list_item")
                and looks_like_json_value(unit.text)
                and (not needs_comma or unit.text.rstrip().endswith(","))
            ):
                sections = [tag for tag in unit.detection if tag.startswith("section:")]
                unit.role, unit.confidence = "code", "high"
                unit.detection = [*sections, "code_sequence"]
                unit.warnings = []
            previous = unit


def _continues(previous: _Unit, block: Block) -> bool:
    """Абзац со строчной буквы продолжает предыдущий фрагмент без конечной пунктуации."""
    last = previous.text.rstrip()[-1:]
    return (
        block.kind == "paragraph"
        and previous.kind in ("paragraph", "list_item")
        and previous.role in _TEXTUAL
        and block.location.section_path == previous.section_path
        and bool(last)
        and last not in _SENTENCE_END
        and block.text[:1].islower()
        and parse_prefix(block.text, block.kind).doc_id is None
    )


# --- 2. связи -------------------------------------------------------------------------


def _link_enumerations(units: list[_Unit]) -> None:
    """Пункты списка сразу после фрагмента, оканчивающегося «:», — части его перечисления."""
    stem: _Unit | None = None
    previous: _Unit | None = None
    for unit in units:
        attached = (
            stem is not None
            and unit.kind == "list_item"
            and unit.prefix.doc_id is None
            and unit.section_path == stem.section_path
            and previous is not None
            and (previous is stem or previous.parent is stem)
        )
        if attached and stem is not None:
            _attach(unit, stem)
            unit.role, unit.category, unit.confidence = stem.role, stem.category, stem.confidence
            sections = [tag for tag in stem.detection if tag.startswith("section:")]
            unit.detection = [*sections, "enumeration"]
            unit.warnings = []
        ends_with_colon = unit.text.rstrip().endswith(":")
        if unit.kind in ("paragraph", "list_item") and unit.role in _TEXTUAL and ends_with_colon:
            stem = unit
        elif not attached:
            stem = None
        previous = unit


def _link_notes(units: list[_Unit]) -> None:
    """Блок-пример или блок-пояснение относится к ближайшему предыдущему требованию раздела."""
    for index, unit in enumerate(units):
        if unit.role not in ("example", "explanation") or unit.parent is not None:
            continue
        for previous in reversed(units[:index]):
            if previous.section_path != unit.section_path:
                break
            if previous.role in _ANALYZED:
                _attach(unit, previous)
                break


def _attach(unit: _Unit, parent: _Unit) -> None:
    unit.parent = parent
    parent.attached.append(unit)


# Предупреждения «вне раздела требований» неверны для текста под заголовком с ID требования.
_OUTSIDE_WARNINGS = frozenset({W_MODAL_OUTSIDE, W_SUBJECT_OUTSIDE, W_MODAL_REFERENCE})


@dataclass
class _HeadingScope:
    heading: _Unit
    path: tuple[str, ...]  # путь раздела, который открывает заголовок
    members: list[_Unit] = field(default_factory=list)  # фрагменты, для которых он ближайший
    has_content: bool = False  # есть ли в разделе (включая вложенные) хоть один фрагмент


def _link_heading_ids(units: list[_Unit]) -> list[str]:
    """ID документа в заголовке («### FR-1 Создание заказа») относится к требованию под ним.

    Область заголовка — фрагменты его раздела, включая вложенные подразделы; ближайший
    заголовок с ID важнее внешнего. Правила, без угадывания:
    - текст под заголовком с ID — требование (неоднозначный абзац становится требованием);
    - требование области со своим таким же ID несёт этот ID;
    - если своих ID в области нет, ID получает первое требование-абзац или пункт верхнего
      уровня, остальные требования области наследуют doc_id;
    - если свои ID другие, ID остаётся у заголовка как ID группы; свой ID, не вложенный в ID
      заголовка («FR-10.1» в «FR-10» — вложенный), — конфликт с предупреждением;
    - заголовок с ID без текста в разделе — общее предупреждение.
    Возвращает общие предупреждения сегментации.
    """
    scopes: list[_HeadingScope] = []
    for unit in units:
        if unit.kind == "heading":
            if unit.declared_id:
                scopes.append(_HeadingScope(unit, (*unit.section_path, one_line(unit.text))))
            continue
        closest: _HeadingScope | None = None
        for scope in scopes:
            if unit.section_path[: len(scope.path)] == scope.path:
                scope.has_content = True
                if closest is None or len(scope.path) >= len(closest.path):
                    closest = scope
        if closest is not None:
            closest.members.append(unit)
    warnings: list[str] = []
    for scope in scopes:
        warnings += _apply_heading_id(scope)
    return warnings


def _apply_heading_id(scope: _HeadingScope) -> list[str]:
    heading, declared = scope.heading, scope.heading.declared_id
    if declared is None:  # область создаётся только для заголовков с ID
        return []
    for unit in scope.members:
        if unit.role == "ambiguous" and unit.kind in ("paragraph", "list_item"):
            unit.role, unit.confidence = "requirement", "medium"
            unit.warnings = [w for w in unit.warnings if w not in _OUTSIDE_WARNINGS]
            _tag(unit, "doc_id:heading_scope")
    candidates = [unit for unit in scope.members if unit.role == "requirement"]
    own = [unit for unit in candidates if unit.id_base]
    inheriting = [unit for unit in candidates if not unit.id_base]
    bearer = next((unit for unit in own if unit.id_base == declared), None)
    if bearer is None and not own:
        bearer = next(
            (u for u in inheriting if u.kind in ("paragraph", "list_item") and u.parent is None),
            None,
        )
        if bearer is not None:
            bearer.id_base = bearer.doc_id = declared
            _tag(bearer, "doc_id:heading")
            inheriting.remove(bearer)
    for unit in inheriting:
        unit.doc_id = declared
        _tag(unit, "doc_id:heading_scope")
    for unit in own:
        if unit is not bearer and not _nested_id(unit.id_base, declared):
            unit.warnings.append(
                f"Идентификатор в тексте «{unit.id_base}» не совпадает "
                f"с идентификатором заголовка «{declared}»."
            )
    heading.doc_id = None if bearer is not None else declared
    if not scope.has_content:
        title = one_line(heading.text)
        return [f"Под заголовком «{title}» с идентификатором «{declared}» нет текста требования."]
    return []


def _nested_id(child: str | None, parent: str) -> bool:
    """«FR-10.1» вложен в «FR-10»; «FR-100» и «R-9» — нет."""
    return child is not None and child.startswith((f"{parent}.", f"{parent}-", f"{parent}_"))


def _tag(unit: _Unit, tag: str) -> None:
    if tag not in unit.detection:
        unit.detection.append(tag)


# --- 3. идентификаторы и дубли ----------------------------------------------------------


def content_id(role: Role, text: str) -> str:
    """Содержательный ID: R- для требований и неоднозначных фрагментов, F- для остальных."""
    letter = "R" if role in _ANALYZED else "F"
    return f"{letter}-{hashlib.sha256(content_key(text).encode('utf-8')).hexdigest()[:8]}"


def _assign_ids(units: list[_Unit]) -> None:
    used: dict[str, int] = {}
    first: dict[str, _Unit] = {}
    by_content: dict[str, _Unit] = {}
    for unit in units:
        key = content_key(unit.body)
        base = unit.id_base or content_id(unit.role, unit.body)
        count = used.get(base, 0) + 1
        used[base] = count
        unit.id = base if count == 1 else f"{base}~{count}"
        if count == 1:
            first[base] = unit
        elif unit.id_base and content_key(first[base].body) != key:
            unit.warnings.append(
                f"Идентификатор «{unit.id_base}» уже использован в документе ({first[base].id})."
            )
        if unit.role in _ANALYZED and key:
            original = by_content.setdefault(key, unit)
            if original is not unit:
                unit.duplicate_of = original.id
                unit.warnings.append(f"Дословно повторяет {original.id}.")


# --- 4. деление на утверждения -----------------------------------------------------------


def _split_statements(units: list[_Unit]) -> None:
    for unit in units:
        if unit.role != "requirement" or unit.kind not in ("paragraph", "list_item"):
            continue
        statements = split_statements(unit.text, unit.prefix.body_start)
        _check_statements(statements, unit)
        if len(statements) > 1:
            unit.statements = statements


def _check_statements(statements: tuple[Statement, ...], unit: _Unit) -> None:
    previous_end = unit.prefix.body_start
    for statement in statements:
        if not previous_end <= statement.start < statement.end <= len(unit.text):
            raise SegmentationError(
                f"Некорректные границы утверждения {statement.start}–{statement.end} "
                f"во фрагменте «{unit.text[:40]}»."
            )
        previous_end = statement.end


# --- 5. сборка Requirement ----------------------------------------------------------------


def _materialize(units: list[_Unit], document: Document) -> list[Requirement]:
    blocks = document.blocks
    doc_ids = {unit.doc_id for unit in units if unit.doc_id}
    headings = [block.text for block in blocks if block.kind == "heading"]

    def references(role: Role, text: str, own_id: str | None) -> tuple[str, ...]:
        if role not in _ANALYZED:
            return ()
        return find_references(text, doc_ids=doc_ids, headings=headings, own_id=own_id)

    fragments: list[Requirement] = []
    for unit in units:
        statement_ids = [f"{unit.id}/{k}" for k in range(1, len(unit.statements) + 1)]
        whole = Requirement(
            id=unit.id,
            text=unit.text,
            normalized=normalize_text(unit.body),
            role=unit.role,
            category=unit.category,
            sources=tuple(
                SourceRef(p.block, 0, p.length, source_location(blocks[p.block], 0, p.length))
                for p in unit.pieces
            ),
            section_path=unit.section_path,
            doc_id=unit.doc_id,
            doc_number=unit.prefix.number,
            parent_id=unit.parent.id if unit.parent else None,
            children=(*statement_ids, *(child.id for child in unit.attached)),
            duplicate_of=unit.duplicate_of,
            references=references(unit.role, unit.text, unit.doc_id),
            detection=tuple(unit.detection),
            confidence=unit.confidence,
            warnings=tuple(unit.warnings),
        )
        fragments.append(whole)
        for statement_id, statement in zip(statement_ids, unit.statements, strict=True):
            text = unit.text[statement.start : statement.end]
            try:
                sources = span_sources(whole, statement.start, statement.end, document)
            except ValueError as exc:  # ошибка агента, а не документа — не пропускаем молча
                raise SegmentationError(f"Утверждение {statement_id}: {exc}") from exc
            fragments.append(
                Requirement(
                    id=statement_id,
                    text=text,
                    normalized=normalize_text(text),
                    role=statement.role,
                    category=unit.category,
                    sources=sources,
                    section_path=unit.section_path,
                    doc_id=unit.doc_id,
                    doc_number=unit.prefix.number,
                    parent_id=unit.id,
                    references=references(statement.role, text, unit.doc_id),
                    detection=(*unit.detection, "sentence_split"),
                    confidence=unit.confidence,
                )
            )
    return fragments


# --- 6. похожие требования и общие предупреждения ------------------------------------------


def _link_similar(fragments: list[Requirement]) -> list[Requirement]:
    containers = {f.parent_id for f in fragments if f.role == "requirement" and f.parent_id}
    candidates = [
        f
        for f in fragments
        if f.role == "requirement" and f.id not in containers and f.duplicate_of is None
    ]
    keys = [content_key(f.normalized) for f in candidates]
    found: dict[str, list[tuple[str, float]]] = {}
    for i, a in enumerate(keys):
        for j in range(i + 1, len(keys)):
            b = keys[j]
            shorter = min(len(a), len(b))
            # верхняя граница сходства по длинам: 2·min / (len a + len b)
            best_possible = 2 * shorter / (len(a) + len(b))
            if a == b or shorter < _MIN_SIMILAR_CHARS or best_possible < SIMILARITY_THRESHOLD:
                continue
            matcher = SequenceMatcher(None, a, b, autojunk=False)
            if matcher.quick_ratio() < SIMILARITY_THRESHOLD:
                continue
            ratio = matcher.ratio()
            if ratio >= SIMILARITY_THRESHOLD:
                found.setdefault(candidates[i].id, []).append((candidates[j].id, ratio))
                found.setdefault(candidates[j].id, []).append((candidates[i].id, ratio))
    if not found:
        return fragments
    result: list[Requirement] = []
    for fragment in fragments:
        similar = found.get(fragment.id)
        if similar:
            notes = tuple(f"Похоже на {other} (сходство {ratio:.0%})." for other, ratio in similar)
            fragment = replace(
                fragment,
                similar_to=tuple(other for other, _ in similar),
                warnings=(*fragment.warnings, *notes),
            )
        result.append(fragment)
    return result


def _global_warnings(fragments: list[Requirement]) -> tuple[str, ...]:
    warnings: list[str] = []
    if not any(fragment.role == "requirement" for fragment in fragments):
        warnings.append(W_NO_REQUIREMENTS)
    ambiguous = sum(1 for fragment in fragments if fragment.role == "ambiguous")
    if ambiguous:
        warnings.append(
            f"Неоднозначных фрагментов: {ambiguous} — нужно проверить, требования это или нет."
        )
    return tuple(warnings)
