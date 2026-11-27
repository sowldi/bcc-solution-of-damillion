"""Структурные правила: полнота (CMP-03…07), согласованность (CON), трассируемость (TRC).

Правила опираются на структуру сегментации (ID, дубли, похожие формулировки, ссылки,
заголовки, таблицы), а не на тексты предупреждений сегментатора.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from difflib import SequenceMatcher

from validator.findings import AbsenceClaim
from validator.ingest.common import one_line
from validator.lint.base import Hit, LintContext, Span
from validator.model import Block
from validator.segmentation import Requirement
from validator.segmentation.text import content_key, is_document_id, locate_references, parse_prefix

_EMPTY_VALUES = frozenset({"", "—", "–", "-"})
_LEAD_LIMIT = 80


def _whole(context: LintContext, fragment: Requirement) -> Span:
    """Весь содержательный текст фрагмента (без ID в начале)."""
    return Span(fragment, context.body_start(fragment), len(fragment.text))


def _lead(context: LintContext, fragment: Requirement) -> Span:
    """Начало формулировки — не длиннее _LEAD_LIMIT символов, по границе слова."""
    start = context.body_start(fragment)
    end = len(fragment.text)
    if end - start > _LEAD_LIMIT:
        cut = fragment.text.rfind(" ", start, start + _LEAD_LIMIT)
        if cut <= start:  # первое слово длиннее лимита — цитата не обрывается посреди слова
            cut = fragment.text.find(" ", start + _LEAD_LIMIT)
        end = cut if cut > start else end
    return Span(fragment, start, end)


def _filled(cell: str) -> bool:
    return cell.strip() not in _EMPTY_VALUES


# --- C. Полнота ------------------------------------------------------------------------

# «Метка: —» в конце строки, ячейки или фразы; метка — до четырёх слов, не часть пути «/a.b».
_LABEL_DASH = re.compile(
    r"(?<![\w/.-])(?P<label>[^\W\d_][\w-]*(?:[ \t.][\w-]+){0,3}):[ \t]*[—–-]"
    r"(?=[ \t]*(?:$|\||[;,](?:\s|$)|\.(?:\s|$)))",
    re.MULTILINE,
)
# Строка из двух столбцов через табуляцию (так TXT, сохранённый из Word, передаёт таблицу).
_TAB_ROW_DASH = re.compile(r"^(?P<label>[^\t\n]*\S)\t+[—–-][ \t]*$", re.MULTILINE)


def _unfilled(fragment: Requirement, start: int, end: int, label: str) -> Hit:
    return Hit(
        (Span(fragment, start, end),),
        f"Значение «{one_line(label).strip()}» не заполнено (пусто или прочерк): "
        "неясно, «значения нет» это или пропуск.",
        absence=AbsenceClaim("requirement", fragment.id, "value"),
    )


def cmp03(context: LintContext) -> Iterator[Hit]:
    for fragment in context.top_level:
        block = context.block(fragment)
        text = fragment.text
        if block is None or not text or fragment.role == "code":
            continue
        if block.kind == "table_row" and len(block.cells) == 2:
            key, value = block.cells
            if _filled(key) and not _filled(value):
                yield _unfilled(fragment, 0, len(text), key)
                continue
        elif block.kind in ("paragraph", "list_item"):
            for row in _TAB_ROW_DASH.finditer(text):
                yield _unfilled(fragment, row.start(), row.end(), row.group("label"))
        elif block.kind != "table_row":
            continue
        for match in _LABEL_DASH.finditer(text):
            yield _unfilled(fragment, match.start(), match.end(), match.group("label"))


def _continued(context: LintContext, unit: Requirement) -> bool:
    """После фрагмента в его разделе (или подразделе) есть продолжение: пункты, таблица,
    текст, вложенный заголовок."""
    if any("/" not in child for child in unit.children):
        return True  # пункты перечисления, пример, пояснение
    index = context.order[unit.id]
    path = unit.section_path
    for fragment in context.segmentation.fragments[index + 1 :]:
        if fragment.is_statement:
            continue
        return fragment.section_path[: len(path)] == path
    return False


_ELLIPSIS_END = re.compile(r"(?:\.{3}|…)\s*$")
_COMMA_ELLIPSIS_END = re.compile(r",\s*(?:\.{3}|…)\s*$")  # открытое перечисление — CMP-02
_DANGLING = re.compile(
    r"(?<![\w-])(?P<word>и|или|либо|а\s+также|если|когда|чтобы|что|для|при|из|от|до|на|по"
    r"|к|о|об|and|or|if|when|to|for|of|with)\s*$",
    re.IGNORECASE,
)


def cmp04(context: LintContext) -> Iterator[Hit]:
    for fragment in context.targets:
        if context.kind(fragment) not in ("paragraph", "list_item"):
            continue
        text = fragment.text.rstrip()
        span = Span(fragment, context.body_start(fragment), len(text))
        if span.start >= span.end:
            continue
        unit = (
            context.segmentation.by_id(fragment.parent_id)
            if fragment.is_statement and fragment.parent_id
            else fragment
        )
        if text.endswith(":"):
            if not _continued(context, unit):
                yield Hit(
                    (span,),
                    "Формулировка заканчивается двоеточием, но перечисления после неё в разделе "
                    "нет.",
                )
        elif _ELLIPSIS_END.search(text) and not _COMMA_ELLIPSIS_END.search(text):
            yield Hit((span,), "Формулировка обрывается многоточием — мысль не закончена.")
        elif match := _DANGLING.search(text):
            yield Hit(
                (span,),
                f"Формулировка обрывается на слове «{match.group('word')}» — текст не дописан.",
            )


@dataclass(frozen=True)
class _Table:
    number: int
    header: list[int]  # индексы блоков
    rows: list[int]


def _tables(context: LintContext) -> list[_Table]:
    tables: dict[int, _Table] = {}
    for index, block in enumerate(context.document.blocks):
        number = block.location.table
        if number is None or block.kind not in ("table_header", "table_row"):
            continue
        table = tables.setdefault(number, _Table(number, [], []))
        (table.header if block.kind == "table_header" else table.rows).append(index)
    return list(tables.values())


def _row_filled(block: Block) -> bool:
    cells = block.cells or (block.text,)
    return any(_filled(cell) for cell in cells)


def cmp05(context: LintContext) -> Iterator[Hit]:
    blocks = context.document.blocks
    for table in _tables(context):
        if not table.header or any(_row_filled(blocks[i]) for i in table.rows):
            continue
        spans: list[Span] = []
        for index in table.header + table.rows:
            fragment = context.owner(index)
            if fragment is None:
                continue
            spans.append(Span(fragment, 0, len(fragment.text)))
        if spans:
            yield Hit(
                tuple(spans),
                f"В таблице нет заполненных строк данных (строк: {len(table.rows)}): "
                "заполнена только шапка.",
                absence=AbsenceClaim("table", str(table.number), "data"),
            )


_SECTION_WORD = re.compile(
    r"(?P<word>(?:под)?раздел\w*|глав[аеуы]|section|chapter)\s*(?P<num>\d+(?:\.\d+)*)",
    re.IGNORECASE,
)
_HEADING_NUMBER = re.compile(r"\s*(\d+(?:\.\d+)*)\.?\s")


def _unresolved(context: LintContext) -> Iterator[tuple[Requirement, str, int, int]]:
    for fragment in context.targets:
        for ref in locate_references(
            fragment.text,
            doc_ids=context.doc_ids,
            headings=context.headings,
            own_id=fragment.doc_id,
        ):
            if ref.label.startswith("unresolved:"):
                yield fragment, ref.label.removeprefix("unresolved:"), ref.start, ref.end


def cmp06(context: LintContext) -> Iterator[Hit]:
    depths = {
        len(match.group(1).split("."))
        for title in context.headings
        if (match := _HEADING_NUMBER.match(one_line(title) + " "))
    }
    for fragment, target, start, end in _unresolved(context):
        if is_document_id(target):
            continue  # ссылка на требование — TRC-03
        match = _SECTION_WORD.fullmatch(one_line(fragment.text[start:end]))
        if match is None or len(match.group("num").split(".")) not in depths:
            continue  # «пункт 3» или документ без нумерации разделов такой глубины
        yield Hit(
            (Span(fragment, start, end),),
            f"Ссылка «{one_line(fragment.text[start:end])}»: раздела с номером "
            f"{match.group('num')} в документе нет.",
            absence=AbsenceClaim("heading_registry", match.group("num"), "heading"),
        )


def _declared_id(context: LintContext, fragment: Requirement) -> str | None:
    """ID документа в заголовке («### Ф-1 Бронирование»)."""
    if context.kind(fragment) != "heading":
        return None
    return parse_prefix(fragment.text, "heading").doc_id


def _heading_path(fragment: Requirement) -> tuple[str, ...]:
    return (*fragment.section_path, one_line(fragment.text))


def cmp07(context: LintContext) -> Iterator[Hit]:
    for heading in context.top_level:
        declared = _declared_id(context, heading)
        if declared is None:
            continue
        path = _heading_path(heading)
        has_text = any(
            context.kind(f) != "heading" and f.section_path[: len(path)] == path
            for f in context.top_level
        )
        if not has_text:
            yield Hit(
                (Span(heading, 0, len(heading.text)),),
                f"Под заголовком с идентификатором «{declared}» нет текста требования: "
                "идентификатор есть, требования нет.",
                absence=AbsenceClaim("section", " › ".join(path), "text"),
            )


# --- D. Согласованность ------------------------------------------------------------------


def con01(context: LintContext) -> Iterator[Hit]:
    for fragment in context.top_level:
        if fragment.duplicate_of is None:
            continue
        original = context.segmentation.by_id(fragment.duplicate_of)
        if "requirement" not in (fragment.role, original.role):
            continue
        yield Hit(
            (_whole(context, original), _whole(context, fragment)),
            f"Требование {fragment.id} дословно повторяет {original.id}: при изменении одного "
            "второе легко забыть.",
        )


def _differences(a: str, b: str, limit: int = 3) -> str:
    left, right = a.split(), b.split()
    parts: list[str] = []
    matcher = SequenceMatcher(None, left, right, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        old = " ".join(left[i1:i2])
        new = " ".join(right[j1:j2])
        parts.append(f"«{old}» → «{new}»" if old and new else f"«{old or new}»")
    shown = "; ".join(parts[:limit])
    return shown + ("; …" if len(parts) > limit else "")


def con02(context: LintContext) -> Iterator[Hit]:
    seen: set[tuple[str, str]] = set()
    by_id = context.segmentation.by_id
    for fragment in context.targets:
        for other_id in fragment.similar_to:
            a, b = sorted((fragment, by_id(other_id)), key=lambda f: context.order[f.id])
            if (a.id, b.id) in seen:
                continue
            seen.add((a.id, b.id))
            ratio = SequenceMatcher(
                None, content_key(a.normalized), content_key(b.normalized), autojunk=False
            ).ratio()
            yield Hit(
                (_whole(context, a), _whole(context, b)),
                f"Формулировки {a.id} и {b.id} совпадают на {ratio:.0%}; различия: "
                f"{_differences(a.normalized, b.normalized)}. Проверьте, не дубль ли это "
                "и согласованы ли различия.",
            )


_VALUE_HEADER = re.compile(r"значени|value|величин", re.IGNORECASE)


def con03(context: LintContext) -> Iterator[Hit]:
    blocks = context.document.blocks
    for table in _tables(context):
        rows = [i for i in table.rows if len(blocks[i].cells) == 2]
        header = blocks[rows[0]].header if rows else ()
        if len(header) != 2 or not _VALUE_HEADER.search(header[1]):
            continue  # только таблицы «параметр — значение»
        groups: dict[str, list[int]] = {}
        for index in rows:
            key, value = blocks[index].cells
            if _filled(key) and _filled(value):
                groups.setdefault(content_key(key), []).append(index)
        for members in groups.values():
            values = {content_key(blocks[i].cells[1]) for i in members}
            if len(values) < 2:
                continue
            spans = []
            for index in members:
                fragment = context.owner(index)
                if fragment is not None:
                    spans.append(Span(fragment, 0, len(fragment.text)))
            key_text = one_line(blocks[members[0]].cells[0]).strip()
            shown = ", ".join(f"«{one_line(blocks[i].cells[1]).strip()}»" for i in members)
            yield Hit(
                tuple(spans),
                f"Параметр «{key_text}» в одной таблице имеет разные значения: {shown} — "
                "неясно, какое действует.",
            )


# --- E. Трассируемость -------------------------------------------------------------------


def trc01(context: LintContext) -> Iterator[Hit]:
    groups: dict[tuple[object, ...], list[Requirement]] = {}
    for fragment in context.top_level:
        if fragment.role != "requirement" or fragment.parent_id is not None:
            continue
        block = context.block(fragment)
        if block is None:
            continue
        if block.kind == "table_row":
            family: tuple[object, ...] = ("table", block.location.table)
        elif block.kind in ("paragraph", "list_item"):
            family = ("text",)
        else:
            continue
        groups.setdefault((fragment.section_path, *family), []).append(fragment)
    for members in groups.values():
        numbered = [f for f in members if f.doc_id]
        bare = [f for f in members if not f.doc_id]
        if len(numbered) < 2 or len(numbered) <= len(bare):
            continue
        for fragment in bare:
            yield Hit(
                (_lead(context, fragment),),
                f"Рядом {len(numbered)} требований с идентификаторами, у этого идентификатора "
                "нет: на него нельзя сослаться и проследить его реализацию.",
                absence=AbsenceClaim("requirement", fragment.id, "document_id"),
            )


_ID_SHAPE = re.compile(r"^(?P<base>[A-ZА-ЯЁ]{1,6}[-_.]?\d{1,4})(?P<levels>(?:\.\d{1,3})*)$")


def _id_pattern(doc_id: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w-]){re.escape(doc_id)}(?![\w-]|\.\d)")


def _id_span(context: LintContext, fragment: Requirement, doc_id: str) -> Span | None:
    """Где в тексте стоит ID: в самом фрагменте или в заголовке, от которого он получен."""
    pattern = _id_pattern(doc_id)
    match = pattern.search(fragment.text)
    if match:
        return Span(fragment, match.start(), match.end())
    index = context.order[fragment.id]
    for previous in reversed(context.segmentation.fragments[:index]):
        if _declared_id(context, previous) == doc_id:
            heading_match = pattern.search(previous.text)
            if heading_match:
                return Span(previous, heading_match.start(), heading_match.end())
    return None


def _declarations(context: LintContext) -> dict[str, list[Requirement]]:
    """Где ID документа объявлен: фрагменты, для которых он собственный ID, и заголовки."""
    found: dict[str, list[Requirement]] = {}
    for fragment in context.top_level:
        declared = _declared_id(context, fragment)
        if declared is None and fragment.doc_id and fragment.id.split("~")[0] == fragment.doc_id:
            declared = fragment.doc_id
        if declared:
            found.setdefault(declared, []).append(fragment)
    return found


def trc02(context: LintContext) -> Iterator[Hit]:
    for doc_id, members in _declarations(context).items():
        distinct: list[Requirement] = []
        keys: set[str] = set()
        for fragment in members:
            if _declared_id(context, fragment):
                continue  # заголовок объявляет ID группы, а не отдельное требование
            key = content_key(fragment.text[context.body_start(fragment) :])
            if key not in keys:
                keys.add(key)
                distinct.append(fragment)
        if len(distinct) < 2:
            continue
        spans = [span for f in distinct if (span := _id_span(context, f, doc_id))]
        yield Hit(
            tuple(spans),
            f"Идентификатор «{doc_id}» присвоен разным требованиям "
            f"({', '.join(f.id for f in distinct)}): ссылка на него неоднозначна.",
        )


def trc03(context: LintContext) -> Iterator[Hit]:
    for fragment, target, start, end in _unresolved(context):
        if is_document_id(target):
            yield Hit(
                (Span(fragment, start, end),),
                f"Ссылка на «{target}»: требования с таким идентификатором в документе нет.",
                absence=AbsenceClaim("id_registry", target, "declaration"),
            )


_LATIN = re.compile(r"[A-Z]")
_CYRILLIC = re.compile(r"[А-ЯЁ]")


def trc04(context: LintContext) -> Iterator[Hit]:
    for doc_id, members in _declarations(context).items():
        letters = re.match(r"[A-ZА-ЯЁ]+", doc_id)
        prefix = letters.group(0) if letters else ""
        if not (_LATIN.search(prefix) and _CYRILLIC.search(prefix)):
            continue
        span = _id_span(context, members[0], doc_id)
        if span:
            yield Hit(
                (span,),
                f"В идентификаторе «{doc_id}» смешаны латинские и кириллические буквы: поиск "
                "и ссылки по нему не совпадут с написанным другой раскладкой.",
            )


def _nested(child: str, parent: str) -> bool:
    return child.startswith((f"{parent}.", f"{parent}-", f"{parent}_"))


def trc05(context: LintContext) -> Iterator[Hit]:
    scopes = [
        (_heading_path(f), declared, f)
        for f in context.top_level
        if (declared := _declared_id(context, f))
    ]
    for fragment in context.top_level:
        own = fragment.doc_id
        if fragment.role != "requirement" or "explicit_id" not in fragment.detection or not own:
            continue
        closest = max(
            (scope for scope in scopes if fragment.section_path[: len(scope[0])] == scope[0]),
            key=lambda scope: len(scope[0]),
            default=None,
        )
        if closest is None:
            continue
        _, declared, heading = closest
        if own == declared or _nested(own, declared):
            continue
        heading_span = _id_span(context, heading, declared)
        own_span = _id_span(context, fragment, own)
        if heading_span and own_span:
            yield Hit(
                (heading_span, own_span),
                f"Заголовок задаёт идентификатор «{declared}», а требование под ним — «{own}»: "
                "неясно, по какому идентификатору на него ссылаться.",
            )


def trc06(context: LintContext) -> Iterator[Hit]:
    declared = _declarations(context)
    known = set(declared) | set(context.doc_ids)
    children: dict[str, list[str]] = {}
    for doc_id in declared:
        shape = _ID_SHAPE.match(doc_id)
        if not shape or not shape.group("levels"):
            continue
        parent = doc_id.rsplit(".", 1)[0]
        if parent not in known:
            children.setdefault(parent, []).append(doc_id)
    for parent, ids in children.items():
        first = ids[0]
        span = _id_span(context, declared[first][0], first)
        if span:
            yield Hit(
                (span,),
                f"Идентификатор «{first}» подразумевает родительский «{parent}», но такого "
                f"идентификатора в документе нет (дочерних идентификаторов: {len(ids)}).",
                absence=AbsenceClaim("id_registry", parent, "declaration"),
            )
