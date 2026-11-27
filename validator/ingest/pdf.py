"""PDF (с текстовым слоем) → блоки с номером страницы и строк на странице.

- Порядок чтения: сверху вниз (одноколоночная вёрстка).
- Строки на странице считаются визуально сверху вниз, включая колонтитулы.
- Колонтитулы (повторяются на большинстве страниц у края) и номера страниц убираются.
- Заголовки: крупнее основного текста или целиком жирные короткие строки с буквами.
- Таблицы: распознаются по линиям сетки (PyMuPDF find_tables); первая строка — шапка.
  Таблица, продолженная в начале следующей страницы, считается той же таблицей.
- OCR не выполняется: PDF без текстового слоя (скан) отклоняется.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

from validator.errors import InputError
from validator.ingest.common import Extraction, SectionTracker, clean_text, one_line, row_text
from validator.model import Block, BlockKind, Location

pymupdf.no_recommend_layout()  # иначе find_tables() печатает рекламную подсказку в stdout
pymupdf.TOOLS.mupdf_display_errors(False)  # ошибки MuPDF не печатаются, а превращаются в InputError
pymupdf.TOOLS.mupdf_display_warnings(False)

_DICT_FLAGS = (
    pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES & ~pymupdf.TEXT_PRESERVE_LIGATURES
)
_CELL_FLAGS = pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES
_MARGIN = 0.07  # доля высоты страницы сверху и снизу, где ищутся колонтитулы
_HEADING_SCALE = 1.15  # во сколько раз заголовок крупнее основного текста
_SIZE_JUMP = 1.10  # смена размера шрифта внутри блока делит его на части
_BOLD_SHARE = 0.8
_MAX_HEADING_LINES = 3
_MAX_HEADING_CHARS = 200
_MIN_HEADING_LETTERS = 2  # «404» жирным — метка, а не заголовок
_SENTENCE_END = (".", ";", ":", ",")
_BOLD_FONT = re.compile(r"bold|black|heavy|semibold|demi", re.IGNORECASE)
_PAGE_NUMBER = re.compile(
    r"\s*(?:(?:стр(?:аница)?|с|page|p)\.?\s*)?\d+(?:\s*(?:из|of|/)\s*\d+)?\s*", re.IGNORECASE
)
# маркеры списков, включая символы шрифтов Symbol/Wingdings из Word (U+F0B7 и др.)
_GLYPH_BULLET = re.compile(r"^\s*[•●○◦▪■□‣⁃∙·\uf0b7\uf0a7\uf0d8\uf076\uf0fc]\s*(?=\S)")
_DASH_BULLET = re.compile(r"^\s*[-*–—]\s+(?=\S)")
# номер пункта — одна-две цифры: «200. Ответ …» описывает HTTP-статус, а не пункт списка
_NUMBERED_ITEM = re.compile(r"^\s*\d{1,2}[.)]\s+(?=\S)")

BBox = tuple[float, float, float, float]


@dataclass
class _Line:
    bbox: BBox
    text: str
    size: float  # размер шрифта, преобладающий в строке
    chars: int
    bold_chars: int
    group: int  # номер текстового блока PyMuPDF на странице
    number: int = 0  # визуальная строка на странице, с единицы


@dataclass
class _Table:
    """Таблица страницы; шапка — всегда первая строка внутри рамки.

    «Внешнюю шапку» PyMuPDF (текст над таблицей) не используем: эвристика ошибается, например
    принимает заголовок раздела над таблицей за шапку, а сам этот текст и так попадает в блоки.
    """

    bbox: BBox
    rows: list[tuple[str, ...]]
    row_lines: list[tuple[int | None, int | None]]


@dataclass
class _Page:
    number: int
    height: float
    lines: list[_Line]
    tables: list[_Table]
    has_images: bool
    flow: list[_Line] = field(default_factory=list)  # строки вне таблиц и колонтитулов


def read_pdf(data: bytes, source: Path) -> Extraction:
    not_pdf = InputError(f"PDF повреждён или не является PDF-файлом: {source}")
    # MuPDF определяет формат по содержимому: картинку или HTML с расширением .pdf он откроет
    # как документ другого типа, поэтому сигнатуру PDF проверяем сами (спецификация допускает
    # её в первых 1024 байтах).
    if b"%PDF-" not in data[:1024]:
        raise not_pdf
    pymupdf.TOOLS.mupdf_warnings()  # сбросить накопленные предупреждения MuPDF
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except pymupdf.FileDataError as exc:
        raise not_pdf from exc
    warnings: list[str] = []
    with document:
        if not document.is_pdf:
            raise not_pdf
        if document.needs_pass:
            raise InputError(f"PDF защищён паролем: {source}. Снимите защиту и повторите.")
        page_count = document.page_count
        mupdf_log = pymupdf.TOOLS.mupdf_warnings()
        pages: list[_Page] = []
        for index in range(page_count):
            try:
                page = document.load_page(index)
            except Exception as exc:  # битая страница не должна ронять весь документ
                warnings.append(f"Страница {index + 1} не прочитана: {exc}")
                continue
            pages.append(_read_page(page, warnings))
        mupdf_log += pymupdf.TOOLS.mupdf_warnings()
    if page_count == 0:
        raise InputError(f"В PDF нет ни одной страницы: {source}")
    if "repair" in mupdf_log.lower():
        warnings.insert(
            0,
            "PDF был повреждён и восстановлен автоматически: часть текста могла не прочитаться.",
        )
    _drop_running_lines(pages)
    for page in pages:
        page.flow = [
            line
            for line in page.lines
            if not any(_inside(line.bbox, table.bbox) for table in page.tables)
        ]
        if not page.lines and not page.tables and page.has_images:
            warnings.append(
                f"Страница {page.number}: нет текстового слоя (возможно, скан) — текст не извлечён."
            )
    blocks = _assemble(pages)
    if not any(block.text.strip() for block in blocks):
        raise InputError(
            f"В PDF нет текстового слоя (похоже на скан). "
            f"Распознавание текста (OCR) не поддерживается: {source}"
        )
    return Extraction(blocks=tuple(blocks), page_count=page_count, warnings=tuple(warnings))


# --- чтение страницы -----------------------------------------------------------


def _read_page(page: pymupdf.Page, warnings: list[str]) -> _Page:
    number = page.number + 1
    height = float(page.rect.height)
    try:
        content = page.get_text("dict", flags=_DICT_FLAGS, sort=True)
    except Exception as exc:  # повреждённое содержимое одной страницы не должно ронять документ
        warnings.append(f"Страница {number} не прочитана: {exc}")
        return _Page(number, height, [], [], has_images=False)
    lines = _page_lines(content)
    _number_lines(lines)
    tables = _page_tables(page, lines, warnings)
    return _Page(number, height, lines, tables, has_images=bool(page.get_images()))


def _page_lines(content: dict[str, Any]) -> list[_Line]:
    lines: list[_Line] = []
    for group, block in enumerate(content["blocks"]):
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            spans = [span for span in line["spans"] if span["text"].strip()]
            text = clean_text("".join(span["text"] for span in line["spans"])).strip()
            if not spans or not text:
                continue
            main = max(spans, key=lambda span: len(span["text"].strip()))
            lines.append(
                _Line(
                    bbox=_bbox(line["bbox"]),
                    text=text,
                    size=float(main["size"]),
                    chars=sum(len(span["text"].strip()) for span in spans),
                    bold_chars=sum(len(span["text"].strip()) for span in spans if _is_bold(span)),
                    group=group,
                )
            )
    return lines


def _is_bold(span: dict[str, Any]) -> bool:
    return bool(span["flags"] & pymupdf.TEXT_FONT_BOLD) or bool(_BOLD_FONT.search(span["font"]))


def _number_lines(lines: list[_Line]) -> None:
    """Визуальные строки: фрагменты на одной высоте (ячейки таблицы и т.п.) — одна строка."""
    number = 0
    anchor = -math.inf
    for line in sorted(lines, key=lambda item: (item.bbox[1], item.bbox[0])):
        height = line.bbox[3] - line.bbox[1]
        if line.bbox[1] - anchor > max(2.0, 0.5 * height):
            number += 1
            anchor = line.bbox[1]
        line.number = number


def _page_tables(page: pymupdf.Page, lines: list[_Line], warnings: list[str]) -> list[_Table]:
    try:
        found = page.find_tables().tables
    except Exception as exc:  # сбой распознавания таблиц — читаем их текст как обычный
        warnings.append(f"Страница {page.number + 1}: таблицы не распознаны ({exc}).")
        return []
    tables: list[_Table] = []
    for table in found:
        rows: list[tuple[str, ...]] = []
        row_lines: list[tuple[int | None, int | None]] = []
        for row in table.rows:
            # Текст ячейки берётся по её рамке через обычное извлечение текста: table.extract()
            # уносит «_» в отдельную строку («limit_kzt» → «limit kzt _»).
            cells = tuple(_cell_text(page, cell) for cell in row.cells if cell is not None)
            if not cells:
                continue
            rows.append(cells)
            inside = [line.number for line in lines if _inside(line.bbox, _bbox(row.bbox))]
            row_lines.append((min(inside), max(inside)) if inside else (None, None))
        if rows:
            tables.append(_Table(_bbox(table.bbox), rows, row_lines))
    return tables


def _cell_text(page: pymupdf.Page, cell: Any) -> str:
    """Переносы внутри ячейки PDF — следствие вёрстки, а не смысла: склеиваем пробелом."""
    text = page.get_text("text", clip=pymupdf.Rect(cell), flags=_CELL_FLAGS)
    return one_line(clean_text(str(text)))


def _bbox(value: Any) -> BBox:
    x0, y0, x1, y1 = (float(v) for v in value)
    return x0, y0, x1, y1


def _inside(inner: BBox, outer: BBox) -> bool:
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] - 1 <= cx <= outer[2] + 1 and outer[1] - 1 <= cy <= outer[3] + 1


# --- колонтитулы -----------------------------------------------------------------


def _in_margin(line: _Line, height: float) -> bool:
    return line.bbox[3] <= height * _MARGIN or line.bbox[1] >= height * (1 - _MARGIN)


def _running_key(text: str) -> str:
    return re.sub(r"\d+", "#", one_line(text).lower())


def _drop_running_lines(pages: list[_Page]) -> None:
    """Убрать номера страниц и строки у края, повторяющиеся на большинстве страниц."""
    seen: Counter[str] = Counter()
    for page in pages:
        seen.update({_running_key(ln.text) for ln in page.lines if _in_margin(ln, page.height)})
    threshold = max(2, math.ceil(len(pages) / 2))
    for page in pages:
        page.lines = [
            line
            for line in page.lines
            if not (
                _in_margin(line, page.height)
                and (
                    _PAGE_NUMBER.fullmatch(line.text) or seen[_running_key(line.text)] >= threshold
                )
            )
        ]


# --- сборка блоков ---------------------------------------------------------------


Entry = tuple[int, list[_Line] | _Table]  # (номер страницы, фрагмент текста или таблица)


@dataclass
class _TableState:
    """Последняя выведенная таблица — чтобы продолжить её на следующей странице."""

    number: int
    columns: int
    header: tuple[str, ...]
    last_row: int


def _assemble(pages: list[_Page]) -> list[Block]:
    body = _body_size(pages)
    entries: list[Entry] = []
    for page in pages:
        placed: list[tuple[float, float, list[_Line] | _Table]] = [
            (chunk[0].bbox[1], chunk[0].bbox[0], chunk) for chunk in _chunks(page.flow)
        ]
        placed += [(table.bbox[1], table.bbox[0], table) for table in page.tables]
        placed.sort(key=lambda item: (item[0], item[1]))
        entries.extend((page.number, item) for _, _, item in placed)

    headings = {
        index: entry
        for index, (_, entry) in enumerate(entries)
        if isinstance(entry, list) and _is_heading(entry, body)
    }
    levels = _heading_levels(headings, body)

    sections = SectionTracker()
    blocks: list[Block] = []
    state: _TableState | None = None
    for index, (page_number, entry) in enumerate(entries):
        if isinstance(entry, _Table):
            if state is not None and _continues_previous_table(entries, index, state):
                blocks.extend(_continued_rows(entry, page_number, state, sections.path))
            else:
                number = state.number + 1 if state is not None else 1
                state = _TableState(number, len(entry.rows[0]), header=(), last_row=0)
                blocks.extend(_new_table(entry, page_number, state, sections.path))
        elif index in levels:
            title = "\n".join(line.text for line in entry)
            parent = sections.enter(levels[index], title)
            location = _location(parent, page_number, entry)
            blocks.append(Block("heading", title, location, level=levels[index]))
        else:
            blocks.extend(_flow_blocks(entry, page_number, sections.path))
    return blocks


def _body_size(pages: list[_Page]) -> float:
    """Основной размер шрифта: тот, которым набрано больше всего символов вне таблиц."""
    sizes: Counter[float] = Counter()
    for page in pages:
        for line in page.flow:
            sizes[_round_size(line.size)] += line.chars
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _round_size(size: float) -> float:
    return round(size * 2) / 2


def _chunks(lines: list[_Line]) -> list[list[_Line]]:
    """Текстовые блоки PyMuPDF, разделённые ещё и по резкой смене шрифта внутри блока."""
    chunks: list[list[_Line]] = []
    for line in lines:
        if chunks:
            previous = chunks[-1][-1]
            same_group = previous.group == line.group
            if same_group and not _style_changes(previous, line):
                chunks[-1].append(line)
                continue
        chunks.append([line])
    return chunks


def _style_changes(previous: _Line, line: _Line) -> bool:
    bigger, smaller = max(previous.size, line.size), min(previous.size, line.size)
    if smaller > 0 and bigger / smaller >= _SIZE_JUMP:
        return True
    return _mostly_bold([previous]) != _mostly_bold([line])


def _mostly_bold(lines: list[_Line]) -> bool:
    chars = sum(line.chars for line in lines)
    return chars > 0 and sum(line.bold_chars for line in lines) / chars >= _BOLD_SHARE


def _dominant_size(lines: list[_Line]) -> float:
    return max(lines, key=lambda line: line.chars).size


def _is_heading(lines: list[_Line], body: float) -> bool:
    text = " ".join(line.text for line in lines)
    if len(lines) > _MAX_HEADING_LINES or len(text) > _MAX_HEADING_CHARS:
        return False
    if text.endswith(_SENTENCE_END) or _GLYPH_BULLET.match(text) or _DASH_BULLET.match(text):
        return False
    if sum(ch.isalpha() for ch in text) < _MIN_HEADING_LETTERS:
        return False
    size = _dominant_size(lines)
    return size >= body * _HEADING_SCALE or (_mostly_bold(lines) and size >= body * 0.95)


def _heading_levels(headings: dict[int, list[_Line]], body: float) -> dict[int, int]:
    """Уровень по размеру: самый крупный — 1; жирные заголовки размера текста — последний."""
    sizes = sorted(
        {
            _round_size(_dominant_size(lines))
            for lines in headings.values()
            if _dominant_size(lines) >= body * _HEADING_SCALE
        },
        reverse=True,
    )
    levels: dict[int, int] = {}
    for index, lines in headings.items():
        size = _round_size(_dominant_size(lines))
        levels[index] = sizes.index(size) + 1 if size in sizes else len(sizes) + 1
    return levels


def _location(path: tuple[str, ...], page: int, lines: list[_Line]) -> Location:
    numbers = [line.number for line in lines]
    return Location(path, page=page, line_start=min(numbers), line_end=max(numbers))


def _flow_blocks(lines: list[_Line], page: int, path: tuple[str, ...]) -> list[Block]:
    """Абзац или несколько пунктов списка внутри одного текстового блока."""
    items: list[tuple[BlockKind, list[_Line]]] = []
    for index, line in enumerate(lines):
        previous_is_item = bool(items) and items[-1][0] == "list_item"
        starts_item = bool(_GLYPH_BULLET.match(line.text)) or (
            (index == 0 or previous_is_item)
            and bool(_DASH_BULLET.match(line.text) or _NUMBERED_ITEM.match(line.text))
        )
        if starts_item:
            items.append(("list_item", [line]))
        elif items:
            items[-1][1].append(line)
        else:
            items.append(("paragraph", [line]))
    blocks: list[Block] = []
    for kind, item_lines in items:
        texts = [line.text for line in item_lines]
        if kind == "list_item":
            texts[0] = _strip_bullet(texts[0])
        blocks.append(Block(kind, "\n".join(texts), _location(path, page, item_lines)))
    return blocks


def _strip_bullet(text: str) -> str:
    for pattern in (_GLYPH_BULLET, _DASH_BULLET):
        match = pattern.match(text)
        if match:
            return text[match.end() :].strip()
    return text


# --- таблицы -------------------------------------------------------------------------


def _continues_previous_table(entries: list[Entry], index: int, state: _TableState) -> bool:
    """Таблица в самом начале страницы сразу после таблицы в конце прошлой страницы,
    с тем же числом колонок, — продолжение той же таблицы."""
    if index == 0:
        return False
    page_number, entry = entries[index]
    previous_page, previous = entries[index - 1]
    return (
        isinstance(entry, _Table)
        and isinstance(previous, _Table)
        and previous_page == page_number - 1
        and len(entry.rows[0]) == state.columns
    )


def _row_block(
    kind: BlockKind,
    cells: tuple[str, ...],
    lines: tuple[int | None, int | None],
    page: int,
    state: _TableState,
    path: tuple[str, ...],
) -> Block:
    location = Location(
        path,
        page=page,
        line_start=lines[0],
        line_end=lines[1],
        table=state.number,
        row=state.last_row,
    )
    return Block(kind, row_text(cells), location, cells=cells, header=state.header)


def _new_table(table: _Table, page: int, state: _TableState, path: tuple[str, ...]) -> list[Block]:
    (header, header_lines), *rows = zip(table.rows, table.row_lines, strict=True)
    state.header = header
    state.last_row = 1
    blocks = [_row_block("table_header", header, header_lines, page, state, path)]
    for cells, lines in rows:
        state.last_row += 1
        blocks.append(_row_block("table_row", cells, lines, page, state, path))
    return blocks


def _continued_rows(
    table: _Table, page: int, state: _TableState, path: tuple[str, ...]
) -> list[Block]:
    rows = list(zip(table.rows, table.row_lines, strict=True))
    if rows and rows[0][0] == state.header:
        rows.pop(0)  # шапка, повторённая на новой странице («повторять как заголовок»)
    blocks: list[Block] = []
    for cells, lines in rows:
        state.last_row += 1
        blocks.append(_row_block("table_row", cells, lines, page, state, path))
    return blocks
