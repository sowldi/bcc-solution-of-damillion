"""Word (DOCX) → блоки: абзацы тела документа, заголовки по стилям, списки, таблицы.

- Место: номер абзаца тела документа (считая пустые) или таблица и строка.
  Номеров страниц в DOCX нет — они зависят от вёрстки.
- Заголовки: стили Title (уровень 0), Heading N / «Заголовок N», либо уровень структуры
  (outlineLvl) в абзаце или его стиле.
- Исправления в режиме рецензирования: вставки учитываются, удаления — нет.
- Колонтитулы, сноски, примечания, надписи (text boxes) и оглавление не извлекаются.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import docx
from docx.document import Document as WordDocument
from docx.opc.exceptions import PackageNotFoundError
from docx.oxml.ns import qn
from docx.table import Table, _Cell, _Row
from docx.text.paragraph import Paragraph

from validator.errors import InputError
from validator.ingest.common import Extraction, SectionTracker, clean_text, row_text
from validator.model import Block, BlockKind, Location

_P, _TBL, _SDT, _SDT_CONTENT = qn("w:p"), qn("w:tbl"), qn("w:sdt"), qn("w:sdtContent")
_TEXT, _TAB, _NO_BREAK_HYPHEN = qn("w:t"), qn("w:tab"), qn("w:noBreakHyphen")
_BREAKS = frozenset({qn("w:br"), qn("w:cr")})
# поддеревья без видимого текста абзаца: свойства, удалённый текст, коды полей, надписи,
# запасное представление графики (дублирует основное)
_SKIP = frozenset(
    {
        qn("w:pPr"),
        qn("w:rPr"),
        qn("w:del"),
        qn("w:moveFrom"),
        qn("w:delText"),
        qn("w:instrText"),
        qn("w:txbxContent"),
        "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback",
    }
)
_HEADING_STYLE = re.compile(r"^(?:heading|заголовок)\s*(\d)$", re.IGNORECASE)
_BODY_TEXT_OUTLINE = 9  # outlineLvl 9 в Word означает «основной текст»


def read_docx(data: bytes, source: Path) -> Extraction:
    try:
        document = docx.Document(io.BytesIO(data))
    except (zipfile.BadZipFile, PackageNotFoundError, KeyError, ValueError, SyntaxError) as exc:
        # SyntaxError — базовый класс ошибок разбора XML (lxml XMLSyntaxError)
        raise InputError(f"DOCX повреждён или не является документом Word: {source}") from exc
    return Extraction(blocks=_DocxWalker(document).walk())


class _DocxWalker:
    def __init__(self, document: WordDocument) -> None:
        self.document = document
        self.sections = SectionTracker()
        self.blocks: list[Block] = []
        self.paragraph_number = 0
        self.table_number = 0

    def walk(self) -> tuple[Block, ...]:
        for element in _body_elements(self.document.element.body):
            if element.tag == _P:
                self._paragraph(Paragraph(element, self.document))
            else:
                self._table(Table(element, self.document))
        return tuple(self.blocks)

    def _paragraph(self, paragraph: Paragraph) -> None:
        self.paragraph_number += 1
        style = paragraph.style
        style_name = (style.name or "") if style is not None else ""
        text = paragraph_text(paragraph._p).strip()
        if not text or style_name.lower().startswith("toc"):
            return
        level = _heading_level(paragraph, style_name)
        if level is not None:
            parent = self.sections.enter(level, text)
            location = Location(parent, paragraph=self.paragraph_number)
            self.blocks.append(Block("heading", text, location, level=level))
            return
        kind: BlockKind = "list_item" if _is_list(paragraph, style_name) else "paragraph"
        location = Location(self.sections.path, paragraph=self.paragraph_number)
        self.blocks.append(Block(kind, text, location))

    def _table(self, table: Table) -> None:
        self.table_number += 1
        header: tuple[str, ...] = ()
        for number, row in enumerate(table.rows, 1):
            cells = tuple(_cell_text(cell) for cell in _unique_cells(row))
            location = Location(self.sections.path, table=self.table_number, row=number)
            if number == 1:
                header = cells
                self.blocks.append(
                    Block("table_header", row_text(cells), location, cells=cells, header=cells)
                )
            else:
                self.blocks.append(
                    Block("table_row", row_text(cells), location, cells=cells, header=header)
                )


def _body_elements(container: Any) -> Iterator[Any]:
    """Абзацы и таблицы тела в порядке документа, включая элементы управления (w:sdt)."""
    for child in container.iterchildren():
        if child.tag in (_P, _TBL):
            yield child
        elif child.tag == _SDT:
            content = child.find(_SDT_CONTENT)
            if content is not None:
                yield from _body_elements(content)


def paragraph_text(element: Any) -> str:
    """Видимый текст абзаца: вставки рецензирования — да, удаления и коды полей — нет."""
    parts: list[str] = []
    _collect_text(element, parts)
    return clean_text("".join(parts))


def _collect_text(element: Any, parts: list[str]) -> None:
    for child in element:
        tag = child.tag
        if not isinstance(tag, str) or tag in _SKIP:  # комментарии XML и служебные поддеревья
            continue
        if tag == _TEXT:
            parts.append(child.text or "")
        elif tag == _TAB:
            parts.append("\t")
        elif tag in _BREAKS:
            parts.append("\n")
        elif tag == _NO_BREAK_HYPHEN:
            parts.append("-")
        else:
            _collect_text(child, parts)


def _heading_level(paragraph: Paragraph, style_name: str) -> int | None:
    name = style_name.strip()
    if name.lower() == "title":
        return 0
    match = _HEADING_STYLE.match(name)
    if match:
        return int(match.group(1))
    holders: list[Any] = [paragraph._p]
    if paragraph.style is not None:
        holders.append(paragraph.style.element)
    for holder in holders:
        values = holder.xpath("./w:pPr/w:outlineLvl/@w:val")
        if values and str(values[0]).isdigit() and int(values[0]) < _BODY_TEXT_OUTLINE:
            return int(values[0]) + 1
    return None


def _is_list(paragraph: Paragraph, style_name: str) -> bool:
    properties = paragraph._p.pPr
    if properties is not None and properties.numPr is not None:
        return True
    lowered = style_name.lower()
    return lowered.startswith("list") or "список" in lowered


def _unique_cells(row: _Row) -> list[_Cell]:
    """Объединённая по горизонтали ячейка возвращается python-docx несколько раз подряд."""
    cells: list[_Cell] = []
    for cell in row.cells:
        if cells and cell._tc is cells[-1]._tc:
            continue
        cells.append(cell)
    return cells


def _cell_text(cell: _Cell) -> str:
    texts = (paragraph_text(paragraph._p).strip() for paragraph in cell.paragraphs)
    return "\n".join(text for text in texts if text)
