"""Единая модель извлечённого документа: блоки текста с точным местом в источнике.

Все читатели форматов (Markdown, TXT, PDF, DOCX) возвращают одни и те же типы,
поэтому дальнейшие этапы не зависят от формата входного файла.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DocumentFormat = Literal["markdown", "text", "pdf", "docx"]
BlockKind = Literal["heading", "paragraph", "list_item", "table_header", "table_row", "code"]

FORMAT_TITLES: dict[DocumentFormat, str] = {
    "markdown": "Markdown",
    "text": "Текст (TXT)",
    "pdf": "PDF",
    "docx": "Word (DOCX)",
}


@dataclass(frozen=True)
class Location:
    """Где находится блок в исходном документе.

    section_path — цепочка заголовков, внутри которых стоит блок.
    page — номер страницы (PDF). line_start/line_end — строки файла (MD, TXT) или
    строки на странице (PDF). paragraph — номер абзаца тела документа (DOCX).
    table/row — номер таблицы в документе и номер строки в ней (шапка — строка 1).
    Для MD, TXT и PDF строка k текста блока соответствует строке line_start + k.
    """

    section_path: tuple[str, ...] = ()
    page: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    paragraph: int | None = None
    table: int | None = None
    row: int | None = None

    def describe(self) -> str:
        """Место по-русски, например «Раздел › Подраздел — стр. 2, строки 10–11»."""
        details: list[str] = []
        if self.page is not None:
            details.append(f"стр. {self.page}")
        if self.table is not None:
            row = f", строка {self.row}" if self.row is not None else ""
            details.append(f"таблица {self.table}{row}")
        if self.paragraph is not None:
            details.append(f"абзац {self.paragraph}")
        lines = _describe_lines(self.line_start, self.line_end)
        if lines:
            if self.table is None:
                details.append(lines)
            elif self.page is None:
                details[-1] += f" ({lines} файла)"
        place = ", ".join(details)
        section = " › ".join(self.section_path)
        if section and place:
            return f"{section} — {place}"
        return section or place or "место не определено"


def _describe_lines(start: int | None, end: int | None) -> str:
    if start is None:
        return ""
    if end is None or end == start:
        return f"строка {start}"
    return f"строки {start}–{end}"


@dataclass(frozen=True)
class Block:
    """Смысловой фрагмент документа в порядке чтения.

    text — текст без разметки (переводы строк источника сохранены).
    level — уровень заголовка (только для heading).
    cells/header — ячейки строки таблицы и шапка её таблицы (только для таблиц).
    Текст строки таблицы — ячейки через « | »; у полностью пустой строки он пустой.
    """

    kind: BlockKind
    text: str
    location: Location
    level: int | None = None
    cells: tuple[str, ...] = ()
    header: tuple[str, ...] = ()


@dataclass(frozen=True)
class Document:
    """Результат чтения файла любого поддерживаемого формата."""

    source: Path
    format: DocumentFormat
    sha256: str
    blocks: tuple[Block, ...]
    encoding: str | None = None
    page_count: int | None = None
    warnings: tuple[str, ...] = ()

    @property
    def has_text(self) -> bool:
        return any(block.text.strip() for block in self.blocks)

    @property
    def char_count(self) -> int:
        return sum(len(block.text) for block in self.blocks)

    @property
    def table_count(self) -> int:
        return len({b.location.table for b in self.blocks if b.location.table is not None})

    def count(self, kind: BlockKind) -> int:
        return sum(1 for block in self.blocks if block.kind == kind)
