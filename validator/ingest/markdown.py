"""Markdown (CommonMark + таблицы GFM) → блоки с номерами строк файла."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.token import Token

from validator.ingest.common import Extraction, SectionTracker, clean_text, decode_text, row_text
from validator.model import Block, BlockKind, Location

# html=True (по умолчанию в commonmark) — чтобы HTML-блоки не терялись; typographer выключен,
# чтобы кавычки и тире оставались ровно такими, как в источнике.
_PARSER = MarkdownIt("commonmark").enable("table")
_LINE_BREAK_TAG = re.compile(r"<br\s*/?>", re.IGNORECASE)


def read_markdown(data: bytes, source: Path) -> Extraction:
    text, encoding, warnings = decode_text(data, source)
    return Extraction(blocks=parse_markdown(text), encoding=encoding, warnings=warnings)


def parse_markdown(text: str) -> tuple[Block, ...]:
    return _MarkdownWalker(_PARSER.parse(clean_text(text))).walk()


@dataclass
class _ListItem:
    prefix: str  # «3. » для нумерованных списков, пусто для маркированных
    started: bool = False  # первый абзац пункта уже выведен


class _MarkdownWalker:
    """Один проход по токенам markdown-it в порядке документа."""

    def __init__(self, tokens: Sequence[Token]) -> None:
        self.tokens = tokens
        self.blocks: list[Block] = []
        self.sections = SectionTracker()
        self.ordered_lists: list[bool] = []  # стек вложенных списков: нумерованный?
        self.items: list[_ListItem] = []
        self.table_number = 0
        self.row_number = 0
        self.in_table_head = False
        self.header: tuple[str, ...] = ()

    def walk(self) -> tuple[Block, ...]:
        index = 0
        while index < len(self.tokens):
            index = self._step(index)
        return tuple(self.blocks)

    def _step(self, index: int) -> int:
        """Обработать токен; вернуть индекс следующего необработанного токена."""
        token = self.tokens[index]
        kind = token.type
        if kind == "heading_open":
            self._heading(token, _inline_text(self.tokens[index + 1]))
            return index + 3
        if kind == "paragraph_open":
            self._paragraph(token, _inline_text(self.tokens[index + 1]))
            return index + 3
        if kind == "tr_open":
            return self._table_row(index)
        if kind in ("bullet_list_open", "ordered_list_open"):
            self.ordered_lists.append(kind == "ordered_list_open")
        elif kind in ("bullet_list_close", "ordered_list_close"):
            self.ordered_lists.pop()
        elif kind == "list_item_open":
            ordered = bool(self.ordered_lists) and self.ordered_lists[-1] and bool(token.info)
            self.items.append(_ListItem(f"{token.info}{token.markup} " if ordered else ""))
        elif kind == "list_item_close":
            self.items.pop()
        elif kind == "table_open":
            self.table_number += 1
            self.row_number = 0
            self.header = ()
        elif kind in ("thead_open", "thead_close"):
            self.in_table_head = kind == "thead_open"
        elif kind in ("fence", "code_block"):
            self._code(token)
        elif kind == "html_block":
            self._html(token)
        return index + 1

    def _location(
        self, token: Token, *, table: int | None = None, row: int | None = None
    ) -> Location:
        start, end = _lines(token)
        return Location(self.sections.path, line_start=start, line_end=end, table=table, row=row)

    def _heading(self, token: Token, title: str) -> None:
        level = int(token.tag[1:])
        start, end = _lines(token)
        parent = self.sections.enter(level, title)
        location = Location(parent, line_start=start, line_end=end)
        self.blocks.append(Block("heading", title, location, level=level))

    def _paragraph(self, token: Token, text: str) -> None:
        if not text.strip():
            return
        kind: BlockKind = "paragraph"
        if self.items and not self.items[-1].started:
            # первый абзац пункта — сам пункт; следующие абзацы пункта — обычные абзацы
            kind = "list_item"
            text = self.items[-1].prefix + text
            self.items[-1].started = True
        self.blocks.append(Block(kind, text, self._location(token)))

    def _table_row(self, index: int) -> int:
        cells: list[str] = []
        cursor = index + 1
        while self.tokens[cursor].type != "tr_close":
            if self.tokens[cursor].type == "inline":
                cells.append(_inline_text(self.tokens[cursor]))
            cursor += 1
        self.row_number += 1
        row = tuple(cells)
        location = self._location(self.tokens[index], table=self.table_number, row=self.row_number)
        if self.in_table_head:
            self.header = row
            self.blocks.append(
                Block("table_header", row_text(row), location, cells=row, header=row)
            )
        else:
            block = Block("table_row", row_text(row), location, cells=row, header=self.header)
            self.blocks.append(block)
        return cursor + 1

    def _code(self, token: Token) -> None:
        content = token.content.rstrip("\n")
        if not content.strip() or token.map is None:
            return
        # у fence первая строка — сама «```», содержимое начинается со следующей
        first = token.map[0] + (2 if token.type == "fence" else 1)
        last = first + content.count("\n")
        location = Location(self.sections.path, line_start=first, line_end=last)
        self.blocks.append(Block("code", content, location))

    def _html(self, token: Token) -> None:
        content = token.content.strip()
        is_comment = content.startswith("<!--") and content.endswith("-->")
        if content and not is_comment:
            self.blocks.append(Block("paragraph", content, self._location(token)))


def _lines(token: Token) -> tuple[int | None, int | None]:
    """map markdown-it: [начало, конец) с нуля → номера строк файла с единицы включительно."""
    if token.map is None:
        return None, None
    return token.map[0] + 1, token.map[1]


def _inline_text(token: Token) -> str:
    """Текст без разметки; мягкие и жёсткие переносы → \\n (строки источника сохраняются)."""
    parts: list[str] = []
    for child in token.children or []:
        if child.type in ("text", "code_inline", "image"):
            parts.append(child.content)
        elif child.type in ("softbreak", "hardbreak"):
            parts.append("\n")
        elif child.type == "html_inline" and _LINE_BREAK_TAG.fullmatch(child.content.strip()):
            parts.append(" ")
    return "".join(parts).strip()
