"""Простой текст (TXT) → блоки с номерами строк файла.

Структура восстанавливается эвристиками (без угадывания смысла):
- абзацы разделяются пустыми строками;
- заголовки: «# …», нумерованные «1.», «2.1 …» без точки в конце, строки ПРОПИСНЫМИ,
  строки, подчёркнутые «===» или «---»;
- пункты списков: строки с маркером «-», «*», «•», «—» или «1.»/«1)».
Таблицы в TXT не распознаются и остаются абзацами.
"""

from __future__ import annotations

import re
from pathlib import Path

from validator.ingest.common import Extraction, SectionTracker, clean_text, decode_text
from validator.model import Block, BlockKind, Location

_ATX = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_NUMBERED_HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,4})\.?\s+[^\W\d_]")
_UNDERLINE = re.compile(r"^\s*(?:={3,}|-{3,})\s*$")
# маркеры списков, включая символы шрифтов Symbol/Wingdings из Word (U+F0B7 и др.)
_GLYPH_BULLET = re.compile(r"^\s*[•●○◦▪■□‣⁃∙·\uf0b7\uf0a7\uf0d8\uf076\uf0fc]\s*(?=\S)")
_DASH_BULLET = re.compile(r"^\s*[-*–—]\s+(?=\S)")
# номер пункта — одна-две цифры: «200. Ответ …» описывает HTTP-статус, а не пункт списка
_NUMBERED_ITEM = re.compile(r"^\s*\d{1,2}[.)]\s+(?=\S)")
_SENTENCE_END = (".", ";", ":", ",", "!", "?")
_MAX_HEADING_CHARS = 120

Line = tuple[int, str]  # (номер строки файла, текст строки)


def read_text(data: bytes, source: Path) -> Extraction:
    text, encoding, warnings = decode_text(data, source)
    return Extraction(blocks=parse_text(text), encoding=encoding, warnings=warnings)


def parse_text(text: str) -> tuple[Block, ...]:
    sections = SectionTracker()
    blocks: list[Block] = []
    for group in _groups(clean_text(text).split("\n")):
        _parse_group(group, sections, blocks)
    return tuple(blocks)


def _groups(lines: list[str]) -> list[list[Line]]:
    """Группы непустых строк, разделённые пустыми строками."""
    groups: list[list[Line]] = []
    current: list[Line] = []
    for number, line in enumerate(lines, 1):
        if line.strip():
            current.append((number, line))
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _parse_group(group: list[Line], sections: SectionTracker, blocks: list[Block]) -> None:
    index = 0
    while index < len(group):
        number, line = group[index]
        following = group[index + 1][1] if index + 1 < len(group) else None
        if following is not None and _UNDERLINE.match(following) and not _is_list_start(line):
            underline_level = 1 if "=" in following else 2
            _heading(blocks, sections, underline_level, line.strip(), number, group[index + 1][0])
            index += 2
            continue
        level = _heading_level(line, following) if index == 0 else _atx_level(line)
        if level is not None:
            match = _ATX.match(line)
            title = match.group(2) if match else line.strip()
            _heading(blocks, sections, level, title, number, number)
            index += 1
            continue
        index = _flow(group, index, sections, blocks)


def _flow(group: list[Line], index: int, sections: SectionTracker, blocks: list[Block]) -> int:
    """Абзац или пункт списка с продолжениями; вернуть индекс первой необработанной строки."""
    first_number, first = group[index]
    kind: BlockKind = "list_item" if _is_list_start(first) else "paragraph"
    lines = [_strip_bullet(first)]
    last_number = first_number
    index += 1
    while index < len(group):
        number, line = group[index]
        starts_item = _GLYPH_BULLET.match(line) or (
            kind == "list_item" and (_DASH_BULLET.match(line) or _NUMBERED_ITEM.match(line))
        )
        if starts_item or _ATX.match(line):
            break
        following = group[index + 1][1] if index + 1 < len(group) else None
        if following is not None and _UNDERLINE.match(following):
            break
        lines.append(line.strip())
        last_number = number
        index += 1
    location = Location(sections.path, line_start=first_number, line_end=last_number)
    blocks.append(Block(kind, "\n".join(lines), location))
    return index


def _heading(
    blocks: list[Block], sections: SectionTracker, level: int, title: str, start: int, end: int
) -> None:
    parent = sections.enter(level, title)
    location = Location(parent, line_start=start, line_end=end)
    blocks.append(Block("heading", title, location, level=level))


def _atx_level(line: str) -> int | None:
    match = _ATX.match(line)
    return len(match.group(1)) if match else None


def _heading_level(line: str, following: str | None) -> int | None:
    """Уровень заголовка для первой строки группы или None."""
    atx = _atx_level(line)
    if atx is not None:
        return atx
    stripped = line.strip()
    if len(stripped) > _MAX_HEADING_CHARS or stripped.endswith(_SENTENCE_END):
        return None
    numbered = _NUMBERED_HEADING.match(stripped)
    if numbered:
        # «1. Проверить сессию» + «2. Сохранить …» — это список, а не заголовок
        if following is not None and _NUMBERED_ITEM.match(following):
            return None
        return numbered.group(1).count(".") + 1
    letters = [ch for ch in stripped if ch.isalpha()]
    if len(letters) >= 4 and all(ch.isupper() for ch in letters):
        return 1
    return None


def _is_list_start(line: str) -> bool:
    return bool(_GLYPH_BULLET.match(line) or _DASH_BULLET.match(line) or _NUMBERED_ITEM.match(line))


def _strip_bullet(line: str) -> str:
    """Маркер «•»/«-» убирается; номер «1.» сохраняется — он часть текста пункта."""
    for pattern in (_GLYPH_BULLET, _DASH_BULLET):
        match = pattern.match(line)
        if match:
            return line[match.end() :].strip()
    return line.strip()
