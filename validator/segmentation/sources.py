"""Источники цитат: место цитаты в блоке и отображение отрезка текста фрагмента в блоки.

Публичный интерфейс для этапов, которым нужно доказательство из документа (lint, Evidence Gate):
найденный в тексте требования отрезок переводится в точные ссылки на блоки документа,
без обращения к внутренним структурам сегментатора.
"""

from __future__ import annotations

from dataclasses import replace

from validator.model import Block, Document, Location
from validator.segmentation.model import Requirement, SourceRef


def source_location(block: Block, start: int, end: int) -> Location:
    """Место цитаты block.text[start:end].

    Строки уточняются до цитаты, только если число строк текста блока совпадает с диапазоном
    строк источника (MD, TXT, PDF); иначе возвращается место всего блока — без выдуманной точности.
    """
    location = block.location
    if location.line_start is None or location.line_end is None:
        return location
    text = block.text
    if text.count("\n") != location.line_end - location.line_start:
        return location
    first = location.line_start + text.count("\n", 0, start)
    last = location.line_start + text.count("\n", 0, max(start, end - 1))
    return replace(location, line_start=first, line_end=last)


def span_sources(
    fragment: Requirement, start: int, end: int, document: Document
) -> tuple[SourceRef, ...]:
    """Отрезок fragment.text[start:end] → ссылки на блоки документа, в порядке текста.

    Текст фрагмента — цитаты его источников, соединённые «\\n». Отрезок может пересекать
    границу блоков: тогда вернётся несколько ссылок, а сам разделитель «\\n» ни к одному блоку
    не относится и в ссылки не попадает. Склеенные через «\\n» цитаты возвращённых ссылок
    в точности равны fragment.text[start:end].

    ValueError, если:
    - отрезок пуст или выходит за границы текста фрагмента;
    - отрезок начинается или заканчивается на разделителе блоков;
    - фрагмент не согласован с документом (несуществующий блок, границы вне блока, текст
      не совпадает с цитатами источников) — фиктивные ссылки не создаются.
    """
    if not 0 <= start < end <= len(fragment.text):
        raise ValueError(
            f"Некорректный отрезок {start}–{end} для фрагмента {fragment.id} "
            f"длиной {len(fragment.text)}."
        )
    pieces = _pieces(fragment, document)
    for ref, offset in pieces[:-1]:
        separator = offset + (ref.end - ref.start)
        if start == separator or end - 1 == separator:
            raise ValueError(
                f"Отрезок {start}–{end} фрагмента {fragment.id} начинается или заканчивается "
                "на разделителе блоков."
            )
    refs: list[SourceRef] = []
    for ref, offset in pieces:
        begin, finish = max(start, offset), min(end, offset + (ref.end - ref.start))
        if begin < finish:
            local_start, local_end = ref.start + begin - offset, ref.start + finish - offset
            location = source_location(document.blocks[ref.block], local_start, local_end)
            refs.append(SourceRef(ref.block, local_start, local_end, location))
    return tuple(refs)


def _pieces(fragment: Requirement, document: Document) -> list[tuple[SourceRef, int]]:
    """Источники фрагмента с их смещениями в тексте фрагмента; проверка согласованности."""
    pieces: list[tuple[SourceRef, int]] = []
    slices: list[str] = []
    offset = 0
    for ref in fragment.sources:
        if not 0 <= ref.block < len(document.blocks):
            raise ValueError(
                f"Фрагмент {fragment.id} ссылается на несуществующий блок {ref.block}."
            )
        text = document.blocks[ref.block].text
        if not 0 <= ref.start <= ref.end <= len(text):
            raise ValueError(
                f"Фрагмент {fragment.id}: границы {ref.start}–{ref.end} "
                f"вне текста блока {ref.block}."
            )
        pieces.append((ref, offset))
        slices.append(text[ref.start : ref.end])
        offset += ref.end - ref.start + 1  # +1 — разделитель «\n» между цитатами блоков
    if "\n".join(slices) != fragment.text:
        raise ValueError(f"Текст фрагмента {fragment.id} не совпадает с цитатами его источников.")
    return pieces
