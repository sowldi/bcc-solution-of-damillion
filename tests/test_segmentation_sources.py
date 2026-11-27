"""Публичное отображение отрезка текста фрагмента в исходные блоки (span_sources)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.seg_helpers import unit
from validator.ingest import load_document
from validator.model import Document
from validator.segmentation import (
    Requirement,
    Segmentation,
    SourceRef,
    segment_document,
    source_location,
    span_sources,
)

SPEC = """# Сервис

## Функциональные требования

Система хранит историю брони

и показывает её читателю.

* Ф-1. Читатель задаёт срок. По умолчанию бронь не продлевается.

| Поле | Значение |
| --- | --- |
| | |
"""


@pytest.fixture
def pair(tmp_path: Path) -> tuple[Segmentation, Document]:
    path = tmp_path / "spec.md"
    path.write_text(SPEC, encoding="utf-8")
    document = load_document(path)
    return segment_document(document), document


def _quote(document: Document, refs: tuple[SourceRef, ...]) -> str:
    return "\n".join(document.blocks[r.block].text[r.start : r.end] for r in refs)


def _merged(seg: Segmentation) -> Requirement:
    fragment = unit(seg, "Система хранит историю")
    assert [ref.block for ref in fragment.sources] == [2, 3]  # фрагмент из двух блоков
    return fragment


def test_span_inside_one_block(pair: tuple[Segmentation, Document]) -> None:
    seg, document = pair
    fragment = _merged(seg)

    refs = span_sources(fragment, 0, 7, document)

    assert [(r.block, r.start, r.end) for r in refs] == [(2, 0, 7)]
    assert _quote(document, refs) == "Система"
    assert refs[0].location.line_start == 5


def test_span_across_blocks_skips_separator_and_keeps_order(
    pair: tuple[Segmentation, Document],
) -> None:
    seg, document = pair
    fragment = _merged(seg)
    start = fragment.text.index("брони")
    end = fragment.text.index("показывает") + len("показывает")

    refs = span_sources(fragment, start, end, document)

    assert [r.block for r in refs] == [2, 3]
    assert _quote(document, refs) == fragment.text[start:end] == "брони\nи показывает"
    assert refs[1].start == 0
    assert refs[1].location.line_start == 7


def test_span_inside_second_block(pair: tuple[Segmentation, Document]) -> None:
    seg, document = pair
    fragment = _merged(seg)
    start = fragment.text.index("показывает")

    refs = span_sources(fragment, start, start + len("показывает"), document)

    assert [(r.block, r.start, r.end) for r in refs] == [(3, 2, 12)]


def test_span_of_statement_is_relative_to_its_own_source(
    pair: tuple[Segmentation, Document],
) -> None:
    seg, document = pair
    statement = seg.by_id("Ф-1/2")
    own = statement.sources[0]

    refs = span_sources(statement, 3, 9, document)

    assert [(r.block, r.start, r.end) for r in refs] == [(own.block, own.start + 3, own.start + 9)]
    assert _quote(document, refs) == statement.text[3:9]


def test_locations_match_source_rules(pair: tuple[Segmentation, Document]) -> None:
    seg, document = pair
    fragment = _merged(seg)

    for ref in span_sources(fragment, 2, len(fragment.text) - 2, document):
        block = document.blocks[ref.block]
        assert ref.location == source_location(block, ref.start, ref.end)


@pytest.mark.parametrize(("start", "end"), [(-1, 3), (0, 10_000), (5, 5), (6, 2)])
def test_out_of_range_or_empty_span_is_rejected(
    pair: tuple[Segmentation, Document], start: int, end: int
) -> None:
    seg, document = pair

    with pytest.raises(ValueError, match="отрез"):
        span_sources(_merged(seg), start, end, document)


def test_span_starting_or_ending_on_separator_is_rejected(
    pair: tuple[Segmentation, Document],
) -> None:
    seg, document = pair
    fragment = _merged(seg)
    separator = fragment.text.index("\n")

    with pytest.raises(ValueError, match="разделител"):
        span_sources(fragment, separator, separator + 3, document)
    with pytest.raises(ValueError, match="разделител"):
        span_sources(fragment, 0, separator + 1, document)


def test_fragment_inconsistent_with_document_is_rejected(
    pair: tuple[Segmentation, Document],
) -> None:
    seg, document = pair
    changed = "Система удаляет историю брони\nи показывает её читателю."
    tampered = replace(_merged(seg), text=changed)

    with pytest.raises(ValueError, match="не совпадает"):
        span_sources(tampered, 0, 7, document)


def test_fragment_with_foreign_block_is_rejected(pair: tuple[Segmentation, Document]) -> None:
    seg, document = pair
    fragment = _merged(seg)
    broken = replace(fragment, sources=(replace(fragment.sources[0], block=999),))

    with pytest.raises(ValueError, match="999"):
        span_sources(broken, 0, 3, document)


def test_empty_fragment_has_no_spans(pair: tuple[Segmentation, Document]) -> None:
    seg, document = pair
    empty = next(f for f in seg.fragments if f.text == "" and f.role != "table_header")

    with pytest.raises(ValueError, match="отрез"):
        span_sources(empty, 0, 0, document)
