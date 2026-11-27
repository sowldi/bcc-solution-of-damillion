from __future__ import annotations

from pathlib import Path

import pytest

from validator.model import Block, Document, Location


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        (
            Location(("Часть A", "Требования"), line_start=3, line_end=4),
            "Часть A › Требования — строки 3–4",
        ),
        (Location(line_start=7, line_end=7), "строка 7"),
        (
            Location(("Критерии",), line_start=16, line_end=16, table=2, row=3),
            "Критерии — таблица 2, строка 3 (строка 16 файла)",
        ),
        (
            Location(("Раздел",), page=2, line_start=10, line_end=11),
            "Раздел — стр. 2, строки 10–11",
        ),
        (
            Location(page=1, line_start=12, line_end=12, table=1, row=2),
            "стр. 1, таблица 1, строка 2",
        ),
        (Location(("Раздел",), paragraph=14), "Раздел — абзац 14"),
        (Location(("Раздел",), table=3, row=1), "Раздел — таблица 3, строка 1"),
        (Location(("Только раздел",)), "Только раздел"),
        (Location(), "место не определено"),
    ],
)
def test_location_describe(location: Location, expected: str) -> None:
    assert location.describe() == expected


def _block(kind: str, text: str, **location: int) -> Block:
    return Block(kind=kind, text=text, location=Location(**location))  # type: ignore[arg-type]


def test_document_statistics() -> None:
    document = Document(
        source=Path("spec.md"),
        format="markdown",
        sha256="0" * 64,
        blocks=(
            _block("heading", "Раздел"),
            _block("paragraph", "Абзац"),
            _block("table_header", "A | B", table=1, row=1),
            _block("table_row", "1 | 2", table=1, row=2),
            _block("table_row", "", table=1, row=3),
            _block("table_header", "C", table=2, row=1),
        ),
    )

    assert document.has_text
    assert document.count("table_row") == 2
    assert document.table_count == 2
    assert document.char_count == len("Раздел") + len("Абзац") + len("A | B") + len("1 | 2") + 1


def test_document_without_text() -> None:
    document = Document(
        source=Path("x.md"),
        format="markdown",
        sha256="0" * 64,
        blocks=(_block("table_row", "", table=1, row=2),),
    )

    assert not document.has_text
