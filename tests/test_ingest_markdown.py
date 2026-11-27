from __future__ import annotations

from pathlib import Path

import pytest

from validator.errors import InputError
from validator.ingest import load_document
from validator.model import Block

SPEC = """# Спецификация

Вводный абзац
на двух строках.

## Функциональные требования

* **Ф-1.** Система хранит порог
  от 1 000 до 1 000 000 ₸.
* Второй пункт
  1. Вложенный шаг
  2. Второй шаг

3. Нумерованный с трёх

### Таблица параметров

| Поле | Значение |
| :--- | :--- |
| A \\| B | 10 ₸ |
| | |

```json
{"result": "OK"}
```

<!-- служебный комментарий -->

Итог
----
"""

ROOT = ("Спецификация",)
FR = ("Спецификация", "Функциональные требования")
TABLE = (*FR, "Таблица параметров")

EXPECTED = [
    ("heading", "Спецификация", 1, 1, ()),
    ("paragraph", "Вводный абзац\nна двух строках.", 3, 4, ROOT),
    ("heading", "Функциональные требования", 6, 6, ROOT),
    ("list_item", "Ф-1. Система хранит порог\nот 1 000 до 1 000 000 ₸.", 8, 9, FR),
    ("list_item", "Второй пункт", 10, 10, FR),
    ("list_item", "1. Вложенный шаг", 11, 11, FR),
    ("list_item", "2. Второй шаг", 12, 12, FR),
    ("list_item", "3. Нумерованный с трёх", 14, 14, FR),
    ("heading", "Таблица параметров", 16, 16, FR),
    ("table_header", "Поле | Значение", 18, 18, TABLE),
    ("table_row", "A | B | 10 ₸", 20, 20, TABLE),
    ("table_row", "", 21, 21, TABLE),
    ("code", '{"result": "OK"}', 24, 24, TABLE),
    ("heading", "Итог", 29, 30, ROOT),
]


def _shape(blocks: tuple[Block, ...]) -> list[tuple[object, ...]]:
    return [
        (b.kind, b.text, b.location.line_start, b.location.line_end, b.location.section_path)
        for b in blocks
    ]


def _load(tmp_path: Path, content: str | bytes, name: str = "spec.md") -> tuple[Block, ...]:
    path = tmp_path / name
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8", newline="")
    else:
        path.write_bytes(content)
    return load_document(path).blocks


def test_blocks_text_lines_and_sections(tmp_path: Path) -> None:
    assert _shape(_load(tmp_path, SPEC)) == EXPECTED


def test_heading_levels(tmp_path: Path) -> None:
    levels = [(b.text, b.level) for b in _load(tmp_path, SPEC) if b.kind == "heading"]

    assert levels == [
        ("Спецификация", 1),
        ("Функциональные требования", 2),
        ("Таблица параметров", 3),
        ("Итог", 2),
    ]


def test_table_cells_header_and_numbering(tmp_path: Path) -> None:
    rows = [b for b in _load(tmp_path, SPEC) if b.kind in ("table_header", "table_row")]

    assert [(b.location.table, b.location.row) for b in rows] == [(1, 1), (1, 2), (1, 3)]
    assert rows[0].cells == ("Поле", "Значение")
    assert rows[1].cells == ("A | B", "10 ₸")
    assert rows[1].header == ("Поле", "Значение")
    assert rows[2].cells == ("", "")


def test_windows_line_endings_keep_line_numbers(tmp_path: Path) -> None:
    crlf = SPEC.replace("\n", "\r\n")

    assert _shape(_load(tmp_path, crlf)) == EXPECTED


def test_utf8_bom_is_ignored(tmp_path: Path) -> None:
    blocks = _load(tmp_path, "﻿" + SPEC)

    assert blocks[0].text == "Спецификация"


def test_text_lines_map_to_source_lines(tmp_path: Path) -> None:
    """Строка k текста блока соответствует строке файла line_start + k."""
    source = SPEC.split("\n")
    for block in _load(tmp_path, SPEC):
        if block.kind not in ("paragraph", "code"):
            continue
        assert block.location.line_start is not None
        for k, line in enumerate(block.text.split("\n")):
            assert line in source[block.location.line_start - 1 + k]


def test_markdown_escapes_and_entities_are_decoded(tmp_path: Path) -> None:
    blocks = _load(tmp_path, "Цена&nbsp;100 \\* 2 и `код`<br>после\n")

    assert blocks[0].text == "Цена 100 * 2 и код после"


def test_cp1251_markdown_is_decoded_with_warning(tmp_path: Path) -> None:
    path = tmp_path / "legacy.md"
    path.write_bytes("# Требования\n\nТекст в кодировке Windows.\n".encode("cp1251"))

    document = load_document(path)

    assert document.blocks[0].text == "Требования"
    assert document.encoding == "cp1251"
    assert any("Windows-1251" in w for w in document.warnings)


def test_binary_file_with_markdown_extension_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "image.md"
    path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + bytes(range(256)))

    with pytest.raises(InputError, match="двоичн"):
        load_document(path)


def test_markdown_without_text_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "only-comment.md"
    path.write_text("<!-- только комментарий -->\n\n---\n", encoding="utf-8")

    with pytest.raises(InputError, match="пуст"):
        load_document(path)
