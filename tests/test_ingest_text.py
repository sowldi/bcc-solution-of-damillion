from __future__ import annotations

from pathlib import Path

import pytest

from validator.errors import InputError
from validator.ingest import load_document
from validator.model import Block

TXT = """СПЕЦИФИКАЦИЯ СЕРВИСА

1. Общие сведения

Сервис отправляет уведомления
клиентам банка.

1.1 Требования
- Первый пункт
  с продолжением
- Второй пункт

1. Проверить читательский билет.
2. Выдать книгу.

Итоги
=====
Текст после заголовка.

## Приложение
Значение 1 000 ₸ указано без пояснений.
"""

GENERAL = ("1. Общие сведения",)
REQS = ("1. Общие сведения", "1.1 Требования")

EXPECTED = [
    ("heading", 1, "СПЕЦИФИКАЦИЯ СЕРВИСА", 1, 1, ()),
    ("heading", 1, "1. Общие сведения", 3, 3, ()),
    ("paragraph", None, "Сервис отправляет уведомления\nклиентам банка.", 5, 6, GENERAL),
    ("heading", 2, "1.1 Требования", 8, 8, GENERAL),
    ("list_item", None, "Первый пункт\nс продолжением", 9, 10, REQS),
    ("list_item", None, "Второй пункт", 11, 11, REQS),
    ("list_item", None, "1. Проверить читательский билет.", 13, 13, REQS),
    ("list_item", None, "2. Выдать книгу.", 14, 14, REQS),
    ("heading", 1, "Итоги", 16, 17, ()),
    ("paragraph", None, "Текст после заголовка.", 18, 18, ("Итоги",)),
    ("heading", 2, "Приложение", 20, 20, ("Итоги",)),
    ("paragraph", None, "Значение 1 000 ₸ указано без пояснений.", 21, 21, ("Итоги", "Приложение")),
]


def _load(tmp_path: Path, data: bytes, name: str = "spec.txt") -> tuple[Block, ...]:
    path = tmp_path / name
    path.write_bytes(data)
    return load_document(path).blocks


def _shape(blocks: tuple[Block, ...]) -> list[tuple[object, ...]]:
    return [
        (
            b.kind,
            b.level,
            b.text,
            b.location.line_start,
            b.location.line_end,
            b.location.section_path,
        )
        for b in blocks
    ]


def test_paragraphs_headings_and_lists(tmp_path: Path) -> None:
    assert _shape(_load(tmp_path, TXT.encode())) == EXPECTED


def test_numbered_sentence_is_not_a_heading(tmp_path: Path) -> None:
    blocks = _load(tmp_path, "1. Проверить срок действия ключа.\n".encode())

    assert [(b.kind, b.text) for b in blocks] == [
        ("list_item", "1. Проверить срок действия ключа.")
    ]


def test_word_symbol_bullet_starts_list_item(tmp_path: Path) -> None:
    """Word при сохранении в текст ставит маркер из шрифта Symbol (U+F0B7)."""
    word_bullet = chr(0xF0B7)
    blocks = _load(tmp_path, f"{word_bullet}\tПервый пункт\n{word_bullet}\tВторой пункт\n".encode())

    assert [(b.kind, b.text) for b in blocks] == [
        ("list_item", "Первый пункт"),
        ("list_item", "Второй пункт"),
    ]


def test_status_code_line_is_paragraph_not_list_item(tmp_path: Path) -> None:
    blocks = _load(tmp_path, "200. Запрос обработан без ошибок.\n".encode())

    assert blocks[0].kind == "paragraph"


def test_number_followed_by_digits_is_not_a_heading(tmp_path: Path) -> None:
    blocks = _load(tmp_path, "1 000 ₸ — минимальный порог\n".encode())

    assert blocks[0].kind == "paragraph"


def test_dash_inside_paragraph_does_not_start_a_list(tmp_path: Path) -> None:
    blocks = _load(tmp_path, "Порог задаёт клиент\n— только целым числом.\n".encode())

    assert [(b.kind, b.text) for b in blocks] == [
        ("paragraph", "Порог задаёт клиент\n— только целым числом.")
    ]


def test_text_lines_map_to_source_lines(tmp_path: Path) -> None:
    source = TXT.split("\n")
    for block in _load(tmp_path, TXT.encode()):
        if block.kind != "paragraph":
            continue
        assert block.location.line_start is not None
        for k, line in enumerate(block.text.split("\n")):
            assert line == source[block.location.line_start - 1 + k].strip()


def test_cp1251_text_is_decoded_with_warning(tmp_path: Path) -> None:
    path = tmp_path / "legacy.txt"
    path.write_bytes("Требование хранится в кодировке Windows.\n".encode("cp1251"))

    document = load_document(path)

    assert document.blocks[0].text == "Требование хранится в кодировке Windows."
    assert document.encoding == "cp1251"
    assert any("Windows-1251" in w for w in document.warnings)


def test_utf16_text_with_bom_is_decoded(tmp_path: Path) -> None:
    path = tmp_path / "notepad.txt"
    path.write_bytes("Сохранено Блокнотом в Юникоде.\r\n".encode("utf-16"))

    document = load_document(path)

    assert document.blocks[0].text == "Сохранено Блокнотом в Юникоде."
    assert document.encoding == "utf-16"


def test_invisible_characters_are_removed(tmp_path: Path) -> None:
    blocks = _load(tmp_path, "Пере­нос и​ невидимый пробел.\n".encode())

    assert blocks[0].text == "Перенос и невидимый пробел."


def test_binary_text_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="двоичн"):
        _load(tmp_path, b"\x00\x01\x02text\x00\xff")


@pytest.mark.parametrize("content", [b"", b"   \r\n\t\n", "﻿  \n".encode()])
def test_blank_text_file_is_empty(tmp_path: Path, content: bytes) -> None:
    with pytest.raises(InputError, match="пуст"):
        _load(tmp_path, content)
