from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from tests.builders import (
    IMG,
    PDF_HEADER,
    SAMPLE,
    H,
    P,
    T,
    expected_blocks,
    simplify,
    write_pdf,
)
from validator.errors import InputError
from validator.ingest import load_document
from validator.model import Block, Document

PAGES = [SAMPLE[:6], SAMPLE[6:]]


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Document:
    return load_document(write_pdf(tmp_path / "spec.pdf", PAGES))


def _find(document: Document, text: str) -> Block:
    return next(b for b in document.blocks if b.text.startswith(text))


def test_blocks_match_source_structure(sample_pdf: Document) -> None:
    assert simplify(sample_pdf) == expected_blocks(SAMPLE)


def test_format_and_page_count(sample_pdf: Document) -> None:
    assert sample_pdf.format == "pdf"
    assert sample_pdf.page_count == 2
    assert sample_pdf.encoding is None


def test_running_header_and_page_numbers_are_removed(sample_pdf: Document) -> None:
    texts = " ".join(b.text for b in sample_pdf.blocks)

    assert PDF_HEADER not in texts
    assert "Страница" not in texts


def test_pages_sections_and_visual_lines(sample_pdf: Document) -> None:
    heading = _find(sample_pdf, "Ограничения")

    assert heading.location.page == 2
    assert heading.location.section_path == ("Сервис уведомлений",)
    # строка 1 на странице — колонтитул (в блоки не попадает, но счёт строк сохраняет)
    assert heading.location.line_start == 2
    assert _find(sample_pdf, "Время ответа").location.line_start == 3


def test_lines_increase_down_the_page(sample_pdf: Document) -> None:
    page_one = [b for b in sample_pdf.blocks if b.location.page == 1]
    starts = [b.location.line_start for b in page_one]

    assert all(isinstance(s, int) for s in starts)
    assert starts == sorted(starts)  # type: ignore[type-var]


def test_table_rows_have_header_and_position(sample_pdf: Document) -> None:
    row = _find(sample_pdf, "Порог")

    assert row.kind == "table_row"
    assert row.cells == ("Порог", "1 000 ₸")
    assert row.header == ("Поле", "Значение")
    assert (row.location.page, row.location.table, row.location.row) == (1, 1, 2)
    assert row.location.section_path == ("Сервис уведомлений", "Функциональные требования")


def test_heading_levels_follow_font_size(sample_pdf: Document) -> None:
    levels = [(b.text, b.level) for b in sample_pdf.blocks if b.kind == "heading"]

    assert levels == [
        ("Сервис уведомлений", 1),
        ("Функциональные требования", 2),
        ("Ограничения", 2),
    ]


BODY = "Основной текст документа, по которому определяется обычный размер шрифта."


def test_bold_line_of_body_size_is_heading(tmp_path: Path) -> None:
    path = write_pdf(
        tmp_path / "bold.pdf",
        [[P(BODY), P("Особые правила", html="<b>Особые правила</b>"), P(BODY)]],
    )

    block = _find(load_document(path), "Особые правила")

    assert block.kind == "heading"


def test_partially_bold_paragraph_is_not_heading(tmp_path: Path) -> None:
    text = "Ф-2. Жирный только префикс требования."
    path = write_pdf(
        tmp_path / "prefix.pdf",
        [[P(BODY), P(text, html="<b>Ф-2.</b> Жирный только префикс требования."), P(BODY)]],
    )

    block = _find(load_document(path), "Ф-2.")

    assert (block.kind, block.text) == ("paragraph", text)


def test_status_code_line_is_paragraph_not_list_item(tmp_path: Path) -> None:
    """Трёхзначное число в начале («200. Ответ…») — не номер пункта списка."""
    text = "200. Запрос обработан без ошибок."
    path = write_pdf(tmp_path / "status.pdf", [[P(BODY), P(text)]])

    block = _find(load_document(path), "200.")

    assert (block.kind, block.text) == ("paragraph", text)


def test_bold_label_without_letters_is_not_heading(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "label.pdf", [[P(BODY), P("404", html="<b>404</b>"), P(BODY)]])

    block = _find(load_document(path), "404")

    assert block.kind == "paragraph"


def test_identifiers_with_underscores_survive_in_table_cells(tmp_path: Path) -> None:
    rows = (("Поле", "Колонка БД"), ("Порог", "limit_kzt"), ("Карта", "card_id"))
    path = write_pdf(tmp_path / "db.pdf", [[P(BODY), T(rows)]], header=None)

    cells = [b.cells for b in load_document(path).blocks if b.location.table == 1]

    assert cells == [("Поле", "Колонка БД"), ("Порог", "limit_kzt"), ("Карта", "card_id")]


def test_empty_table_row_is_kept(tmp_path: Path) -> None:
    """Пустая строка таблицы (например, незаполненный блок подписей) не теряется."""
    rows = (("Подразделение", "Подпись"), ("", ""))
    path = write_pdf(tmp_path / "sign.pdf", [[P(BODY), T(rows)]], header=None)

    table = [b for b in load_document(path).blocks if b.location.table == 1]

    assert [(b.kind, b.text, b.cells) for b in table] == [
        ("table_header", "Подразделение | Подпись", ("Подразделение", "Подпись")),
        ("table_row", "", ("", "")),
    ]


HEAD = ("Проверка", "Отметка")


@pytest.mark.parametrize("repeat_header", [True, False])
def test_table_continued_on_next_page_is_one_table(tmp_path: Path, repeat_header: bool) -> None:
    first = T((HEAD, ("Версия указана", "Нет"), ("Есть откат", "Нет")))
    rest = ((HEAD,) if repeat_header else ()) + (("Есть тесты", "Да"), ("Есть схема", "Да"))
    path = write_pdf(tmp_path / "long.pdf", [[H(1, "Чек-лист"), first], [T(rest)]])

    rows = [b for b in load_document(path).blocks if b.location.table is not None]

    assert [(b.kind, b.location.table, b.location.row, b.location.page) for b in rows] == [
        ("table_header", 1, 1, 1),
        ("table_row", 1, 2, 1),
        ("table_row", 1, 3, 1),
        ("table_row", 1, 4, 2),
        ("table_row", 1, 5, 2),
    ]
    assert all(b.header == HEAD for b in rows)
    assert rows[3].cells == ("Есть тесты", "Да")


def test_different_tables_on_one_page_are_numbered_separately(tmp_path: Path) -> None:
    first = T((("A", "B"), ("1", "2")))
    second = T((("C", "D"), ("3", "4")))
    path = write_pdf(tmp_path / "two.pdf", [[first, P(BODY), second]], header=None)

    tables = [b.location.table for b in load_document(path).blocks if b.location.table]

    assert tables == [1, 1, 2, 2]


def test_page_without_text_layer_gives_warning(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "mixed.pdf", [[H(1, "Текст")], [IMG()]], header=None, footer=False)

    document = load_document(path)

    assert document.page_count == 2
    assert any("2" in w and "текст" in w.lower() for w in document.warnings)


def test_scanned_pdf_without_text_is_rejected(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "scan.pdf", [[IMG()]], header=None, footer=False)

    with pytest.raises(InputError, match="текстового слоя"):
        load_document(path)


def test_password_protected_pdf_is_rejected(tmp_path: Path) -> None:
    plain = write_pdf(tmp_path / "plain.pdf", PAGES)
    locked = tmp_path / "locked.pdf"
    with pymupdf.open(plain) as source:
        source.save(
            str(locked), encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="user", owner_pw="owner"
        )

    with pytest.raises(InputError, match="паролем"):
        load_document(locked)


def test_owner_password_only_pdf_is_readable(tmp_path: Path) -> None:
    plain = write_pdf(tmp_path / "plain.pdf", PAGES)
    restricted = tmp_path / "restricted.pdf"
    with pymupdf.open(plain) as source:
        source.save(str(restricted), encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner")

    assert simplify(load_document(restricted)) == expected_blocks(SAMPLE)


def test_damaged_xref_is_repaired_with_warning(tmp_path: Path) -> None:
    good = write_pdf(tmp_path / "good.pdf", PAGES).read_bytes()
    damaged = good.replace(b"startxref", b"startxrex")
    assert damaged != good
    path = tmp_path / "damaged.pdf"
    path.write_bytes(damaged)

    document = load_document(path)

    assert simplify(document) == expected_blocks(SAMPLE)
    assert any("восстановлен" in w for w in document.warnings)


@pytest.mark.parametrize(
    "content",
    [
        b"%PDF-1.7\n" + bytes(range(256)) * 8,
        b"<html>not a pdf</html>",
        b"\x89PNG\r\n\x1a\n" + bytes(range(256)),
    ],
)
def test_corrupted_or_fake_pdf_is_rejected(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(content)

    with pytest.raises(InputError, match="PDF повреждён"):
        load_document(path)


def test_empty_pdf_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"")

    with pytest.raises(InputError, match="пуст"):
        load_document(path)
