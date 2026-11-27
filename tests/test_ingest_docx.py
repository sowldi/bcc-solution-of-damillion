from __future__ import annotations

import io
import zipfile
from pathlib import Path

import docx
import pytest
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

from tests.builders import SAMPLE, expected_blocks, simplify, write_docx
from validator.errors import InputError
from validator.ingest import load_document
from validator.model import Block, Document

W = nsdecls("w")


def _save(document: docx.document.Document, tmp_path: Path, name: str = "spec.docx") -> Document:
    path = tmp_path / name
    document.save(str(path))
    return load_document(path)


def _find(document: Document, text: str) -> Block:
    return next(b for b in document.blocks if b.text.startswith(text))


def test_blocks_match_source_structure(tmp_path: Path) -> None:
    document = load_document(write_docx(tmp_path / "spec.docx", SAMPLE))

    assert simplify(document) == expected_blocks(SAMPLE)
    assert document.format == "docx"
    assert document.page_count is None


def test_sections_paragraph_numbers_and_title_level(tmp_path: Path) -> None:
    source = docx.Document()
    source.add_heading("Сервис", level=0)  # стиль Title
    source.add_paragraph("")  # пустой абзац тоже учитывается в нумерации
    source.add_heading("Требования", level=1)
    source.add_paragraph("Текст требования.")

    document = _save(source, tmp_path)

    title, heading, text = document.blocks
    assert (title.kind, title.level, title.location.paragraph) == ("heading", 0, 1)
    assert (heading.level, heading.location.section_path) == (1, ("Сервис",))
    assert text.location.paragraph == 4
    assert text.location.section_path == ("Сервис", "Требования")


def test_list_styles_and_numbering_are_list_items(tmp_path: Path) -> None:
    source = docx.Document()
    source.add_paragraph("Маркированный пункт", style="List Bullet")
    source.add_paragraph("Нумерованный пункт", style="List Number")
    numbered = source.add_paragraph("Пункт с numPr")
    numbered._p.get_or_add_pPr().append(
        parse_xml(f'<w:numPr {W}><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>')
    )
    source.add_paragraph("Обычный абзац")

    kinds = [b.kind for b in _save(source, tmp_path).blocks]

    assert kinds == ["list_item", "list_item", "list_item", "paragraph"]


def test_table_rows_header_and_merged_cells(tmp_path: Path) -> None:
    source = docx.Document()
    source.add_heading("Параметры", level=1)
    table = source.add_table(rows=3, cols=3)
    for j, value in enumerate(["Поле", "Значение", "Комментарий"]):
        table.cell(0, j).text = value
    table.cell(1, 0).text = "Порог"
    table.cell(1, 1).text = "10 ₸"
    table.cell(1, 1).merge(table.cell(1, 2))

    document = _save(source, tmp_path)
    rows = [b for b in document.blocks if b.location.table is not None]

    assert [(b.kind, b.location.table, b.location.row) for b in rows] == [
        ("table_header", 1, 1),
        ("table_row", 1, 2),
        ("table_row", 1, 3),
    ]
    assert rows[1].cells == ("Порог", "10 ₸")
    assert rows[1].header == ("Поле", "Значение", "Комментарий")
    assert rows[1].location.section_path == ("Параметры",)
    assert rows[2].text == ""


def test_line_breaks_and_tabs_are_kept(tmp_path: Path) -> None:
    source = docx.Document()
    paragraph = source.add_paragraph("Первая строка")
    paragraph.add_run().add_break()
    paragraph.add_run("вторая\tчасть")

    assert _save(source, tmp_path).blocks[0].text == "Первая строка\nвторая\tчасть"


def test_tracked_insertions_kept_and_deletions_dropped(tmp_path: Path) -> None:
    source = docx.Document()
    paragraph = source.add_paragraph("Базовый текст ")
    stamp = 'w:id="{}" w:author="t" w:date="2026-01-01T00:00:00Z"'
    paragraph._p.append(
        parse_xml(f"<w:ins {W} {stamp.format(1)}><w:r><w:t>вставка</w:t></w:r></w:ins>")
    )
    paragraph._p.append(
        parse_xml(f"<w:del {W} {stamp.format(2)}><w:r><w:delText>удалено</w:delText></w:r></w:del>")
    )

    assert _save(source, tmp_path).blocks[0].text == "Базовый текст вставка"


def test_custom_style_with_outline_level_is_heading(tmp_path: Path) -> None:
    source = docx.Document()
    style = source.styles.add_style("Мой заголовок", WD_STYLE_TYPE.PARAGRAPH)
    style.element.get_or_add_pPr().append(parse_xml(f'<w:outlineLvl {W} w:val="1"/>'))
    source.add_paragraph("Особые правила", style=style)

    block = _save(source, tmp_path).blocks[0]

    assert (block.kind, block.level) == ("heading", 2)


def test_table_of_contents_is_skipped(tmp_path: Path) -> None:
    source = docx.Document()
    source.styles.add_style("toc 1", WD_STYLE_TYPE.PARAGRAPH)
    source.add_paragraph("Требования\t3", style="toc 1")
    source.add_heading("Требования", level=1)

    assert [b.kind for b in _save(source, tmp_path).blocks] == ["heading"]


def test_content_controls_are_read(tmp_path: Path) -> None:
    source = docx.Document()
    source.add_paragraph("До элемента")
    body = source.element.body
    body.insert(
        len(body) - 1,
        parse_xml(
            f"<w:sdt {W}><w:sdtPr/><w:sdtContent><w:p><w:r>"
            "<w:t>Внутри элемента управления</w:t></w:r></w:p></w:sdtContent></w:sdt>"
        ),
    )

    texts = [b.text for b in _save(source, tmp_path).blocks]

    assert texts == ["До элемента", "Внутри элемента управления"]


def _zip(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _broken_document_xml(tmp_path: Path) -> bytes:
    good = write_docx(tmp_path / "good.docx", SAMPLE).read_bytes()
    with zipfile.ZipFile(io.BytesIO(good)) as source:
        files = {name: source.read(name) for name in source.namelist()}
    files["word/document.xml"] = b"<w:document><w:body><w:p>"  # обрезанный XML
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "make",
    [
        lambda _: bytes(range(256)) * 4,
        lambda _: _zip({"hello.txt": "не документ"}),
        lambda _: _zip(
            {
                "[Content_Types].xml": (
                    '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/'
                    'package/2006/content-types"><Default Extension="xml" '
                    'ContentType="application/xml"/></Types>'
                ),
                "_rels/.rels": (
                    '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/'
                    'package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.'
                    'openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                    'Target="xl/workbook.xml"/></Relationships>'
                ),
                "xl/workbook.xml": "<workbook/>",
            }
        ),
        _broken_document_xml,
    ],
    ids=["garbage", "zip-without-document", "spreadsheet", "broken-xml"],
)
def test_corrupted_docx_is_rejected(tmp_path: Path, make: object) -> None:
    path = tmp_path / "broken.docx"
    path.write_bytes(make(tmp_path))  # type: ignore[operator]

    with pytest.raises(InputError, match="DOCX повреждён"):
        load_document(path)


def test_docx_without_text_is_empty(tmp_path: Path) -> None:
    source = docx.Document()
    source.add_paragraph("")

    with pytest.raises(InputError, match="пуст"):
        _save(source, tmp_path)
