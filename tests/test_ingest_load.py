from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import validator.ingest as ingest
from tests.builders import (
    SAMPLE,
    Writer,
    expected_blocks,
    simplify,
    write_docx,
    write_markdown,
    write_sample_pdf,
    write_text,
)
from validator.errors import InputError
from validator.ingest import SUPPORTED_SUFFIXES, load_document


@pytest.mark.parametrize(
    ("writer", "name", "fmt", "tables"),
    [
        (write_markdown, "spec.md", "markdown", True),
        (write_text, "spec.txt", "text", False),
        (write_docx, "spec.docx", "docx", True),
        (write_sample_pdf, "spec.pdf", "pdf", True),
    ],
    ids=["md", "txt", "docx", "pdf"],
)
def test_same_content_gives_same_blocks_in_every_format(
    tmp_path: Path, writer: Writer, name: str, fmt: str, tables: bool
) -> None:
    """Единый результат: одно содержание → одинаковые блоки независимо от формата."""
    path = writer(tmp_path / name, SAMPLE)

    document = load_document(path)

    assert document.format == fmt
    assert simplify(document) == expected_blocks(SAMPLE, tables=tables)
    assert document.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert document.source == path


def test_every_block_has_a_location(tmp_path: Path) -> None:
    for writer, name in [
        (write_markdown, "a.md"),
        (write_text, "a.txt"),
        (write_docx, "a.docx"),
        (write_sample_pdf, "a.pdf"),
    ]:
        document = load_document(writer(tmp_path / name, SAMPLE))
        for block in document.blocks:
            assert block.location.describe() != "место не определено", (name, block)


def test_supported_suffixes() -> None:
    assert set(SUPPORTED_SUFFIXES) == {".md", ".markdown", ".txt", ".pdf", ".docx"}


def test_uppercase_extension_is_accepted(tmp_path: Path) -> None:
    path = write_markdown(tmp_path / "SPEC.MD", SAMPLE)

    assert load_document(path).format == "markdown"


@pytest.mark.parametrize(
    ("name", "hint"),
    [("old.doc", ".docx"), ("table.xlsx", "Поддерживаются"), ("noext", "без расширения")],
)
def test_unsupported_formats_explain_what_to_do(tmp_path: Path, name: str, hint: str) -> None:
    path = tmp_path / name
    path.write_bytes(b"data")

    with pytest.raises(InputError, match="Неподдерживаемый формат") as exc:
        load_document(path)

    assert hint in str(exc.value)


def test_missing_file_and_directory(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="не найден"):
        load_document(tmp_path / "nope.md")
    with pytest.raises(InputError, match="папка"):
        load_document(tmp_path)


def test_too_large_file_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingest, "MAX_FILE_BYTES", 10)
    path = tmp_path / "big.md"
    path.write_text("Слишком длинный документ", encoding="utf-8")

    with pytest.raises(InputError, match="слишком большой"):
        load_document(path)
