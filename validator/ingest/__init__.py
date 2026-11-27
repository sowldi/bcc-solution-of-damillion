"""Чтение документа требований любого поддерживаемого формата в единую модель Document."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

from validator.errors import InputError
from validator.ingest.common import Extraction
from validator.ingest.markdown import read_markdown
from validator.ingest.pdf import read_pdf
from validator.ingest.text import read_text
from validator.ingest.word import read_docx
from validator.model import Document, DocumentFormat

Reader = Callable[[bytes, Path], Extraction]

_READERS: dict[str, tuple[DocumentFormat, Reader]] = {
    ".md": ("markdown", read_markdown),
    ".markdown": ("markdown", read_markdown),
    ".txt": ("text", read_text),
    ".pdf": ("pdf", read_pdf),
    ".docx": ("docx", read_docx),
}
SUPPORTED_SUFFIXES = tuple(_READERS)

_HINTS = {
    ".doc": "старый формат Word — сохраните документ как .docx или PDF",
    ".rtf": "сохраните документ как .docx или PDF",
    ".odt": "сохраните документ как .docx или PDF",
}

MAX_FILE_BYTES = 50 * 1024 * 1024


def load_document(path: Path) -> Document:
    """Прочитать файл и вернуть Document. Любая проблема с файлом — InputError с пояснением."""
    if not path.exists():
        raise InputError(f"Документ не найден: {path}")
    if not path.is_file():
        raise InputError(f"Ожидался файл, а это папка: {path}")
    suffix = path.suffix.lower()
    if suffix not in _READERS:
        hint = _HINTS.get(suffix, f"Поддерживаются: {', '.join(SUPPORTED_SUFFIXES)}")
        raise InputError(f"Неподдерживаемый формат «{suffix or 'без расширения'}»: {hint}.")
    try:
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise InputError(
                f"Файл слишком большой ({size / 1024 / 1024:.1f} МБ, "
                f"максимум {MAX_FILE_BYTES / 1024 / 1024:.0f} МБ): {path}"
            )
        data = path.read_bytes()
    except OSError as exc:
        raise InputError(f"Не удалось прочитать документ {path}: {exc.strerror or exc}") from None
    if not data.strip():
        raise InputError(f"Документ пуст: {path}")

    document_format, reader = _READERS[suffix]
    extraction = reader(data, path)
    document = Document(
        source=path,
        format=document_format,
        sha256=hashlib.sha256(data).hexdigest(),
        blocks=extraction.blocks,
        encoding=extraction.encoding,
        page_count=extraction.page_count,
        warnings=extraction.warnings,
    )
    if not document.has_text:
        raise InputError(f"Документ пуст: в нём не найден текст: {path}")
    return document


__all__ = ["MAX_FILE_BYTES", "SUPPORTED_SUFFIXES", "load_document"]
