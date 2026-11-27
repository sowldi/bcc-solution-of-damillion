"""Общие помощники читателей: результат разбора, очистка текста, кодировки, разделы."""

from __future__ import annotations

import codecs
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from validator.errors import InputError
from validator.model import Block


@dataclass(frozen=True)
class Extraction:
    """Что возвращает читатель формата; Document собирается в load_document."""

    blocks: tuple[Block, ...]
    encoding: str | None = None
    page_count: int | None = None
    warnings: tuple[str, ...] = ()


# Невидимые символы, которые не несут текста, но мешают точному цитированию:
# мягкий перенос, пробел нулевой ширины, «склейка слов», BOM внутри текста.
_INVISIBLE = dict.fromkeys(map(ord, "­​⁠﻿"))


def clean_text(text: str) -> str:
    """Единые переводы строк (\\n), нормализация Unicode NFC, без невидимых символов."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text).translate(_INVISIBLE)


def one_line(text: str) -> str:
    """Схлопнуть пробельные символы — для заголовков в пути раздела."""
    return " ".join(text.split())


def row_text(cells: tuple[str, ...]) -> str:
    """Текст строки таблицы: ячейки через « | »; пустая строка таблицы — пустой текст."""
    return " | ".join(cells) if any(cells) else ""


class SectionTracker:
    """Стек заголовков: по уровню нового заголовка закрывает вложенные разделы."""

    def __init__(self) -> None:
        self._stack: list[tuple[int, str]] = []

    @property
    def path(self) -> tuple[str, ...]:
        return tuple(title for _, title in self._stack)

    def enter(self, level: int, title: str) -> tuple[str, ...]:
        """Открыть раздел; вернуть путь родителя (место самого заголовка)."""
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        parent = self.path
        self._stack.append((level, one_line(title)))
        return parent


_ALLOWED_CONTROLS = frozenset("\t\n\r\f")
_MAX_CONTROL_SHARE = 0.01


def _looks_binary(text: str) -> bool:
    if not text:
        return False
    controls = sum(
        1 for ch in text if unicodedata.category(ch) == "Cc" and ch not in _ALLOWED_CONTROLS
    )
    return controls / len(text) > _MAX_CONTROL_SHARE


def decode_text(data: bytes, source: Path) -> tuple[str, str, tuple[str, ...]]:
    """Определить кодировку текстового файла: UTF-8 (с BOM или без), UTF-16 с BOM, Windows-1251.

    Возвращает (текст, кодировка, предупреждения). Двоичные данные — InputError.
    """
    binary = InputError(f"Файл не похож на текстовый (двоичные данные): {source}")
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        try:
            text = data.decode("utf-16")
        except UnicodeDecodeError:
            raise binary from None
        if _looks_binary(text):
            raise binary
        return text, "utf-16", ()
    if b"\x00" in data:
        raise binary
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    else:
        if _looks_binary(text):
            raise binary
        return text, "utf-8", ()
    try:
        text = data.decode("cp1251")
    except UnicodeDecodeError:
        raise binary from None
    if _looks_binary(text):
        raise binary
    warning = (
        "Файл не в UTF-8: кодировка определена как Windows-1251 (cp1251). "
        "Проверьте, что текст в отчёте прочитан верно."
    )
    return text, "cp1251", (warning,)
