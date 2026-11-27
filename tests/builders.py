"""Генераторы тестовых документов: одно и то же содержание в разных форматах.

Документы собственные (вымышленный «Сервис уведомлений»), к официальному заданию не относятся.
"""

from __future__ import annotations

import html
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import docx
import pymupdf

from validator.model import Document


@dataclass(frozen=True)
class H:
    """Заголовок."""

    level: int
    text: str


@dataclass(frozen=True)
class P:
    """Абзац. html — своя разметка для PDF (например, жирный фрагмент)."""

    text: str
    html: str | None = None


@dataclass(frozen=True)
class LI:
    """Пункт маркированного списка."""

    text: str


@dataclass(frozen=True)
class T:
    """Таблица: первая строка — шапка."""

    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class IMG:
    """Картинка без текста (только для PDF: имитация скана)."""


Element = H | P | LI | T | IMG
Writer = Callable[[Path, Sequence[Element]], Path]

SAMPLE: tuple[Element, ...] = (
    H(1, "Сервис уведомлений"),
    P("Документ описывает требования к сервису."),
    H(2, "Функциональные требования"),
    LI("R-1. Система хранит настройку порога."),
    LI("R-2. Система отправляет уведомление клиенту."),
    T((("Поле", "Значение"), ("Порог", "1 000 ₸"), ("Канал", "push"))),
    H(2, "Ограничения"),
    P("Время ответа не более 2 секунд."),
)

Simple = tuple[str, int | None, str]


def normalize(text: str) -> str:
    return " ".join(text.split())


def expected_blocks(elements: Sequence[Element], *, tables: bool = True) -> list[Simple]:
    out: list[Simple] = []
    for element in elements:
        if isinstance(element, H):
            out.append(("heading", element.level, element.text))
        elif isinstance(element, P):
            out.append(("paragraph", None, element.text))
        elif isinstance(element, LI):
            out.append(("list_item", None, element.text))
        elif isinstance(element, T) and tables:
            header, *rows = element.rows
            out.append(("table_header", None, " | ".join(header)))
            out.extend(("table_row", None, " | ".join(row)) for row in rows)
    return out


def simplify(document: Document) -> list[Simple]:
    return [(b.kind, b.level, normalize(b.text)) for b in document.blocks]


# --- Markdown / TXT ----------------------------------------------------------


def write_markdown(path: Path, elements: Sequence[Element]) -> Path:
    chunks: list[str] = []
    for element in elements:
        if isinstance(element, H):
            chunks.append("#" * element.level + " " + element.text)
        elif isinstance(element, P):
            chunks.append(element.text)
        elif isinstance(element, LI):
            chunks.append("* " + element.text)
        elif isinstance(element, T):
            header, *rows = element.rows
            lines = [
                "| " + " | ".join(header) + " |",
                "| " + " | ".join("---" for _ in header) + " |",
                *("| " + " | ".join(row) + " |" for row in rows),
            ]
            chunks.append("\n".join(lines))
    path.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
    return path


def write_text(path: Path, elements: Sequence[Element]) -> Path:
    """TXT: заголовки в стиле «# …», пункты «- …», таблицы не переносятся."""
    chunks: list[str] = []
    for element in elements:
        if isinstance(element, H):
            chunks.append("#" * element.level + " " + element.text)
        elif isinstance(element, P):
            chunks.append(element.text)
        elif isinstance(element, LI):
            chunks.append("- " + element.text)
    path.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
    return path


# --- DOCX --------------------------------------------------------------------


def write_docx(path: Path, elements: Sequence[Element]) -> Path:
    document = docx.Document()
    for element in elements:
        if isinstance(element, H):
            document.add_heading(element.text, level=element.level)
        elif isinstance(element, P):
            document.add_paragraph(element.text)
        elif isinstance(element, LI):
            document.add_paragraph(element.text, style="List Bullet")
        elif isinstance(element, T):
            table = document.add_table(rows=len(element.rows), cols=len(element.rows[0]))
            for i, row in enumerate(element.rows):
                for j, value in enumerate(row):
                    table.cell(i, j).text = value
    document.save(str(path))
    return path


# --- PDF ---------------------------------------------------------------------

PAGE_WIDTH, PAGE_HEIGHT = 595, 842
HEADING_SIZES = {1: 18, 2: 14, 3: 12}
PDF_HEADER = "Сервис уведомлений — черновик"


def write_pdf(
    path: Path,
    pages: Sequence[Sequence[Element]],
    *,
    header: str | None = PDF_HEADER,
    footer: bool = True,
) -> Path:
    """Многостраничный PDF: колонтитул сверху, «Страница N из M» снизу, таблицы с линиями."""
    document = pymupdf.open()
    for number, elements in enumerate(pages, 1):
        page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        if header:
            _text(page, 22, header, 9)
        y = 70.0
        for element in elements:
            if isinstance(element, H):
                y = _text(page, y, element.text, HEADING_SIZES[element.level], bold=True) + 10
            elif isinstance(element, P):
                y = _text(page, y, element.text, 11, markup=element.html) + 8
            elif isinstance(element, LI):
                y = _text(page, y, "• " + element.text, 11) + 4
            elif isinstance(element, T):
                y = _table(page, y, element.rows) + 12
            else:
                y = _image(page, y) + 10
        if footer:
            _text(page, 808, f"Страница {number} из {len(pages)}", 9)
    document.save(str(path))
    document.close()
    return path


def write_sample_pdf(path: Path, elements: Sequence[Element]) -> Path:
    """Одна страница без верхнего колонтитула: на одной странице его не отличить от текста."""
    return write_pdf(path, [elements], header=None)


def _text(
    page: pymupdf.Page,
    y: float,
    text: str,
    size: int,
    *,
    bold: bool = False,
    markup: str | None = None,
) -> float:
    body = markup if markup is not None else html.escape(text)
    weight = "bold" if bold else "normal"
    rect = pymupdf.Rect(50, y, 545, y + 300)
    spare, _ = page.insert_htmlbox(
        rect,
        f'<p style="font-size:{size}pt;font-weight:{weight};margin:0">{body}</p>',
        scale_low=1,
    )
    assert spare >= 0, f"текст не поместился: {text!r}"
    return float(y + rect.height - spare)


def _table(page: pymupdf.Page, y: float, rows: tuple[tuple[str, ...], ...]) -> float:
    # таблица должна умещаться между полями страницы (x от 50 до 545)
    col_width, row_height = min(200, 495 // len(rows[0])), 22
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            rect = pymupdf.Rect(
                50 + j * col_width,
                y + i * row_height,
                50 + (j + 1) * col_width,
                y + (i + 1) * row_height,
            )
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            padded = pymupdf.Rect(rect.x0 + 4, rect.y0 + 4, rect.x1 - 4, rect.y1 - 2)
            spare, _ = page.insert_htmlbox(
                padded,
                f'<p style="font-size:10pt;margin:0">{html.escape(value)}</p>',
                scale_low=1,
            )
            assert spare >= 0, f"ячейка не поместилась: {value!r}"
    return y + len(rows) * row_height


def _image(page: pymupdf.Page, y: float) -> float:
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40), False)
    pixmap.clear_with(180)
    page.insert_image(pymupdf.Rect(50, y, 250, y + 200), pixmap=pixmap)
    return y + 200
