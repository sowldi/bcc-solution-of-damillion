from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from validator.ingest import load_document
from validator.model import Block, Document, Location
from validator.passport import RunPassport
from validator.report import render_report
from validator.segmentation import Segmentation, segment_document


def _passport(started_at: datetime) -> RunPassport:
    return RunPassport(
        document_name="spec.md",
        document_path="C:/docs/spec.md",
        document_sha256="0" * 64,
        document_chars=10,
        started_at=started_at,
        agent_version="0.1.0+abc1234",
        provider="gemini",
        model="gemini-3.6-flash",
        llm_used=False,
    )


def _document(**changes: object) -> Document:
    document = Document(
        source=Path("spec.pdf"),
        format="pdf",
        sha256="0" * 64,
        page_count=3,
        blocks=(
            Block("heading", "Раздел", Location(page=1), level=1),
            Block("paragraph", "Абзац", Location(page=1)),
            Block("list_item", "Пункт", Location(page=1)),
            Block("table_header", "A | B", Location(page=2, table=1, row=1), cells=("A", "B")),
            Block("table_row", "1 | 2", Location(page=2, table=1, row=2), cells=("1", "2")),
            Block("table_row", "3 | 4", Location(page=2, table=1, row=3), cells=("3", "4")),
            Block("code", "{}", Location(page=3)),
        ),
    )
    return replace(document, **changes)  # type: ignore[arg-type]


NOON = datetime(2026, 1, 1, 12, tzinfo=UTC)


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (timedelta(hours=5), "2026-11-27 10:20:29 UTC+05:00"),
        (timedelta(0), "2026-11-27 10:20:29 UTC+00:00"),
        (timedelta(hours=-3, minutes=-30), "2026-11-27 10:20:29 UTC-03:30"),
    ],
)
def test_run_time_uses_explicit_utc_offset(offset: timedelta, expected: str) -> None:
    """Время прогона однозначно на любой ОС: смещение от UTC, а не локализованное имя пояса."""
    started = datetime(2026, 11, 27, 10, 20, 29, tzinfo=timezone(offset, "Кызылорда (зима)"))

    text = render_report(_passport(started), _document())

    assert f"| Дата и время прогона | {expected} |" in text
    assert "Кызылорда" not in text


def test_table_cells_are_escaped() -> None:
    passport = replace(_passport(NOON), document_name="a|b\nc.md")

    text = render_report(passport, _document())

    assert "a\\|b c.md" in text


def test_extraction_summary() -> None:
    text = render_report(_passport(NOON), _document())

    assert "## 2. Извлечение текста" in text
    for row in [
        "| Формат | PDF |",
        "| Страниц | 3 |",
        "| Заголовков | 1 |",
        "| Абзацев | 1 |",
        "| Пунктов списков | 1 |",
        "| Таблиц | 1 |",
        "| Строк таблиц (без шапки) | 2 |",
        "| Фрагментов кода | 1 |",
        "| Всего блоков | 7 |",
    ]:
        assert row in text
    assert "Кодировка" not in text  # для PDF кодировка не применяется


def test_report_without_segmentation_keeps_stage_one_layout() -> None:
    text = render_report(_passport(NOON), _document())

    assert "Нарезка требований" not in text


def _segmentation(tmp_path: Path, markdown: str) -> tuple[Document, Segmentation]:
    path = tmp_path / "spec.md"
    path.write_text(markdown, encoding="utf-8")
    document = load_document(path)
    return document, segment_document(document)


def test_segmentation_section_lists_requirements_with_sources(tmp_path: Path) -> None:
    document, segmentation = _segmentation(
        tmp_path,
        "# Сервис\n\n## Функциональные требования\n\n"
        "* Ф-1. Читатель задаёт срок брони. По умолчанию бронь не продлевается.\n"
        "* Система отправляет push | SMS.\n\n# Описание\n\nСистема должна хранить журнал.\n",
    )

    text = render_report(_passport(NOON), document, segmentation)

    assert "## 3. Нарезка требований" in text
    assert "| ID | ID в документе | Тип | Формулировка | Место |" in text
    assert "| Ф-1/1 | Ф-1 | функциональное | Читатель задаёт срок брони. |" in text
    assert "Функциональные требования — строка 5" in text
    assert "push \\| SMS" in text  # ячейки таблицы экранируются
    assert "### Неоднозначные фрагменты" in text
    assert "Система должна хранить журнал." in text


def test_report_quotes_original_text_not_normalized(tmp_path: Path) -> None:
    document, segmentation = _segmentation(
        tmp_path,
        "# Сервис\n\n## Функциональные требования\n\n"
        "* R-1. Кнопка «Забронировать» активна после выбора книги.\n",
    )

    text = render_report(_passport(NOON), document, segmentation)

    assert (
        "| R-1 | R-1 | функциональное | Кнопка «Забронировать» активна после выбора книги. |"
        in text
    )


def test_segmentation_section_when_nothing_found(tmp_path: Path) -> None:
    document, segmentation = _segmentation(
        tmp_path, "# Заметки\n\nДокумент содержит справочную информацию.\n"
    )

    text = render_report(_passport(NOON), document, segmentation)

    assert "Требования не обнаружены" in text


def test_encoding_and_warnings_are_shown() -> None:
    document = _document(
        format="text", page_count=None, encoding="cp1251", warnings=("Кодировка угадана.",)
    )

    text = render_report(_passport(NOON), document)

    assert "| Кодировка | cp1251 |" in text
    assert "Страниц" not in text
    assert "- Кодировка угадана." in text
