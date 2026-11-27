"""Markdown-отчёт на русском.

Этап 2: паспорт прогона, сводка извлечения текста и нарезка требований.
Разделы анализа добавляются далее.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from validator.ingest.common import one_line
from validator.model import FORMAT_TITLES, Document
from validator.passport import RunPassport
from validator.segmentation import CATEGORY_TITLES, Requirement, Segmentation
from validator.segmentation.text import parse_prefix

_BRIEF_LIMIT = 140


def _cell(value: str) -> str:
    """Экранировать значение для ячейки таблицы Markdown."""
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _table(rows: list[tuple[str, str]]) -> str:
    lines = ["| Поле | Значение |", "| :--- | :--- |"]
    lines += [f"| {_cell(name)} | {_cell(value)} |" for name, value in rows]
    return "\n".join(lines)


def _format_timestamp(moment: datetime) -> str:
    """«2026-11-27 10:20:29 UTC+05:00».

    Имя часового пояса (%Z) на Windows локализовано и неоднозначно («Кызылорда (зима)»),
    поэтому в отчёт пишется явное смещение от UTC.
    """
    offset = moment.utcoffset() or timedelta(0)
    sign = "-" if offset < timedelta(0) else "+"
    minutes = abs(int(offset.total_seconds())) // 60
    return f"{moment:%Y-%m-%d %H:%M:%S} UTC{sign}{minutes // 60:02d}:{minutes % 60:02d}"


def _passport_section(passport: RunPassport) -> str:
    model = (
        f"{passport.provider} / {passport.model}"
        if passport.llm_used
        else f"не вызывалась (настроено: {passport.provider} / {passport.model})"
    )
    rows = [
        ("Документ", passport.document_name),
        ("Путь", passport.document_path),
        ("SHA-256 документа", passport.document_sha256),
        ("Размер текста, символов", str(passport.document_chars)),
        ("Дата и время прогона", _format_timestamp(passport.started_at)),
        ("Версия агента", passport.agent_version),
        ("Модель", model),
    ]
    return "## 1. Паспорт прогона\n\n" + _table(rows)


def _extraction_section(document: Document) -> str:
    rows: list[tuple[str, str]] = [("Формат", FORMAT_TITLES[document.format])]
    if document.page_count is not None:
        rows.append(("Страниц", str(document.page_count)))
    if document.encoding is not None:
        rows.append(("Кодировка", document.encoding))
    rows += [
        ("Заголовков", str(document.count("heading"))),
        ("Абзацев", str(document.count("paragraph"))),
        ("Пунктов списков", str(document.count("list_item"))),
        ("Таблиц", str(document.table_count)),
        ("Строк таблиц (без шапки)", str(document.count("table_row"))),
        ("Фрагментов кода", str(document.count("code"))),
        ("Всего блоков", str(len(document.blocks))),
    ]
    section = "## 2. Извлечение текста\n\n" + _table(rows).replace("| Поле |", "| Показатель |", 1)
    if document.warnings:
        section += "\n\nПредупреждения при чтении:\n\n" + "\n".join(
            f"- {warning}" for warning in document.warnings
        )
    return section


def _brief(fragment: Requirement) -> str:
    """Дословная цитата в одну строку; ID документа в начале убран — он в отдельной колонке."""
    text = fragment.text
    if fragment.doc_id:
        text = text[parse_prefix(text, "paragraph").body_start :]
    text = one_line(text)
    if len(text) <= _BRIEF_LIMIT:
        return text
    cut = text[:_BRIEF_LIMIT].rsplit(" ", 1)[0]
    return f"{cut}…"


def _grid(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(":---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(_cell(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def _segmentation_section(segmentation: Segmentation) -> str:
    fragments = segmentation.fragments
    composite = {f.parent_id for f in fragments if f.is_statement and f.role == "requirement"}
    summary = [
        ("Атомарных требований", str(len(segmentation.atomic_requirements))),
        ("Составных фрагментов (разделены на утверждения)", str(len(composite))),
        ("Неоднозначных фрагментов", str(len(segmentation.ambiguous))),
        (
            "Пояснений и примеров",
            str(sum(1 for f in fragments if f.role in ("explanation", "example"))),
        ),
        ("Справочных фрагментов", str(sum(1 for f in fragments if f.role == "reference"))),
        (
            "Заголовков, шапок таблиц и кода",
            str(sum(1 for f in fragments if f.role in ("heading", "table_header", "code"))),
        ),
    ]
    parts = [
        "## 3. Нарезка требований",
        _table(summary).replace("| Поле |", "| Показатель |", 1),
        "### Требования",
    ]
    atomic = segmentation.atomic_requirements
    if atomic:
        parts.append(
            _grid(
                ("ID", "ID в документе", "Тип", "Формулировка", "Место"),
                [
                    (
                        f.id,
                        f.doc_id or "—",
                        CATEGORY_TITLES[f.category],
                        _brief(f),
                        f.location.describe(),
                    )
                    for f in atomic
                ],
            )
        )
    else:
        parts.append("Требования не обнаружены.")
    if segmentation.ambiguous:
        parts += [
            "### Неоднозначные фрагменты",
            "Не считаются требованиями, пока это не подтверждено проверкой.",
            _grid(
                ("ID", "Формулировка", "Почему неоднозначно", "Место"),
                [
                    (f.id, _brief(f), f.warnings[0] if f.warnings else "—", f.location.describe())
                    for f in segmentation.ambiguous
                ],
            ),
        ]
    notes = [f"- {warning}" for warning in segmentation.warnings]
    notes += [
        f"- `{f.id}`: {warning}"
        for f in fragments
        if f.role == "requirement"
        for warning in f.warnings
    ]
    if notes:
        parts += ["### Предупреждения сегментации", "\n".join(notes)]
    return "\n\n".join(parts)


def render_report(
    passport: RunPassport, document: Document, segmentation: Segmentation | None = None
) -> str:
    sections = [
        f"# Отчёт о проверке требований: {_cell(passport.document_name)}",
        _passport_section(passport),
        _extraction_section(document),
    ]
    if segmentation is not None:
        sections.append(_segmentation_section(segmentation))
    sections.append("> Агент на этапе 2: требования выделены, анализ качества ещё не подключён.")
    return "\n\n".join(sections) + "\n"
