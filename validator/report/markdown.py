"""Markdown-отчёт на русском.

Этап 0: только паспорт прогона. Разделы анализа добавляются на следующих этапах.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from validator.passport import RunPassport


def _cell(value: str) -> str:
    """Экранировать значение для ячейки таблицы Markdown."""
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _format_timestamp(moment: datetime) -> str:
    """«2026-11-27 10:20:29 UTC+05:00».

    Имя часового пояса (%Z) на Windows локализовано и неоднозначно («Кызылорда (зима)»),
    поэтому в отчёт пишется явное смещение от UTC.
    """
    offset = moment.utcoffset() or timedelta(0)
    sign = "-" if offset < timedelta(0) else "+"
    minutes = abs(int(offset.total_seconds())) // 60
    return f"{moment:%Y-%m-%d %H:%M:%S} UTC{sign}{minutes // 60:02d}:{minutes % 60:02d}"


def render_report(passport: RunPassport) -> str:
    started = _format_timestamp(passport.started_at)
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
        ("Дата и время прогона", started),
        ("Версия агента", passport.agent_version),
        ("Модель", model),
    ]
    table = "\n".join(f"| {_cell(name)} | {_cell(value)} |" for name, value in rows)
    return (
        f"# Отчёт о проверке требований: {_cell(passport.document_name)}\n\n"
        "## 1. Паспорт прогона\n\n"
        "| Поле | Значение |\n"
        "| :--- | :--- |\n"
        f"{table}\n\n"
        "> Каркас агента (этап 0): анализ требований ещё не подключён.\n"
    )
