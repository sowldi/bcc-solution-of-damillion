"""Markdown-отчёт на русском.

Этап 4: паспорт прогона, сводка извлечения текста, нарезка требований, детерминированная
проверка (замечания правил, прошедшие проверку доказательств) и гипотезы LLM — отдельным
разделом и со статусом «не подтверждено»: с замечаниями правил они не смешиваются.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from validator.criteria import criterion
from validator.findings import (
    REVIEW_CATEGORY_TITLES,
    RULE_CATEGORY_TITLES,
    SEVERITIES,
    SEVERITY_TITLES,
    AbsenceClaim,
    Evidence,
    Finding,
    Severity,
)
from validator.ingest.common import one_line
from validator.lint import LintResult
from validator.model import FORMAT_TITLES, Document
from validator.passport import RunPassport
from validator.segmentation import CATEGORY_TITLES, Requirement, Segmentation
from validator.segmentation.text import parse_prefix

if TYPE_CHECKING:  # отчёт не загружает LLM-слой: в режиме --rules-only его нет
    from validator.llm import LLMAnalysis

# Причины, по которым требование не отправлено модели (коды validator.llm.analysis).
_SKIP_TITLES: dict[str, str] = {
    "budget": "лимит запросов LLM_MAX_REQUESTS",
    "empty": "пустой текст",
    "no_source": "некорректная привязка к источнику",
}

_BRIEF_LIMIT = 140
_SEVERITY_GROUPS: dict[Severity, str] = {
    "critical": "Критичные",
    "major": "Существенные",
    "minor": "Незначительные",
    "info": "Информационные",
}
# Символы, которые Markdown превращает в разметку: в цитате они должны остаться текстом.
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_\[\]<>~|])")


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
    if passport.mode == "rules-only":
        mode = "только детерминированные правила (--rules-only)"
        model = "не используется (режим --rules-only)"
    else:
        mode = "стандартный"
        configured = f"{passport.provider} / {passport.model}"
        model = configured if passport.llm_used else f"не вызывалась (настроено: {configured})"
    rows = [
        ("Документ", passport.document_name),
        ("Путь", passport.document_path),
        ("SHA-256 документа", passport.document_sha256),
        ("Размер текста, символов", str(passport.document_chars)),
        ("Дата и время прогона", _format_timestamp(passport.started_at)),
        ("Версия агента", passport.agent_version),
        ("Режим", mode),
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


def _escape(text: str) -> str:
    return _MARKDOWN_SPECIAL.sub(r"\\\1", one_line(text))


def _quote(text: str) -> str:
    """Цитата в одну строку (переводы строк — пробелы), разметка Markdown экранирована."""
    return f"«{_escape(text)}»" if text else "(пусто)"


def _where(evidence: Evidence) -> str:
    places = dict.fromkeys(ref.location.describe() for ref in evidence.sources)
    return "; ".join(places) or "место не определено"


def _finding_block(number: int, finding: Finding) -> str:
    item = criterion(finding.criterion)
    lines = [f"**4.{number}. {finding.rule_id} · {finding.title}**", ""]
    if len(finding.evidence) == 1:
        evidence = finding.evidence[0]
        lines.append(f"- Требование: `{evidence.requirement_id}`")
        lines.append(f"- Цитата: {_quote(evidence.quote)} — {_where(evidence)}")
    else:
        lines.append("- Требования: " + ", ".join(f"`{rid}`" for rid in finding.requirement_ids))
        lines.append("- Цитаты:")
        lines += [
            f"  - `{e.requirement_id}`: {_quote(e.quote)} — {_where(e)}" for e in finding.evidence
        ]
    if finding.absence is not None:
        lines.append(f"- Отсутствие проверено в области: {_absence_scope(finding.absence)}")
    lines += [
        f"- Почему: {_escape(finding.explanation)}",
        f"- Как исправить: {_escape(finding.recommendation)}",
        f"- Критерий: {item.reference} — {item.title}",
        f"- ID замечания: `{finding.id}`",
    ]
    return "\n".join(lines)


def _absence_scope(claim: AbsenceClaim) -> str:
    target = _escape(claim.target)
    return {
        "table": f"таблица {target}",
        "section": f"раздел «{target}»",
        "requirement": f"требование `{claim.target}`",
        "id_registry": f"идентификаторы требований документа (нет «{target}»)",
        "heading_registry": f"нумерация разделов документа (нет раздела {target})",
    }[claim.scope]


def _lint_section(lint: LintResult, segmentation: Segmentation | None) -> str:
    atomic = len(segmentation.atomic_requirements) if segmentation else 0
    parts = [
        "## 4. Детерминированная проверка",
        f"Каталог правил {lint.catalog_version}, правил: {len(lint.rules)} — проверяемость, "
        "однозначность, полнота, согласованность, трассируемость. Правила формулировок "
        f"применены к атомарным требованиям ({atomic}), структурные — ко всему документу. "
        "Каждое замечание прошло Evidence Gate: оно привязано к этой версии документа "
        f"(SHA-256 {lint.document_sha256[:12]}…), цитата — точный фрагмент текста, "
        "извлечённого из документа (разметка формата в него не входит), место вычислено по "
        "исходному файлу, утверждение об отсутствии перепроверено в указанной области.",
    ]
    if segmentation is not None and segmentation.ambiguous:
        parts.append(
            f"Неоднозначные фрагменты ({len(segmentation.ambiguous)}) правилами формулировок "
            "не проверялись: сначала нужно подтвердить, что это требования."
        )
    parts += [
        _grid(
            ("Важность", "Замечаний"),
            [(SEVERITY_TITLES[severity], str(lint.count(severity))) for severity in SEVERITIES],
        ),
        f"Проверка доказательств: подтверждено — {len(lint.findings)}, "
        f"отклонено — {len(lint.rejected)}.",
    ]
    if not lint.findings:
        parts.append(
            "Детерминированные правила замечаний не нашли. Это не означает, что документ "
            "корректен: правила проверяют только формальные признаки — оценочные слова, "
            "заглушки, ссылки, идентификаторы, дубли; смысловые пропуски и противоречия ими "
            "не выявляются."
        )
    number = 0
    for severity in SEVERITIES:
        group = [finding for finding in lint.findings if finding.severity == severity]
        if not group:
            continue
        parts.append(f"### {_SEVERITY_GROUPS[severity]} ({len(group)})")
        for finding in group:
            number += 1
            parts.append(_finding_block(number, finding))
    if lint.rejected:
        parts += [
            "### Отклонено проверкой доказательств",
            "Эти замечания не включены в отчёт: их доказательства не подтвердились по документу "
            "(это ошибка правила, а не документа).",
            "\n".join(
                f"- `{rejected.finding.id}`: {_escape('; '.join(rejected.reasons))}"
                for rejected in lint.rejected
            ),
        ]
    return "\n\n".join(parts)


def _hypothesis_block(number: int, finding: Finding) -> str:
    item = criterion(finding.criterion)
    evidence = finding.evidence[0]
    kind = (
        REVIEW_CATEGORY_TITLES[finding.review_category]
        if finding.review_category
        else RULE_CATEGORY_TITLES[finding.category]
    )
    lines = [
        f"**5.{number}. {kind} · {_escape(finding.title)}**",
        "",
        f"- Требование: `{evidence.requirement_id}`",
        f"- Цитата: {_quote(evidence.quote)} — {_where(evidence)}",
        f"- Предполагаемая важность: {SEVERITY_TITLES[finding.severity]} (по категории)",
        f"- Обоснование модели: {_escape(finding.explanation)}",
        f"- Рекомендация модели: {_escape(finding.recommendation)}",
    ]
    if finding.confidence is not None:
        lines.append(f"- Уверенность модели: {finding.confidence:g} — на статус не влияет")
    lines += [
        f"- Критерий: {item.reference} — {item.title}",
        "- Происхождение цитаты: проверено Evidence Gate (точный текст этого требования, "
        "место в файле)",
        "- Смысл замечания: не проверен — это гипотеза модели",
        "- Статус: не подтверждено",
    ]
    return "\n".join(lines)


def _no_hypotheses(analysis: LLMAnalysis) -> str:
    """Почему в разделе нет гипотез — так, чтобы непроверенное не выглядело проверенным."""
    if analysis.requirements_sent == 0:
        return "LLM-обзор не выполнен: модель не получила ни одного требования."
    if analysis.rejected:
        return (
            f"Принятых гипотез нет: все предложенные моделью гипотезы ({len(analysis.rejected)}) "
            "отклонены — причины ниже."
        )
    if analysis.skipped:
        return (
            "Модель не предложила гипотез по проверенным требованиям "
            f"({analysis.requirements_sent} из {analysis.requirements_total})."
        )
    return "Модель не предложила гипотез."


def _llm_section(passport: RunPassport, analysis: LLMAnalysis | None) -> str | None:
    if analysis is None:
        if passport.mode == "rules-only":
            return "## 5. Анализ LLM\n\nLLM не использовался (режим --rules-only)."
        return None
    coverage = (
        f"Провайдер: {analysis.provider}, модель: {analysis.model}, версия промпта: "
        f"{analysis.prompt_version}. Проверено требований: {analysis.requirements_sent} из "
        f"{analysis.requirements_total} (по одному запросу на требование)."
    )
    counts = Counter(item.code for item in analysis.skipped)
    if counts:
        listed = ", ".join(f"{_SKIP_TITLES[code]} — {count}" for code, count in counts.items())
        coverage += f" Не проверены: {listed}."
    parts = [
        "## 5. Анализ LLM (гипотезы, не подтверждены)",
        coverage,
        "> Гипотезы LLM не являются замечаниями: модель только предполагает дефект. Ниже — "
        "гипотезы, прошедшие Evidence Gate: цитата точно и однозначно найдена в тексте "
        "названного требования этой версии документа. Это доказывает происхождение цитаты, "
        "но не смысл замечания — статус остаётся «не подтверждено» до смысловой проверки.",
    ]
    if not analysis.hypotheses:
        parts.append(_no_hypotheses(analysis))
    parts += [
        _hypothesis_block(number, finding) for number, finding in enumerate(analysis.hypotheses, 1)
    ]
    if analysis.rejected:
        parts += [
            f"### Отклонённые ответы модели ({len(analysis.rejected)})",
            "\n".join(
                f"- элемент ответа № {item.index + 1}"
                + (f" (`{item.requirement_id}`)" if item.requirement_id else "")
                + f": {_escape(' '.join(item.reasons))}"
                for item in analysis.rejected
            ),
        ]
    if analysis.skipped:
        parts += [
            f"### Требования без LLM-обзора ({len(analysis.skipped)})",
            "Модель их не получала — это не значит, что проблем нет.",
            "\n".join(
                f"- `{item.requirement_id}` — {_SKIP_TITLES[item.code]}"
                for item in analysis.skipped
            ),
        ]
    return "\n\n".join(parts)


def render_report(
    passport: RunPassport,
    document: Document,
    segmentation: Segmentation | None = None,
    lint: LintResult | None = None,
    llm: LLMAnalysis | None = None,
) -> str:
    sections = [
        f"# Отчёт о проверке требований: {_cell(passport.document_name)}",
        _passport_section(passport),
        _extraction_section(document),
    ]
    if segmentation is not None:
        sections.append(_segmentation_section(segmentation))
    if lint is not None:
        sections.append(_lint_section(lint, segmentation))
    llm_section = _llm_section(passport, llm)
    if llm_section is not None:
        sections.append(llm_section)
    sections.append(
        "> Агент на этапе 6: замечания правил подтверждены Evidence Gate; LLM-обзор каждого "
        "требования даёт гипотезы, прошедшие проверку цитат, но не смысла — смысловая проверка "
        "появится на следующих этапах."
    )
    return "\n\n".join(sections) + "\n"
