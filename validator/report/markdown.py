"""Markdown-отчёт на русском.

Разделы: паспорт прогона, сводка извлечения текста, нарезка требований, детерминированная
проверка (замечания правил, прошедшие проверку доказательств), гипотезы LLM-обзора —
отдельным разделом и со статусом «не подтверждено», и Fact Graph (этап 7): подтверждённые
машинной проверкой связи отдельно от гипотез модели о связях; проверка цитат и логический
вывод не смешиваются, о неподтверждённой связи не говорится как об установленной.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, cast

from validator.criteria import criterion
from validator.findings import (
    RELATION_TITLES,
    RELATION_TYPES,
    REVIEW_CATEGORY_TITLES,
    RULE_CATEGORY_TITLES,
    SEVERITIES,
    SEVERITY_TITLES,
    SYMMETRIC_RELATIONS,
    AbsenceClaim,
    Evidence,
    Finding,
    ReviewCategory,
    Severity,
)
from validator.graph import FactGraph, GraphEdge
from validator.ingest.common import one_line
from validator.lint import LintResult
from validator.model import FORMAT_TITLES, Document
from validator.passport import RunPassport
from validator.quality import Contradiction, QualityResult
from validator.scoring import (
    STATUS_TITLES,
    VERDICT_TITLES,
    DefectRef,
    ReviewRef,
    Scorecard,
)
from validator.segmentation import CATEGORY_TITLES, Requirement, Segmentation
from validator.segmentation.text import parse_prefix

if TYPE_CHECKING:  # отчёт не загружает LLM-слой: в режиме --rules-only его нет
    from validator.llm import CrossCheckAnalysis, LLMAnalysis

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


_LIST_LIMIT = 10  # сколько пар и основ показывать в списке
# Тип связи в гипотезе — предположение модели, а не установленный факт.
_ASSUMED_RELATIONS: dict[str, str] = {
    "depends_on": "возможная зависимость",
    "conflicts_with": "возможное противоречие",
    "related_to": "возможная связь",
    "requires_condition": "возможное условие выполнения",
}
_EXCLUDE_TITLES: dict[str, str] = {
    "empty": "пустой текст",
    "no_source": "нет корректной привязки к источнику",
}


def _edge_block(number: int, edge: GraphEdge) -> str:
    source, target = edge.evidence
    if edge.rule_id == "XRF-01":
        basis = (
            "обе цитаты точно найдены в своих требованиях (Evidence Gate); цитата "
            f"`{edge.source_id}` — ID требования `{edge.target_id}`, и этот ID принадлежит "
            "только ему"
        )
    else:
        basis = (
            f"замечание `{edge.finding_id}` (раздел 4): один параметр с разными значениями "
            "в одной таблице «параметр — значение»"
        )
    return "\n".join(
        [
            f"**6.{number}. `{edge.source_id}` {RELATION_TITLES[edge.relation]} "
            f"`{edge.target_id}`** — {edge.rule_id} · {_escape(edge.title)}",
            "",
            f"- Цитата `{edge.source_id}`: {_quote(source.quote)} — {_where(source)}",
            f"- Цитата `{edge.target_id}`: {_quote(target.quote)} — {_where(target)}",
            f"- Проверено: {basis}",
            "- Статус: подтверждено",
        ]
    )


def _relation_hypothesis_block(number: int, finding: Finding) -> str:
    item = criterion(finding.criterion)
    source, target = finding.evidence
    relation = finding.relation
    assert relation is not None
    if relation in SYMMETRIC_RELATIONS:
        ends = f"`{source.requirement_id}` и `{target.requirement_id}`"
    else:
        ends = f"`{source.requirement_id}` {RELATION_TITLES[relation]} `{target.requirement_id}`"
    lines = [
        f"**6.{number}. {_escape(finding.title)}: {ends}**",
        "",
        f"- Цитата `{source.requirement_id}`: {_quote(source.quote)} — {_where(source)}",
        f"- Цитата `{target.requirement_id}`: {_quote(target.quote)} — {_where(target)}",
        f"- Предполагаемый тип связи: {relation} — {_ASSUMED_RELATIONS[relation]}",
        f"- Предполагаемая важность: {SEVERITY_TITLES[finding.severity]} (по типу связи)",
        f"- Обоснование модели: {_escape(finding.explanation)}",
        f"- Рекомендация модели: {_escape(finding.recommendation)}",
    ]
    if finding.confidence is not None:
        lines.append(f"- Уверенность модели: {finding.confidence:g} — на статус не влияет")
    lines += [
        f"- Критерий: {item.reference} — {item.title}",
        "- Происхождение цитат: проверено Evidence Gate (точный текст обоих требований, "
        "место в файле)",
        "- Связь: не проверена — это гипотеза модели",
        "- Статус: не подтверждено",
    ]
    return "\n".join(lines)


def _pairs(pairs: tuple[tuple[str, str], ...]) -> str:
    shown = ", ".join(f"`{a}`–`{b}`" for a, b in pairs[:_LIST_LIMIT])
    rest = len(pairs) - _LIST_LIMIT
    return shown + (f" и ещё {rest}" if rest > 0 else "")


def _selection_lines(
    passport: RunPassport, graph: FactGraph, crosscheck: CrossCheckAnalysis | None
) -> list[str]:
    selection = graph.candidates
    found = str(selection.found)
    if selection.truncated:
        found = (
            f"найдено {selection.found}, оставлено {len(selection.pairs)} лучших по оценке — "
            f"ограничение отбора {selection.limit} пар; исключено ограничением: "
            f"{selection.excluded_by_limit}, моделью они не проверялись"
        )
    lines = [
        f"- Возможных пар: {selection.possible_pairs} — полный перебор не выполняется.",
        f"- Оценено пар (есть общее значимое слово): {selection.evaluated_pairs}.",
        f"- Пар-кандидатов (общих значимых слов не меньше {selection.min_shared}): {found}. "
        "Учитываются слова из 4 и более букв, формы одного слова объединяются; числа, ID, "
        "отрицания, служебные и модальные слова не учитываются — одинаковые числа пару "
        "не образуют.",
    ]
    if selection.common_stems:
        stems = ", ".join(selection.common_stems[:_LIST_LIMIT])
        more = " …" if len(selection.common_stems) > _LIST_LIMIT else ""
        lines.append(
            "- Слишком общие слова (больше чем у половины требований) не учитывались: "
            f"{_escape(stems)}{more}."
        )
    excluded = []
    if selection.duplicate_count:
        excluded.append(f"дословные повторы — {selection.duplicate_count}")
    if selection.linked:
        excluded.append(f"уже связаны подтверждённой связью — {len(selection.linked)}")
    if excluded:
        lines.append("- Не кандидаты: " + "; ".join(excluded) + ".")
    if crosscheck is None:
        mode = " (режим --rules-only)" if passport.mode == "rules-only" else ""
        lines.append(f"- Пары-кандидаты моделью не проверялись{mode}.")
    elif crosscheck.disabled:
        lines.append("- Проверка пар моделью отключена (LLM_MAX_PAIR_REQUESTS = 0).")
    else:
        lines += [
            f"- Проверено моделью: {crosscheck.requests} из {crosscheck.candidates_total} "
            f"(лимит LLM_MAX_PAIR_REQUESTS = {crosscheck.budget}).",
            "- Гипотезы модели: прошли проверку цитат (Evidence Gate) — "
            f"{len(crosscheck.hypotheses)}, отклонены — {len(crosscheck.rejected)}; сама "
            "связь не проверена.",
            "- Покрытие: требований хотя бы в одной проверенной паре — "
            f"{len(crosscheck.covered)} из {len(graph.nodes)}.",
        ]
    return lines


def _gap_lines(graph: FactGraph, crosscheck: CrossCheckAnalysis | None) -> list[str]:
    lines = [
        f"- Вне графа: `{item.requirement_id}` — {_EXCLUDE_TITLES[item.code]}"
        for item in graph.excluded
    ]
    for gap in graph.gaps:
        if gap.kind == "conflict_outside_graph":
            lines.append(f"- Замечание `{gap.target}`: {_escape(gap.detail)}")
        else:
            lines.append(
                f"- Ссылка `{gap.requirement_id}` → «{_escape(gap.target)}»: {_escape(gap.detail)}"
            )
    unchecked = (
        crosscheck.skipped
        if crosscheck is not None
        else tuple((pair.source_id, pair.target_id) for pair in graph.candidates.pairs)
    )
    if unchecked:
        lines.append(
            f"- Пары-кандидаты, не проверенные моделью: {len(unchecked)} — {_pairs(unchecked)}"
        )
    selection = graph.candidates
    if selection.truncated:
        lines.append(
            "- Пары-кандидаты, исключённые ограничением отбора: "
            f"{selection.excluded_by_limit} — не перечислены поимённо, моделью не проверялись."
        )
    if selection.outside_pairs > 0:
        lines.append(
            f"- Пары вне отбора (общих значимых слов меньше {selection.min_shared}): "
            f"{selection.outside_pairs} — моделью не проверялись."
        )
    if crosscheck is not None and not crosscheck.disabled:
        covered = set(crosscheck.covered)
        alone = [node.requirement_id for node in graph.nodes if node.requirement_id not in covered]
        if alone:
            lines.append(
                f"- Требования без проверенной пары: {len(alone)} — "
                + ", ".join(f"`{rid}`" for rid in alone)
                + ". Модель не сравнивала их ни с одним требованием — это не значит, что "
                "связей нет."
            )
    return lines


def _no_relation_hypotheses(crosscheck: CrossCheckAnalysis) -> str:
    if crosscheck.rejected:
        return (
            f"Принятых гипотез нет: все предложенные моделью связи ({len(crosscheck.rejected)}) "
            "отклонены — причины ниже."
        )
    return (
        "Модель не предложила связей по проверенным парам "
        f"({crosscheck.requests} из {crosscheck.candidates_total})."
    )


def _graph_section(
    passport: RunPassport, graph: FactGraph, crosscheck: CrossCheckAnalysis | None
) -> str:
    parts = ["## 6. Связи между требованиями (Fact Graph)"]
    if not graph.nodes:
        parts.append(
            "Узлов нет: в документе не найдено атомарных требований с текстом и привязкой к "
            "источнику — связи не строились."
        )
        gaps = _gap_lines(graph, crosscheck)
        if gaps:
            parts += ["### Пропуски", "\n".join(gaps)]
        return "\n\n".join(parts)
    hypotheses = crosscheck.hypotheses if crosscheck is not None else ()
    parts += [
        "Узлы графа — атомарные требования с текстом и привязкой к источнику: "
        f"{len(graph.nodes)} из {graph.requirements_total}. Подтверждённая связь — только "
        "доказанная машинной проверкой обоих концов: явная ссылка на ID требования (XRF-01) "
        "или разные значения одного параметра в одной таблице (CON-03); обе цитаты прошли "
        "Evidence Gate. Совпадение слов или чисел связью не считается — по нему только "
        "отбираются пары для проверки моделью. Связи, предложенные моделью, — гипотезы: "
        "проверены цитаты, а не сама связь.",
        _grid(
            ("Тип связи", "Подтверждено", "Гипотезы LLM"),
            [
                (
                    f"{relation} — {RELATION_TITLES[relation]}",
                    str(graph.count(relation, "rule")),
                    str(sum(1 for finding in hypotheses if finding.relation == relation)),
                )
                for relation in RELATION_TYPES
            ],
        ),
        "### Отбор пар для проверки моделью",
        "\n".join(_selection_lines(passport, graph, crosscheck)),
    ]
    verified = [edge for edge in graph.edges if edge.origin == "rule"]
    parts.append(f"### Подтверждённые связи ({len(verified)})")
    if not verified:
        parts.append(
            "Подтверждённых связей нет: явных ссылок между требованиями и разных значений "
            "одного параметра в одной таблице не найдено."
        )
    number = 0
    for edge in verified:
        number += 1
        parts.append(_edge_block(number, edge))
    if crosscheck is not None and crosscheck.requests:
        parts += [
            f"### Гипотезы модели о связях ({len(hypotheses)}, не подтверждены)",
            "> Гипотеза — предположение модели о связи двух требований. Evidence Gate "
            "проверил только цитаты: они точно и однозначно найдены в текстах обоих "
            "требований этой версии документа. Сама связь не проверена: гипотеза о "
            "противоречии не означает, что противоречие установлено.",
        ]
        if not hypotheses:
            parts.append(_no_relation_hypotheses(crosscheck))
        for finding in hypotheses:
            number += 1
            parts.append(_relation_hypothesis_block(number, finding))
    if crosscheck is not None and crosscheck.rejected:
        parts += [
            f"### Отклонённые гипотезы связей ({len(crosscheck.rejected)})",
            "\n".join(
                f"- пара `{item.pair[0]}`–`{item.pair[1]}`, элемент ответа № {item.index + 1}: "
                f"{_escape(' '.join(item.reasons))}"
                for item in crosscheck.rejected
            ),
        ]
    if graph.rejected:
        parts += [
            f"### Связи правил, отклонённые проверкой доказательств ({len(graph.rejected)})",
            "\n".join(
                f"- `{item.finding.id}`: {_escape('; '.join(item.reasons))}"
                for item in graph.rejected
            ),
        ]
    gaps = _gap_lines(graph, crosscheck)
    parts += [
        "### Пропуски",
        "\n".join(gaps)
        if gaps
        else "Непроверенных пар и ссылок без связи нет: каждая возможная пара — кандидат, "
        "проверенный моделью, дословный повтор или уже подтверждённая связь.",
    ]
    return "\n\n".join(parts)


_SOURCE_TITLES: dict[str, str] = {
    "CTR-01": "CTR-01 · Несовместимые ограничения одного параметра",
    "CTR-02": "CTR-02 · Несовместимые значения одноимённого параметра в разных контекстах",
    "CON-03": "CON-03 · Разные значения одного параметра в одной таблице",
    "LLM-PAIR": "гипотеза модели о противоречии",
}
_UNCHECKED: dict[str, str] = {
    "CTR-02": "один ли это параметр в одних условиях — если да, это противоречие",
    "CON-03": "уточнение это или противоречие: как ограничения значения совместимы",
    "LLM-PAIR": "что речь об одном объекте при одних условиях — это вывод модели (раздел 6)",
}
_COMPLETENESS_REVIEW = ("completeness", "missing_condition")


def _contradiction_block(number: str, item: Contradiction, confirmed: bool) -> str:
    source, target = item.evidence[0], item.evidence[-1]
    lines = [
        f"**{number}. `{item.requirement_ids[0]}` и `{item.requirement_ids[1]}`** — "
        f"{_SOURCE_TITLES[item.source]}",
        "",
        f"- Цитата `{source.requirement_id}`: {_quote(source.quote)} — {_where(source)}",
        f"- Цитата `{target.requirement_id}`: {_quote(target.quote)} — {_where(target)}",
        f"- Почему: {_escape(item.explanation)}",
    ]
    if item.numeric is not None:
        if confirmed and item.numeric.verdict == "disjoint":
            lines.append(f"- Контрпример: {_escape(item.numeric.detail)}")
        else:
            note = (
                " — числа не пересекаются, но относятся ли они к одному объекту при одних "
                "условиях, не проверено"
                if item.numeric.verdict == "disjoint" and not confirmed
                else ""
            )
            lines.append(f"- Сравнение чисел: {_escape(item.numeric.detail)}{note}")
    elif item.source == "LLM-PAIR":
        lines.append(
            "- Сравнение чисел: не выполнялось — в цитатах нет однозначного числового ограничения"
        )
    if confirmed:
        basis = (
            "машинная проверка ограничений (Evidence Gate проверил обе цитаты)"
            if item.source == "CTR-01"
            else f"правило CON-03, замечание `{item.finding_id}` (раздел 4)"
        )
        lines += [f"- Основание: {basis}", "- Статус: подтверждено"]
    else:
        lines += [
            f"- Не проверено: {_UNCHECKED[item.source]}",
            "- Статус: не подтверждено (потенциальное противоречие)",
        ]
    return "\n".join(lines)


def _rule_block(number: str, finding: Finding) -> str:
    item = criterion(finding.criterion)
    lines = [f"**{number}. {finding.rule_id} · {finding.title}**", ""]
    lines += [
        f"- Цитата `{e.requirement_id}`: {_quote(e.quote)} — {_where(e)}" for e in finding.evidence
    ]
    lines += [
        f"- Почему: {_escape(finding.explanation)}",
        f"- Как исправить: {_escape(finding.recommendation)}",
        f"- Важность: {SEVERITY_TITLES[finding.severity]}",
        f"- Критерий: {item.reference} — {item.title}",
        "- Статус: подтверждено",
    ]
    return "\n".join(lines)


def _ids(ids: Iterable[str]) -> str:
    return ", ".join(f"`{rid}`" for rid in ids)


def _coverage_lines(
    passport: RunPassport,
    quality: QualityResult,
    graph: FactGraph | None,
    crosscheck: CrossCheckAnalysis | None,
) -> list[str]:
    lines = [
        "- Детерминированно сравниваются только утверждения «параметр — значение» с числовым "
        f"значением (строки таблиц «параметр | значение» и фрагменты «Параметр: значение»): "
        f"{quality.statements}."
    ]
    reasons: list[str] = []
    if crosscheck is None:
        mode = " (режим --rules-only)" if passport.mode == "rules-only" else ""
        lines.append(f"- Смысловые противоречия моделью не проверялись{mode}.")
        reasons.append("модель пары не проверяла")
    elif crosscheck.disabled:
        lines.append("- Проверка пар моделью отключена (LLM_MAX_PAIR_REQUESTS = 0).")
        reasons.append("проверка пар моделью отключена")
    else:
        lines.append(
            f"- Модель проверила {crosscheck.requests} из {crosscheck.candidates_total} "
            "пар-кандидатов (раздел 6)."
        )
        if crosscheck.skipped:
            reasons.append(f"не проверено моделью пар-кандидатов: {len(crosscheck.skipped)}")
    if graph is not None:
        selection = graph.candidates
        if selection.truncated:
            reasons.append(f"исключено ограничением отбора: {selection.excluded_by_limit}")
        if selection.outside_pairs > 0:
            reasons.append(
                f"пар вне отбора (общих значимых слов меньше {selection.min_shared}): "
                f"{selection.outside_pairs}"
            )
    if reasons:
        lines.append("- Отсутствие противоречий не утверждается: " + "; ".join(reasons) + ".")
    else:
        lines.append(
            "- Все пары-кандидаты проверены, но это не доказательство отсутствия противоречий: "
            "правила сравнивают только числовые значения параметров, а модель — только пары "
            "с общими словами."
        )
    return lines


def _acceptance_lines(
    quality: QualityResult, lint: LintResult | None, analysis: LLMAnalysis | None
) -> list[str]:
    accepted = set(quality.acceptance)
    if not quality.acceptance:
        return [
            "Критерии приёмки в документе не выделены: нет раздела, заголовок которого "
            "называет критерии приёмки, — проверить приёмку требований по документу нельзя. "
            "Это наблюдение по структуре документа: критерии могут быть описаны иначе, и тогда "
            "агент их не распознал."
        ]
    lines = [
        f"- Критериев приёмки (фрагменты раздела критериев приёмки): {len(quality.acceptance)} — "
        f"{_ids(quality.acceptance)}."
    ]
    if lint is not None:
        rules = [f for f in lint.findings if accepted & set(f.requirement_ids)]
        if rules:
            lines.append(
                "- Замечания правил к критериям (раздел 4): "
                + "; ".join(f"`{f.id}` — {_escape(f.title)}" for f in rules)
                + "."
            )
    if analysis is not None:
        hypotheses = [f for f in analysis.hypotheses if accepted & set(f.requirement_ids)]
        if hypotheses:
            lines.append(
                "- Гипотезы модели к критериям (раздел 5, не подтверждены): "
                + "; ".join(f"`{f.requirement_ids[0]}` — {_escape(f.title)}" for f in hypotheses)
                + "."
            )
        skipped = [s.requirement_id for s in analysis.skipped if s.requirement_id in accepted]
        if skipped:
            lines.append(f"- Модель не проверяла критерии: {_ids(skipped)}.")
    else:
        lines.append("- Смысловая проверка критериев моделью не выполнялась.")
    if quality.acceptance_links:
        lines.append(
            f"- Критерии ссылаются на требования: {_ids(quality.acceptance_links)}; без ссылки "
            f"из критериев: {len(quality.unlinked_requirements)}"
            + (
                f" — {_ids(quality.unlinked_requirements)}."
                if quality.unlinked_requirements
                else "."
            )
        )
    else:
        lines.append(
            "- Связь критериев с требованиями не прослеживается: ни один критерий не ссылается "
            "на ID требования — какие требования покрыты приёмкой, определить нельзя."
        )
    return lines


def _completeness_lines(
    passport: RunPassport,
    lint: LintResult | None,
    analysis: LLMAnalysis | None,
    graph: FactGraph | None,
) -> list[str]:
    lines: list[str] = []
    if lint is not None:
        rules = [f for f in lint.findings if f.category == "completeness"]
        lines.append(
            f"- Замечания правил полноты (раздел 4): {len(rules)}"
            + (
                " — " + "; ".join(f"`{f.id}` — {_escape(f.title)}" for f in rules) + "."
                if rules
                else "."
            )
        )
    if graph is not None and graph.gaps:
        lines.append(
            f"- Ссылки, не сведённые к требованию (раздел 6): {len(graph.gaps)} — "
            + _ids(dict.fromkeys(gap.requirement_id for gap in graph.gaps))
            + "."
        )
    if analysis is not None:
        hypotheses = [f for f in analysis.hypotheses if f.review_category in _COMPLETENESS_REVIEW]
        lines.append(
            f"- Гипотезы модели о неполноте (раздел 5, не подтверждены): {len(hypotheses)}"
            + (
                " — "
                + "; ".join(f"`{f.requirement_ids[0]}` — {_escape(f.title)}" for f in hypotheses)
                + "."
                if hypotheses
                else "."
            )
        )
        if analysis.skipped:
            lines.append(
                f"- Модель не проверяла требований: {len(analysis.skipped)} (раздел 5) — "
                "их смысловая полнота не оценивалась."
            )
        scope = (
            f"смысловые пропуски — гипотезами модели для {analysis.requirements_sent} требований"
        )
    else:
        mode = " (режим --rules-only)" if passport.mode == "rules-only" else ""
        lines.append(f"- Смысловая полнота моделью не проверялась{mode}.")
        scope = "смысловые пропуски не проверялись"
    lines.append(
        "- Полнота документа не доказывается: правила проверяют формальные признаки "
        f"(заглушки, пустые значения, оборванные формулировки, ссылки), {scope}; "
        "отсутствующие требования агент не придумывает."
    )
    return lines


def _quality_section(
    passport: RunPassport,
    quality: QualityResult,
    lint: LintResult | None,
    analysis: LLMAnalysis | None,
    graph: FactGraph | None,
    crosscheck: CrossCheckAnalysis | None,
) -> str:
    confirmed, potential = quality.confirmed, quality.potential
    model_items = sum(1 for item in potential if item.source == "LLM-PAIR")
    parts = [
        "## 7. Противоречия, критерии приёмки и полнота",
        "Проверка набора требований целиком. Подтверждённое противоречие — только с машинным "
        "контрпримером (ограничения одного параметра в одном разделе не пересекаются) или по "
        "правилу CON-03; потенциальное — когда контекст или тождество объекта не проверены. "
        "Гипотезы модели остаются потенциальными. Новых запросов к модели этот раздел не "
        "делает: он сводит результаты разделов 4–6 и добавляет правила CTR-01, CTR-02, ACC-01.",
        "### 7.1. Противоречия",
        f"Подтверждено: {len(confirmed)}; потенциальных: {len(potential)}"
        + (f" (из них гипотез модели: {model_items})" if model_items else "")
        + ".",
    ]
    if confirmed:
        parts.append(f"#### Подтверждённые противоречия ({len(confirmed)})")
        parts += [
            _contradiction_block(f"7.1.{n}", item, True) for n, item in enumerate(confirmed, 1)
        ]
    if potential:
        parts.append(f"#### Потенциальные противоречия ({len(potential)}, не подтверждены)")
        start = len(confirmed)
        parts += [
            _contradiction_block(f"7.1.{start + n}", item, False)
            for n, item in enumerate(potential, 1)
        ]
    if quality.hypotheses_confirmed:
        parts.append(
            "Гипотезы модели о противоречии, совпавшие с подтверждёнными (не дублируются): "
            + _ids(quality.hypotheses_confirmed)
            + "."
        )
    parts += [
        "#### Полнота проверки противоречий",
        "\n".join(_coverage_lines(passport, quality, graph, crosscheck)),
        "### 7.2. Критерии приёмки",
        "\n".join(_acceptance_lines(quality, lint, analysis)),
    ]
    scenario = [f for f in quality.findings if f.rule_id == "ACC-01"]
    parts += [_rule_block(f"7.2.{n}", finding) for n, finding in enumerate(scenario, 1)]
    parts += [
        "### 7.3. Полнота",
        "\n".join(_completeness_lines(passport, lint, analysis, graph)),
    ]
    if quality.rejected:
        parts += [
            f"### Отклонено проверкой доказательств ({len(quality.rejected)})",
            "\n".join(
                f"- `{item.finding.id}`: {_escape('; '.join(item.reasons))}"
                for item in quality.rejected
            ),
        ]
    return "\n\n".join(parts)


_ITEMS_LIMIT = 10  # сколько пунктов «исправить» и «проверить» показывать в итоге
_LLM_REVIEW_TITLES: dict[str, str] = {
    "reviewed": "проверено",
    "budget": "нет: лимит",
    "empty": "нет: пустой текст",
    "no_source": "нет: нет источника",
    "not_used": "не использовался",
}
_PAIR_TITLES: dict[str, str] = {
    "checked": "проверено",
    "not_checked": "кандидат не проверен",
    "no_candidate": "нет пары-кандидата",
    "excluded": "вне графа",
    "not_used": "не использовалась",
    "disabled": "отключена",
}
_REJECTED_TITLES: dict[str, str] = {
    "rules": "замечания правил (раздел 4)",
    "stage8-rules": "замечания правил этапа 8 (раздел 7)",
    "graph-rules": "связи правил (раздел 6)",
    "llm-review": "ответы модели при обзоре требований (раздел 5)",
    "llm-pairs": "ответы модели о парах (раздел 6)",
}


def _requirements_section(scorecard: Scorecard) -> str:
    parts = [
        "## 8. Проверка каждого требования",
        "Карточка каждого атомарного требования по результатам разделов 4–7: подтверждённые "
        "дефекты (замечания со статусом verified), участие в противоречиях, гипотезы модели и "
        "какие проверки выполнены. Статус — категория, а не балл. «Замечаний нет» значит "
        "только, что перечисленные проверки ничего не нашли, а не что требование корректно.",
    ]
    if not scorecard.requirements:
        parts.append("Атомарных требований нет — карточки не строятся.")
        return "\n\n".join(parts)
    parts.append(
        _grid(
            (
                "ID",
                "ID в документе",
                "Статус",
                "Подтверждено: сущ. / незнач.",
                "Противоречия: подтв. / потенц.",
                "Гипотезы модели",
                "LLM-обзор",
                "Проверка пар",
            ),
            [
                (
                    item.requirement_id,
                    item.doc_id or "—",
                    STATUS_TITLES[item.status],
                    f"{len(item.confirmed_major)} / {len(item.confirmed_minor)}",
                    f"{len(item.contradictions_confirmed)} / {len(item.contradictions_potential)}",
                    str(len(item.hypotheses)),
                    _LLM_REVIEW_TITLES[item.llm_review],
                    _PAIR_TITLES[item.pair_check],
                )
                for item in scorecard.requirements
            ],
        )
    )
    return "\n\n".join(parts)


def _defect_line(number: int, item: DefectRef) -> str:
    mark = " — подтверждённое противоречие" if item.contradiction else ""
    return (
        f"{number}. `{item.finding_id}` — {item.rule_id} · {_escape(item.title)} "
        f"({SEVERITY_TITLES[item.severity].lower()}, "
        f"{RULE_CATEGORY_TITLES[item.category].lower()}){mark}; требования: "
        f"{_ids(item.requirement_ids)}"
    )


def _review_line(number: int, item: ReviewRef) -> str:
    source = "гипотеза модели" if item.source == "llm-review" else "потенциальное противоречие"
    return (
        f"{number}. `{item.item_id}` — {_escape(item.title)} ({source}, предполагаемая "
        f"важность: {SEVERITY_TITLES[item.severity].lower()}); требования: "
        f"{_ids(item.requirement_ids)}"
    )


def _summary_section(scorecard: Scorecard) -> str:
    coverage = scorecard.coverage
    defects = scorecard.defects
    parts = [
        "## 9. Сводка",
        f"Требований (атомарных): {coverage.requirements}; неоднозначных фрагментов: "
        f"{coverage.ambiguous}. Подтверждённых дефектов: {len(defects)}; подтверждённых "
        f"противоречий: {scorecard.contradictions_confirmed}; на проверку человеку: "
        f"{len(scorecard.review)}.",
        _grid(
            ("Статус требования", "Требований"),
            [(STATUS_TITLES[status], str(count)) for status, count in scorecard.status_counts],
        ),
        "### Подтверждённые дефекты",
    ]
    if defects:
        parts.append(
            _grid(
                ("Важность", "Дефектов"),
                [
                    (SEVERITY_TITLES[severity], str(count))
                    for severity in SEVERITIES
                    if (count := sum(1 for d in defects if d.severity == severity))
                ],
            )
        )
        by_criterion = Counter(d.criterion for d in defects)
        parts.append(
            _grid(
                ("Критерий", "Дефектов"),
                [
                    (f"{criterion(code).reference} — {criterion(code).title}", str(count))
                    for code, count in sorted(by_criterion.items())
                ],
            )
        )
        if scorecard.structural:
            parts.append(
                f"Из них к структуре документа (заголовки, таблицы), а не к отдельному "
                f"требованию: {len(scorecard.structural)} — {_ids(scorecard.structural)}."
            )
    else:
        parts.append(
            "Подтверждённых дефектов не найдено — это результат перечисленных проверок, а не "
            "доказательство качества документа."
        )
    critical = [d for d in defects if d.finding_id in scorecard.critical]
    parts.append("### Критичные дефекты (непроверяемые, противоречивые, неполные формулировки)")
    parts.append(
        "\n".join(_defect_line(n, item) for n, item in enumerate(critical, 1))
        if critical
        else "Подтверждённых критичных дефектов не найдено."
    )
    lines = [
        f"- Противоречия (раздел 7): подтверждено — {scorecard.contradictions_confirmed}, "
        f"потенциальных — {scorecard.contradictions_potential}.",
    ]
    if coverage.llm_used:
        categories = ", ".join(
            f"{REVIEW_CATEGORY_TITLES[cast(ReviewCategory, key)].lower()} — {count}"
            for key, count in scorecard.review_categories
        )
        relations = ", ".join(
            f"{_ASSUMED_RELATIONS[key]} — {count}" for key, count in scorecard.relation_hypotheses
        )
        lines.append(
            "- Гипотезы модели о дефектах (раздел 5, не подтверждены): "
            f"{sum(c for _, c in scorecard.review_categories)}"
            + (f" — {categories}." if categories else ".")
        )
        lines.append(
            "- Гипотезы модели о связях (раздел 6, не подтверждены): "
            f"{sum(c for _, c in scorecard.relation_hypotheses)}"
            + (f" — {relations}." if relations else ".")
        )
    else:
        lines.append("- Гипотезы модели: модель не использовалась.")
    rejected = [(source, count) for source, count in scorecard.rejected if count]
    lines.append(
        "- Отклонено проверкой доказательств или схемой ответа (в отчёт как замечания не "
        "вошли): "
        + (
            "; ".join(f"{_REJECTED_TITLES[source]} — {count}" for source, count in rejected)
            if rejected
            else "ничего"
        )
        + "."
    )
    lines.append(
        f"- Информационные замечания (повод для проверки, не дефекты): {len(scorecard.info)}."
    )
    lines.append(
        f"- Критерии приёмки: {scorecard.acceptance}; сценариев без ожидаемого результата: "
        f"{scorecard.acceptance_scenarios}; требований, на которые ссылаются критерии: "
        f"{scorecard.acceptance_links}."
    )
    review = (
        f"{coverage.reviewed} из {coverage.requirements}"
        if coverage.llm_used
        else "не использовалась"
    )
    if coverage.pairs_disabled:
        pairs = "отключена (LLM_MAX_PAIR_REQUESTS = 0)"
    elif coverage.llm_used:
        pairs = f"{coverage.pairs_checked} из {coverage.pairs_candidates} пар-кандидатов"
    else:
        pairs = "не использовалась"
    lines.append(
        f"- Покрытие: правила — все атомарные требования ({coverage.requirements}); модель — "
        f"{review}; проверка пар моделью — {pairs}; исключено ограничением отбора — "
        f"{coverage.truncated_pairs}; пар вне отбора — {coverage.outside_pairs}; требований "
        f"вне графа — {coverage.excluded}."
    )
    parts += ["### Противоречия, гипотезы, отклонённое и покрытие", "\n".join(lines)]
    return "\n\n".join(parts)


def _verdict_section(scorecard: Scorecard) -> str:
    parts = [
        "## 10. Итог",
        f"**Вердикт: {VERDICT_TITLES[scorecard.verdict]}.**",
        "Основание:\n\n"
        + "\n".join(f"- {_escape(reason)}" for reason in scorecard.verdict_reasons),
        "Вердикт строится только на подтверждённых замечаниях: подтверждённое противоречие или "
        "подтверждённый существенный дефект проверяемости или полноты — разработку начинать не "
        "рекомендуется; другие подтверждённые дефекты — начинать после их исправления. "
        "Гипотезы модели и потенциальные противоречия вердикт не меняют.",
        "### Приоритет исправлений",
    ]
    shown = scorecard.defects[:_ITEMS_LIMIT]
    if shown:
        lines = [_defect_line(n, item) for n, item in enumerate(shown, 1)]
        rest = len(scorecard.defects) - len(shown)
        if rest > 0:
            lines.append(f"\nИ ещё {rest} — полный список в разделах 4 и 7.")
        parts.append("\n".join(lines))
    else:
        parts.append("Подтверждённых дефектов нет.")
    parts.append("### Проверить человеком (не подтверждено)")
    review = scorecard.review[:_ITEMS_LIMIT]
    if review:
        lines = [_review_line(n, item) for n, item in enumerate(review, 1)]
        rest = len(scorecard.review) - len(review)
        if rest > 0:
            lines.append(f"\nИ ещё {rest} — полный список в разделах 5–7.")
        parts.append("\n".join(lines))
    else:
        parts.append(
            "Пунктов на проверку человеку нет: гипотезы модели и потенциальные противоречия "
            "не найдены."
            if scorecard.coverage.llm_used
            else "Пунктов на проверку человеку нет: потенциальных противоречий правила не нашли, "
            "а модель не использовалась — гипотез о дефектах нет, потому что их никто не искал."
        )
    parts += [
        "### Ограничения вывода",
        "\n".join(f"- {_escape(note)}" for note in scorecard.caveats),
    ]
    return "\n\n".join(parts)


def render_report(
    passport: RunPassport,
    document: Document,
    segmentation: Segmentation | None = None,
    lint: LintResult | None = None,
    llm: LLMAnalysis | None = None,
    *,
    graph: FactGraph | None = None,
    crosscheck: CrossCheckAnalysis | None = None,
    quality: QualityResult | None = None,
    scorecard: Scorecard | None = None,
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
    if graph is not None:
        sections.append(_graph_section(passport, graph, crosscheck))
    if quality is not None:
        sections.append(_quality_section(passport, quality, lint, llm, graph, crosscheck))
    if scorecard is not None:
        sections += [
            _requirements_section(scorecard),
            _summary_section(scorecard),
            _verdict_section(scorecard),
        ]
    sections.append(
        "> Агент на этапе 9: замечания правил, связи графа и противоречия с машинным "
        "контрпримером подтверждены Evidence Gate; гипотезы модели (обзор требований, "
        "проверка пар) — с проверенными цитатами, но непроверенным смыслом; итог построен "
        "только на подтверждённых замечаниях."
    )
    return "\n\n".join(sections) + "\n"
