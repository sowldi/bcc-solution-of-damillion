from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.graph_helpers import load, relation, relations_json
from tests.llm_helpers import findings_json, hypothesis
from validator.graph import build_fact_graph
from validator.ingest import load_document
from validator.lint import LintResult, run_lint
from validator.llm import LLMOptions, review_requirements
from validator.llm.crosscheck import crosscheck_pairs
from validator.llm.fake import FakeProvider
from validator.model import Block, Document, Location
from validator.passport import RunPassport
from validator.quality import analyze_quality
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


# --- раздел 4: детерминированная проверка ----------------------------------------------


def _linted(tmp_path: Path, markdown: str) -> tuple[Document, Segmentation, LintResult]:
    document, segmentation = _segmentation(tmp_path, markdown)
    return document, segmentation, run_lint(document, segmentation)


def test_lint_section_shows_verified_findings(tmp_path: Path) -> None:
    document, segmentation, lint = _linted(
        tmp_path,
        "# Сервис\n\n## Функциональные требования\n\n"
        "* Ф-1. Каталог книг должен быть удобным.\n* Ф-2. Срок брони: TBD.\n",
    )

    text = render_report(_passport(NOON), document, segmentation, lint)

    assert "## 4. Детерминированная проверка" in text
    assert "| Существенное | 2 |" in text
    assert "Проверка доказательств: подтверждено — 2, отклонено — 0." in text
    assert "### Существенные (2)" in text
    assert "TST-01 · Субъективная оценка качества" in text
    assert "- Требование: `Ф-1`" in text
    assert "- Цитата: «удобным» — Сервис › Функциональные требования — строка 5" in text
    assert "- Почему: «удобным»" in text
    assert "- Как исправить:" in text
    assert "- Критерий: ISO/IEC/IEEE 29148:2018, п. 5.2.5 g — Проверяемость" in text
    assert text.index("## 3.") < text.index("## 4.")


def test_lint_section_lists_every_participant(tmp_path: Path) -> None:
    document, segmentation, lint = _linted(
        tmp_path,
        "# Сервис\n\n## Функциональные требования\n\n"
        "* Ф-1. Читатель бронирует книгу через каталог.\n"
        "* Ф-2. Читатель бронирует книгу через каталог.\n",
    )

    text = render_report(_passport(NOON), document, segmentation, lint)

    assert "- Требования: `Ф-1`, `Ф-2`" in text
    assert "  - `Ф-1`: «Читатель бронирует книгу через каталог.» — " in text
    assert "  - `Ф-2`: «Читатель бронирует книгу через каталог.» — " in text


def test_lint_section_without_findings_does_not_claim_correctness(tmp_path: Path) -> None:
    document, segmentation, lint = _linted(
        tmp_path,
        "# Сервис\n\n## Функциональные требования\n\n"
        "* Ф-1. Система сохраняет бронь не позднее 2 секунд после подтверждения.\n",
    )

    text = render_report(_passport(NOON), document, segmentation, lint)

    assert lint.findings == ()
    assert "Детерминированные правила замечаний не нашли" in text
    assert "не означает, что документ корректен" in text


def test_lint_quotes_are_escaped_for_markdown(tmp_path: Path) -> None:
    document, segmentation, lint = _linted(
        tmp_path,
        "# Сервис\n\n## Функциональные требования\n\n"
        "* Ф-1. Поле limit_kzt [по умолчанию] хранит лимит.\n"
        "* Ф-2. Поле limit_kzt [по умолчанию] хранит лимит.\n",
    )

    text = render_report(_passport(NOON), document, segmentation, lint)

    assert r"«Поле limit\_kzt \[по умолчанию\] хранит лимит.»" in text


def test_passport_shows_rules_only_mode() -> None:
    passport = replace(_passport(NOON), provider=None, model=None, mode="rules-only")

    text = render_report(passport, _document())

    assert "| Режим | только детерминированные правила (--rules-only) |" in text
    assert "| Модель | не используется (режим --rules-only) |" in text


def test_report_footer_names_current_stage(tmp_path: Path) -> None:
    document, segmentation, lint = _linted(
        tmp_path, "# Сервис\n\n## Функциональные требования\n\n* Ф-1. Читатель бронирует книгу.\n"
    )

    text = render_report(_passport(NOON), document, segmentation, lint)

    assert "> Агент на этапе 8:" in text
    assert text.rstrip().endswith("смысловая проверка гипотез появятся на следующих этапах.")


# --- раздел 5: гипотезы LLM ---------------------------------------------------------------

LLM_SPEC = (
    "# Сервис\n\n## Функциональные требования\n\n"
    "* Ф-1. Библиотекарь подтверждает бронь в течение рабочего дня.\n"
)


def _with_llm(tmp_path: Path, reply: str) -> str:
    document, segmentation, lint = _linted(tmp_path, LLM_SPEC)
    analysis = review_requirements(
        document, segmentation, FakeProvider(reply), LLMOptions(max_retries=0)
    )
    passport = replace(_passport(NOON), llm_used=True, provider="fake", model="fake-model")
    return render_report(passport, document, segmentation, lint, analysis)


def test_llm_section_marks_hypotheses_unverified(tmp_path: Path) -> None:
    reply = findings_json(
        hypothesis("Ф-1", "в течение рабочего дня"), hypothesis("Ф-1", "нет такой цитаты")
    )

    text = _with_llm(tmp_path, reply)

    section = text.split("## 5. Анализ LLM (гипотезы, не подтверждены)")[1]
    assert "| Модель | fake / fake-model |" in text
    assert "Провайдер: fake, модель: fake-model" in section
    assert "не являются замечаниями" in section
    assert "Проверено требований: 1 из 1 (по одному запросу на требование)" in section
    assert "**5.1. Неясная формулировка · Неоднозначный срок**" in section
    assert "- Требование: `Ф-1`" in section
    assert "- Цитата: «в течение рабочего дня» — Сервис › Функциональные требования" in section
    assert "- Обоснование модели: Срок не задан в единицах времени." in section
    assert "- Уверенность модели: 0.7 — на статус не влияет" in section
    assert "- Критерий: ISO/IEC/IEEE 29148:2018, п. 5.2.5 c — Однозначность" in section
    assert "- Статус: не подтверждено" in section
    assert "### Отклонённые ответы модели (1)" in section
    assert "цитата не найдена" in section
    # гипотезы LLM не смешиваются с подтверждёнными замечаниями правил
    assert "LLM" not in text.split("## 4.")[1].split("## 5.")[0]


def test_llm_section_without_hypotheses(tmp_path: Path) -> None:
    text = _with_llm(tmp_path, '{"findings": []}')

    assert "Модель не предложила гипотез." in text


def test_rules_only_report_says_llm_was_not_used(tmp_path: Path) -> None:
    document, segmentation, lint = _linted(tmp_path, LLM_SPEC)
    passport = replace(_passport(NOON), provider=None, model=None, mode="rules-only")

    text = render_report(passport, document, segmentation, lint)

    assert "## 5. Анализ LLM" in text
    assert "LLM не использовался (режим --rules-only)." in text


def test_absence_findings_show_their_bounded_scope(tmp_path: Path) -> None:
    document, segmentation, lint = _linted(
        tmp_path,
        "# Сервис\n\n## Функциональные требования\n\n* Ф-1. Читатель бронирует книгу.\n\n"
        "## Согласование\n\n| Роль | Дата |\n| --- | --- |\n| | |\n",
    )

    text = render_report(_passport(NOON), document, segmentation, lint)

    assert "CMP-05" in text
    assert "- Отсутствие проверено в области: таблица 1" in text
    assert "извлечённого из документа" in text


def test_llm_section_lists_unreviewed_requirements_by_id(tmp_path: Path) -> None:
    """Stage 6.1 (C): непроверенные требования названы по ID и не выглядят проверенными."""
    document, segmentation, lint = _linted(
        tmp_path,
        LLM_SPEC + "* Ф-2. Читатель отменяет бронь.\n* Ф-3. Читатель отменяет бронь.\n",
    )
    analysis = review_requirements(
        document, segmentation, FakeProvider(), LLMOptions(max_retries=0, max_requests=1)
    )
    passport = replace(_passport(NOON), llm_used=True, provider="fake", model="fake-model")

    text = render_report(passport, document, segmentation, lint, analysis)

    assert "Проверено требований: 1 из 3" in text
    assert "### Требования без LLM-обзора (2)" in text
    assert "- `Ф-2` — лимит запросов LLM_MAX_REQUESTS" in text
    assert "- `Ф-3` — лимит запросов LLM_MAX_REQUESTS" in text
    assert "Модель не предложила гипотез по проверенным требованиям (1 из 3)." in text
    assert "Модель не предложила гипотез.\n" not in text


def test_llm_hypothesis_separates_quote_origin_from_meaning(tmp_path: Path) -> None:
    """Stage 6.1 (B): происхождение цитаты проверено, смысл — нет; это видно у гипотезы."""
    text = _with_llm(tmp_path, findings_json(hypothesis("Ф-1", "в течение рабочего дня")))

    assert "- Происхождение цитаты: проверено Evidence Gate" in text
    assert "- Смысл замечания: не проверен — это гипотеза модели" in text


# --- Stage 6.2 (F): честное покрытие в разделе 5 -------------------------------------------


def _section5(
    tmp_path: Path, markdown: str, reply: str = '{"findings": []}', **options: int
) -> str:
    document, segmentation, lint = _linted(tmp_path, markdown)
    analysis = review_requirements(
        document, segmentation, FakeProvider(reply), LLMOptions(max_retries=0, **options)
    )
    passport = replace(_passport(NOON), llm_used=True, provider="fake", model="fake-model")
    text = render_report(passport, document, segmentation, lint, analysis)
    return text.split("## 5.")[1].split("> Агент на этапе")[0]


def test_only_rejected_hypotheses_are_not_reported_as_no_hypotheses(tmp_path: Path) -> None:
    reply = findings_json(hypothesis("Ф-1", "нет такой цитаты"))

    section = _section5(tmp_path, LLM_SPEC, reply)

    assert "Модель не предложила гипотез" not in section
    assert "Принятых гипотез нет: все предложенные моделью гипотезы (1) отклонены" in section
    assert "### Отклонённые ответы модели (1)" in section


def test_zero_coverage_does_not_look_like_a_review(tmp_path: Path) -> None:
    section = _section5(tmp_path, "# Заметки\n\nСправочный текст.\n")

    assert "Проверено требований: 0 из 0" in section
    assert "LLM-обзор не выполнен: модель не получила ни одного требования." in section
    assert "Модель не предложила гипотез" not in section


def test_full_partial_and_budget_coverage_wording(tmp_path: Path) -> None:
    spec = LLM_SPEC + "* Ф-2. Читатель отменяет бронь.\n"

    full = _section5(tmp_path, spec)
    partial = _section5(tmp_path, spec, max_requests=1)

    assert "Модель не предложила гипотез." in full
    assert "Требования без LLM-обзора" not in full
    assert "по проверенным требованиям (1 из 2)" in partial
    assert "- `Ф-2` — лимит запросов LLM_MAX_REQUESTS" in partial


# --- раздел 6: Fact Graph -----------------------------------------------------------------

CONFLICT = relation(
    "Ф-3", "не хранит бронь книги дольше 3 дней", "Ф-4", "хранит бронь книги 5 дней"
)


def _section6(
    tmp_path: Path,
    *replies: str,
    max_pair_requests: int = 10,
    rules_only: bool = False,
    max_candidates: int | None = None,
) -> str:
    built = load(tmp_path)
    if max_candidates is not None:
        graph = build_fact_graph(
            built.document, built.segmentation, built.lint, max_candidates=max_candidates
        )
        built = replace(built, graph=graph)
    passport = _passport(NOON)
    crosscheck = None
    if rules_only:
        passport = replace(passport, provider=None, model=None, mode="rules-only")
    else:
        options = LLMOptions(max_retries=0, max_pair_requests=max_pair_requests)
        crosscheck = crosscheck_pairs(
            built.document, built.segmentation, built.graph, FakeProvider(*replies), options
        )
    text = render_report(
        passport,
        built.document,
        built.segmentation,
        built.lint,
        graph=built.graph,
        crosscheck=crosscheck,
    )
    start = text.index("## 6. Связи между требованиями (Fact Graph)")
    return text[start : text.index("> Агент на этапе 8:")]


def test_graph_section_shows_verified_dependency_with_both_quotes(tmp_path: Path) -> None:
    section = _section6(tmp_path)

    assert "### Подтверждённые связи (1)" in section
    assert "**6.1. `Ф-2` зависит от `Ф-1`** — XRF-01 · Явная ссылка на требование" in section
    assert "- Цитата `Ф-2`: «Ф-1» — " in section
    assert "- Цитата `Ф-1`: «Ф-1. Библиотекарь подтверждает бронь книги" in section
    assert "- Статус: подтверждено" in section
    assert "| depends_on — зависит от | 1 | 0 |" in section


def test_graph_hypotheses_are_never_reported_as_established(tmp_path: Path) -> None:
    section = _section6(tmp_path, relations_json(CONFLICT))

    assert "### Гипотезы модели о связях (1, не подтверждены)" in section
    assert "**6.2. Возможное противоречие: `Ф-3` и `Ф-4`**" in section
    assert "- Цитата `Ф-3`: «не хранит бронь книги дольше 3 дней» — " in section
    assert "- Происхождение цитат: проверено Evidence Gate" in section
    assert "- Связь: не проверена — это гипотеза модели" in section
    assert "- Статус: не подтверждено" in section
    assert "| conflicts_with — противоречит | 0 | 1 |" in section
    lowered = section.casefold()
    for claim in ("содержит противоречие", "противоречие подтверждено", "противоречие установлено"):
        assert claim not in lowered.replace("не означает, что противоречие установлено", "")


def test_graph_section_explains_selection_budget_and_coverage(tmp_path: Path) -> None:
    built = load(tmp_path)
    total = len(built.graph.candidates.pairs)

    section = _section6(tmp_path, max_pair_requests=2)

    assert "- Возможных пар: 36 — полный перебор не выполняется." in section
    assert f"- Пар-кандидатов (общих значимых слов не меньше 2): {total}." in section
    assert "одинаковые числа пару не образуют" in section
    assert "уже связаны подтверждённой связью — 1" in section
    assert f"- Проверено моделью: 2 из {total} (лимит LLM_MAX_PAIR_REQUESTS = 2)." in section
    assert "- Покрытие: требований хотя бы в одной проверенной паре — 3 из 9." in section
    assert f"- Пары-кандидаты, не проверенные моделью: {total - 2}" in section
    assert "- Требования без проверенной пары: 6" in section


def test_graph_section_lists_gaps_and_rejected_hypotheses(tmp_path: Path) -> None:
    foreign = relation("Ф-3", "хранит бронь книги 5 дней", "Ф-4", "хранит бронь книги 5 дней")

    section = _section6(tmp_path, relations_json(foreign))

    assert "### Отклонённые гипотезы связей (1)" in section
    assert "- пара `Ф-3`–`Ф-4`, элемент ответа № 1: Цитата из Ф-3" in section
    assert "Принятых гипотез нет: все предложенные моделью связи (1) отклонены" in section
    assert "- Ссылка `Ф-7` → «Ф-6»: составное требование" in section


def test_rules_only_graph_section_says_pairs_were_not_checked(tmp_path: Path) -> None:
    section = _section6(tmp_path, rules_only=True)

    assert "### Подтверждённые связи (1)" in section
    assert "- Пары-кандидаты моделью не проверялись (режим --rules-only)." in section
    assert "Гипотезы модели о связях" not in section


def test_disabled_pair_check_is_reported(tmp_path: Path) -> None:
    section = _section6(tmp_path, max_pair_requests=0)

    assert "- Проверка пар моделью отключена (LLM_MAX_PAIR_REQUESTS = 0)." in section


def test_empty_graph_section(tmp_path: Path) -> None:
    document, segmentation, lint = _linted(tmp_path, "# Заметки\n\nСправочный текст.\n")

    graph = build_fact_graph(document, segmentation, lint)
    text = render_report(_passport(NOON), document, segmentation, lint, graph=graph)

    assert "Узлов нет: в документе не найдено атомарных требований" in text


# --- раздел 6: формулировки и ограничение отбора (этап 7.1) --------------------------------


def test_hypothesis_relation_type_is_worded_as_an_assumption(tmp_path: Path) -> None:
    section = _section6(tmp_path, relations_json(CONFLICT))

    assert "- Предполагаемый тип связи: conflicts_with — возможное противоречие" in section
    assert "- Тип связи:" not in section
    assert "- Статус: не подтверждено" in section


def test_stages_of_the_pair_check_are_reported_separately(tmp_path: Path) -> None:
    section = _section6(tmp_path, relations_json(CONFLICT))

    assert "- Пар-кандидатов (общих значимых слов не меньше 2): 6." in section
    assert "- Проверено моделью: 6 из 6 (лимит LLM_MAX_PAIR_REQUESTS = 10)." in section
    assert (
        "- Гипотезы модели: прошли проверку цитат (Evidence Gate) — 1, отклонены — 0; "
        "сама связь не проверена." in section
    )
    assert "- Пары вне отбора (общих значимых слов меньше 2): 29 — моделью не проверялись." in (
        section
    )
    assert "Пропусков нет" not in section
    assert "все пары" not in section.casefold()


def test_truncated_candidates_are_counted_and_never_reported_as_checked(tmp_path: Path) -> None:
    section = _section6(tmp_path, max_pair_requests=1, max_candidates=2)

    assert (
        "- Пар-кандидатов (общих значимых слов не меньше 2): найдено 6, оставлено 2 лучших по "
        "оценке — ограничение отбора 2 пар; исключено ограничением: 4, моделью они не "
        "проверялись." in section
    )
    assert "- Проверено моделью: 1 из 2 (лимит LLM_MAX_PAIR_REQUESTS = 1)." in section
    assert (
        "- Пары-кандидаты, исключённые ограничением отбора: 4 — не перечислены поимённо, "
        "моделью не проверялись." in section
    )
    assert "- Пары-кандидаты, не проверенные моделью: 1 — `Ф-1`–`Ф-4`" in section


def test_zero_candidates_report_is_honest(tmp_path: Path) -> None:
    document, segmentation, lint = _linted(
        tmp_path,
        "# Сервис\n\n## Функциональные требования\n\n"
        "* Ф-1. Читатель бронирует книгу.\n* Ф-2. Курьер доставляет посылку.\n",
    )
    graph = build_fact_graph(document, segmentation, lint)
    crosscheck = crosscheck_pairs(
        document, segmentation, graph, FakeProvider(), LLMOptions(max_retries=0)
    )

    text = render_report(
        _passport(NOON), document, segmentation, lint, graph=graph, crosscheck=crosscheck
    )

    assert "- Пар-кандидатов (общих значимых слов не меньше 2): 0." in text
    assert "- Пары вне отбора (общих значимых слов меньше 2): 1 — моделью не проверялись." in text
    assert "Пропусков нет" not in text


def test_disabled_budget_lists_every_unchecked_pair(tmp_path: Path) -> None:
    section = _section6(tmp_path, max_pair_requests=0)

    assert "- Проверка пар моделью отключена (LLM_MAX_PAIR_REQUESTS = 0)." in section
    assert "- Пары-кандидаты, не проверенные моделью: 6 —" in section
    assert "- Пары вне отбора (общих значимых слов меньше 2): 29" in section


# --- раздел 7: противоречия, критерии приёмки, полнота (этап 8) ----------------------------

QUALITY_SPEC = (
    "# Сервис бронирования книг\n\n## Функциональные требования\n\n"
    "* Ф-1. Срок хранения брони: не более 3 дней.\n"
    "* Ф-2. Срок хранения брони — 5 дней.\n"
    "* Ф-3. Система хранит бронь книги не более 3 дней после подтверждения.\n"
    "* Ф-4. Система хранит бронь книги 7 дней после подтверждения.\n"
    "* Ф-5. Срок выдачи: TBD.\n\n"
    "## Критерии приёмки\n\n"
    "* Дано: читатель авторизован.\n"
    "* Когда читатель бронирует книгу.\n"
)
PAIR = relation("Ф-3", "не более 3 дней после подтверждения", "Ф-4", "7 дней после подтверждения")


def _section7(tmp_path: Path, markdown: str = QUALITY_SPEC, *, rules_only: bool = False) -> str:
    built = load(tmp_path, markdown)
    passport = _passport(NOON)
    quality = analyze_quality(built.document, built.segmentation, built.lint, built.graph)
    analysis = crosscheck = None
    if rules_only:
        passport = replace(passport, provider=None, model=None, mode="rules-only")
    else:
        options = LLMOptions(max_retries=0, max_requests=3, max_pair_requests=1)
        analysis = review_requirements(built.document, built.segmentation, FakeProvider(), options)
        crosscheck = crosscheck_pairs(
            built.document,
            built.segmentation,
            built.graph,
            FakeProvider(relations_json(PAIR)),
            options,
        )
        quality = quality.with_hypotheses(crosscheck.hypotheses)
    text = render_report(
        passport,
        built.document,
        built.segmentation,
        built.lint,
        analysis,
        graph=built.graph,
        crosscheck=crosscheck,
        quality=quality,
    )
    return text[text.index("## 7.") : text.index("> Агент на этапе 8:")]


def test_confirmed_contradiction_shows_counterexample(tmp_path: Path) -> None:
    section = _section7(tmp_path)

    confirmed = section[section.index("#### Подтверждённые") : section.index("#### Потенциальные")]
    assert "**7.1.1. `Ф-1` и `Ф-2`** — CTR-01" in confirmed
    assert "- Контрпример: «не более 3 дней» и «5 дней» не пересекаются" in confirmed
    assert "- Статус: подтверждено" in confirmed


def test_model_contradiction_stays_potential_without_counterexample_wording(
    tmp_path: Path,
) -> None:
    section = _section7(tmp_path)

    potential = section[section.index("#### Потенциальные") : section.index("#### Полнота")]
    assert "`Ф-3` и `Ф-4`** — гипотеза модели о противоречии" in potential
    assert "- Сравнение чисел: «не более 3 дней» и «7 дней» не пересекаются" in potential
    assert "не проверено" in potential
    assert "Контрпример" not in potential
    assert "- Статус: не подтверждено (потенциальное противоречие)" in potential


def test_absence_of_contradictions_is_never_claimed(tmp_path: Path) -> None:
    section = _section7(tmp_path)

    assert "- Отсутствие противоречий не утверждается:" in section
    assert "противоречий нет" not in section.casefold()


def test_acceptance_and_completeness_are_reported(tmp_path: Path) -> None:
    section = _section7(tmp_path)

    assert "**7.2.1. ACC-01 · Сценарий приёмки без ожидаемого результата**" in section
    assert "- Связь критериев с требованиями не прослеживается" in section
    assert "- Замечания правил полноты (раздел 4): 1 — `CMP-01:" in section
    assert "- Полнота документа не доказывается" in section


def test_rules_only_quality_section_names_what_was_not_checked(tmp_path: Path) -> None:
    section = _section7(tmp_path, rules_only=True)

    assert "- Смысловые противоречия моделью не проверялись (режим --rules-only)." in section
    assert "- Смысловая полнота моделью не проверялась (режим --rules-only)." in section
    assert "- Смысловая проверка критериев моделью не выполнялась." in section
    assert "гипотеза модели" not in section


def test_document_without_acceptance_criteria_is_an_explicit_observation(tmp_path: Path) -> None:
    section = _section7(
        tmp_path, "# Сервис\n\n## Функциональные требования\n\n* Ф-1. Читатель бронирует книгу.\n"
    )

    assert "Критерии приёмки в документе не выделены" in section
    assert "агент их не распознал" in section
