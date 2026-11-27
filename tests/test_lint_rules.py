"""Детерминированные правила: срабатывание, учёт контекста и точные цитаты.

Все документы — собственные синтетические примеры (домен «библиотека»).
"""

from __future__ import annotations

from pathlib import Path

from tests.lint_helpers import HEAD, lint_markdown, quotes

# --- A. Проверяемость ------------------------------------------------------------------


def test_tst01_subjective_word(tmp_path: Path) -> None:
    linted = lint_markdown(tmp_path, HEAD + "* Ф-1. Каталог книг должен быть удобным.\n")

    [finding] = linted.of("TST-01")
    assert finding.requirement_ids == ("Ф-1",)
    assert quotes(finding) == ["удобным"]
    assert finding.severity == "major"
    assert finding.criterion == "VER"
    assert "удобным" in finding.explanation
    assert finding.evidence_status == "verified"


def test_tst01_skips_measured_quality(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path, HEAD + "* Ф-1. Поиск книги работает быстро: не более 2 секунд.\n"
    )

    assert linted.of("TST-01") == []


def test_tst01_respects_definitions(tmp_path: Path) -> None:
    markdown = (
        "# Сервис бронирования книг\n\n## Термины\n\n"
        "* Быстрый поиск — поиск, который возвращает результат не дольше 1 секунды.\n\n"
        "## Функциональные требования\n\n* Ф-1. Каталог поддерживает быстрый поиск по автору.\n"
    )

    linted = lint_markdown(tmp_path, markdown)

    assert linted.of("TST-01") == []


def test_tst01_ignores_established_terms(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path,
        HEAD + "* Ф-1. Штраф оплачивается через систему быстрых платежей.\n"
        "* Ф-2. Максимально допустимый срок брони задаёт администратор.\n",
    )

    assert linted.of("TST-01") == []


def test_tst02_optional_wording(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path,
        HEAD + "* Ф-1. По возможности система продлевает бронь автоматически.\n"
        "* Ф-2. Желательно показывать обложку книги.\n"
        "* Ф-3. Система показывает возможность продления брони.\n",
    )

    found = linted.of("TST-02")
    assert [(f.requirement_ids, quotes(f)) for f in found] == [
        (("Ф-1",), ["По возможности"]),
        (("Ф-2",), ["Желательно"]),
    ]


def test_tst03_undefined_condition(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path,
        HEAD + "* Ф-1. При необходимости библиотекарь отменяет бронь.\n"
        "* Ф-2. Библиотекарь отменяет бронь, если читатель не пришёл за книгой.\n",
    )

    [finding] = linted.of("TST-03")
    assert quotes(finding) == ["При необходимости"]
    assert finding.severity == "minor"


def test_tst04_goal_without_target(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path,
        HEAD + "* Ф-1. Система минимизирует время ожидания читателя.\n"
        "* Ф-2. Система оптимизирует поиск: не более 3 запросов к каталогу.\n",
    )

    [finding] = linted.of("TST-04")
    assert finding.requirement_ids == ("Ф-1",)
    assert quotes(finding) == ["минимизирует"]


def test_tst05_quality_attribute_without_number(tmp_path: Path) -> None:
    markdown = (
        "# Сервис бронирования книг\n\n## Производительность\n\n"
        "* Система должна выдерживать пиковую нагрузку.\n"
        "* Система выдерживает нагрузку 200 запросов в секунду.\n\n"
        "## Функциональные требования\n\n"
        "* Система проверяет доступность книги перед бронированием.\n"
    )

    linted = lint_markdown(tmp_path, markdown)

    [finding] = linted.of("TST-05")
    assert quotes(finding) == ["нагрузку"]
    assert finding.evidence_kind == "absence"


# --- B. Однозначность ------------------------------------------------------------------


def test_amb01_vague_quantity(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path,
        HEAD + "* Ф-1. Читатель может забронировать несколько книг.\n"
        "* Ф-2. Читатель может продлить несколько броней (не более 3).\n"
        "* Ф-3. Каталог поддерживает несколько форматов: PDF, EPUB.\n",
    )

    [finding] = linted.of("AMB-01")
    assert finding.requirement_ids == ("Ф-1",)
    assert quotes(finding) == ["несколько"]


def test_amb02_approximation(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path,
        HEAD + "* Ф-1. Бронь хранится около 3 дней.\n"
        "* Ф-2. Уведомление приходит примерно в полдень.\n"
        "* Ф-3. Бронь хранится около 72 часов (±1 час).\n"
        "* Ф-4. Кнопка продления расположена около обложки.\n",
    )

    found = linted.of("AMB-02")
    assert [(f.requirement_ids, quotes(f)) for f in found] == [
        (("Ф-1",), ["около 3"]),
        (("Ф-2",), ["примерно"]),
    ]


def test_amb03_vague_time(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path,
        HEAD + "* Ф-1. Система своевременно уведомляет читателя.\n"
        "* Ф-2. Каталог обновляется в режиме реального времени.\n"
        "* Ф-3. Система своевременно, не позднее чем за 24 часа, уведомляет читателя.\n"
        "* Ф-4. Отчёт формируется регулярно: ежедневно в 9:00.\n",
    )

    found = linted.of("AMB-03")
    assert [(f.requirement_ids, quotes(f)) for f in found] == [
        (("Ф-1",), ["своевременно"]),
        (("Ф-2",), ["в режиме реального времени"]),
    ]


def test_overlapping_matches_keep_the_more_specific_finding(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path, HEAD + "* Ф-1. Система уведомляет читателя в течение нескольких минут.\n"
    )

    assert linted.rule_ids == ["AMB-03"]
    assert quotes(linted.of("AMB-03")[0]) == ["в течение нескольких минут"]
    assert linted.result.suppressed == 1


def test_amb04_pronoun_instead_of_subject(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path,
        HEAD + "* Ф-1. Читатель бронирует книгу.\n"
        "* Она хранится три дня.\n"
        "* Ф-2. Бронь хранится три дня, она продлевается один раз.\n"
        "* Данные читателя хранятся в зашифрованном виде.\n",
    )

    [finding] = linted.of("AMB-04")
    assert quotes(finding) == ["Она"]


def test_amb05_undefined_abbreviation_once_per_document(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path,
        HEAD + "* Ф-1. Бронь передаётся в АИБС.\n"
        "* Ф-2. АИБС хранит историю броней.\n"
        "* Ф-3. Читатель указывает ФИО.\n"
        "* Ф-4. Бронь НЕ продлевается автоматически.\n",
    )

    [finding] = linted.of("AMB-05")
    assert finding.requirement_ids == ("Ф-1",)
    assert quotes(finding) == ["АИБС"]
    assert "2" in finding.explanation  # число употреблений


def test_amb05_respects_expansion_in_text(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path,
        HEAD + "* Ф-1. Автоматизированная информационная библиотечная система (АИБС) "
        "хранит каталог.\n"
        "* Ф-2. Бронь передаётся в АИБС.\n",
    )

    assert linted.of("AMB-05") == []


def test_amb05_respects_glossary_table(tmp_path: Path) -> None:
    markdown = (
        "# Сервис бронирования книг\n\n## Сокращения\n\n"
        "| Сокращение | Расшифровка |\n| --- | --- |\n| ЧБ | Читательский билет |\n\n"
        "## Функциональные требования\n\n* Ф-1. Бронь привязана к ЧБ.\n"
    )

    linted = lint_markdown(tmp_path, markdown)

    assert linted.of("AMB-05") == []


# --- C. Полнота ------------------------------------------------------------------------


def test_cmp01_placeholder_section(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path, HEAD + "* Ф-1. Читатель бронирует книгу.\n\n## Отчёты\n\nTBD\n"
    )

    [finding] = linted.of("CMP-01")
    assert quotes(finding) == ["TBD"]
    assert finding.evidence[0].sources[0].location.section_path[-1] == "Отчёты"


def test_cmp01_inline_placeholders(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path,
        HEAD + "* Ф-1. Срок брони: TBD.\n"
        "* Ф-2. Размер штрафа будет определён позже.\n"
        "* Ф-3. Система определяет размер штрафа по тарифу.\n",
    )

    found = linted.of("CMP-01")
    assert [(f.requirement_ids, quotes(f)) for f in found] == [
        (("Ф-1",), ["TBD"]),
        (("Ф-2",), ["будет определён позже"]),
    ]


def test_cmp02_open_enumeration(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path,
        HEAD + "* Ф-1. Каталог хранит автора, жанр, год и т.д.\n"
        "* Ф-2. Поддерживаются коды 1, 2, 3 и далее.\n"
        "* Ф-3. Система проверяет бронь и далее выдаёт книгу.\n"
        "* Ф-4. Доступные форматы: PDF, EPUB, …\n"
        "* Ф-5. Коды 7, 8, 9 и далее — ошибочные.\n",
    )

    found = linted.of("CMP-02")
    assert [(f.requirement_ids, quotes(f)) for f in found] == [
        (("Ф-1",), ["и т.д."]),
        (("Ф-2",), ["и далее"]),
        (("Ф-4",), [", …"]),
        (("Ф-5",), ["и далее"]),
    ]
    assert linted.of("CMP-04") == []


def test_cmp01_placeholder_line_inside_merged_text(tmp_path: Path) -> None:
    """В TXT неоформленный заголовок склеивается со следующей строкой — заглушка всё равно
    находится, по своей строке."""
    linted = lint_markdown(
        tmp_path,
        "Сервис бронирования книг\n\nОтчёты по выдаче\nзаполняется аналитиком\n",
        name="spec.txt",
    )

    [finding] = linted.of("CMP-01")
    assert quotes(finding) == ["заполняется аналитиком"]


def test_cmp03_unfilled_value(tmp_path: Path) -> None:
    markdown = (
        HEAD + "* Ф-1. Читатель бронирует книгу.\n* Адрес тестового стенда: —\n\n"
        "## Параметры\n\n| Параметр | Значение |\n| --- | --- |\n"
        "| Срок брони | 3 дня |\n| Штраф | — |\n\n"
        "## Поля\n\n| Поле | Тип | Комментарий |\n| --- | --- | --- |\n| Автор | строка | — |\n"
    )

    linted = lint_markdown(tmp_path, markdown)

    found = linted.of("CMP-03")
    assert [quotes(f) for f in found] == [["Адрес тестового стенда: —"], ["Штраф | —"]]
    assert all(f.evidence_kind == "absence" for f in found)


def test_cmp03_unfilled_label_inside_text(tmp_path: Path) -> None:
    markdown = (
        HEAD + "* Ф-1. Читатель бронирует книгу.\n\n## Адреса\n\n"
        "| Параметр | Значение |\n| --- | --- |\n| Адрес | Тест: /books . Бой: — |\n"
        "| Время: — отсчитывается от подтверждения | 1 час |\n"
    )

    linted = lint_markdown(tmp_path, markdown)

    [finding] = linted.of("CMP-03")
    assert quotes(finding) == ["Бой: —"]


def test_cmp03_tab_separated_text_row(tmp_path: Path) -> None:
    """TXT, сохранённый из Word, превращает таблицу в строки через табуляцию."""
    text = (
        "Сервис бронирования книг\n\nПараметры\nСрок брони\t3 дня\nШтраф\t—\n"
        "Поле\tстрока\tКомментарий\t—\n"
    )

    linted = lint_markdown(tmp_path, text, name="spec.txt")

    [finding] = linted.of("CMP-03")
    assert quotes(finding) == ["Штраф\t—"]


def test_cmp04_colon_without_items(tmp_path: Path) -> None:
    markdown = (
        HEAD + "Библиотекарь видит поля:\n\n* автор;\n* название.\n\n"
        "Поддерживаются форматы:\n\n| Формат | Описание |\n| --- | --- |\n| PDF | Документ |\n\n"
        "* Ф-1. Библиотекарь получает отчёты:\n\n## Доступ\n\n"
        "* Ф-2. Библиотекарь видит журнал выдачи.\n"
    )

    linted = lint_markdown(tmp_path, markdown)

    [finding] = linted.of("CMP-04")
    assert finding.requirement_ids == ("Ф-1",)
    assert quotes(finding) == ["Библиотекарь получает отчёты:"]


def test_cmp04_truncated_wording(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path,
        HEAD + "* Ф-1. Если читатель не забрал книгу…\n"
        "* Ф-2. Система уведомляет читателя и\n"
        "* Ф-3. Система уведомляет читателя по почте.\n",
    )

    found = linted.of("CMP-04")
    assert [(f.requirement_ids, quotes(f)) for f in found] == [
        (("Ф-1",), ["Если читатель не забрал книгу…"]),
        (("Ф-2",), ["Система уведомляет читателя и"]),
    ]


def test_cmp05_empty_table(tmp_path: Path) -> None:
    markdown = (
        HEAD + "* Ф-1. Читатель бронирует книгу.\n\n## Согласование\n\n"
        "| Роль | ФИО | Дата |\n| --- | --- | --- |\n| | | |\n| | | |\n\n"
        "## История\n\n| Версия | Дата |\n| --- | --- |\n| 1.0 | 01.02.2026 |\n"
    )

    linted = lint_markdown(tmp_path, markdown)

    [finding] = linted.of("CMP-05")
    assert finding.evidence_kind == "absence"
    assert quotes(finding) == ["Роль | ФИО | Дата", "", ""]
    assert len(finding.requirement_ids) == 3
    assert finding.evidence_status == "verified"


def test_cmp06_missing_numbered_section(tmp_path: Path) -> None:
    markdown = (
        "# 1. Сервис бронирования книг\n\n## 1.1 Функциональные требования\n\n"
        "* Ф-1. Формат отчёта описан в разделе 1.5.\n"
        "* Ф-2. Поля брони перечислены в разделе 1.2.\n\n"
        "## 1.2 Поля брони\n\n* Ф-3. Бронь содержит автора и название книги.\n"
    )

    linted = lint_markdown(tmp_path, markdown)

    [finding] = linted.of("CMP-06")
    assert finding.requirement_ids == ("Ф-1",)
    assert quotes(finding) == ["разделе 1.5"]


def test_cmp06_skips_documents_without_section_numbers(tmp_path: Path) -> None:
    linted = lint_markdown(tmp_path, HEAD + "* Ф-1. Формат отчёта описан в разделе 4.\n")

    assert linted.of("CMP-06") == []


def test_cmp07_heading_id_without_text(tmp_path: Path) -> None:
    markdown = (
        HEAD + "### Ф-1 Бронирование книги\n\n### Ф-2 Продление брони\n\n"
        "Читатель продлевает бронь один раз.\n"
    )

    linted = lint_markdown(tmp_path, markdown)

    [finding] = linted.of("CMP-07")
    assert quotes(finding) == ["Ф-1 Бронирование книги"]


# --- D. Согласованность ----------------------------------------------------------------


def test_con01_exact_duplicate_references_both_requirements(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path,
        HEAD + "* Ф-1. Читатель бронирует книгу через каталог.\n"
        "* Ф-2. Читатель бронирует книгу через каталог.\n",
    )

    [finding] = linted.of("CON-01")
    assert finding.requirement_ids == ("Ф-1", "Ф-2")
    assert quotes(finding) == ["Читатель бронирует книгу через каталог."] * 2
    assert [e.sources[0].location.line_start for e in finding.evidence] == [5, 6]
    assert finding.evidence_kind == "comparison"


def test_con02_similar_but_not_duplicate(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path,
        HEAD + "* Ф-1. Читатель может продлить бронь книги в личном кабинете до 3 раз.\n"
        "* Ф-2. Читатель может продлить бронь книги в личном кабинете до 5 раз.\n",
    )

    [finding] = linted.of("CON-02")
    assert finding.severity == "info"
    assert finding.requirement_ids == ("Ф-1", "Ф-2")
    assert "«3» → «5»" in finding.explanation
    assert "противореч" not in (finding.title + finding.explanation).casefold()
    assert linted.of("CON-01") == []


def test_con03_conflicting_values_in_one_table(tmp_path: Path) -> None:
    markdown = (
        HEAD + "* Ф-1. Читатель бронирует книгу.\n\n## Параметры\n\n"
        "| Параметр | Значение |\n| --- | --- |\n"
        "| Срок брони | 3 дня |\n| Штраф | 100 ₸ |\n| Срок брони | 5 дней |\n\n"
        "## Статусы\n\n| Статус | Описание |\n| --- | --- |\n"
        "| Отменена | Читатель отменил бронь |\n| Отменена | Истёк срок брони |\n"
    )

    linted = lint_markdown(tmp_path, markdown)

    [finding] = linted.of("CON-03")
    assert quotes(finding) == ["Срок брони | 3 дня", "Срок брони | 5 дней"]
    assert finding.severity == "major"


# --- E. Трассируемость -----------------------------------------------------------------


def test_trc01_requirement_without_id_among_numbered(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path,
        HEAD + "* Ф-1. Читатель бронирует книгу.\n* Ф-2. Читатель отменяет бронь.\n"
        "* Читатель продлевает бронь.\n",
    )

    [finding] = linted.of("TRC-01")
    assert quotes(finding) == ["Читатель продлевает бронь."]


def test_trc01_silent_when_document_has_no_ids(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path, HEAD + "* Читатель бронирует книгу.\n* Читатель отменяет бронь.\n"
    )

    assert linted.of("TRC-01") == []


def test_trc02_reused_id(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path, HEAD + "* Ф-1. Читатель бронирует книгу.\n* Ф-1. Читатель продлевает бронь.\n"
    )

    [finding] = linted.of("TRC-02")
    assert finding.requirement_ids == ("Ф-1", "Ф-1~2")
    assert quotes(finding) == ["Ф-1", "Ф-1"]
    assert finding.severity == "major"


def test_trc03_reference_to_missing_requirement(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path,
        HEAD + "* Ф-1. Читатель бронирует книгу.\n"
        "* Ф-2. Продление выполняется согласно Ф-7.\n"
        "* Ф-3. Отмена выполняется согласно Ф-1.\n",
    )

    [finding] = linted.of("TRC-03")
    assert finding.requirement_ids == ("Ф-2",)
    assert quotes(finding) == ["Ф-7"]


def test_trc04_mixed_alphabet_id(tmp_path: Path) -> None:
    mixed = "F" + chr(0x0420) + "-2"  # кириллическая «Р» среди латиницы
    linted = lint_markdown(
        tmp_path, HEAD + f"* FR-1. Читатель бронирует книгу.\n* {mixed}. Читатель отменяет бронь.\n"
    )

    [finding] = linted.of("TRC-04")
    assert quotes(finding) == [mixed]


def test_trc05_heading_and_text_id_conflict(tmp_path: Path) -> None:
    markdown = (
        HEAD + "### Ф-1 Бронирование\n\n* Ф-5. Читатель бронирует книгу.\n\n"
        "### Ф-2 Продление\n\n* Ф-2.1 Читатель продлевает бронь.\n"
    )

    linted = lint_markdown(tmp_path, markdown)

    [finding] = linted.of("TRC-05")
    assert quotes(finding) == ["Ф-1", "Ф-5"]
    assert finding.evidence_kind == "comparison"


def test_trc06_missing_parent_id(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path,
        HEAD + "* Ф-2.1 Читатель бронирует книгу.\n* Ф-2.2 Читатель отменяет бронь.\n"
        "* Ф-3. Читатель продлевает бронь.\n* Ф-3.1 Продление доступно один раз.\n",
    )

    [finding] = linted.of("TRC-06")
    assert quotes(finding) == ["Ф-2.1"]
    assert "Ф-2" in finding.explanation
