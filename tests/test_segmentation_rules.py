"""Классификация фрагментов: роль и тип требования определяются по нескольким сигналам."""

from __future__ import annotations

from pathlib import Path

import docx
import pytest

from tests.seg_helpers import segment_markdown, unit
from validator.ingest import load_document
from validator.segmentation import Segmentation, segment_document


def test_functional_list_items_with_and_without_ids(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        "# Сервис\n\n## Функциональные требования\n\n"
        "* R-1. Система хранит порог.\n* Система отправляет уведомление клиенту.\n",
    )

    explicit = unit(seg, "R-1.")
    implicit = unit(seg, "Система отправляет")

    assert (explicit.id, explicit.doc_id, explicit.role, explicit.category) == (
        "R-1",
        "R-1",
        "requirement",
        "functional",
    )
    assert explicit.confidence == "high"
    assert "explicit_id" in explicit.detection
    assert "section:functional" in explicit.detection
    assert (implicit.role, implicit.category, implicit.confidence) == (
        "requirement",
        "functional",
        "high",
    )
    assert implicit.doc_id is None


@pytest.mark.parametrize(
    ("text", "category", "cue"),
    [
        (
            "Сервис отправляет уведомление в течение 2 секунд после события.",
            "performance",
            "performance",
        ),
        (
            "Доступ к настройкам возможен только после авторизации пользователя.",
            "security",
            "security",
        ),
        ("Доступность сервиса не ниже 99,9 % в месяц.", "availability", "availability"),
        ("Интерфейс поддерживает тёмную тему оформления.", "non_functional", None),
    ],
)
def test_non_functional_category_is_refined_by_content(
    tmp_path: Path, text: str, category: str, cue: str | None
) -> None:
    seg = segment_markdown(tmp_path, f"# Сервис\n\n## Нефункциональные требования\n\n{text}\n")

    fragment = unit(seg, text[:20])

    assert (fragment.role, fragment.category) == ("requirement", category)
    if cue:
        assert f"cue:{cue}" in fragment.detection


@pytest.mark.parametrize(
    ("section", "category"),
    [
        ("Ограничения", "constraint"),
        ("Бизнес-правила", "business_rule"),
        ("Требования к данным", "data"),
        ("Требования к интеграции", "integration"),
        ("REST API", "api"),
        ("Дизайн интерфейса", "interface"),
        ("Цели", "goal"),
        ("Архитектура решения", "constraint"),
        ("Требования безопасности", "security"),
    ],
)
def test_section_defines_category(tmp_path: Path, section: str, category: str) -> None:
    seg = segment_markdown(
        tmp_path, f"# Сервис\n\n## {section}\n\nСистема хранит историю изменений порога.\n"
    )

    fragment = unit(seg, "Система хранит")

    assert (fragment.role, fragment.category) == ("requirement", category)


def test_innermost_section_wins(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        "# Сервис\n\n## Требования к изменениям\n\n### Нефункциональные требования\n\n"
        "Сервис хранит журнал операций.\n",
    )

    assert unit(seg, "Сервис хранит").category == "non_functional"


def test_acceptance_table_rows_and_header(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        "# Сервис\n\n## Критерии приёмки\n\n| Условие | Действие | Результат |\n"
        "| --- | --- | --- |\n| Бронь на 14 дней | Прошло 14 дней | Одно напоминание |\n",
    )

    header = unit(seg, "Условие")
    row = unit(seg, "Бронь на 14")

    assert header.role == "table_header"
    assert (row.role, row.category, row.confidence) == ("requirement", "acceptance", "high")


def test_reference_tables_and_empty_rows(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        "# Сервис\n\n## Общие сведения\n\n| Поле | Значение |\n| --- | --- |\n"
        "| Продукт | Мобильный банк |\n\n## Блок согласующих\n\n"
        "| Подразделение | Подпись |\n| --- | --- |\n| | |\n",
    )

    meta = unit(seg, "Продукт")
    empty = next(f for f in seg.fragments if f.text == "" and f.role != "table_header")

    assert meta.role == "reference"
    assert empty.role == "reference"
    assert any("Пустая строка таблицы" in w for w in empty.warnings)


def test_example_and_note_attach_to_previous_requirement(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        "# Сервис\n\n## Функциональные требования\n\n* R-1. Порог задаётся целым числом.\n\n"
        "Например, клиент задаёт 5 000 ₸ и должен получить уведомление.\n\n"
        "Примечание: копейки не используются.\n",
    )

    example = unit(seg, "Например")
    note = unit(seg, "Примечание")

    assert (example.role, example.parent_id) == ("example", "R-1")
    assert (note.role, note.parent_id) == ("explanation", "R-1")
    assert {example.id, note.id} <= set(unit(seg, "R-1").children)


@pytest.mark.parametrize("placeholder", ["заполняется после согласования", "TBD", "—"])
def test_placeholders_are_reference_with_warning(tmp_path: Path, placeholder: str) -> None:
    seg = segment_markdown(tmp_path, f"# Сервис\n\n## Требования к тестированию\n\n{placeholder}\n")

    fragment = unit(seg, placeholder)

    assert fragment.role == "reference"
    assert any("заглушк" in w for w in fragment.warnings)


def test_normative_text_outside_requirement_sections_is_ambiguous(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        "# Описание сервиса\n\nСистема должна хранить порог.\n\n"
        "Документ описывает сервис уведомлений.\n\nКлиент открывает приложение утром.\n\n"
        "Сервис появился в 2024 году как пилот.\n",
    )

    normative = unit(seg, "Система должна")
    self_reference = unit(seg, "Документ описывает")
    behaviour = unit(seg, "Клиент открывает")
    history = unit(seg, "Сервис появился")

    assert normative.role == "ambiguous"
    assert normative.confidence == "low"
    assert normative.warnings
    assert self_reference.role == "reference"
    assert behaviour.role == "ambiguous"
    assert history.role == "ambiguous"


def test_plain_narrative_outside_sections_is_reference(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path, "# Заметки\n\nКомплект собран из двух исходных файлов без правок.\n"
    )

    assert unit(seg, "Комплект").role == "reference"


def test_modal_word_in_reference_section_is_ambiguous_not_requirement(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path, "# Сервис\n\n## Глоссарий\n\nПорог должен быть целым числом тенге.\n"
    )

    assert unit(seg, "Порог должен").role == "ambiguous"


def test_heading_that_looks_like_requirement(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        "# Сервис\n\n## Система должна шифровать данные карты\n\nПодробности ниже.\n\n"
        "## Обычный раздел\n\nТекст раздела.\n",
    )

    suspicious = unit(seg, "Система должна шифровать")
    normal = unit(seg, "Обычный раздел")

    assert suspicious.role == "ambiguous"
    assert any("Заголовок" in w for w in suspicious.warnings)
    assert normal.role == "heading"


def test_code_block_is_technical_fragment(tmp_path: Path) -> None:
    seg = segment_markdown(tmp_path, '# Сервис\n\n## REST API\n\n```json\n{"result": "OK"}\n```\n')

    code = unit(seg, '{"result"')

    assert (code.role, code.category) == ("code", "api")


@pytest.mark.parametrize(
    "text",
    [
        '{"status": "DONE", "fine": "2500"}',
        '[{"id": 1}, {"id": 2}]',
        "GET /books/7 HTTP/1.1",
        'curl -s "$BASE/books?id=7"',
        "<request><id>1</id></request>",
    ],
)
def test_code_like_paragraph_is_technical_fragment(tmp_path: Path, text: str) -> None:
    """В TXT, DOCX и PDF нет разметки кода: пример JSON или запроса узнаётся по содержанию."""
    seg = segment_markdown(tmp_path, f"# Сервис\n\n## REST API\n\n{text}\n", name="spec.txt")

    assert unit(seg, text[:8]).role == "code"


def test_prose_mentioning_braces_is_not_code(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path, "# Сервис\n\n## REST API\n\nОтвет содержит поле {result} со значением OK.\n"
    )

    assert unit(seg, "Ответ содержит").role == "requirement"


def test_word_between_code_examples_is_not_glued_to_code(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        '# Сервис\n\n## REST API\n\n{"result": "OK"}\n\nили\n\n{"result": "ERROR"}\n',
    )

    assert [(f.role, f.text) for f in seg.fragments[2:]] == [
        ("code", '{"result": "OK"}'),
        ("reference", "или"),
        ("code", '{"result": "ERROR"}'),
    ]


@pytest.mark.parametrize(
    "intro",
    [
        "В этом разделе описаны функциональные требования к сервису.",
        "В данном разделе приведены правила бронирования.",
        "Ниже перечислены основные сценарии работы.",
        "Далее описаны требования к интерфейсу каталога.",
        "Данный раздел описывает требования к выдаче книг.",
        "Раздел содержит требования к уведомлениям.",
        "This section describes the booking requirements.",
    ],
)
def test_section_intro_is_reference_not_requirement(tmp_path: Path, intro: str) -> None:
    seg = segment_markdown(
        tmp_path,
        f"# Сервис\n\n## Функциональные требования\n\n{intro}\n\n* Система хранит бронь.\n",
    )

    fragment = unit(seg, intro[:15])

    assert fragment.role == "reference"
    assert "section_intro" in fragment.detection
    assert unit(seg, "Система хранит бронь").role == "requirement"


@pytest.mark.parametrize(
    "text",
    [
        "Ниже порога уведомление не отправляется.",
        "Далее система отправляет напоминание читателю.",
        "В этом месяце читатель получает одно напоминание.",
        "Раздел каталога открывается без авторизации.",
    ],
)
def test_requirements_resembling_intro_stay_requirements(tmp_path: Path, text: str) -> None:
    seg = segment_markdown(tmp_path, f"# Сервис\n\n## Функциональные требования\n\n{text}\n")

    assert unit(seg, text[:15]).role == "requirement"


def _docx_segmentation(tmp_path: Path, paragraphs: list[str]) -> Segmentation:
    document = docx.Document()
    document.add_heading("REST API", level=1)
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    path = tmp_path / "api.docx"
    document.save(str(path))
    return segment_document(load_document(path))


def test_json_split_into_docx_paragraphs_is_code(tmp_path: Path) -> None:
    """В Word пример JSON часто набран абзацем на строку — это не требования."""
    lines = [
        "{",
        '  "result": "OK",',
        '  "amount": 500,',
        '  "items": [',
        "    1,",
        "    2",
        "  ]",
        "}",
    ]

    seg = _docx_segmentation(tmp_path, lines)
    body = seg.fragments[1:]

    assert [f.role for f in body] == ["code"] * len(lines)
    assert [f.text for f in body] == [line.strip() for line in lines]  # текст не потерян
    assert seg.requirements == ()


def test_json_string_values_next_to_code_are_code(tmp_path: Path) -> None:
    seg = _docx_segmentation(tmp_path, ["[", '"push",', '"sms"', "]"])

    assert [f.role for f in seg.fragments[1:]] == ["code", "code", "code", "code"]


@pytest.mark.parametrize(
    "text",
    [
        '"Порог" задаётся читателем в настройках профиля.',
        "Ответ содержит поле result со значением OK.",
        "Список форматов задаётся в настройках: PDF, EPUB.",
    ],
)
def test_prose_with_quotes_or_colons_is_not_code(tmp_path: Path, text: str) -> None:
    seg = _docx_segmentation(tmp_path, [text])

    assert seg.fragments[1].role == "requirement"


def test_status_label_before_json_example_is_not_code(tmp_path: Path) -> None:
    """Метка «404» перед примером ответа — не элемент JSON, хотя похожа на число."""
    seg = _docx_segmentation(
        tmp_path, ["Для чужой книги возвращается 404.", "404", '{"message": "Книга не найдена"}']
    )

    assert [f.role for f in seg.fragments[1:]] == ["requirement", "reference", "code"]


def test_lonely_quoted_word_is_not_code_without_code_neighbours(tmp_path: Path) -> None:
    seg = _docx_segmentation(tmp_path, ["Ответ содержит статус.", '"OK"'])

    assert seg.fragments[2].role != "code"


def test_explicit_id_outside_requirement_section_is_flagged(tmp_path: Path) -> None:
    seg = segment_markdown(tmp_path, "# Введение\n\nR-9. Сервис хранит порог.\n")

    fragment = unit(seg, "R-9.")

    assert (fragment.role, fragment.confidence) == ("requirement", "medium")
    assert fragment.warnings


def test_metadata_rows_in_api_table_are_reference(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        "# Сервис\n\n## REST API\n\n| Поле | Значение |\n| --- | --- |\n"
        "| Method | POST |\n| Author | Команда разработки |\n| Swagger | — |\n",
    )

    method = unit(seg, "Method")
    author = unit(seg, "Author")
    swagger = unit(seg, "Swagger")

    assert (method.role, method.category) == ("requirement", "api")
    assert author.role == "reference"
    assert swagger.role == "requirement"
    assert any("не заполнено" in w for w in swagger.warnings)


def test_choice_cells_of_checklist_are_not_normative(tmp_path: Path) -> None:
    """«Готово / В работе / Не требуется» — варианты выбора шаблона, а не формулировка
    требования: нормативное слово внутри варианта строку требованием не делает."""
    seg = segment_markdown(
        tmp_path,
        "# Сервис\n\n## Ревью требований\n\n| Проверка | Создано | Проверено |\n"
        "| --- | --- | --- |\n"
        "| Диаграммы предусмотрены | Готово / В работе / Не требуется | Принято / Отклонено |\n"
        "| Доступ к журналу | Авторизация не требуется для чтения журнала "
        "| Принято / Отклонено |\n",
    )

    assert unit(seg, "Диаграммы предусмотрены").role == "reference"
    assert unit(seg, "Доступ к журналу").role == "ambiguous"


def test_metadata_key_must_be_whole_word(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        "# Сервис\n\n## REST API\n\n| Поле | Значение |\n| --- | --- |\n"
        "| Authorization | Bearer-токен читателя |\n| Gitlab project | Внутренний репозиторий |\n",
    )

    assert (unit(seg, "Authorization").role, unit(seg, "Authorization").category) == (
        "requirement",
        "api",
    )
    assert unit(seg, "Gitlab project").role == "reference"


def test_single_word_paragraph_is_not_requirement(tmp_path: Path) -> None:
    seg = segment_markdown(tmp_path, "# Сервис\n\n## REST API\n\nили\n")

    assert unit(seg, "или").role == "reference"
