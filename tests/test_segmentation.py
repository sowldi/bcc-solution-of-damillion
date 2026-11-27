"""Структура сегментации: границы требований, связи, доказательная привязка к источнику."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.builders import (
    LI,
    Element,
    H,
    P,
    T,
    Writer,
    write_docx,
    write_markdown,
    write_sample_pdf,
    write_text,
)
from tests.seg_helpers import child, segment_markdown, unit
from validator.ingest import load_document
from validator.model import Document
from validator.segmentation import Segmentation, segment_document, verify_segmentation

SPEC = """# Сервис уведомлений

Документ описывает требования к сервису.

## Функциональные требования

* Ф-1. Читатель бронирует книгу
  на срок от 1 до 14 дней включительно. По умолчанию бронь не продлевается.
* Ф-2. Когда срок брони истекает, читатель получает напоминание.
  Повторное напоминание в тот же день не отправляется.
* Система поддерживает способы оповещения:
* e-mail;
* SMS.

## Критерии приёмки

| Условие | Действие | Результат |
| --- | --- | --- |
| Бронь на 14 дней | Прошло 14 дней | Одно напоминание |
"""


@pytest.fixture
def spec(tmp_path: Path) -> tuple[Segmentation, Document]:
    path = tmp_path / "spec.md"
    path.write_text(SPEC, encoding="utf-8")
    document = load_document(path)
    return segment_document(document), document


def test_multiline_requirement_is_one_unit_split_into_statements(
    spec: tuple[Segmentation, Document],
) -> None:
    seg, document = spec
    parent = unit(seg, "Ф-1.")
    first, second = child(seg, "Читатель"), child(seg, "По умолчанию")

    assert parent.id == "Ф-1"
    assert parent.text == document.blocks[parent.sources[0].block].text  # цитата без потерь
    assert parent.children == ("Ф-1/1", "Ф-1/2")
    assert (first.id, first.parent_id, first.doc_id) == ("Ф-1/1", "Ф-1", "Ф-1")
    assert first.text == "Читатель бронирует книгу\nна срок от 1 до 14 дней включительно."
    assert first.normalized == "Читатель бронирует книгу на срок от 1 до 14 дней включительно."
    # строки уточнены до утверждения: первое — строки 7–8, второе — только строка 8
    assert (first.sources[0].location.line_start, first.sources[0].location.line_end) == (7, 8)
    assert (second.sources[0].location.line_start, second.sources[0].location.line_end) == (8, 8)


def test_composite_parent_is_not_atomic(spec: tuple[Segmentation, Document]) -> None:
    seg, _ = spec
    atomic_ids = {f.id for f in seg.atomic_requirements}

    assert {"Ф-1/1", "Ф-1/2", "Ф-2/1", "Ф-2/2"} <= atomic_ids
    assert "Ф-1" not in atomic_ids
    assert "Ф-2" not in atomic_ids


def test_enumeration_items_belong_to_colon_stem(spec: tuple[Segmentation, Document]) -> None:
    seg, _ = spec
    stem = unit(seg, "Система поддерживает способы")
    push, sms = unit(seg, "e-mail;"), unit(seg, "SMS.")

    assert push.parent_id == stem.id
    assert sms.parent_id == stem.id
    assert stem.children == (push.id, sms.id)
    assert "enumeration" in push.detection
    assert stem.id not in {f.id for f in seg.atomic_requirements}


def test_table_row_is_one_requirement(spec: tuple[Segmentation, Document]) -> None:
    seg, _ = spec
    row = unit(seg, "Бронь на 14")

    assert row.text == "Бронь на 14 дней | Прошло 14 дней | Одно напоминание"
    assert (row.sources[0].location.table, row.sources[0].location.row) == (1, 2)
    assert row.children == ()


def test_every_block_is_covered_exactly_once(spec: tuple[Segmentation, Document]) -> None:
    seg, document = spec
    units = [f for f in seg.fragments if "/" not in f.id]
    covered = [ref.block for f in units for ref in f.sources]

    assert sorted(covered) == list(range(len(document.blocks)))


def test_segmentation_passes_source_verification(spec: tuple[Segmentation, Document]) -> None:
    seg, document = spec

    assert verify_segmentation(seg, document) == ()


def test_continuation_paragraph_is_merged_with_all_sources(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        "# Сервис\n\n## Функциональные требования\n\nСистема хранит историю изменений порога\n\n"
        "и показывает её в приложении.\n",
    )

    merged = unit(seg, "Система хранит")

    assert merged.text == "Система хранит историю изменений порога\nи показывает её в приложении."
    assert [ref.block for ref in merged.sources] == [2, 3]
    assert "continuation" in merged.detection


def test_duplicates_get_suffix_and_link(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        "# Сервис\n\n## Функциональные требования\n\n* Система хранит порог.\n"
        "* Система отправляет push.\n* Система хранит порог.\n",
    )
    first, *rest = [f for f in seg.fragments if f.text == "Система хранит порог."]
    (second,) = rest

    assert second.id == f"{first.id}~2"
    assert second.duplicate_of == first.id
    assert any(first.id in w for w in second.warnings)


def test_reused_explicit_id_is_disambiguated(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        "# Сервис\n\n## Функциональные требования\n\n* R-1. Система хранит порог.\n"
        "* R-1. Система удаляет порог.\n",
    )

    ids = [f.id for f in seg.fragments if f.doc_id == "R-1"]

    assert ids == ["R-1", "R-1~2"]
    assert any("R-1" in w for w in unit(seg, "R-1. Система удаляет").warnings)


def test_similar_requirements_are_linked(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        "# Сервис\n\n## Функциональные требования\n\n"
        "* Система отправляет push клиенту при достижении порога.\n"
        "* Система отправляет push клиенту при достижении порога расходов.\n",
    )
    a = unit(seg, "Система отправляет push клиенту при достижении порога.")
    b = unit(seg, "Система отправляет push клиенту при достижении порога расходов.")

    assert a.similar_to == (b.id,)
    assert b.similar_to == (a.id,)
    assert b.duplicate_of is None


def test_references_to_requirements_and_sections(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        "# Сервис\n\n## Функциональные требования\n\n* Ф-1. Система хранит порог.\n"
        "* Ф-2. Порог меняется как в Ф-1, формат в разделе «Требования к данным».\n\n"
        "## Требования к данным\n\nПорог хранится в тенге.\n",
    )

    assert unit(seg, "Ф-2.").references == ("req:Ф-1", "section:Требования к данным")


def test_http_status_paragraph_is_split_by_status(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        "# Сервис\n\n## REST API\n\n"
        "Коды 200 и 201 означают успешную обработку. "
        "Коды 400, 401 и выше означают ошибку клиента.\n",
    )
    parent = unit(seg, "Коды 200 и 201")

    assert [seg.by_id(c).text for c in parent.children] == [
        "Коды 200 и 201 означают успешную обработку.",
        "Коды 400, 401 и выше означают ошибку клиента.",
    ]


def test_document_without_requirements(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        "# Заметки\n\nДокумент содержит справочную информацию.\n\n"
        "Комплект собран из двух файлов без правок.\n",
    )

    assert seg.requirements == ()
    assert seg.atomic_requirements == ()
    assert any("не обнаружены" in w for w in seg.warnings)


def test_empty_document_model_gives_empty_segmentation() -> None:
    document = Document(source=Path("empty.md"), format="markdown", sha256="0" * 64, blocks=())

    seg = segment_document(document)

    assert seg.fragments == ()
    assert seg.document_sha256 == "0" * 64
    assert any("не обнаружены" in w for w in seg.warnings)


# --- ID документа в заголовке («одно требование — один заголовок») ----------------

HEADING_IDS = """# Сервис

## Функциональные требования

### FR-1 Создание брони

Система должна позволять читателю забронировать книгу.

### FR-2 Отмена брони

Система отменяет бронь до выдачи книги. Отмена фиксируется в журнале.

### Общие положения

Сервис работает круглосуточно.

## Связи

Система отменяет бронь так же, как описано в FR-1.
"""


def test_heading_id_goes_to_requirement_not_heading(tmp_path: Path) -> None:
    seg = segment_markdown(tmp_path, HEADING_IDS)
    heading = unit(seg, "FR-1 Создание брони")
    requirement = seg.by_id("FR-1")

    assert heading.role == "heading"
    assert heading.id.startswith("F-")
    assert heading.doc_id is None
    assert requirement.role == "requirement"
    assert requirement.text == "Система должна позволять читателю забронировать книгу."
    assert requirement.doc_id == "FR-1"
    assert "doc_id:heading" in requirement.detection
    # источник — абзац требования, а не заголовок
    assert requirement.sources[0].location.line_start == 7


def test_heading_id_reaches_statements_report_and_json(tmp_path: Path) -> None:
    seg = segment_markdown(tmp_path, HEADING_IDS)
    composite = seg.by_id("FR-2")

    assert composite.children == ("FR-2/1", "FR-2/2")
    assert {seg.by_id(c).doc_id for c in composite.children} == {"FR-2"}
    assert Segmentation.from_json(seg.to_json()) == seg
    assert unit(seg, "Сервис работает").doc_id is None


def test_reference_to_heading_id_resolves_to_requirement(tmp_path: Path) -> None:
    seg = segment_markdown(tmp_path, HEADING_IDS)
    referring = unit(seg, "Система отменяет бронь так же")

    assert "req:FR-1" in referring.references
    assert seg.by_id("FR-1").role == "requirement"


def test_heading_id_makes_text_a_requirement_outside_requirement_sections(
    tmp_path: Path,
) -> None:
    seg = segment_markdown(
        tmp_path, "# Каталог\n\n### FR-7 Поиск книги\n\nСистема должна искать книгу по названию.\n"
    )

    found = seg.by_id("FR-7")
    assert (found.role, found.doc_id) == ("requirement", "FR-7")
    assert seg.ambiguous == ()


def test_several_requirements_under_one_id_heading(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        "# Сервис\n\n## Функциональные требования\n\n### FR-3 Выдача\n\n"
        "Система выдаёт книгу по читательскому билету.\n\n"
        "Система отправляет квитанцию на e-mail.\n",
    )
    first = unit(seg, "Система выдаёт книгу")
    second = unit(seg, "Система отправляет квитанцию")

    assert (first.id, first.doc_id) == ("FR-3", "FR-3")
    assert second.id.startswith("R-")
    assert second.doc_id == "FR-3"


def test_nested_headings_and_closest_id_wins(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        "# Сервис\n\n## Функциональные требования\n\n### FR-4 Профиль\n\n#### Описание\n\n"
        "Система хранит профиль читателя.\n\n### FR-5 Поиск\n\n#### FR-5.1 Фильтры\n\n"
        "Система фильтрует результаты поиска.\n",
    )

    assert unit(seg, "Система хранит профиль").id == "FR-4"
    assert unit(seg, "Система фильтрует").id == "FR-5.1"
    group = unit(seg, "FR-5 Поиск")
    assert (group.role, group.doc_id) == ("heading", "FR-5")  # ID группы не теряется
    assert not any("FR-5" in w for w in seg.warnings)


def test_id_heading_without_requirement_text_is_reported(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        "# Сервис\n\n## Функциональные требования\n\n### FR-6 Экспорт\n\n### FR-7 Импорт\n\n"
        "Система импортирует каталог.\n",
    )

    empty = unit(seg, "FR-6 Экспорт")
    assert (empty.role, empty.doc_id) == ("heading", "FR-6")
    assert any("FR-6" in w for w in seg.warnings)
    assert unit(seg, "Система импортирует").id == "FR-7"


def test_plain_heading_text_is_not_an_id(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        "# Сервис\n\n## Функциональные требования\n\n### 2.1 Создание брони\n\n"
        "Система создаёт бронь.\n",
    )

    created = unit(seg, "Система создаёт бронь")
    assert created.id.startswith("R-")
    assert created.doc_id is None


def test_repeated_heading_id_is_disambiguated(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        "# Сервис\n\n## Функциональные требования\n\n### FR-1 Бронь\n\nСистема создаёт бронь.\n\n"
        "### FR-1 Выдача\n\nСистема выдаёт книгу.\n",
    )

    second = unit(seg, "Система выдаёт книгу")
    assert (second.id, second.doc_id) == ("FR-1~2", "FR-1")
    assert any("FR-1" in w for w in second.warnings)


def test_conflicting_id_in_text_and_heading(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        "# Сервис\n\n## Функциональные требования\n\n### FR-8 Возврат\n\n"
        "R-9. Система оформляет возврат книги.\n",
    )

    own = seg.by_id("R-9")
    assert own.doc_id == "R-9"
    assert any("R-9" in w and "FR-8" in w for w in own.warnings)
    assert unit(seg, "FR-8 Возврат").doc_id == "FR-8"  # ID заголовка не потерян


def test_same_id_in_text_and_heading_is_not_a_conflict(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        "# Сервис\n\n## Функциональные требования\n\n### FR-9 Бронь\n\n"
        "FR-9. Система бронирует книгу.\n",
    )

    booked = seg.by_id("FR-9")
    assert (booked.role, booked.warnings) == ("requirement", ())
    assert unit(seg, "FR-9 Бронь").doc_id is None


def test_hierarchical_ids_under_group_heading(tmp_path: Path) -> None:
    seg = segment_markdown(
        tmp_path,
        "# Сервис\n\n## Функциональные требования\n\n### FR-10 Заказы\n\n"
        "* FR-10.1 Система создаёт заказ.\n* FR-10.2 Система отменяет заказ.\n\n"
        "Заказы хранятся пять лет.\n",
    )

    assert seg.by_id("FR-10.1").warnings == ()
    assert seg.by_id("FR-10.2").warnings == ()
    assert unit(seg, "Заказы хранятся").doc_id == "FR-10"
    assert unit(seg, "FR-10 Заказы").doc_id == "FR-10"


def test_heading_ids_are_stable_on_rerun(tmp_path: Path) -> None:
    first = segment_markdown(tmp_path, HEADING_IDS, "a.md")
    second = segment_markdown(tmp_path, HEADING_IDS, "b.md")

    assert first == second


# --- одинаковая логика во всех форматах ------------------------------------------

SEG_SAMPLE: tuple[Element, ...] = (
    H(1, "Сервис уведомлений"),
    P("Документ описывает требования к сервису."),
    H(2, "Функциональные требования"),
    LI("R-1. Читатель бронирует книгу на срок до 14 дней. По умолчанию бронь не продлевается."),
    LI("R-2. Когда срок брони истекает, читатель получает напоминание."),
    P("Например, бронь от 1 марта истекает 15 марта."),
    H(2, "Нефункциональные требования"),
    P("Сервис отправляет push в течение 2 секунд после события."),
    H(2, "Критерии приёмки"),
    T(
        (
            ("Условие", "Действие", "Результат"),
            ("Бронь на 14 дней", "Прошло 14 дней", "Одно напоминание"),
        )
    ),
)


def _shape(seg: Segmentation) -> list[tuple[object, ...]]:
    return [
        (f.id, f.role, f.category, f.normalized, f.parent_id, f.doc_id, f.confidence)
        for f in seg.fragments
    ]


@pytest.mark.parametrize(
    ("writer", "name", "tables"),
    [
        (write_text, "spec.txt", False),
        (write_docx, "spec.docx", True),
        (write_sample_pdf, "spec.pdf", True),
    ],
    ids=["txt", "docx", "pdf"],
)
def test_same_content_gives_same_segmentation_in_every_format(
    tmp_path: Path, writer: Writer, name: str, tables: bool
) -> None:
    reference = segment_document(load_document(write_markdown(tmp_path / "ref.md", SEG_SAMPLE)))
    other_document = load_document(writer(tmp_path / name, SEG_SAMPLE))
    other = segment_document(other_document)

    expected = _shape(reference)
    if not tables:
        expected = [
            s for s in expected if s[1] not in ("table_header",) and "Бронь на 14" not in str(s[3])
        ]
    assert _shape(other) == expected
    assert verify_segmentation(other, other_document) == ()
