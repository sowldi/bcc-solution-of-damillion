"""Stage 10: сама оценка — разметка, сопоставление и подсчёт метрик.

Проверяется методика, а не числа корпуса: они записываются в EVALUATION.md скриптом
`python -m evaluation.run_evaluation`, и тесты не подгоняются под них.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.metrics import (
    Label,
    LabelError,
    check_anchors,
    defect_findings,
    load_labels,
    match,
    resolve,
)
from evaluation.run_evaluation import corpus_documents, markdown_elements, reversed_lists
from tests.builders import LI, H, T
from tests.lint_helpers import HEAD, lint_markdown

DOC = HEAD + (
    "* Ф-1. Система должна быстро показывать каталог.\n"
    "* Ф-2. Бронь хранится около 3 дней.\n"
    "* Ф-3. Курьер доставляет посылку адресату.\n"
    "* Ф-4. Срок выдачи: TBD.\n"
)


def _write(tmp_path: Path, data: dict[str, object]) -> Path:
    path = tmp_path / "doc.labels.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_counts_true_false_positives_and_misses(tmp_path: Path) -> None:
    linted = lint_markdown(tmp_path, DOC)
    labels = (
        Label("Ф-1", "TST-01"),  # найдено
        Label("Ф-3", "AMB-03"),  # такого дефекта правила не находят — FN
        Label("Ф-4", "CMP-01"),  # найдено
    )

    result = match(labels, defect_findings(linted.result.findings), linted.segmentation)

    assert result.counts("TST-01").tp == 1
    assert result.counts("CMP-01").tp == 1
    assert (result.counts("AMB-03").tp, result.counts("AMB-03").fn) == (0, 1)
    assert [f.rule_id for f in result.false_positives] == ["AMB-02"]  # не размечено — FP
    assert result.counts("AMB-02").precision == 0.0
    assert result.counts("TST-01").recall == 1.0


def test_alternative_rules_on_one_requirement_are_both_correct(tmp_path: Path) -> None:
    linted = lint_markdown(
        tmp_path,
        HEAD.replace("Функциональные", "Нефункциональные")
        + "* Н-1. Время отклика минимизируется.\n",
    )
    findings = defect_findings(linted.result.findings)
    label = Label("Н-1", "TST-04", alt=("TST-05",))

    result = match((label,), findings, linted.segmentation)

    assert {f.rule_id for f in findings} <= {"TST-04", "TST-05"}
    assert result.false_positives == [] and result.false_negatives == []


def test_missing_anchor_is_a_labelling_error(tmp_path: Path) -> None:
    linted = lint_markdown(tmp_path, DOC)

    with pytest.raises(LabelError, match="Ф-9"):
        resolve("Ф-9", linted.segmentation)
    with pytest.raises(LabelError):
        resolve("quote:такого текста нет", linted.segmentation)
    assert resolve("quote:около 3 дней", linted.segmentation) == frozenset({"Ф-2"})


@pytest.mark.parametrize(
    "data",
    [
        {"document": "d.md", "split": "train"},
        {"document": "d.md", "split": "dev", "extra": 1},
        {"document": "d.md", "split": "dev", "expected": [{"anchor": "Ф-1"}]},
        {
            "document": "d.md",
            "split": "dev",
            "contradictions": [{"anchor": "Ф-1", "status": "maybe"}],
        },
    ],
)
def test_malformed_labels_are_rejected(tmp_path: Path, data: dict[str, object]) -> None:
    with pytest.raises(LabelError):
        load_labels(_write(tmp_path, data))


def test_every_corpus_anchor_exists() -> None:
    from validator.ingest import load_document
    from validator.segmentation import segment_document

    documents = corpus_documents()

    assert {labels.split for _, labels in documents} == {"dev", "test"}
    for path, labels in documents:
        check_anchors(labels, segment_document(load_document(path)))


def test_markdown_elements_for_format_generation() -> None:
    text = "# Сервис\n\n## Раздел\n\n* Ф-1. Пункт.\n\n| А | Б |\n| --- | --- |\n| 1 | 2 |\n"

    elements = markdown_elements(text)

    assert elements == [
        H(1, "Сервис"),
        H(2, "Раздел"),
        LI("Ф-1. Пункт."),
        T((("А", "Б"), ("1", "2"))),
    ]


def test_reversed_lists_keeps_everything_but_list_order() -> None:
    text = "# С\n\n* Ф-1. А.\n* Ф-2. Б.\n\nАбзац.\n"

    assert reversed_lists(text) == "# С\n\n* Ф-2. Б.\n* Ф-1. А.\n\nАбзац.\n"
