"""Доказательная привязка к источнику и сериализация в JSON."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import validator.segmentation.segmenter as segmenter
from tests.seg_helpers import unit
from validator.ingest import load_document
from validator.model import Document
from validator.segmentation import (
    SCHEMA,
    Segmentation,
    SegmentationError,
    Statement,
    segment_document,
    verify_segmentation,
)

SPEC = """# Сервис

## Функциональные требования

* Ф-1. Читатель задаёт срок брони. По умолчанию бронь не продлевается.
* Система отправляет push.

| Поле | Значение |
| --- | --- |
| Срок | 14 дней |
"""


@pytest.fixture
def pair(tmp_path: Path) -> tuple[Segmentation, Document]:
    path = tmp_path / "spec.md"
    path.write_text(SPEC, encoding="utf-8")
    document = load_document(path)
    return segment_document(document), document


def _replace_fragment(seg: Segmentation, fragment_id: str, **changes: object) -> Segmentation:
    fragments = tuple(
        replace(f, **changes) if f.id == fragment_id else f  # type: ignore[arg-type]
        for f in seg.fragments
    )
    return replace(seg, fragments=fragments)


def test_json_round_trip(pair: tuple[Segmentation, Document]) -> None:
    seg, _ = pair

    text = seg.to_json()
    restored = Segmentation.from_json(text)

    assert restored == seg
    data = json.loads(text)
    assert data["schema"] == SCHEMA
    assert "Читатель" in text  # кириллица без \\u-экранирования


def test_from_json_rejects_unknown_schema_and_values(pair: tuple[Segmentation, Document]) -> None:
    seg, _ = pair
    data = json.loads(seg.to_json())

    with pytest.raises(ValueError, match="схем"):
        Segmentation.from_json(json.dumps({**data, "schema": "other/9"}))
    data["fragments"][0]["role"] = "wish"
    with pytest.raises(ValueError, match="role"):
        Segmentation.from_json(json.dumps(data))


def test_verification_detects_wrong_offsets(pair: tuple[Segmentation, Document]) -> None:
    seg, document = pair
    target = unit(seg, "Система отправляет")
    broken_ref = replace(target.sources[0], end=target.sources[0].end + 500)

    problems = verify_segmentation(
        _replace_fragment(seg, target.id, sources=(broken_ref,)), document
    )

    assert any(target.id in p for p in problems)


def test_verification_detects_wrong_block(pair: tuple[Segmentation, Document]) -> None:
    seg, document = pair
    target = unit(seg, "Система отправляет")
    broken_ref = replace(target.sources[0], block=999)

    problems = verify_segmentation(
        _replace_fragment(seg, target.id, sources=(broken_ref,)), document
    )

    assert any("999" in p for p in problems)


def test_verification_detects_changed_quote(pair: tuple[Segmentation, Document]) -> None:
    seg, document = pair
    target = unit(seg, "Система отправляет")

    problems = verify_segmentation(
        _replace_fragment(seg, target.id, text="Система отправляет SMS."), document
    )

    assert any("цитат" in p for p in problems)


def test_verification_detects_wrong_location(pair: tuple[Segmentation, Document]) -> None:
    seg, document = pair
    target = unit(seg, "Система отправляет")
    ref = target.sources[0]
    broken_ref = replace(ref, location=replace(ref.location, line_start=1, line_end=1))

    problems = verify_segmentation(
        _replace_fragment(seg, target.id, sources=(broken_ref,)), document
    )

    assert any("мест" in p for p in problems)


def test_verification_detects_child_outside_parent(pair: tuple[Segmentation, Document]) -> None:
    seg, document = pair
    child = seg.by_id("Ф-1/1")
    foreign = unit(seg, "Система отправляет").sources[0]

    problems = verify_segmentation(
        _replace_fragment(
            seg,
            "Ф-1/1",
            sources=(foreign,),
            text=document.blocks[foreign.block].text[foreign.start : foreign.end],
        ),
        document,
    )

    assert child.parent_id == "Ф-1"
    assert any("Ф-1/1" in p for p in problems)


def test_verification_survives_corrupted_parent_ref(pair: tuple[Segmentation, Document]) -> None:
    """Повреждённые данные (например, из JSON) дают список нарушений, а не исключение."""
    seg, document = pair
    parent = seg.by_id("Ф-1")
    broken_ref = replace(parent.sources[0], end=parent.sources[0].end + 1000)

    problems = verify_segmentation(_replace_fragment(seg, "Ф-1", sources=(broken_ref,)), document)

    assert any("Ф-1" in p for p in problems)


def test_verification_detects_lost_statement_text(pair: tuple[Segmentation, Document]) -> None:
    seg, document = pair
    statement = seg.by_id("Ф-1/2")
    ref = statement.sources[0]
    shorter = replace(ref, end=ref.end - 5)
    block_text = document.blocks[ref.block].text

    problems = verify_segmentation(
        _replace_fragment(
            seg, "Ф-1/2", sources=(shorter,), text=block_text[shorter.start : shorter.end]
        ),
        document,
    )

    assert any("не покрывают" in p for p in problems)


def test_verification_detects_other_document(pair: tuple[Segmentation, Document]) -> None:
    seg, document = pair

    problems = verify_segmentation(replace(seg, document_sha256="f" * 64), document)

    assert any("другого документа" in p for p in problems)


def test_internal_inconsistency_is_not_silently_accepted(
    pair: tuple[Segmentation, Document], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, document = pair

    def broken_split(text: str, start: int = 0) -> tuple[Statement, ...]:
        return (Statement(start, len(text) + 10, "requirement"), Statement(0, 1, "requirement"))

    monkeypatch.setattr(segmenter, "split_statements", broken_split)

    with pytest.raises(SegmentationError):
        segment_document(document)
