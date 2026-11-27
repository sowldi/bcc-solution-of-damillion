"""Стабильные идентификаторы и сопоставление версий документа."""

from __future__ import annotations

import re
from pathlib import Path

from tests.seg_helpers import segment_markdown
from validator.segmentation import Segmentation, match_segmentations

BASE = """# Сервис

## Функциональные требования

* Система хранит порог.
* Система отправляет push.

## Нефункциональные требования

Сервис отвечает в течение 2 секунд.
"""


def _ids(seg: Segmentation) -> dict[str, str]:
    return {f.normalized: f.id for f in seg.requirements}


def test_rerun_gives_identical_result(tmp_path: Path) -> None:
    first = segment_markdown(tmp_path, BASE, "a.md")
    second = segment_markdown(tmp_path, BASE, "b.md")  # имя файла на ID не влияет

    assert first == second
    assert first.to_json() == second.to_json()


def test_hash_ids_have_documented_format(tmp_path: Path) -> None:
    seg = segment_markdown(tmp_path, BASE)

    for fragment in seg.fragments:
        prefix = "R" if fragment.role in ("requirement", "ambiguous") else "F"
        assert re.fullmatch(rf"{prefix}-[0-9a-f]{{8}}", fragment.id), fragment


def test_inserting_requirement_keeps_other_ids(tmp_path: Path) -> None:
    before = _ids(segment_markdown(tmp_path, BASE, "a.md"))
    changed = BASE.replace(
        "* Система хранит порог.", "* Система удаляет порог.\n* Система хранит порог."
    )
    after = _ids(segment_markdown(tmp_path, changed, "b.md"))

    assert {k: after[k] for k in before} == before
    assert len(after) == len(before) + 1


def test_moving_requirement_to_other_section_keeps_id(tmp_path: Path) -> None:
    before = _ids(segment_markdown(tmp_path, BASE, "a.md"))
    moved = BASE.replace("\nСервис отвечает в течение 2 секунд.\n", "\n").replace(
        "* Система отправляет push.",
        "* Система отправляет push.\n\nСервис отвечает в течение 2 секунд.",
    )
    after = _ids(segment_markdown(tmp_path, moved, "b.md"))

    assert (
        after["Сервис отвечает в течение 2 секунд."]
        == before["Сервис отвечает в течение 2 секунд."]
    )


def test_text_change_changes_hash_id_but_not_explicit_id(tmp_path: Path) -> None:
    plain_before = _ids(segment_markdown(tmp_path, BASE, "a.md"))
    plain_after = _ids(
        segment_markdown(tmp_path, BASE.replace("хранит порог.", "хранит порог в тенге."), "b.md")
    )
    explicit = "# Сервис\n\n## Функциональные требования\n\n* R-1. Система хранит порог.\n"
    explicit_before = segment_markdown(tmp_path, explicit, "c.md")
    explicit_after = segment_markdown(
        tmp_path, explicit.replace("хранит порог.", "хранит порог в тенге."), "d.md"
    )

    assert plain_before["Система хранит порог."] != plain_after["Система хранит порог в тенге."]
    assert [f.id for f in explicit_before.requirements] == ["R-1"]
    assert [f.id for f in explicit_after.requirements] == ["R-1"]


def test_match_versions(tmp_path: Path) -> None:
    old = segment_markdown(tmp_path, BASE, "old.md")
    new_text = (
        BASE.replace("* Система хранит порог.", "* Система хранит порог в тенге.")
        .replace(
            "* Система отправляет push.", "* Система отправляет push.\n* Система удаляет порог."
        )
        .replace("\nСервис отвечает в течение 2 секунд.\n", "\n")
        + "\n## Ограничения\n\nСервис отвечает в течение 2 секунд.\n"
    )
    new = segment_markdown(tmp_path, new_text, "new.md")

    matches = {(m.old_text, m.new_text): m for m in match_segmentations(old, new)}

    modified = matches[("Система хранит порог.", "Система хранит порог в тенге.")]
    unchanged = matches[("Система отправляет push.", "Система отправляет push.")]
    moved = matches[("Сервис отвечает в течение 2 секунд.", "Сервис отвечает в течение 2 секунд.")]
    added = matches[(None, "Система удаляет порог.")]
    assert modified.status == "modified"
    assert 0.75 <= modified.similarity < 1
    assert unchanged.status == "unchanged"
    assert (moved.status, moved.section_changed) == ("unchanged", True)
    assert moved.old_id == moved.new_id
    assert added.status == "added"


def test_match_reports_removed(tmp_path: Path) -> None:
    old = segment_markdown(tmp_path, BASE, "old.md")
    new = segment_markdown(tmp_path, BASE.replace("* Система отправляет push.\n", ""), "new.md")

    removed = [m for m in match_segmentations(old, new) if m.status == "removed"]

    assert [(m.old_text, m.new_id) for m in removed] == [("Система отправляет push.", None)]


def test_same_content_in_other_format_matches_fully(tmp_path: Path) -> None:
    from tests.builders import SAMPLE, write_docx, write_markdown
    from validator.ingest import load_document
    from validator.segmentation import segment_document

    md = segment_document(load_document(write_markdown(tmp_path / "a.md", SAMPLE)))
    docx = segment_document(load_document(write_docx(tmp_path / "a.docx", SAMPLE)))

    assert {m.status for m in match_segmentations(md, docx)} == {"unchanged"}
