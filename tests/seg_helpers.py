"""Помощники тестов сегментации (собственные синтетические документы)."""

from __future__ import annotations

from pathlib import Path

from validator.ingest import load_document
from validator.segmentation import Requirement, Segmentation, segment_document


def segment_markdown(tmp_path: Path, markdown: str, name: str = "spec.md") -> Segmentation:
    path = tmp_path / name
    path.write_text(markdown, encoding="utf-8")
    return segment_document(load_document(path))


def unit(segmentation: Segmentation, text_start: str) -> Requirement:
    """Фрагмент верхнего уровня (не утверждение внутри требования), начинающийся с текста."""
    return next(
        f for f in segmentation.fragments if "/" not in f.id and f.text.startswith(text_start)
    )


def child(segmentation: Segmentation, text_start: str) -> Requirement:
    """Утверждение внутри требования (ID вида «X/1»)."""
    return next(f for f in segmentation.fragments if "/" in f.id and f.text.startswith(text_start))
