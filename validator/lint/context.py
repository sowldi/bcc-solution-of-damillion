"""Сборка контекста проверки из документа и сегментации."""

from __future__ import annotations

from validator.lint.base import LintContext
from validator.lint.definitions import extract_definitions
from validator.model import Document
from validator.segmentation import Segmentation


def build_context(document: Document, segmentation: Segmentation) -> LintContext:
    fragments = segmentation.fragments
    containers = {f.parent_id for f in fragments if f.is_statement and f.parent_id}
    return LintContext(
        document=document,
        segmentation=segmentation,
        definitions=extract_definitions(document),
        targets=segmentation.atomic_requirements,
        top_level=tuple(f for f in fragments if not f.is_statement),
        leaves=tuple(f for f in fragments if f.id not in containers),
        order={f.id: index for index, f in enumerate(fragments)},
        doc_ids=frozenset(f.doc_id for f in fragments if f.doc_id),
        headings=tuple(block.text for block in document.blocks if block.kind == "heading"),
    )
