"""Помощники тестов детерминированных правил (собственные документы, домен «библиотека»)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from validator.findings import Finding
from validator.ingest import load_document
from validator.lint import CATALOG, LintResult, Rule, run_lint
from validator.model import Document
from validator.segmentation import Segmentation, segment_document

HEAD = "# Сервис бронирования книг\n\n## Функциональные требования\n\n"


@dataclass(frozen=True)
class Linted:
    document: Document
    segmentation: Segmentation
    result: LintResult

    def of(self, rule_id: str) -> list[Finding]:
        return [f for f in self.result.findings if f.rule_id == rule_id]

    @property
    def rule_ids(self) -> list[str]:
        return [f.rule_id for f in self.result.findings]


def lint_markdown(
    tmp_path: Path, markdown: str, *, rules: Sequence[Rule] = CATALOG, name: str = "spec.md"
) -> Linted:
    path = tmp_path / name
    path.write_text(markdown, encoding="utf-8")
    document = load_document(path)
    segmentation = segment_document(document)
    return Linted(document, segmentation, run_lint(document, segmentation, rules=rules))


def items(*texts: str) -> str:
    """Раздел функциональных требований из пунктов списка."""
    return HEAD + "".join(f"* {text}\n" for text in texts)


def quotes(finding: Finding) -> list[str]:
    return [evidence.quote for evidence in finding.evidence]
