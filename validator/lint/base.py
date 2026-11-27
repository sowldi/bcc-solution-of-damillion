"""Основа детерминированных правил: правило каталога, совпадение и контекст проверки.

Правило — чистая функция контекста: получает документ, сегментацию и определения, возвращает
совпадения (Hit) с точными отрезками текста фрагментов. Замечание (Finding) из совпадения
собирает движок; доказательства проверяет Evidence Gate. Правила не зависят друг от друга,
от CLI, отчёта и LLM.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

from validator.findings import AbsenceClaim, EvidenceKind, RuleCategory, Severity
from validator.lint.definitions import Definitions
from validator.model import Block, BlockKind, Document
from validator.segmentation import Requirement, Segmentation
from validator.segmentation.text import parse_prefix


@dataclass(frozen=True)
class Span:
    """Отрезок fragment.text[start:end] — будущая цитата. (0, 0) у пустого фрагмента —
    доказательство отсутствия содержания."""

    fragment: Requirement
    start: int
    end: int


@dataclass(frozen=True)
class Hit:
    """Срабатывание правила: участники (первый — основной) и конкретное объяснение.

    absence — ограниченная область, где правило утверждает отсутствие (для правил с видом
    доказательства «absence»); Evidence Gate перепроверяет её по документу.
    """

    spans: tuple[Span, ...]
    explanation: str
    absence: AbsenceClaim | None = None


@dataclass(frozen=True)
class LintContext:
    document: Document
    segmentation: Segmentation
    definitions: Definitions
    targets: tuple[Requirement, ...]  # атомарные требования — объект правил формулировок
    top_level: tuple[Requirement, ...]  # фрагменты без утверждений «X/k»
    leaves: tuple[Requirement, ...]  # фрагменты, не разделённые на утверждения, и утверждения
    order: Mapping[str, int]  # ID фрагмента → позиция в сегментации
    doc_ids: frozenset[str]  # ID документа у фрагментов (свои и унаследованные)
    headings: tuple[str, ...]  # тексты заголовков (для разрешения ссылок)

    @cached_property
    def _owners(self) -> dict[int, Requirement]:
        owners: dict[int, Requirement] = {}
        for fragment in self.top_level:
            for ref in fragment.sources:
                owners.setdefault(ref.block, fragment)
        return owners

    def owner(self, block_index: int) -> Requirement | None:
        """Фрагмент верхнего уровня, которому принадлежит блок документа."""
        return self._owners.get(block_index)

    def block(self, fragment: Requirement) -> Block | None:
        if not fragment.sources:
            return None
        return self.document.blocks[fragment.sources[0].block]

    def kind(self, fragment: Requirement) -> BlockKind | None:
        block = self.block(fragment)
        return block.kind if block else None

    def body_start(self, fragment: Requirement) -> int:
        """Начало содержательного текста: после «Ф-1. » или «3. » у абзаца и пункта списка.

        У утверждения «X/k» префикса нет — оно начинается после ID фрагмента.
        """
        kind = self.kind(fragment)
        if fragment.is_statement or kind not in ("paragraph", "list_item"):
            return 0
        return parse_prefix(fragment.text, kind).body_start


Check = Callable[[LintContext], Iterable[Hit]]
Granularity = Literal["phrase", "fragment"]


@dataclass(frozen=True)
class Rule:
    """Правило каталога. rule_id и version стабильны: замечание воспроизводимо по ним."""

    rule_id: str
    version: str
    category: RuleCategory
    severity: Severity
    criterion: str  # код из validator.criteria
    evidence_kind: EvidenceKind
    title: str  # короткое название замечания
    description: str  # что проверяет правило
    trigger: str  # когда срабатывает (и когда нет)
    rationale: str  # почему это дефект
    recommendation: str  # как исправить
    check: Check
    # phrase — дефект конкретной фразы; fragment — замечание ко всему требованию или
    # к структуре. Пересекающиеся цитаты схлопываются только у замечаний к фразам.
    granularity: Granularity = "phrase"
