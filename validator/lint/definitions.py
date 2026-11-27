"""Термины и сокращения, определённые в самом документе.

Правила не называют слово неопределённым или субъективным, если документ его определяет:
в глоссарии (раздел «Термины», «Сокращения», «Определения» или таблица с шапкой «Термин» /
«Сокращение» / «Определение»), расшифровкой в скобках («Система быстрых платежей (СБП)»,
«СБП (система быстрых платежей)»), оборотом «(далее — Сервис)» или «Под X понимается…».

Совпадение учитывает словоформы: слова сравниваются по основе (начало слова без окончания),
поэтому определённый «быстрый поиск» покрывает и «быстрого поиска».
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

from validator.model import Block, Document
from validator.segmentation.text import normalize_text

# Раздел-глоссарий определяется по ближайшему заголовку (а не по заголовку документа).
_GLOSSARY_SECTION = re.compile(
    r"глоссари|терминолог|\bтермин(?:ы|ов|ам)?\b|\bопределени[яй]\b|\bсокращени[яй]\b"
    r"|аббревиатур|обозначени[яй]\b|glossary|definitions|abbreviations|acronyms|\bterms\b",
    re.IGNORECASE,
)
_GLOSSARY_HEADER = re.compile(
    r"\bтермин\b|сокращени|аббревиатур|определени|расшифровк|обозначени|понятие"
    r"|\bterm\b|abbreviation|acronym|definition|meaning",
    re.IGNORECASE,
)
_ABBR = r"[A-ZА-ЯЁ][A-ZА-ЯЁ0-9]{1,9}"
# «Полное название (АБВ)», «(далее — АБВ)», «(далее «АБВ»)»
_ABBR_IN_PARENS = re.compile(
    rf"\(\s*(?:(?i:далее)\s*(?:[—–-]\s*)?)?[«\"]?(?P<abbr>{_ABBR})[»\"]?\s*\)"
)
# «АБВ (полное название)» — расшифровка из нескольких слов сразу после сокращения
_ABBR_EXPANDED = re.compile(rf"(?<![\w-])(?P<abbr>{_ABBR})\s*\((?P<expansion>[^()]{{3,150}})\)")
# «АБВ — расшифровка» / «АБВ: расшифровка» в начале блока
_ABBR_DASH = re.compile(rf"^\s*(?P<abbr>{_ABBR})\s*(?:—|–|-|:)\s+\S")
# «Термин — определение» / «Термин: определение» в начале блока глоссария
_TERM_DASH = re.compile(r"^\s*[«\"]?(?P<term>[^—–:«»\"\n]{1,80}?)[»\"]?\s*(?:—|–|-|:)\s+\S")
_DEFINED_AS = re.compile(
    r"\bпод\s+(?:термином\s+)?[«\"]?(?P<term>[^«»\"\n]{2,60}?)[»\"]?\s+"
    r"(?:понимается|понимаются|подразумевается|подразумеваются)\b",
    re.IGNORECASE,
)
_IS_DEFINITION = re.compile(r"^\s*[«\"]?(?P<term>[^—–«»\"\n]{2,60}?)[»\"]?\s+[—–]\s+это\b")
_HEREINAFTER = re.compile(
    r"\(\s*далее\s*(?:по\s+тексту\s*)?(?:[—–-]\s*)?[«\"]?(?P<term>[^»\")]{2,60}?)[»\"]?\s*\)",
    re.IGNORECASE,
)
_WORD = re.compile(r"[\w-]+")
_TRIM = " \t«»\"'.,;:()"
_MIN_STEM = 4


def _fold(word: str) -> str:
    return word.casefold().replace("ё", "е")


def same_word(a: str, b: str) -> bool:
    """Две словоформы одного слова: общее начало не короче четырёх букв и отличается
    от более короткой формы не больше чем на окончание из двух букв."""
    a, b = _fold(a), _fold(b)
    common = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        common += 1
    return common >= _MIN_STEM and common >= min(len(a), len(b)) - 2


def _phrase_pattern(term: str) -> re.Pattern[str]:
    parts = []
    for word in term.split():
        folded = _fold(word)
        base = folded[: max(_MIN_STEM, len(folded) - 2)]
        parts.append(re.escape(base).replace("е", "[её]") + r"\w*")
    return re.compile(r"(?<!\w)" + r"\s+".join(parts) + r"(?!\w)", re.IGNORECASE)


@dataclass(frozen=True)
class Definitions:
    terms: frozenset[str] = frozenset()  # нормализованные, в нижнем регистре
    abbreviations: frozenset[str] = frozenset()

    def defines_abbreviation(self, abbreviation: str) -> bool:
        return abbreviation in self.abbreviations

    @cached_property
    def _single_words(self) -> tuple[str, ...]:
        return tuple(sorted(term for term in self.terms if " " not in term))

    @cached_property
    def _phrases(self) -> tuple[re.Pattern[str], ...]:
        return tuple(_phrase_pattern(term) for term in sorted(self.terms) if " " in term)

    def covers(self, text: str, start: int, end: int) -> bool:
        """text[start:end] входит в определённый документом термин (с учётом словоформ)."""
        words = _WORD.findall(text[start:end])
        if len(words) == 1 and any(same_word(words[0], term) for term in self._single_words):
            return True
        for pattern in self._phrases:
            for match in pattern.finditer(text):
                if match.start() < end and start < match.end():
                    return True
        return False


def extract_definitions(document: Document) -> Definitions:
    terms: set[str] = set()
    abbreviations: set[str] = set()

    def add_term(raw: str) -> None:
        term = normalize_text(raw).strip(_TRIM).casefold()
        if term and len(term.split()) <= 8:
            terms.add(term)

    def add_entry(raw: str) -> None:
        entry = raw.strip(_TRIM)
        if re.fullmatch(_ABBR, entry):
            abbreviations.add(entry)
        else:
            add_term(entry)

    for block in document.blocks:
        _glossary_entry(block, add_entry)
        text = block.text
        for match in _ABBR_IN_PARENS.finditer(text):
            abbreviation = match.group("abbr")
            abbreviations.add(abbreviation)
            expansion = _expansion_before(text, match.start(), abbreviation)
            if expansion:
                add_term(expansion)
        for match in _ABBR_EXPANDED.finditer(text):
            expansion = match.group("expansion")
            if len(expansion.split()) >= 2 and not expansion.isupper():
                abbreviations.add(match.group("abbr"))
                add_term(expansion)
        if dash := _ABBR_DASH.match(text):
            abbreviations.add(dash.group("abbr"))
        for pattern in (_DEFINED_AS, _IS_DEFINITION, _HEREINAFTER):
            for match in pattern.finditer(text):
                add_term(match.group("term"))
    return Definitions(frozenset(terms), frozenset(abbreviations))


def _glossary_entry(block: Block, add_entry: Callable[[str], None]) -> None:
    """Термин из строки глоссария: первая ячейка строки таблицы или текст до «—» / «:»."""
    path = block.location.section_path
    in_section = bool(path) and bool(_GLOSSARY_SECTION.search(path[-1]))
    if block.kind == "table_row":
        in_table = any(_GLOSSARY_HEADER.search(cell) for cell in block.header)
        cells = block.cells
        defined = bool(cells) and bool(cells[0].strip()) and any(c.strip() for c in cells[1:])
        if (in_section or in_table) and defined:
            add_entry(cells[0])
    elif in_section and block.kind in ("paragraph", "list_item"):
        match = _TERM_DASH.match(block.text)
        if match:
            add_entry(match.group("term"))


def _expansion_before(text: str, position: int, abbreviation: str) -> str | None:
    """Расшифровка перед «(АБВ)»: последние слова, первые буквы которых дают сокращение."""
    letters = [char for char in abbreviation if char.isalpha()]
    words = list(_WORD.finditer(text[:position]))
    if not letters or len(words) < len(letters):
        return None
    candidate = words[-len(letters) :]
    initials = [_fold(word.group(0))[0] for word in candidate]
    if initials != [_fold(letter) for letter in letters]:
        return None
    return text[candidate[0].start() : candidate[-1].end()]
