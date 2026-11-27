"""Текстовые правила сегментации: нормализация, префиксы ID, деление на утверждения, ссылки.

Все правила детерминированы и покрыты тестами (tests/test_segmentation_text.py).
Исходный текст никогда не изменяется: функции возвращают новые строки или смещения в исходной.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from validator.ingest.common import one_line

StatementRole = Literal["requirement", "example", "explanation"]

# --- нормализация --------------------------------------------------------------------

_QUOTES = str.maketrans(
    {
        "«": '"',
        "»": '"',
        "„": '"',
        "“": '"',
        "”": '"',
        "‟": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
    }
)
_DASHES = str.maketrans(dict.fromkeys("‐‑‒–—―−", "-"))


def normalize_text(text: str) -> str:
    """Текст для анализа: NFC, единые кавычки, любые пробельные символы (в т.ч. переводы строк
    и неразрывные пробелы) схлопнуты в один пробел. Регистр и тире сохраняются."""
    return " ".join(unicodedata.normalize("NFC", text).translate(_QUOTES).split())


def content_key(text: str) -> str:
    """Ключ содержания для ID и поиска дублей: регистр, вид тире и концевая пунктуация не важны."""
    return normalize_text(text).translate(_DASHES).casefold().rstrip(" .;:,!?")


# --- префиксы: идентификатор требования и номер пункта ---------------------------------

# Идентификатор: 1–6 заглавных букв (латиница/кириллица), необязательный разделитель, номер с
# подуровнями: «Ф-1», «FR-2.1», «REQ_007», «R1». Не идентификаторы — стандарты и протоколы.
ID_PATTERN = r"[A-ZА-ЯЁ]{1,6}[-_.]?\d{1,4}(?:\.\d{1,3}){0,3}"
_LEADING_ID = re.compile(rf"\s*\[?(?P<id>{ID_PATTERN})\]?(?:[.):]|\s+[—–-])?\s+(?=\S)")
_ORDINAL = re.compile(r"\s*(?P<num>\d{1,2})[.)]\s+(?=\S)")
_ID_LETTERS = re.compile(r"^[A-ZА-ЯЁ]+")
_NOT_ID_PREFIXES = frozenset(
    {
        "HTTP",
        "HTTPS",
        "ISO",
        "IEC",
        "IEEE",
        "RFC",
        "UTF",
        "ГОСТ",
        "GOST",
        "SHA",
        "MD",
        "TLS",
        "SSL",
        "PCI",
        "DSS",
        "ISBN",
        "OAUTH",
        "KZT",
        "USD",
        "EUR",
        "RUB",
        "TCP",
        "UDP",
        "IP",
        "IPV",
    }
)


@dataclass(frozen=True)
class Prefix:
    """Служебное начало фрагмента: «Ф-1. » (идентификатор) или «3. » (номер пункта списка)."""

    doc_id: str | None
    number: str | None
    body_start: int  # смещение начала содержательного текста


def is_document_id(candidate: str) -> bool:
    letters = _ID_LETTERS.match(candidate)
    return bool(re.fullmatch(ID_PATTERN, candidate)) and bool(
        letters and letters.group(0) not in _NOT_ID_PREFIXES
    )


def parse_prefix(text: str, kind: str) -> Prefix:
    """Найти идентификатор требования в начале текста; номер пункта — только у пунктов списка."""
    match = _LEADING_ID.match(text)
    if match and is_document_id(match.group("id")):
        return Prefix(match.group("id"), None, match.end())
    if kind == "list_item":
        ordinal = _ORDINAL.match(text)
        if ordinal:
            return Prefix(None, ordinal.group("num"), ordinal.end())
    return Prefix(None, None, 0)


# --- деление на утверждения ------------------------------------------------------------

_TERMINATORS = ".!?…"
_TRAILERS = '.!?…»”’")]'  # что может идти сразу за концом предложения
_OPENING = '«„“"(['
_DEPTH_OPEN = "«(["
_DEPTH_CLOSE = "»)]"
_ABBREVIATIONS = frozenset(
    {
        "т.е",
        "т.к",
        "т.д",
        "т.п",
        "т.ч",
        "т.н",
        "др",
        "пр",
        "см",
        "ср",
        "п",
        "пп",
        "стр",
        "рис",
        "табл",
        "гг",
        "вв",
        "руб",
        "коп",
        "тыс",
        "млн",
        "млрд",
        "мин",
        "сек",
        "напр",
        "прим",
        "ул",
        "кв",
        "корп",
        "св",
        "ст",
        "гл",
        "разд",
        "ред",
        "изд",
        "вып",
        "мес",
        "нед",
        "ед",
        "e.g",
        "i.e",
        "etc",
        "vs",
        "no",
        "fig",
        "ref",
        "approx",
    }
)
# Начало предложения, которое продолжает мысль предыдущего (не самостоятельное требование).
_CONNECTIVES = (
    "иначе",
    "тогда",
    "однако",
    "но",
    "затем",
    "далее",
    "при этом",
    "в этом случае",
    "в таком случае",
    "в противном случае",
    "после этого",
    "причём",
    "причем",
    "соответственно",
    "аналогично",
    "в том числе",
    "otherwise",
    "then",
    "however",
)
# Местоимения и указательные слова: предложение ссылается на предыдущее («она», «такой push»).
_ANAPHORA = frozenset(
    {
        "он",
        "она",
        "оно",
        "они",
        "его",
        "её",
        "ее",
        "их",
        "ему",
        "ей",
        "им",
        "него",
        "неё",
        "нее",
        "них",
        "нему",
        "ней",
        "ним",
        "нём",
        "нем",
        "такой",
        "такая",
        "такое",
        "такие",
        "такого",
        "такую",
        "таким",
        "таких",
        "такому",
        "такими",
        "этот",
        "эта",
        "это",
        "эти",
        "этого",
        "этой",
        "этих",
        "этим",
        "этому",
        "эту",
        "этими",
        "этом",
    }
)
EXAMPLE_MARKERS = ("например", "пример:", "пример.", "пример ", "к примеру", "for example", "e.g.")
EXPLANATION_MARKERS = (
    "примечание",
    "прим.",
    "пояснение",
    "комментарий",
    "note:",
    "note.",
    "важно:",
    "это означает",
    "это значит",
    "т.е.",
    "то есть",
    "иными словами",
    "другими словами",
)
_WORDS = re.compile(r"\w+")


@dataclass(frozen=True)
class Statement:
    """Утверждение внутри фрагмента: смещения в исходном тексте и роль."""

    start: int
    end: int
    role: StatementRole


def statement_role(text: str) -> StatementRole:
    lowered = text.casefold().lstrip()
    if lowered.startswith(EXAMPLE_MARKERS):
        return "example"
    if lowered.startswith(EXPLANATION_MARKERS):
        return "explanation"
    return "requirement"


def split_statements(text: str, start: int = 0) -> tuple[Statement, ...]:
    """Разделить текст (начиная со start) на утверждения.

    Граница — конец предложения (. ! ? …) перед заглавной буквой, цифрой или кавычкой, либо «;».
    Не граница: сокращения («т.е.», «г.»), инициалы, числа и версии («1.5», «1.2.3»), текст
    в скобках и «ёлочках». Утверждение без букв («200.», «404») присоединяется к соседнему.
    Предложение, которое продолжает предыдущее (связка в начале, местоимение, строчная буква
    после точки), с ним объединяется. «Например…» и «Примечание…» — отдельные утверждения
    с ролью example/explanation. Перечисления через запятую и союз «и» не делятся.
    """
    segments = _merge_dependent(text, _merge_letterless(text, _raw_segments(text, start)))
    return tuple(Statement(s, e, statement_role(text[s:e])) for s, e, _ in segments)


Segment = tuple[int, int, str]  # (начало, конец, знак, которым закончился сегмент)


def _raw_segments(text: str, start: int) -> list[Segment]:
    segments: list[Segment] = []
    seg_start, depth, index, length = start, 0, start, len(text)
    while index < length:
        char = text[index]
        if char in _DEPTH_OPEN:
            depth += 1
        elif char in _DEPTH_CLOSE and depth > 0:
            depth -= 1
        elif depth == 0 and (char in _TERMINATORS or char == ";"):
            end = index + 1
            while end < length and text[end] in _TRAILERS:
                end += 1
            nxt = end
            while nxt < length and text[nxt].isspace():
                nxt += 1
            if nxt > end and nxt < length and _is_boundary(text, index, nxt):
                segments.append((seg_start, end, char))
                seg_start = index = nxt
                continue
            index = end
            continue
        index += 1
    segments.append((seg_start, length, ""))
    return [_strip(text, s, e, t) for s, e, t in segments if text[s:e].strip()]


def _is_boundary(text: str, terminator: int, next_char: int) -> bool:
    if text[terminator] == ";":
        return True
    nxt = text[next_char]
    if not (nxt.isupper() or nxt.isdigit() or nxt in _OPENING):
        return False
    if text[terminator] != ".":
        return True
    word = _word_before(text, terminator)
    return not (word.casefold() in _ABBREVIATIONS or (len(word) == 1 and word.isalpha()))


def _word_before(text: str, position: int) -> str:
    """Слово (с точками внутри, как «т.е») непосредственно перед позицией."""
    begin = position
    while begin > 0 and (text[begin - 1].isalpha() or text[begin - 1] == "."):
        begin -= 1
    return text[begin:position].strip(".")


def _strip(text: str, start: int, end: int, terminator: str) -> Segment:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end, terminator


def _has_letters(text: str) -> bool:
    return any(char.isalpha() for char in text)


def _merge_letterless(text: str, segments: list[Segment]) -> list[Segment]:
    """«200.» и «404» — не утверждения: присоединить к следующему (последний — к предыдущему)."""
    merged: list[Segment] = []
    pending: Segment | None = None
    for start, end, terminator in segments:
        if not _has_letters(text[start:end]):
            pending = (pending[0] if pending else start, end, terminator)
            continue
        if pending is not None:
            start, pending = pending[0], None
        merged.append((start, end, terminator))
    if pending is not None:
        if merged:
            merged[-1] = (merged[-1][0], pending[1], pending[2])
        else:
            merged.append(pending)
    return merged


def _depends_on_previous(segment: str, previous_terminator: str) -> bool:
    if previous_terminator != ";" and segment[:1].islower():
        return True
    lowered = segment.casefold()
    for connective in _CONNECTIVES:
        if (
            lowered.startswith(connective)
            and not lowered[len(connective) : len(connective) + 1].isalnum()
        ):
            return True
    return bool(set(_WORDS.findall(lowered)) & _ANAPHORA)


def _merge_dependent(text: str, segments: list[Segment]) -> list[Segment]:
    if not segments:
        return []
    merged = [segments[0]]
    for start, end, terminator in segments[1:]:
        segment = text[start:end]
        previous = merged[-1]
        if statement_role(segment) == "requirement" and _depends_on_previous(segment, previous[2]):
            merged[-1] = (previous[0], end, terminator)
        else:
            merged.append((start, end, terminator))
    return merged


# --- ссылки на другие требования и разделы ---------------------------------------------

_ID_ANYWHERE = re.compile(rf"(?<![\w-])(?P<id>{ID_PATTERN})(?![\w-])")
_REFERENCE_CUE = re.compile(
    r"(?:см\.|смотри|согласно|в соответствии с|аналогично|как в|по)\s*$", re.IGNORECASE
)
_SECTION_REFERENCE = re.compile(
    r"(?<!\w)(?P<word>(?:под)?раздел\w*|пункт\w*|п\.|глав[аеуы]|section|clause)"
    r"\s*(?P<num>\d+(?:\.\d+)*)",
    re.IGNORECASE,
)
_QUOTED = re.compile(r"«([^«»]{2,150})»")
_HEADING_NUMBER = re.compile(r"\s*(\d+(?:\.\d+)*)\.?\s")


def find_references(
    text: str, *, doc_ids: Iterable[str], headings: Iterable[str], own_id: str | None
) -> tuple[str, ...]:
    """Ссылки в тексте в порядке появления.

    «req:Ф-2» — на требование документа; «section:<заголовок>» — на раздел (по номеру
    «раздел 4.2» или по названию в «ёлочках»); «unresolved:…» — ссылка, цель которой
    в документе не найдена (только явные: «см. Ф-9», «раздел 7»).
    """
    known_ids = set(doc_ids)
    titles = [one_line(title) for title in headings]
    by_number = {m.group(1): t for t in titles if (m := _HEADING_NUMBER.match(t + " "))}
    by_title = {title.casefold(): title for title in titles}
    found: list[tuple[int, str]] = []
    for match in _ID_ANYWHERE.finditer(text):
        ref = match.group("id")
        if ref == own_id or not is_document_id(ref):
            continue
        if ref in known_ids:
            found.append((match.start(), f"req:{ref}"))
        elif _REFERENCE_CUE.search(text[max(0, match.start() - 25) : match.start()]):
            found.append((match.start(), f"unresolved:{ref}"))
    for match in _SECTION_REFERENCE.finditer(text):
        target = by_number.get(match.group("num"))
        label = f"section:{target}" if target else f"unresolved:{one_line(match.group(0))}"
        found.append((match.start(), label))
    for match in _QUOTED.finditer(text):
        title = by_title.get(one_line(match.group(1)).casefold())
        if title:
            found.append((match.start(), f"section:{title}"))
    found.sort()
    return tuple(dict.fromkeys(ref for _, ref in found))
