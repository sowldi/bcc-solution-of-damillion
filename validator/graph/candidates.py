"""Отбор пар-кандидатов для проверки моделью — без полного перебора всех пар.

1. У каждого требования — набор значимых основ слов: текст после ID, слова из 4+ букв без
   служебных, модальных и ссылочных слов («должна», «может», «согласно»), с отсечением одного
   окончания и усечением до 6 букв. Числа и ID не учитываются: одинаковые числа («3 дня» и
   «3 рабочих дня») сами по себе пару не образуют. Отрицания («не», «нет», «нельзя») тоже не
   основы — они не делают требования похожими.
2. Инвертированный индекс «основа → требования»: сравниваются только требования с общей
   основой (evaluated_pairs), а не все n(n-1)/2 пар. Основа, которая есть больше чем у
   половины требований (при 4+ требованиях), не различает их и не учитывается.
3. Кандидат — пара минимум с двумя общими основами. Исключаются дословные повторы (одинаковый
   текст после ID — сравнивать нечего, это замечание CON-01) и пары, уже связанные
   подтверждённой связью (бюджет модели на них не тратится).
4. Оценка — число общих основ, +1 за один раздел; порядок — по убыванию оценки, затем по
   месту в документе. Результат детерминирован.

Это эвристика отбора, а не вывод о связи: кандидат — только вопрос к модели.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence

from validator.graph.model import CandidatePair, CandidateSelection
from validator.model import Document
from validator.segmentation import Requirement
from validator.segmentation.text import ID_PATTERN, content_key
from validator.segmentation.verify import body_start

MIN_SHARED = 2  # общих основ у пары-кандидата
MIN_WORD = 4  # букв в учитываемом слове
STEM_LENGTH = 6
COMMON_SHARE = 0.5  # основа у большей доли требований — слишком общая
MIN_NODES_FOR_COMMON = 4

_WORD = re.compile(r"[^\W\d_]+")
_ID_TOKEN = re.compile(rf"(?<![\w-]){ID_PATTERN}(?![\w-])")
_STOP = frozenset(
    {
        # служебные и местоименные
        "более", "менее", "также", "либо", "если", "когда", "после", "перед", "через",
        "между", "кроме", "чтобы", "только", "лишь", "этот", "этой", "этого", "этом", "этих",
        "который", "которая", "которое", "которые", "которых", "каждый", "каждая", "каждое",
        "каждого", "любой", "любая", "любое", "любого", "всех", "всего", "весь", "вся",
        "течение", "случае", "случай", "рамках",
        # модальность и связки: не делают требования похожими
        "должен", "должна", "должно", "должны", "может", "могут", "можно", "нельзя",
        "следует", "необходимо", "нужно", "быть", "будет", "будут", "является", "являются",
        # ссылки и отрицания
        "согласно", "соответствии", "указанный", "указанные", "нет",
        "shall", "must", "should", "will", "with", "from", "that", "this", "when", "after",
        "before", "more", "less", "than", "into", "only", "each", "never",
    }
)  # fmt: skip
_REFLEXIVE = ("ся", "сь")
_ENDINGS = tuple(
    sorted(
        {
            "иями", "ями", "ами", "ого", "его", "ому", "ему", "ыми", "ими", "ией",
            "ах", "ях", "ов", "ев", "ей", "ой", "ий", "ый", "ая", "яя", "ое", "ее", "ые",
            "ие", "ом", "ем", "ам", "ям", "ую", "юю", "ия", "ию", "ии", "ть", "ет", "ют",
            "ут", "ит", "ат", "ят",
            "а", "я", "о", "е", "ы", "и", "у", "ю", "ь", "й",
            "ing", "ed", "es", "s",
        },
        key=lambda ending: (-len(ending), ending),
    )
)  # fmt: skip


def stem(word: str) -> str | None:
    """Основа слова или None, если слово не учитывается (короткое, служебное, модальное)."""
    word = word.casefold().replace("ё", "е")
    if len(word) < MIN_WORD or word in _STOP:
        return None
    for suffix in _REFLEXIVE:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    for ending in _ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= 3:
            word = word[: -len(ending)]
            break
    return word[:STEM_LENGTH]


def content_stems(fragment: Requirement, document: Document) -> frozenset[str]:
    """Значимые основы текста требования (без ID в начале и без ссылок на ID)."""
    body = _ID_TOKEN.sub(" ", fragment.text[body_start(fragment, document) :])
    return frozenset(s for word in _WORD.findall(body) if (s := stem(word)) is not None)


def select_candidates(
    fragments: Sequence[Requirement],
    document: Document,
    linked: Collection[frozenset[str]] = (),
) -> CandidateSelection:
    """Пары-кандидаты среди узлов графа (fragments — в порядке документа)."""
    count = len(fragments)
    order = {fragment.id: index for index, fragment in enumerate(fragments)}
    postings: dict[str, list[str]] = {}
    for fragment in fragments:
        for value in sorted(content_stems(fragment, document)):
            postings.setdefault(value, []).append(fragment.id)
    common = sorted(
        value
        for value, ids in postings.items()
        if count >= MIN_NODES_FOR_COMMON and len(ids) > count * COMMON_SHARE
    )
    skip = set(common)
    shared: dict[tuple[str, str], list[str]] = {}
    for value in sorted(postings):
        ids = postings[value]
        if value in skip or len(ids) < 2:
            continue
        for index, first in enumerate(ids):
            for second in ids[index + 1 :]:
                shared.setdefault((first, second), []).append(value)
    texts = {
        fragment.id: content_key(fragment.text[body_start(fragment, document) :])
        for fragment in fragments
    }
    sections = {fragment.id: fragment.section_path for fragment in fragments}
    pairs: list[CandidatePair] = []
    duplicates: list[tuple[str, str]] = []
    connected: list[tuple[str, str]] = []
    for (first, second), values in shared.items():
        if len(values) < MIN_SHARED:
            continue
        if texts[first] == texts[second]:
            duplicates.append((first, second))
        elif frozenset((first, second)) in linked:
            connected.append((first, second))
        else:
            same = sections[first] == sections[second]
            pairs.append(CandidatePair(first, second, len(values) + same, tuple(values), same))

    def position(pair: tuple[str, str]) -> tuple[int, int]:
        return order[pair[0]], order[pair[1]]

    pairs.sort(key=lambda p: (-p.score, *position((p.source_id, p.target_id))))
    return CandidateSelection(
        possible_pairs=count * (count - 1) // 2,
        evaluated_pairs=len(shared),
        pairs=tuple(pairs),
        duplicates=tuple(sorted(duplicates, key=position)),
        linked=tuple(sorted(connected, key=position)),
        common_stems=tuple(common),
        min_shared=MIN_SHARED,
    )
