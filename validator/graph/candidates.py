"""Отбор пар-кандидатов для проверки моделью — без полного перебора и с ограничением ресурсов.

1. Значимые слова требования: текст после ID, слова из 4+ букв без служебных, модальных и
   ссылочных слов («должна», «может», «согласно»), в нижнем регистре, «ё» → «е», без
   возвратной частицы «-ся/-сь». Числа и ID не учитываются: одинаковые числа («3 дня» и
   «3 рабочих дня») сами по себе пару не образуют. Отрицания («не», «нет», «нельзя») тоже
   не учитываются — они не делают требования похожими.
2. Формы одного слова (этап 7.1). Варианты основы слова — само слово и слово без одного
   окончания из закрытого списка флексий: после именного окончания остаётся 3+ буквы, после
   глагольного («-ть», «-ет», «-ит»…) — 4+, и тогда же — основа без конечной гласной
   («сохраня|ет», «сохрани|ть» → «сохран»). Две формы — одно слово,
   если у них есть общий вариант основы («ответ» и «ответа» → «ответ», «хранит» и «хранят»
   → «хран») или совпадают первые 7 букв (общий корень: «подтверждает», «подтверждения»).
   Окончание не выбирается жадно и слово не усекается: «ответ» не превращается в «отв», а
   «отчёт» и «отчество», «перевод» и «перевозка» не объединяются. Формы объединяются по
   словарю документа (компоненты связности), представитель — самая короткая форма; результат
   не зависит от порядка требований и от PYTHONHASHSEED.
3. Инвертированный индекс «слово → требования»: сравниваются только требования с общим
   словом (evaluated_pairs), а не все n(n-1)/2 пар. Слово, которое есть больше чем у
   половины требований (при 4+ требованиях), не различает их и не учитывается.
4. Кандидат — пара минимум с двумя общими словами. Исключаются дословные повторы (одинаковый
   текст после ID — сравнивать нечего, это замечание CON-01) и пары, уже связанные
   подтверждённой связью (бюджет модели на них не тратится).
5. Оценка — число общих слов, +1 за один раздел; порядок — по убыванию оценки, затем по
   месту в документе. Результат детерминирован.
6. Ограничение ресурсов (этап 7.1): пары считаются построчно (для требования — только
   требования после него), в памяти — счётчики одной строки и не больше limit лучших пар
   (куча). Список всех кандидатов не строится, но их число (found) и число исключённых
   ограничением считаются точно в том же проходе — дополнительного перебора нет.

Это эвристика отбора, а не вывод о связи: кандидат — только вопрос к модели.
"""

from __future__ import annotations

import heapq
import re
from bisect import bisect_right
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass

from validator.graph.model import MAX_CANDIDATES, CandidatePair, CandidateSelection
from validator.model import Document
from validator.segmentation import Requirement
from validator.segmentation.text import ID_PATTERN, content_key
from validator.segmentation.verify import body_start

MIN_SHARED = 2  # общих слов у пары-кандидата
MIN_WORD = 4  # букв в учитываемом слове
MIN_CORE = 4  # букв в основе после глагольного окончания («ответ» не даёт «отв»)
MIN_NOMINAL_CORE = 3  # букв в основе после именного окончания («поле» → «пол»)
ROOT_PREFIX = 7  # общий корень: совпадают первые 7 букв
COMMON_SHARE = 0.5  # слово у большей доли требований — слишком общее
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
_VERB_ENDINGS = frozenset({"ть", "ет", "ют", "ут", "ит", "ат", "ят"})
_VOWELS = frozenset("аеиоуыэюя")
# Закрытый список флексий (тот же, что на этапе 7): окончания существительных,
# прилагательных и глаголов и английские -s/-es/-ed/-ing.
_ENDINGS = tuple(
    sorted(
        {
            "иями", "ями", "ами", "ого", "его", "ому", "ему", "ыми", "ими", "ией",
            "ах", "ях", "ов", "ев", "ей", "ой", "ий", "ый", "ая", "яя", "ое", "ее", "ые",
            "ие", "ом", "ем", "ам", "ям", "ую", "юю", "ия", "ию", "ии", "ым", "ых",
            "ть", "ет", "ют",
            "ут", "ит", "ат", "ят",
            "а", "я", "о", "е", "ы", "и", "у", "ю", "ь", "й",
            "ing", "ed", "es", "s",
        },
        key=lambda ending: (-len(ending), ending),
    )
)  # fmt: skip


@dataclass(frozen=True)
class CandidateInput:
    """Требование для отбора: ID, значимые слова, раздел и текст после ID (для повторов)."""

    requirement_id: str
    words: frozenset[str]
    section_path: tuple[str, ...]
    text_key: str


def normalize(word: str) -> str | None:
    """Значимое слово в нормальном виде или None (короткое, служебное, модальное)."""
    word = word.casefold().replace("ё", "е")
    if len(word) < MIN_WORD or word in _STOP:
        return None
    for suffix in _REFLEXIVE:
        if word.endswith(suffix) and len(word) - len(suffix) >= MIN_WORD:
            return word[: -len(suffix)]
    return word


def word_cores(word: str) -> frozenset[str]:
    """Варианты основы: само слово и слово без одного окончания.

    После именного окончания основа — от 3 букв («поле», «полем» → «пол»), после глагольного
    — от 4 («ответ» не даёт «отв», «дать» — не «дате»); после глагольного окончания основа
    берётся и без конечной гласной («сохраня|ет», «сохрани|ть» → «сохран»).
    """
    cores = {word}
    for ending in _ENDINGS:
        if not word.endswith(ending):
            continue
        core = word[: -len(ending)]
        if ending == "ь":  # «-ть» инфинитива и «-ь» существительного не различить: «дать»
            if len(core) >= MIN_CORE:
                cores.add(core)
        elif ending not in _VERB_ENDINGS:
            if len(core) >= MIN_NOMINAL_CORE:
                cores.add(core)
        elif len(core) >= MIN_CORE:
            cores.add(core)
            if core[-1] in _VOWELS and len(core) - 1 >= MIN_CORE:
                cores.add(core[:-1])
    return frozenset(cores)


def lexemes(vocabulary: Iterable[str]) -> dict[str, str]:
    """Слово → представитель его группы форм (самая короткая форма, затем по алфавиту)."""
    words = sorted(set(vocabulary))
    parent = {word: word for word in words}

    def find(word: str) -> str:
        while parent[word] != word:
            parent[word] = parent[parent[word]]
            word = parent[word]
        return word

    first: dict[tuple[str, str], str] = {}
    for word in words:
        keys = {("core", core) for core in word_cores(word)}
        if len(word) >= ROOT_PREFIX:
            keys.add(("root", word[:ROOT_PREFIX]))
        for key in sorted(keys):
            if key in first:
                parent[find(word)] = find(first[key])
            else:
                first[key] = word
    groups: dict[str, list[str]] = {}
    for word in words:
        groups.setdefault(find(word), []).append(word)
    result: dict[str, str] = {}
    for members in groups.values():
        representative = min(members, key=lambda word: (len(word), word))
        for word in members:
            result[word] = representative
    return result


def content_words(fragment: Requirement, document: Document) -> frozenset[str]:
    """Значимые слова требования в нормальном виде (без ID в начале и ссылок на ID)."""
    body = _ID_TOKEN.sub(" ", fragment.text[body_start(fragment, document) :])
    return frozenset(w for token in _WORD.findall(body) if (w := normalize(token)) is not None)


def select_candidates(
    fragments: Sequence[Requirement],
    document: Document,
    linked: Collection[frozenset[str]] = (),
    *,
    limit: int = MAX_CANDIDATES,
) -> CandidateSelection:
    """Пары-кандидаты среди узлов графа (fragments — в порядке документа)."""
    items = [
        CandidateInput(
            fragment.id,
            content_words(fragment, document),
            fragment.section_path,
            content_key(fragment.text[body_start(fragment, document) :]),
        )
        for fragment in fragments
    ]
    return rank_candidates(items, linked, limit=limit)


def rank_candidates(
    items: Sequence[CandidateInput],
    linked: Collection[frozenset[str]] = (),
    *,
    limit: int = MAX_CANDIDATES,
) -> CandidateSelection:
    """Не больше limit лучших пар; память — O(n + limit), счётчики — точные."""
    if limit < 1:
        raise ValueError(f"limit: ожидается целое число не меньше 1, получено {limit}.")
    count = len(items)
    representative = lexemes(word for item in items for word in item.words)
    sets = [frozenset(representative[word] for word in item.words) for item in items]
    postings: dict[str, list[int]] = {}
    for index, values in enumerate(sets):
        for value in values:
            postings.setdefault(value, []).append(index)  # индексы по возрастанию
    common = sorted(
        value
        for value, ids in postings.items()
        if count >= MIN_NODES_FOR_COMMON and len(ids) > count * COMMON_SHARE
    )
    skip = set(common)
    active = {value: ids for value, ids in postings.items() if value not in skip and len(ids) > 1}
    best: list[tuple[int, int, int]] = []  # (оценка, -i, -j): в вершине — худшая из лучших
    evaluated = found = duplicate_count = 0
    duplicates: list[tuple[str, str]] = []
    connected: list[tuple[str, str]] = []
    for i, values in enumerate(sets):
        shared: dict[int, int] = {}
        for value in values:
            ids = active.get(value)
            if ids is None:
                continue
            for j in ids[bisect_right(ids, i) :]:
                shared[j] = shared.get(j, 0) + 1
        evaluated += len(shared)
        first = items[i]
        for j in sorted(shared):
            if shared[j] < MIN_SHARED:
                continue
            second = items[j]
            pair = (first.requirement_id, second.requirement_id)
            if first.text_key == second.text_key:
                duplicate_count += 1
                if len(duplicates) < limit:
                    duplicates.append(pair)
            elif frozenset(pair) in linked:
                connected.append(pair)
            else:
                found += 1
                entry = (shared[j] + (first.section_path == second.section_path), -i, -j)
                if len(best) < limit:
                    heapq.heappush(best, entry)
                elif entry > best[0]:
                    heapq.heapreplace(best, entry)
    pairs = []
    for score, minus_i, minus_j in sorted(best, reverse=True):
        i, j = -minus_i, -minus_j
        pairs.append(
            CandidatePair(
                items[i].requirement_id,
                items[j].requirement_id,
                score,
                tuple(sorted((sets[i] & sets[j]) - skip)),
                items[i].section_path == items[j].section_path,
            )
        )
    return CandidateSelection(
        possible_pairs=count * (count - 1) // 2,
        evaluated_pairs=evaluated,
        pairs=tuple(pairs),
        duplicates=tuple(duplicates),
        linked=tuple(connected),
        common_stems=tuple(common),
        min_shared=MIN_SHARED,
        found=found,
        limit=limit,
        duplicate_count=duplicate_count,
    )
