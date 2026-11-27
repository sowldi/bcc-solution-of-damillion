"""Сопоставление требований двух версий документа.

Содержательный ID меняется при правке текста (если у требования нет идентификатора документа),
поэтому сопоставление идёт в три шага, от строгого к мягкому:
1. одинаковый ID (идентификатор документа или одинаковое содержание);
2. одинаковое содержание при разных ID (сдвиг суффикса «~2» у дублей);
3. похожесть нормализованного текста не ниже SIMILARITY_THRESHOLD — «изменено».
Остальное — «удалено» (было только в старой версии) или «добавлено» (только в новой).
Перемещение в другой раздел ID не меняет и отмечается флагом section_changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

from validator.segmentation.model import Requirement, Segmentation
from validator.segmentation.text import content_key

MatchStatus = Literal["unchanged", "modified", "added", "removed"]
SIMILARITY_THRESHOLD = 0.75


@dataclass(frozen=True)
class RequirementMatch:
    status: MatchStatus
    old_id: str | None
    new_id: str | None
    old_text: str | None  # нормализованный текст
    new_text: str | None
    similarity: float
    section_changed: bool


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, content_key(a), content_key(b), autojunk=False).ratio()


def match_segmentations(old: Segmentation, new: Segmentation) -> tuple[RequirementMatch, ...]:
    olds, news = list(old.requirements), list(new.requirements)
    pairs: dict[int, int] = {}  # индекс в olds → индекс в news

    new_by_id = {fragment.id: index for index, fragment in enumerate(news)}
    for i, fragment in enumerate(olds):
        if fragment.id in new_by_id:
            pairs[i] = new_by_id[fragment.id]

    taken = set(pairs.values())
    free_by_key: dict[str, list[int]] = {}
    for j, fragment in enumerate(news):
        if j not in taken:
            free_by_key.setdefault(content_key(fragment.normalized), []).append(j)
    for i, fragment in enumerate(olds):
        candidates = free_by_key.get(content_key(fragment.normalized))
        if i not in pairs and candidates:
            pairs[i] = candidates.pop(0)
            taken.add(pairs[i])

    scored = sorted(
        (
            (similarity(olds[i].normalized, news[j].normalized), i, j)
            for i in range(len(olds))
            if i not in pairs
            for j in range(len(news))
            if j not in taken
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    for score, i, j in scored:
        if score >= SIMILARITY_THRESHOLD and i not in pairs and j not in taken:
            pairs[i] = j
            taken.add(j)

    matches = [
        _pair(olds[i], news[pairs[i]]) if i in pairs else _removed(olds[i])
        for i in range(len(olds))
    ]
    matches += [_added(news[j]) for j in range(len(news)) if j not in taken]
    return tuple(matches)


def _pair(old: Requirement, new: Requirement) -> RequirementMatch:
    same = content_key(old.normalized) == content_key(new.normalized)
    return RequirementMatch(
        status="unchanged" if same else "modified",
        old_id=old.id,
        new_id=new.id,
        old_text=old.normalized,
        new_text=new.normalized,
        similarity=1.0 if same else similarity(old.normalized, new.normalized),
        section_changed=old.section_path != new.section_path,
    )


def _removed(old: Requirement) -> RequirementMatch:
    return RequirementMatch("removed", old.id, None, old.normalized, None, 0.0, False)


def _added(new: Requirement) -> RequirementMatch:
    return RequirementMatch("added", None, new.id, None, new.normalized, 0.0, False)
