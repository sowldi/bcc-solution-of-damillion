"""Машинная проверка доказательной привязки сегментации к исходному документу.

Проверяется, что каждая цитата — точный срез текста блока, место совпадает с вычисленным
по блоку, утверждения не выходят за пределы своего фрагмента и покрывают весь его текст,
а каждый блок документа принадлежит ровно одному фрагменту (текст не потерян).
"""

from __future__ import annotations

from validator.model import Document
from validator.segmentation.model import Requirement, Segmentation, SourceRef

# source_location живёт в sources.py; импорт здесь сохраняет прежний путь
# validator.segmentation.verify.source_location для обратной совместимости.
from validator.segmentation.sources import source_location
from validator.segmentation.text import normalize_text, parse_prefix

__all__ = ["body_start", "source_location", "verify_segmentation"]


def verify_segmentation(segmentation: Segmentation, document: Document) -> tuple[str, ...]:
    """Список нарушений (пустой — всё доказуемо). Безопасна для данных, прочитанных из JSON."""
    problems: list[str] = []
    if segmentation.document_sha256 != document.sha256:
        problems.append("Сегментация построена для другого документа: SHA-256 не совпадает.")
    by_id = {fragment.id: fragment for fragment in segmentation.fragments}
    for fragment in segmentation.fragments:
        quote_problems = _check_quote(fragment, document)
        problems += quote_problems
        if not quote_problems:
            problems += _check_normalized(fragment, document)
        problems += _check_links(fragment, by_id)
    problems += _check_coverage(segmentation, document)
    return tuple(problems)


def _check_quote(fragment: Requirement, document: Document) -> list[str]:
    if not fragment.sources:
        return [f"{fragment.id}: нет ссылок на источник."]
    problems: list[str] = []
    slices: list[str] = []
    for ref in fragment.sources:
        if not 0 <= ref.block < len(document.blocks):
            return [f"{fragment.id}: ссылка на несуществующий блок {ref.block}."]
        block = document.blocks[ref.block]
        if not 0 <= ref.start <= ref.end <= len(block.text):
            return [
                f"{fragment.id}: границы цитаты {ref.start}–{ref.end} вне текста блока {ref.block}."
            ]
        expected = source_location(block, ref.start, ref.end)
        if ref.location != expected:
            problems.append(
                f"{fragment.id}: место «{ref.location.describe()}» не совпадает "
                f"с источником «{expected.describe()}»."
            )
        slices.append(block.text[ref.start : ref.end])
    if "\n".join(slices) != fragment.text:
        problems.append(f"{fragment.id}: текст не совпадает с цитатой из источника.")
    return problems


def body_start(fragment: Requirement, document: Document) -> int:
    """Начало содержательного текста фрагмента — так же, как его считает сегментатор:
    без ID документа или номера пункта в начале абзаца и пункта списка. У строки таблицы,
    заголовка и утверждения «X/k» префикса нет."""
    if fragment.is_statement or not fragment.sources:
        return 0
    block = document.blocks[fragment.sources[0].block]
    if block.kind in ("table_row", "heading"):
        return 0
    return parse_prefix(block.text, block.kind).body_start


def _check_normalized(fragment: Requirement, document: Document) -> list[str]:
    """M4: normalized — производное от точного текста, а не самостоятельные данные."""
    expected = normalize_text(fragment.text[body_start(fragment, document) :])
    if fragment.normalized != expected:
        return [f"{fragment.id}: normalized не соответствует тексту фрагмента."]
    return []


def _check_links(fragment: Requirement, by_id: dict[str, Requirement]) -> list[str]:
    problems: list[str] = []
    if fragment.parent_id is not None:
        parent = by_id.get(fragment.parent_id)
        if parent is None:
            problems.append(f"{fragment.id}: родитель {fragment.parent_id} не найден.")
        else:
            if fragment.id not in parent.children:
                problems.append(f"{fragment.id}: нет в списке детей родителя {parent.id}.")
            if fragment.is_statement and not _within(fragment.sources, parent.sources):
                problems.append(f"{fragment.id}: цитата выходит за пределы фрагмента {parent.id}.")
    for child_id in fragment.children:
        child = by_id.get(child_id)
        if child is None or child.parent_id != fragment.id:
            problems.append(
                f"{fragment.id}: дочерний фрагмент {child_id} не ссылается на родителя."
            )
    return problems


def _within(inner: tuple[SourceRef, ...], outer: tuple[SourceRef, ...]) -> bool:
    return all(
        any(o.block == i.block and o.start <= i.start and i.end <= o.end for o in outer)
        for i in inner
    )


def _check_coverage(segmentation: Segmentation, document: Document) -> list[str]:
    problems: list[str] = []
    owners: dict[int, list[str]] = {}
    for fragment in segmentation.fragments:
        if fragment.is_statement:
            continue
        for ref in fragment.sources:
            if 0 <= ref.block < len(document.blocks):
                owners.setdefault(ref.block, []).append(fragment.id)
                if (ref.start, ref.end) != (0, len(document.blocks[ref.block].text)):
                    problems.append(f"{fragment.id}: блок {ref.block} взят не целиком.")
    for index in range(len(document.blocks)):
        holders = owners.get(index, [])
        if len(holders) != 1:
            problems.append(f"Блок {index} принадлежит фрагментам: {holders or 'ни одному'}.")
    problems += _check_statement_coverage(segmentation, document)
    return problems


def _check_statement_coverage(segmentation: Segmentation, document: Document) -> list[str]:
    """Утверждения покрывают весь содержательный текст своего фрагмента — ничего не потеряно."""
    statements: dict[str, list[SourceRef]] = {}
    for fragment in segmentation.fragments:
        if fragment.is_statement and fragment.parent_id:
            statements.setdefault(fragment.parent_id, []).extend(fragment.sources)
    by_id = {fragment.id: fragment for fragment in segmentation.fragments}
    problems: list[str] = []
    for parent_id, refs in statements.items():
        parent = by_id.get(parent_id)
        if parent is None:
            continue  # отсутствующий родитель уже отмечен в _check_links
        for position, ref in enumerate(parent.sources):
            if not 0 <= ref.block < len(document.blocks):
                continue  # битая ссылка уже отмечена в _check_quote
            block = document.blocks[ref.block]
            end = min(ref.end, len(block.text))
            begin = parse_prefix(block.text, block.kind).body_start if position == 0 else 0
            covered = [False] * len(block.text)
            for statement in refs:
                if statement.block == ref.block:
                    for offset in range(max(statement.start, 0), min(statement.end, end)):
                        covered[offset] = True
            if any(not covered[i] and not block.text[i].isspace() for i in range(begin, end)):
                problems.append(f"{parent_id}: утверждения не покрывают часть текста фрагмента.")
                break
    return problems
