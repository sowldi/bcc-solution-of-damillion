"""Отбор пар-кандидатов (этап 7.1): формы одного слова и ограниченный по ресурсам отбор.

Кандидат — только вопрос к модели, а не вывод о связи. Проверяются гарантии: формы одного
слова объединяются, разные слова — нет; число пар и память ограничены; счётчики покрытия
честные; порядок детерминирован.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tracemalloc
from pathlib import Path

import pytest

from tests.graph_helpers import GRAPH_SPEC, load
from tests.lint_helpers import HEAD
from validator.graph import CandidateInput, rank_candidates

ROOT = Path(__file__).resolve().parent.parent


def _pairs(tmp_path: Path, *requirements: str) -> list[tuple[str, str]]:
    lines = "".join(f"* Ф-{n}. {text}\n" for n, text in enumerate(requirements, 1))
    graph = load(tmp_path, HEAD + lines).graph
    return [(pair.source_id, pair.target_id) for pair in graph.candidates.pairs]


# --- A. формы одного слова ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Система хранит ответ заемщика.", "Срок ответа заемщика — один день."),
        ("Система хранит отчёт заемщика.", "Срок отчёта заемщика — один день."),
        ("Система хранит пакет заемщика.", "Срок пакета заемщика — один день."),
        ("Система хранит лимит заемщика.", "Срок лимита заемщика — один день."),
        ("Система хранит кредит заемщика.", "Срок кредита заемщика — один день."),
        ("Система хранит депозит заемщика.", "Срок депозита заемщика — один день."),
        ("Ожидание заемщика длится 5 минут.", "Таймер заемщика показывает минуты."),
    ],
)
def test_noun_forms_make_a_candidate_pair(tmp_path: Path, first: str, second: str) -> None:
    assert _pairs(tmp_path, first, second) == [("Ф-1", "Ф-2")]


def test_verb_forms_including_reflexive_are_one_word(tmp_path: Path) -> None:
    assert _pairs(tmp_path, "Банк хранит выписку заемщика.", "Выписки хранятся отдельно.") == [
        ("Ф-1", "Ф-2")
    ]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Банк сохраняет выписку.", "Сотрудник может сохранить выписку."),  # основа глагола
        ("Анкета содержит поле адреса.", "Оператор управляет полем адреса."),  # основа 3 буквы
        ("Оператор вводит целое значение.", "Значение задаётся целым числом."),
    ],
)
def test_short_and_verb_stems_keep_word_forms_together(
    tmp_path: Path, first: str, second: str
) -> None:
    assert _pairs(tmp_path, first, second) == [("Ф-1", "Ф-2")]


def test_three_letter_stems_are_not_taken_from_verb_endings(tmp_path: Path) -> None:
    # «дате» и «дать», «поле» и «полёт» — разные слова: глагольное окончание не даёт основу
    # из трёх букв.
    assert (
        _pairs(
            tmp_path, "Отчёт сортируется по дате платежа.", "Клиент может дать отсрочку платежа."
        )
        == []
    )
    assert (
        _pairs(tmp_path, "Анкета содержит поле рейса.", "Диспетчер отслеживает полёт рейса.") == []
    )


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Банк хранит отчёт заемщика.", "Анкета содержит отчество заемщика."),
        ("Система хранит ответ заемщика.", "Договор описывает ответственность заемщика."),
        ("Банк оформляет перевод заемщика.", "Курьер выполняет перевозку заемщика."),
        ("Ожидание заемщика длится минуту.", "Задаётся минимальный платёж заемщика."),
    ],
)
def test_different_words_with_a_common_start_are_not_merged(
    tmp_path: Path, first: str, second: str
) -> None:
    assert _pairs(tmp_path, first, second) == []


def test_negations_and_modal_words_do_not_make_pairs_similar(tmp_path: Path) -> None:
    pairs = _pairs(
        tmp_path,
        "Оператор не должен изменять анкету.",
        "Клиент не должен подтверждать выписку.",
    )

    assert pairs == []


def test_english_word_forms(tmp_path: Path) -> None:
    pairs = _pairs(tmp_path, "The service stores request limits.", "Each request limit is logged.")

    assert pairs == [("Ф-1", "Ф-2")]


def test_short_words_ids_and_numbers_do_not_make_pairs(tmp_path: Path) -> None:
    pairs = _pairs(
        tmp_path,
        "Код ПИН 5000 KZT по Ф-3 за 1 час.",
        "Код ПИН 5000 KZT по Ф-3 за 2 дня.",
        "Справка выдаётся бесплатно.",
    )

    assert pairs == []


def test_same_words_in_different_sections_rank_below_the_same_section(tmp_path: Path) -> None:
    graph = load(
        tmp_path,
        HEAD
        + "* Ф-1. Система хранит выписку заемщика.\n"
        + "* Ф-2. Оператор печатает выписку заемщика.\n\n"
        + "## Нефункциональные требования\n\n* Ф-3. Аудитор проверяет выписку заемщика.\n",
    ).graph

    scores = {(p.source_id, p.target_id): (p.score, p.same_section) for p in graph.candidates.pairs}
    assert scores[("Ф-1", "Ф-2")] == (3, True)
    assert scores[("Ф-1", "Ф-3")] == (2, False)


# --- B. ограниченный отбор -------------------------------------------------------------------


_CONSONANTS = "бвгдклмнпрстфхчш"


def _code(number: int) -> str:
    """Уникальное буквенное начало слова (как у настоящих слов — без цифр)."""
    letters = []
    for _ in range(3):
        number, digit = divmod(number, len(_CONSONANTS))
        letters.append(_CONSONANTS[digit] + "о")
    return "".join(letters)


def _items(groups: int, size: int, *, common: int = 0) -> list[CandidateInput]:
    """Группы требований с общими словами внутри группы; common — сколько требований
    дополнительно содержат два общих слова (неблагоприятный случай)."""
    items = []
    for index in range(groups * size):
        group = index // size
        words = {_code(group) + "альфа", _code(group) + "бета", _code(index) + "уник"}
        if index < common:
            words |= {"журнальный", "архивный"}
        items.append(CandidateInput(f"R-{index}", frozenset(words), ("Раздел",), f"текст {index}"))
    return items


def test_cap_keeps_the_best_ranked_pairs_and_counts_the_rest(tmp_path: Path) -> None:
    items = _items(groups=4, size=5)  # 4 × C(5,2) = 40 кандидатов
    full = rank_candidates(items, limit=1000)
    capped = rank_candidates(items, limit=7)

    assert (full.found, len(full.pairs), full.truncated) == (40, 40, False)
    assert (capped.found, len(capped.pairs), capped.truncated) == (40, 7, True)
    assert capped.excluded_by_limit == 33
    assert capped.pairs == full.pairs[:7]
    assert capped.evaluated_pairs == full.evaluated_pairs
    kept = min(p.score for p in capped.pairs)
    assert all(p.score <= kept for p in full.pairs[7:])


def test_cap_reached_exactly_is_not_truncation() -> None:
    selection = rank_candidates(_items(groups=2, size=5), limit=20)  # 2 × C(5,2) = 20

    assert (selection.found, len(selection.pairs), selection.truncated) == (20, 20, False)


def test_invalid_cap_is_rejected() -> None:
    with pytest.raises(ValueError, match="limit"):
        rank_candidates(_items(groups=1, size=3), limit=0)


def test_adversarial_common_words_stay_within_memory_and_cap() -> None:
    items = _items(groups=1500, size=2, common=700)  # C(700,2) ≈ 245 тыс. кандидатов
    tracemalloc.start()
    selection = rank_candidates(items, limit=500)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert selection.found == 700 * 699 // 2 + 1500 - 350  # общие слова + пары групп
    assert len(selection.pairs) == 500
    assert selection.excluded_by_limit == selection.found - 500
    assert peak < 10_000_000  # память не растёт с числом кандидатов (~245 тыс.)


def test_common_words_are_excluded_by_document_frequency() -> None:
    items = _items(groups=2, size=4, common=6)  # общие слова у 6 из 8 — больше половины

    selection = rank_candidates(items, limit=1000)

    assert set(selection.common_stems) == {"журнальный", "архивный"}
    assert selection.found == 2 * 6  # только пары внутри групп


def test_pairs_are_unique_unordered_and_in_document_order() -> None:
    selection = rank_candidates(_items(groups=3, size=6, common=10), limit=1000)
    order = {f"R-{i}": i for i in range(18)}

    keys = [frozenset((p.source_id, p.target_id)) for p in selection.pairs]
    assert len(keys) == len(set(keys))
    assert all(order[p.source_id] < order[p.target_id] for p in selection.pairs)


def test_permutation_keeps_candidates_and_scores(tmp_path: Path) -> None:
    items = _items(groups=3, size=4, common=5)
    forward = rank_candidates(items, limit=1000)
    backward = rank_candidates(items[::-1], limit=1000)

    def scored(selection: object) -> set[tuple[frozenset[str], int]]:
        return {
            (frozenset((p.source_id, p.target_id)), p.score)
            for p in selection.pairs  # type: ignore[attr-defined]
        }

    assert scored(forward) == scored(backward)
    assert (forward.found, forward.evaluated_pairs) == (backward.found, backward.evaluated_pairs)


def test_truncation_after_permutation_keeps_only_top_scores() -> None:
    items = _items(groups=3, size=4, common=5)
    capped = rank_candidates(items[::-1], limit=6)
    full = rank_candidates(items[::-1], limit=1000)

    assert min(p.score for p in capped.pairs) >= max(p.score for p in full.pairs[6:])


def test_repeated_runs_and_hash_seeds_give_the_same_graph(tmp_path: Path) -> None:
    spec = tmp_path / "graph.md"
    spec.write_text(GRAPH_SPEC, encoding="utf-8")
    script = (
        "import sys, json; from pathlib import Path\n"
        "from validator.ingest import load_document\n"
        "from validator.segmentation import segment_document\n"
        "from validator.lint import run_lint\n"
        "from validator.graph import build_fact_graph\n"
        "d = load_document(Path(sys.argv[1])); s = segment_document(d)\n"
        "print(build_fact_graph(d, s, run_lint(d, s), max_candidates=3).to_json())\n"
    )
    outputs = set()
    for seed in ("0", "1", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(ROOT)}
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", script, str(spec)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            check=True,
        )
        outputs.add(result.stdout)
    assert len(outputs) == 1
    data = json.loads(outputs.pop())
    assert len(data["candidates"]["pairs"]) == 3
    assert data["candidates"]["found"] == 6


def test_graph_build_respects_the_cap(tmp_path: Path) -> None:
    built = load(tmp_path)
    from validator.graph import build_fact_graph

    graph = build_fact_graph(built.document, built.segmentation, built.lint, max_candidates=2)

    assert [(p.source_id, p.target_id) for p in graph.candidates.pairs] == [
        ("Ф-3", "Ф-4"),
        ("Ф-1", "Ф-4"),
    ]
    assert (graph.candidates.found, graph.candidates.excluded_by_limit) == (6, 4)
    assert graph.candidates.linked == (("Ф-1", "Ф-2"),)  # подтверждённая связь не кандидат
