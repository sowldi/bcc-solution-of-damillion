"""Stage 9: оценка требований, сводка и итог без ложной точности.

Итог строится только на подтверждённых (verified) замечаниях правил и подтверждённых
противоречиях. Гипотезы модели и потенциальные противоречия идут «на проверку человеку» и
итог не меняют; непроверенное перечисляется. Документы — собственные, сети и ключей нет.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tests.graph_helpers import GRAPH_SPEC, Built, load
from tests.lint_helpers import HEAD
from tests.llm_helpers import findings_json, hypothesis
from validator.graph import CandidateSelection, ExcludedRequirement, build_fact_graph
from validator.llm import LLMOptions, review_requirements
from validator.llm.crosscheck import crosscheck_pairs
from validator.llm.fake import FakeProvider
from validator.quality import analyze_quality
from validator.scoring import Scorecard, assess

ROOT = Path(__file__).resolve().parent.parent
OPTIONS = LLMOptions(max_retries=0, max_requests=10, max_pair_requests=10)


def _score(
    tmp_path: Path,
    markdown: str,
    *,
    rules_only: bool = True,
    review: tuple[str, ...] = (),
    pairs: tuple[str, ...] = (),
    options: LLMOptions = OPTIONS,
    max_candidates: int | None = None,
) -> tuple[Scorecard, Built]:
    built = load(tmp_path, markdown)
    graph = built.graph
    if max_candidates is not None:
        graph = build_fact_graph(
            built.document, built.segmentation, built.lint, max_candidates=max_candidates
        )
    quality = analyze_quality(built.document, built.segmentation, built.lint, graph)
    analysis = crosscheck = None
    if not rules_only:
        analysis = review_requirements(
            built.document, built.segmentation, FakeProvider(*review), options
        )
        crosscheck = crosscheck_pairs(
            built.document, built.segmentation, graph, FakeProvider(*pairs), options
        )
        quality = quality.with_hypotheses(crosscheck.hypotheses)
    card = assess(
        built.segmentation,
        built.lint,
        graph,
        quality,
        analysis=analysis,
        crosscheck=crosscheck,
        mode="rules-only" if rules_only else "standard",
    )
    return card, built


def _doc(*requirements: str, tail: str = "") -> str:
    lines = "".join(f"* Ф-{n}. {text}\n" for n, text in enumerate(requirements, 1))
    return HEAD + lines + tail


def _status(card: Scorecard, rid: str) -> str:
    return next(r.status for r in card.requirements if r.requirement_id == rid)


CLEAN = "Курьер доставляет посылку адресату."
CONFLICT = ("Срок хранения брони: не более 3 дней.", "Срок хранения брони — 5 дней.")


# --- статусы требований и вердикт -----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        ("Система должна быстро подтверждать бронь.", "TST-01"),  # проверяемость
        ("Срок выдачи: TBD.", "CMP-01"),  # полнота
    ],
)
def test_major_testability_or_completeness_defect_blocks_development(
    tmp_path: Path, text: str, rule: str
) -> None:
    card, _ = _score(tmp_path, _doc(text, CLEAN))

    assert _status(card, "Ф-1") == "confirmed_major"
    assert _status(card, "Ф-2") == "clean_rules_only"
    assert card.verdict == "not_recommended"
    assert [d.rule_id for d in card.defects] == [rule]
    assert card.critical == tuple(d.finding_id for d in card.defects)


def test_minor_defects_only_allow_starting_after_fixes(tmp_path: Path) -> None:
    card, _ = _score(tmp_path, _doc("Бронь хранится около 3 дней.", CLEAN))

    assert _status(card, "Ф-1") == "confirmed_minor"
    assert card.verdict == "after_fixes"
    assert card.critical == ()


def test_major_defect_of_other_category_does_not_block(tmp_path: Path) -> None:
    card, _ = _score(tmp_path, _doc("Система хранит несколько броней читателя.", CLEAN))

    [defect] = card.defects
    assert (defect.severity, defect.category) == ("major", "ambiguity")
    assert _status(card, "Ф-1") == "confirmed_major"
    assert card.verdict == "after_fixes"


def test_confirmed_contradiction_blocks_and_comes_first(tmp_path: Path) -> None:
    card, _ = _score(tmp_path, _doc("Бронь хранится около 3 дней.", *CONFLICT, "Срок выдачи: TBD."))

    assert card.verdict == "not_recommended"
    assert [d.rule_id for d in card.defects] == ["CTR-01", "CMP-01", "AMB-02"]
    assert card.defects[0].contradiction is True
    assert {_status(card, "Ф-2"), _status(card, "Ф-3")} == {"confirmed_major"}


def test_priority_order_is_severity_then_category_then_document(tmp_path: Path) -> None:
    card, _ = _score(
        tmp_path,
        _doc(
            "Бронь хранится около 3 дней.",  # AMB-02, minor
            "Срок выдачи: TBD.",  # CMP-01, major, полнота
            "Система должна быстро подтверждать бронь.",  # TST-01, major, проверяемость
            "Она хранит историю броней.",  # AMB-04, minor
        ),
    )

    assert [d.rule_id for d in card.defects] == ["TST-01", "CMP-01", "AMB-02", "AMB-04"]


def test_potential_ctr02_is_for_review_not_a_confirmed_defect(tmp_path: Path) -> None:
    card, _ = _score(
        tmp_path,
        _doc("Срок хранения: 3 дня.")
        + "\n## Нефункциональные требования\n\n* Н-1. Срок хранения: 5 дней.\n",
    )

    assert card.defects == ()
    assert [r.source for r in card.review] == ["potential-contradiction"]
    assert card.review[0].item_id.startswith("CTR-02:")
    assert _status(card, "Ф-1") == "needs_review"
    assert card.verdict == "no_blockers"
    assert any("потенциальн" in note.casefold() for note in card.caveats)


@pytest.mark.parametrize(
    ("values", "contradiction", "verdict"),
    [
        (("до 5 броней", "3 брони"), False, "after_fixes"),  # совместимы: уточнение
        (("не более 3 дней", "5 дней"), True, "not_recommended"),  # не пересекаются
    ],
)
def test_con03_contradiction_status_follows_stage8(
    tmp_path: Path, values: tuple[str, str], contradiction: bool, verdict: str
) -> None:
    table = (
        "\n## Параметры\n\n| Параметр | Значение |\n| --- | --- |\n"
        f"| Лимит | {values[0]} |\n| Лимит | {values[1]} |\n"
    )
    card, _ = _score(tmp_path, _doc(CLEAN) + table)

    [defect] = card.defects
    assert defect.rule_id == "CON-03"  # подтверждённое замечание правила (раздел 4)
    assert defect.contradiction is contradiction
    assert card.verdict == verdict
    assert any(r.source == "potential-contradiction" for r in card.review) is not contradiction


def test_info_findings_are_not_defects(tmp_path: Path) -> None:
    card, _ = _score(
        tmp_path,
        _doc(
            "Читатель продлевает бронь книги на 7 дней в личном кабинете.",
            "Читатель продлевает бронь книги на 7 дней в мобильном кабинете.",
        ),
    )

    assert card.defects == ()
    assert len(card.info) == 1 and card.info[0].startswith("CON-02:")
    assert card.verdict == "no_blockers"


# --- гипотезы модели ------------------------------------------------------------------------


def test_llm_hypotheses_need_review_and_do_not_change_the_verdict(tmp_path: Path) -> None:
    review = findings_json(hypothesis("Ф-1", "доставляет посылку", category="testability"))

    card, _ = _score(
        tmp_path, _doc(CLEAN, "Читатель отменяет бронь."), rules_only=False, review=(review,)
    )

    assert card.defects == ()
    assert _status(card, "Ф-1") == "needs_review"
    assert _status(card, "Ф-2") == "clean_full"
    assert [r.source for r in card.review] == ["llm-review"]
    assert card.verdict == "no_blockers"


def test_model_confidence_does_not_order_the_review_list(tmp_path: Path) -> None:
    def run(first: float, second: float) -> list[str]:
        replies = (
            findings_json(hypothesis("Ф-1", "доставляет посылку", confidence=first)),
            findings_json(hypothesis("Ф-2", "отменяет бронь", confidence=second)),
        )
        card, _ = _score(
            tmp_path, _doc(CLEAN, "Читатель отменяет бронь."), rules_only=False, review=replies
        )
        return [r.requirement_ids[0] for r in card.review]

    assert run(0.1, 0.99) == run(0.99, 0.1) == ["Ф-1", "Ф-2"]


def test_rejected_candidates_are_counted_by_source(tmp_path: Path) -> None:
    bad = findings_json(hypothesis("Ф-1", "такой цитаты нет"))

    card, _ = _score(tmp_path, _doc(CLEAN), rules_only=False, review=(bad,))

    assert dict(card.rejected)["llm-review"] == 1
    assert card.review == ()


# --- покрытие и оговорки ----------------------------------------------------------------------


def test_rules_only_verdict_says_the_model_was_not_used(tmp_path: Path) -> None:
    card, _ = _score(tmp_path, _doc(CLEAN))

    assert card.verdict == "no_blockers"
    assert any("--rules-only" in note for note in card.caveats)
    assert not card.coverage.llm_used


def test_partial_budget_is_a_caveat_and_unreviewed_are_rules_only(tmp_path: Path) -> None:
    card, _ = _score(
        tmp_path,
        _doc(CLEAN, "Читатель отменяет бронь."),
        rules_only=False,
        options=replace(OPTIONS, max_requests=1),
    )

    assert _status(card, "Ф-1") == "clean_full"
    assert _status(card, "Ф-2") == "clean_rules_only"
    assert card.coverage.reviewed == 1
    assert any("1 из 2" in note for note in card.caveats)


def test_disabled_pair_check_and_truncation_are_caveats(tmp_path: Path) -> None:
    disabled, _ = _score(
        tmp_path, GRAPH_SPEC, rules_only=False, options=replace(OPTIONS, max_pair_requests=0)
    )
    truncated, _ = _score(tmp_path, GRAPH_SPEC, rules_only=False, max_candidates=2)

    assert any("LLM_MAX_PAIR_REQUESTS = 0" in note for note in disabled.caveats)
    assert any("ограничение отбора" in note.casefold() for note in truncated.caveats)
    assert truncated.coverage.truncated_pairs == 4


def test_requirement_outside_the_graph_is_unchecked(tmp_path: Path) -> None:
    built = load(tmp_path, _doc(CLEAN, "Читатель отменяет бронь."))
    graph = replace(
        built.graph,
        nodes=built.graph.nodes[:1],
        excluded=(ExcludedRequirement("Ф-2", "no_source", "Нет ссылок на источник."),),
        candidates=CandidateSelection(),
    )
    quality = analyze_quality(built.document, built.segmentation, built.lint, built.graph)

    card = assess(built.segmentation, built.lint, graph, quality, mode="rules-only")

    assert _status(card, "Ф-2") == "unchecked"
    assert any("вне графа" in note for note in card.caveats)


@pytest.mark.parametrize(
    "markdown",
    [
        "# Сервис\n\n## Введение\n\nОбщее описание библиотеки.\n",
        "# Сервис\n\n## Введение\n\nСистема должна хранить брони читателей.\n",
    ],
)
def test_document_without_requirements_is_not_assessable(tmp_path: Path, markdown: str) -> None:
    card, built = _score(tmp_path, markdown)

    assert card.requirements == ()
    assert card.verdict == "not_assessable"
    assert card.ambiguous == len(built.segmentation.ambiguous)


def test_structural_findings_are_counted_at_document_level(tmp_path: Path) -> None:
    table = "\n## Согласование\n\n| Роль | Дата |\n| --- | --- |\n| | |\n"

    card, _ = _score(tmp_path, _doc(CLEAN) + table)

    assert [d.rule_id for d in card.defects] == ["CMP-05"]
    assert card.structural == (card.defects[0].finding_id,)
    assert _status(card, "Ф-1") == "clean_rules_only"


# --- воспроизводимость ----------------------------------------------------------------------


def test_scorecard_json_is_stable_across_runs_and_hash_seeds(tmp_path: Path) -> None:
    spec = tmp_path / "spec.md"
    spec.write_text(_doc("Бронь хранится около 3 дней.", *CONFLICT, CLEAN), encoding="utf-8")
    script = (
        "import sys; from pathlib import Path\n"
        "from validator.ingest import load_document\n"
        "from validator.segmentation import segment_document\n"
        "from validator.lint import run_lint\n"
        "from validator.graph import build_fact_graph\n"
        "from validator.quality import analyze_quality\n"
        "from validator.scoring import assess\n"
        "d = load_document(Path(sys.argv[1])); s = segment_document(d); l = run_lint(d, s)\n"
        "g = build_fact_graph(d, s, l); q = analyze_quality(d, s, l, g)\n"
        "print(assess(s, l, g, q, mode='rules-only').to_json())\n"
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
    assert data["schema"] == "validator.scorecard/1"
    assert data["verdict"] == "not_recommended"
