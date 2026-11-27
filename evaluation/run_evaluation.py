"""Stage 10: воспроизводимая оценка валидатора.

    .venv\\Scripts\\python -m evaluation.run_evaluation [--official PATH] [--quick]

Без сети: модель не вызывается (смысловой LLM-слой помечается «не измерено»), сбои API
имитирует локальная HTTP-заглушка на 127.0.0.1. Результаты — evaluation/results/results.json
и EVALUATION.md (метрики — из этого прогона, разбор — из evaluation/analysis.md).
--official PATH — локальный прогон официального документа (--rules-only): в результаты
попадают только сводные числа, без текста документа. --quick — без замеров 1000/2000.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import tracemalloc
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.metrics import (  # noqa: E402
    DocumentLabels,
    LabelError,
    RuleCounts,
    check_anchors,
    defect_findings,
    fmt,
    load_labels,
    match,
    resolve,
    trap_hits,
)
from tests.builders import LI, H, P, T, write_docx, write_pdf, write_text  # noqa: E402
from validator.graph import build_fact_graph  # noqa: E402
from validator.ingest import load_document  # noqa: E402
from validator.lint import run_lint  # noqa: E402
from validator.quality import analyze_quality  # noqa: E402
from validator.scoring import assess  # noqa: E402
from validator.segmentation import segment_document  # noqa: E402

CORPUS = ROOT / "evaluation" / "corpus"
RESULTS = ROOT / "evaluation" / "results"
ANALYSIS = ROOT / "evaluation" / "analysis.md"
REPORT = ROOT / "EVALUATION.md"


# --- конвейер ----------------------------------------------------------------------------


@dataclass
class Run:
    document: Any
    segmentation: Any
    lint: Any
    graph: Any
    quality: Any
    scorecard: Any

    @property
    def findings(self) -> list[Any]:
        return [*self.lint.findings, *self.quality.findings]

    def fingerprint(self) -> str:
        data = self.lint.to_json() + self.quality.to_json() + self.scorecard.to_json()
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def signatures(self) -> set[tuple[str, tuple[str, ...]]]:
        return {(f.rule_id, tuple(sorted(f.requirement_ids))) for f in self.findings}


def pipeline(path: Path, timings: dict[str, float] | None = None) -> Run:
    def step(name: str, call: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        value = call()
        if timings is not None:
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - started
        return value

    document = step("чтение", lambda: load_document(path))
    segmentation = step("сегментация", lambda: segment_document(document))
    lint = step("правила", lambda: run_lint(document, segmentation))
    graph = step("граф", lambda: build_fact_graph(document, segmentation, lint))
    quality = step("этап 8", lambda: analyze_quality(document, segmentation, lint, graph))
    scorecard = step(
        "оценка", lambda: assess(segmentation, lint, graph, quality, mode="rules-only")
    )
    return Run(document, segmentation, lint, graph, quality, scorecard)


# --- корпус ------------------------------------------------------------------------------


def corpus_documents() -> list[tuple[Path, DocumentLabels]]:
    items = []
    for labels_path in sorted(CORPUS.glob("*/*.labels.json")):
        labels = load_labels(labels_path)
        doc = labels_path.with_name(labels.document)
        if labels.split != labels_path.parent.name:
            raise LabelError(f"{labels_path}: split не совпадает с папкой")
        items.append((doc, labels))
    return items


def evaluate_document(run: Run, labels: DocumentLabels) -> dict[str, Any]:
    check_anchors(labels, run.segmentation)
    findings = defect_findings(run.findings)
    result = match(labels.expected, findings, run.segmentation)
    traps = []
    for trap in labels.traps:
        hits = trap_hits(trap, findings, run.segmentation)
        traps.append(
            {"anchor": trap.anchor, "rules": list(trap.rules), "note": trap.note, "hits": hits}
        )
    contradictions = []
    for label in labels.contradictions:
        anchor = resolve(label.anchor, run.segmentation)
        pair = resolve(label.pair, run.segmentation) if label.pair else None
        found = [
            c
            for c in run.quality.contradictions
            if anchor & set(c.requirement_ids) and (pair is None or pair & set(c.requirement_ids))
        ]
        contradictions.append(
            {
                "anchor": label.anchor,
                "pair": label.pair,
                "expected": label.status,
                "observed": [c.status for c in found],
                "ok": any(c.status == label.status for c in found),
                "note": label.note,
            }
        )
    semantic = []
    for label in labels.semantic:
        anchor = resolve(label.anchor, run.segmentation)
        flagged = [f.rule_id for f in findings if anchor & set(f.requirement_ids)]
        semantic.append(
            {
                "anchor": label.anchor,
                "category": label.category,
                "note": label.note,
                "flagged_by_rules": flagged,
            }
        )
    graph = run.graph
    linked = {frozenset(p) for p in graph.candidates.linked} | {
        frozenset(p) for p in graph.candidates.duplicates
    }
    linked |= {frozenset((e.source_id, e.target_id)) for e in graph.edges}
    candidates = {frozenset((p.source_id, p.target_id)) for p in graph.candidates.pairs}
    related = []
    for a, b in labels.related_pairs:
        ids_a, ids_b = resolve(a, run.segmentation), resolve(b, run.segmentation)
        pairs = {frozenset((x, y)) for x in ids_a for y in ids_b if x != y}
        related.append(
            {"pair": [a, b], "candidate": bool(pairs & candidates), "linked": bool(pairs & linked)}
        )
    return {
        "document": labels.document,
        "split": labels.split,
        "requirements": len(run.segmentation.atomic_requirements),
        "ambiguous": len(run.segmentation.ambiguous),
        "findings": len(findings),
        "per_rule": {rule: vars(c) for rule, c in sorted(result.per_rule.items())},
        "tp": [
            {"label": lab.rule, "anchor": lab.anchor, "finding": f.id, "rule": f.rule_id}
            for lab, f in result.true_positives
        ],
        "fn": [
            {"rule": lab.rule, "anchor": lab.anchor, "note": lab.note}
            for lab in result.false_negatives
        ],
        "fp": [
            {
                "rule": f.rule_id,
                "finding": f.id,
                "requirements": list(f.requirement_ids),
                "quote": f.evidence[0].quote[:120] if f.evidence else "",
            }
            for f in result.false_positives
        ],
        "traps": traps,
        "contradictions": contradictions,
        "semantic": semantic,
        "related_pairs": related,
        "candidates": len(graph.candidates.pairs),
        "verdict": {"expected": labels.verdict, "observed": run.scorecard.verdict},
    }


def aggregate(documents: Sequence[dict[str, Any]]) -> dict[str, Any]:
    per_rule: dict[str, RuleCounts] = {}
    for doc in documents:
        for rule, counts in doc["per_rule"].items():
            per_rule.setdefault(rule, RuleCounts()).add(RuleCounts(**counts))
    total = RuleCounts()
    for counts in per_rule.values():
        total.add(counts)
    rules = {
        rule: {**vars(c), "precision": c.precision, "recall": c.recall}
        for rule, c in sorted(per_rule.items())
    }
    macro_p = [c.precision for c in per_rule.values() if c.precision is not None]
    macro_r = [c.recall for c in per_rule.values() if c.recall is not None]
    traps = [t for d in documents for t in d["traps"]]
    contradictions = [c for d in documents for c in d["contradictions"]]
    related = [r for d in documents for r in d["related_pairs"]]
    return {
        "documents": [d["document"] for d in documents],
        "requirements": sum(d["requirements"] for d in documents),
        "expected": total.tp + total.fn,
        "per_rule": rules,
        "micro": {**vars(total), "precision": total.precision, "recall": total.recall},
        "macro": {
            "precision": sum(macro_p) / len(macro_p) if macro_p else None,
            "recall": sum(macro_r) / len(macro_r) if macro_r else None,
            "rules_with_precision": len(macro_p),
            "rules_with_recall": len(macro_r),
        },
        "traps": {"total": len(traps), "passed": sum(1 for t in traps if not t["hits"])},
        "contradictions": {
            "total": len(contradictions),
            "ok": sum(1 for c in contradictions if c["ok"]),
        },
        "related_pairs": {
            "total": len(related),
            "found": sum(1 for r in related if r["candidate"] or r["linked"]),
        },
        "verdicts": {
            "total": len(documents),
            "ok": sum(1 for d in documents if d["verdict"]["expected"] == d["verdict"]["observed"]),
        },
    }


# --- форматы ------------------------------------------------------------------------------


def markdown_elements(text: str) -> list[Any]:
    """Простой разбор Markdown корпуса в элементы генераторов tests/builders.py."""
    elements: list[Any] = []
    table: list[tuple[str, ...]] = []
    for raw in [*text.splitlines(), ""]:
        line = raw.rstrip()
        if line.startswith("|"):
            cells = tuple(cell.strip() for cell in line.strip("|").split("|"))
            if not all(set(cell) <= {"-", " ", ":"} for cell in cells):
                table.append(cells)
            continue
        if table:
            elements.append(T(tuple(table)))
            table = []
        if not line:
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            elements.append(H(min(level, 3), line[level:].strip()))
        elif line.startswith("* "):
            elements.append(LI(line[2:]))
        else:
            elements.append(P(line))
    return elements


def write_format(source: Path, target: Path) -> Path:
    elements = markdown_elements(source.read_text(encoding="utf-8"))
    if target.suffix == ".docx":
        return write_docx(target, elements)
    if target.suffix == ".txt":
        return write_text(target, elements)
    pages = [elements[i : i + 18] for i in range(0, len(elements), 18)]
    return write_pdf(target, pages, header=None)


def evaluate_formats(documents: Sequence[tuple[Path, DocumentLabels]]) -> list[dict[str, Any]]:
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for source, labels in documents:
            reference = pipeline(source)
            for suffix in (".md", ".docx", ".pdf", ".txt"):
                path = (
                    source
                    if suffix == ".md"
                    else write_format(source, Path(tmp) / (source.stem + suffix))
                )
                run = reference if suffix == ".md" else pipeline(path)
                usable = []
                missing = 0
                for label in labels.expected:
                    try:
                        resolve(label.anchor, run.segmentation)
                        if label.pair:
                            resolve(label.pair, run.segmentation)
                        usable.append(label)
                    except LabelError:
                        missing += 1
                result = match(usable, defect_findings(run.findings), run.segmentation)
                tp = len(result.true_positives)
                out.append(
                    {
                        "document": labels.document,
                        "format": suffix.lstrip("."),
                        "requirements": len(run.segmentation.atomic_requirements),
                        "findings": len(defect_findings(run.findings)),
                        "labels_lost_in_format": missing,
                        "tp": tp,
                        "fn": len(result.false_negatives) + missing,
                        "fp": len(result.false_positives),
                        "same_findings_as_md": run.signatures() == reference.signatures(),
                        "diff_rules": sorted(
                            {rule for rule, _ in run.signatures() ^ reference.signatures()}
                        ),
                        "verdict": run.scorecard.verdict,
                    }
                )
    return out


# --- стабильность ---------------------------------------------------------------------------


_SEED_SCRIPT = (
    "import sys, hashlib; from pathlib import Path\n"
    "sys.path.insert(0, sys.argv[2])\n"
    "from evaluation.run_evaluation import pipeline\n"
    "print(pipeline(Path(sys.argv[1])).fingerprint())\n"
)


def reversed_lists(text: str) -> str:
    """Тот же документ с обратным порядком пунктов в каждом списке."""
    lines, out, block = text.splitlines(), [], []
    for line in [*lines, ""]:
        if line.startswith("* "):
            block.append(line)
            continue
        out += block[::-1]
        block = []
        out.append(line)
    return "\n".join(out[:-1]) + "\n"


def evaluate_stability(documents: Sequence[tuple[Path, DocumentLabels]]) -> list[dict[str, Any]]:
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for source, labels in documents:
            first, second = pipeline(source), pipeline(source)
            seeds = set()
            for seed in ("0", "1", "random"):
                env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONIOENCODING": "utf-8"}
                result = subprocess.run(
                    [sys.executable, "-X", "utf8", "-c", _SEED_SCRIPT, str(source), str(ROOT)],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    env=env,
                    check=True,
                )
                seeds.add(result.stdout.strip())
            shuffled = Path(tmp) / source.name
            shuffled.write_text(
                reversed_lists(source.read_text(encoding="utf-8")), encoding="utf-8"
            )
            permuted = pipeline(shuffled)
            out.append(
                {
                    "document": labels.document,
                    "repeat_identical": first.fingerprint() == second.fingerprint(),
                    "hash_seeds_identical": len(seeds) == 1 and first.fingerprint() in seeds,
                    "permutation_same_findings": permuted.signatures() == first.signatures(),
                    "permutation_diff_rules": sorted(
                        {rule for rule, _ in permuted.signatures() ^ first.signatures()}
                    ),
                    "permutation_same_verdict": permuted.scorecard.verdict
                    == first.scorecard.verdict,
                }
            )
    return out


# --- производительность ----------------------------------------------------------------------


def synthetic(count: int, adversarial: bool, seed: int = 7) -> str:
    rng = random.Random(seed)
    letters, vowels = "бвгдклмнпрстфхчш", "аеиоу"
    words = sorted(
        {
            "".join(rng.choice(letters) + rng.choice(vowels) for _ in range(3)) + "ка"
            for _ in range(3000)
        }
    )
    lines = []
    for i in range(1, count + 1):
        picked = [rng.choice(words) for _ in range(8)]
        if adversarial and i % 2 == 0 and i < count:
            picked += ["журнала", "архива"]
        lines.append(f"* Р-{i}. Система " + " ".join(picked) + ".")
    return "# Сервис\n\n## Функциональные требования\n\n" + "\n".join(lines) + "\n"


def evaluate_performance(quick: bool) -> list[dict[str, Any]]:
    sizes = [50, 200] if quick else [50, 200, 1000, 2000]
    cases = [(n, False) for n in sizes] + [(sizes[-1], True)]
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for count, adversarial in cases:
            path = Path(tmp) / f"perf_{count}_{int(adversarial)}.md"
            path.write_text(synthetic(count, adversarial), encoding="utf-8")
            timings: dict[str, float] = {}
            started = time.perf_counter()
            run = pipeline(path, timings)
            total = time.perf_counter() - started
            tracemalloc.start()
            pipeline(path)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            out.append(
                {
                    "requirements": count,
                    "adversarial": adversarial,
                    "seconds_total": round(total, 2),
                    "seconds_by_stage": {k: round(v, 2) for k, v in timings.items()},
                    "peak_mb": round(peak / 1e6, 1),
                    "candidates": len(run.graph.candidates.pairs),
                    "candidates_found": run.graph.candidates.found,
                }
            )
    return out


# --- сбои API --------------------------------------------------------------------------------


@contextmanager
def _env(**values: str) -> Iterator[dict[str, str]]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("LLM_", "GEMINI", "OPENAI", "ANTHROPIC", "OLLAMA"))
    }
    env.update(PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8", **values)
    yield env


def evaluate_api_failures(source: Path) -> list[dict[str, Any]]:
    from tests.llm_helpers import FAKE_KEY, Reply, json_reply, ollama_reply, stub_server

    review = ollama_reply('{"findings": []}')
    scenarios: list[tuple[str, Reply, dict[str, str], str]] = [
        ("таймаут", Reply(200, review.body, {}, delay=3.0), {"LLM_TIMEOUT_SECONDS": "1"}, "ollama"),
        (
            "429 с Retry-After (1 повтор)",
            json_reply(429, {"error": "rate"}, **{"Retry-After": "1"}),
            {"LLM_MAX_RETRIES": "1"},
            "ollama",
        ),
        ("HTTP 503", json_reply(503, {"error": "overloaded"}), {}, "ollama"),
        ("ответ не JSON", ollama_reply("{findings: ["), {}, "ollama"),
        ("пустой ответ", ollama_reply(""), {}, "ollama"),
        (
            "ключ в теле ошибки 500",
            json_reply(500, {"error": f"bad key {FAKE_KEY}"}),
            {"OPENAI_API_KEY": FAKE_KEY, "OPENAI_MODEL": "m"},
            "openai",
        ),
        (
            "ключ отклонён (401)",
            json_reply(401, {"error": f"invalid {FAKE_KEY}"}),
            {"OPENAI_API_KEY": FAKE_KEY, "OPENAI_MODEL": "m"},
            "openai",
        ),
    ]
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for title, reply, extra, provider in scenarios:
            report = Path(tmp) / "report.md"
            report.unlink(missing_ok=True)
            with stub_server(reply) as server:
                base = (
                    {"OLLAMA_BASE_URL": server.url, "OLLAMA_MODEL": "m"}
                    if provider == "ollama"
                    else {"LLM_BASE_URL": server.url}
                )
                settings = {
                    "LLM_PROVIDER": provider,
                    "LLM_MAX_RETRIES": "0",
                    "LLM_RPM": "10000",
                    **base,
                    **extra,
                }
                with _env(**settings) as env:
                    result = subprocess.run(
                        [sys.executable, "-X", "utf8", "-m", "validator", str(source), str(report)],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        env=env,
                        cwd=tmp,
                        timeout=120,
                    )
            leaked = FAKE_KEY in result.stdout + result.stderr or (
                report.exists() and FAKE_KEY in report.read_text(encoding="utf-8")
            )
            message = next(
                (line for line in result.stderr.splitlines() if line.startswith("Ошибка")), ""
            )
            out.append(
                {
                    "scenario": title,
                    "exit_code": result.returncode,
                    "report_written": report.exists(),
                    "key_leaked": leaked,
                    "message": message[:160],
                }
            )
    return out


# --- официальный документ --------------------------------------------------------------------


def evaluate_official(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    run = pipeline(path)
    card = run.scorecard
    return {
        "sha256_prefix": run.document.sha256[:12],
        "requirements": len(run.segmentation.atomic_requirements),
        "confirmed_defects": len(card.defects),
        "major_defects": sum(1 for d in card.defects if d.severity in ("critical", "major")),
        "confirmed_contradictions": card.contradictions_confirmed,
        "candidates": len(run.graph.candidates.pairs),
        "verdict": card.verdict,
    }


# --- отчёт -----------------------------------------------------------------------------------


def _meta() -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip()

    packages = ["pymupdf", "python-docx", "markdown-it-py", "python-dotenv"]
    return {
        "date": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "commit": git("describe", "--always", "--dirty"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": {name: metadata.version(name) for name in packages},
        "labels_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(CORPUS.glob("test/*"))
        },
    }


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(":---" for _ in header) + " |"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def render(results: dict[str, Any]) -> str:
    meta, corpus = results["meta"], results["corpus"]
    parts = [
        "# EVALUATION — оценка качества валидатора",
        "> Файл сгенерирован `python -m evaluation.run_evaluation`; числа ниже — результат этого "
        "прогона. Разбор — `evaluation/analysis.md`. Методика — `evaluation/README.md`.",
        "## 1. Конфигурация",
        _table(
            ("Параметр", "Значение"),
            [
                ("Дата", meta["date"]),
                ("Коммит", meta["commit"]),
                ("Python", meta["python"]),
                ("Платформа", meta["platform"]),
                ("Процессор", meta["processor"]),
                ("Ядер", meta["cpu_count"]),
                ("Зависимости", ", ".join(f"{k} {v}" for k, v in meta["packages"].items())),
                ("Режим", "детерминированный (--rules-only); модель не вызывалась"),
            ],
        ),
        "## 2. Корпус",
        _table(
            ("Документ", "Разбиение", "Требований", "Ожидаемых дефектов", "Ловушек"),
            [
                (
                    d["document"],
                    d["split"],
                    d["requirements"],
                    len(d["tp"]) + len(d["fn"]),
                    len(d["traps"]),
                )
                for d in corpus["documents"]
            ],
        ),
        "Хеши файлов разметки test (зафиксированы до первого прогона, см. analysis.md):\n\n"
        + "\n".join(
            f"- `{name}`: `{digest[:16]}…`" for name, digest in meta["labels_sha256"].items()
        ),
    ]
    for split in ("test", "dev"):
        agg = corpus["splits"][split]
        title = "held-out test (итоговые метрики)" if split == "test" else "dev (отладочный)"
        parts += [
            f"## 3{'a' if split == 'test' else 'b'}. Детерминированный слой — {title}",
            f"Документы: {', '.join(agg['documents'])}; атомарных требований: "
            f"{agg['requirements']}; ожидаемых дефектов: {agg['expected']}.",
            _table(
                ("Правило", "TP", "FP", "FN", "Precision", "Recall"),
                [
                    (rule, c["tp"], c["fp"], c["fn"], fmt(c["precision"]), fmt(c["recall"]))
                    for rule, c in agg["per_rule"].items()
                ],
            ),
            f"Микро-усреднение: TP {agg['micro']['tp']}, FP {agg['micro']['fp']}, FN "
            f"{agg['micro']['fn']}; precision {fmt(agg['micro']['precision'])}, recall "
            f"{fmt(agg['micro']['recall'])}. Макро-усреднение по правилам: precision "
            f"{fmt(agg['macro']['precision'])} ({agg['macro']['rules_with_precision']} правил), "
            f"recall {fmt(agg['macro']['recall'])} ({agg['macro']['rules_with_recall']} правил).",
            f"Ловушки без срабатывания: {agg['traps']['passed']} из {agg['traps']['total']}. "
            f"Статус противоречий этапа 8 совпал с разметкой: {agg['contradictions']['ok']} из "
            f"{agg['contradictions']['total']}. Связанные пары среди кандидатов или связей: "
            f"{agg['related_pairs']['found']} из {agg['related_pairs']['total']}. Вердикт этапа 9 "
            f"совпал: {agg['verdicts']['ok']} из {agg['verdicts']['total']}.",
        ]
    fp = [
        (d["document"], f["rule"], ", ".join(f["requirements"]), f["quote"])
        for d in corpus["documents"]
        for f in d["fp"]
    ]
    fn = [
        (d["document"], f["rule"], f["anchor"], f["note"])
        for d in corpus["documents"]
        for f in d["fn"]
    ]
    traps = [
        (d["document"], t["anchor"], ", ".join(t["rules"]), t["note"], ", ".join(t["hits"]))
        for d in corpus["documents"]
        for t in d["traps"]
        if t["hits"]
    ]
    semantic = [
        (
            d["document"],
            s["anchor"],
            s["category"],
            s["note"],
            ", ".join(s["flagged_by_rules"]) or "—",
        )
        for d in corpus["documents"]
        for s in d["semantic"]
    ]
    wrong_contradictions = [
        (d["document"], c["anchor"], c["expected"], ", ".join(c["observed"]) or "не найдено")
        for d in corpus["documents"]
        for c in d["contradictions"]
        if not c["ok"]
    ]
    parts += [
        "## 4. Ложные срабатывания (все документы)",
        _table(("Документ", "Правило", "Требования", "Цитата"), fp) if fp else "Нет.",
        "Сработавшие ловушки:\n\n"
        + (
            _table(("Документ", "Ловушка", "Правила", "Почему ловушка", "Замечания"), traps)
            if traps
            else "нет."
        ),
        "## 5. Пропуски (все документы)",
        _table(("Документ", "Правило", "Якорь", "Дефект"), fn) if fn else "Нет.",
        "Смысловые дефекты, которые правила по замыслу не ловят:\n\n"
        + _table(("Документ", "Якорь", "Категория", "Дефект", "Сработавшие правила"), semantic),
        "Противоречия с неверным статусом: "
        + (
            "\n\n" + _table(("Документ", "Якорь", "Ожидалось", "Получено"), wrong_contradictions)
            if wrong_contradictions
            else "нет."
        ),
        "## 6. Форматы",
        _table(
            (
                "Документ",
                "Формат",
                "Требований",
                "Замечаний",
                "TP",
                "FN",
                "FP",
                "Меток потеряно форматом",
                "Отличия от MD (правила)",
                "Вердикт",
            ),
            [
                (
                    f["document"],
                    f["format"],
                    f["requirements"],
                    f["findings"],
                    f["tp"],
                    f["fn"],
                    f["fp"],
                    f["labels_lost_in_format"],
                    ", ".join(f["diff_rules"]) or "нет",
                    f["verdict"],
                )
                for f in results["formats"]
            ],
        ),
        "## 7. Стабильность",
        "Перестановка — обратный порядок пунктов в каждом списке документа.",
        _table(
            (
                "Документ",
                "Повтор идентичен",
                "PYTHONHASHSEED 0/1/random",
                "Перестановка: отличия (правила)",
                "Перестановка: вердикт тот же",
            ),
            [
                (
                    s["document"],
                    "да" if s["repeat_identical"] else "нет",
                    "да" if s["hash_seeds_identical"] else "нет",
                    ", ".join(s["permutation_diff_rules"]) or "нет",
                    "да" if s["permutation_same_verdict"] else "нет",
                )
                for s in results["stability"]
            ],
        ),
        "## 8. Производительность и память",
        "Синтетические документы (по 8–10 случайных слов на требование); время — без "
        "трассировки памяти, пик памяти — отдельным прогоном с `tracemalloc`.",
        _table(
            (
                "Требований",
                "Случай",
                "Всего, с",
                "По этапам, с",
                "Пик памяти, МБ",
                "Кандидатов (найдено)",
            ),
            [
                (
                    p["requirements"],
                    "неблагоприятный" if p["adversarial"] else "обычный",
                    p["seconds_total"],
                    ", ".join(f"{k} {v}" for k, v in p["seconds_by_stage"].items()),
                    p["peak_mb"],
                    f"{p['candidates']} ({p['candidates_found']})",
                )
                for p in results["performance"]
            ],
        ),
        "## 9. Сбои API",
        "CLI в стандартном режиме против локальной HTTP-заглушки (127.0.0.1); реальная сеть не "
        "используется.",
        _table(
            ("Сценарий", "Код выхода", "Отчёт записан", "Ключ в выводе или отчёте", "Сообщение"),
            [
                (
                    a["scenario"],
                    a["exit_code"],
                    "да" if a["report_written"] else "нет",
                    "да" if a["key_leaked"] else "нет",
                    a["message"].replace("|", "/"),
                )
                for a in results["api"]
            ],
        ),
        "## 10. LLM-слой",
        "**Не измерено.** Смысловое качество гипотез модели требует реального ключа или локальной "
        "модели и разрешения владельца проекта; в этом прогоне модель не вызывалась. Механика "
        "(схема ответа, Evidence Gate, бюджеты, повторы, сбои) покрыта тестами `tests/` и "
        "разделом 9. Как измерить — `evaluation/README.md`.",
        "## 11. Официальный документ (локально, только сводные числа)",
    ]
    official = results.get("official")
    parts.append(
        _table(("Показатель", "Значение"), [(k, v) for k, v in official.items()])
        if official
        else "Не запускался (параметр --official не указан)."
    )
    if ANALYSIS.is_file():
        parts.append(ANALYSIS.read_text(encoding="utf-8").strip())
    return "\n\n".join(parts) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.run_evaluation")
    parser.add_argument("--official", type=Path, help="официальный документ (локально)")
    parser.add_argument("--quick", action="store_true", help="без замеров 1000/2000")
    args = parser.parse_args(argv)
    documents = corpus_documents()
    evaluated = [evaluate_document(pipeline(path), labels) for path, labels in documents]
    results: dict[str, Any] = {
        "meta": _meta(),
        "corpus": {
            "documents": evaluated,
            "splits": {
                split: aggregate([d for d in evaluated if d["split"] == split])
                for split in ("dev", "test")
            },
        },
        "formats": evaluate_formats(documents),
        "stability": evaluate_stability(documents),
        "performance": evaluate_performance(args.quick),
        "api": evaluate_api_failures(documents[0][0]),
        "official": evaluate_official(args.official),
        "llm": {"measured": False, "reason": "модель не вызывалась: нет ключа и разрешения"},
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "results.json").write_bytes(
        json.dumps(results, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    )
    REPORT.write_bytes(render(results).encode("utf-8"))
    test = results["corpus"]["splits"]["test"]["micro"]
    print(
        f"test: precision {fmt(test['precision'])}, recall {fmt(test['recall'])}; "
        f"отчёт: {REPORT.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
