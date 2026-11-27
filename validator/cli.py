"""Командная строка: python -m validator <документ> <отчёт> [параметры]."""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from validator import __version__
from validator.config import PROVIDERS, Settings, load_settings
from validator.errors import (
    EXIT_INTERNAL_ERROR,
    EXIT_OK,
    InputError,
    ProviderError,
    ValidatorError,
)
from validator.evidence import ensure_source_unchanged
from validator.graph import FactGraph, build_fact_graph
from validator.ingest import SUPPORTED_SUFFIXES, load_document
from validator.lint import run_lint
from validator.model import Document
from validator.passport import build_passport
from validator.quality import QualityResult, analyze_quality
from validator.report import render_report, write_text_atomic
from validator.scoring import VERDICT_TITLES, Scorecard, assess
from validator.segmentation import Segmentation, segment_document

if TYPE_CHECKING:  # LLM-слой импортируется лениво — только в стандартном режиме
    from validator.llm import CrossCheckAnalysis, LLMAnalysis


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m validator",
        description="Статическая проверка документа требований с отчётом на русском языке.",
    )
    parser.add_argument(
        "input", type=Path, help=f"путь к документу требований ({', '.join(SUPPORTED_SUFFIXES)})"
    )
    parser.add_argument("output", type=Path, help="путь к файлу отчёта (Markdown)")
    parser.add_argument("--provider", choices=PROVIDERS, help="LLM-провайдер (иначе из .env)")
    parser.add_argument("--model", help="модель (иначе из .env)")
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="только детерминированные правила: настройки LLM не читаются, ключи и сеть не нужны",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _check_output(output: Path, document: Path) -> None:
    if output.exists() and output.is_dir():
        raise InputError(f"Путь отчёта указывает на папку, нужен файл: {output}")
    if output.resolve() == document.resolve():
        raise InputError("Путь отчёта совпадает с документом — документ был бы перезаписан.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.rules_only and (args.provider or args.model):
        parser.error("--rules-only несовместим с --provider и --model: в этом режиме LLM не нужен.")
    started_at = datetime.now().astimezone()
    try:
        # В режиме --rules-only настройки LLM не читаются вовсе: ни .env, ни ключи, ни сеть.
        settings = (
            None
            if args.rules_only
            else load_settings(provider_override=args.provider, model_override=args.model)
        )
        _check_output(args.output, args.input)
        document = load_document(args.input)
        for warning in document.warnings:
            print(f"Предупреждение: {warning}", file=sys.stderr)
        segmentation = segment_document(document)
        lint = run_lint(document, segmentation)
        graph = build_fact_graph(document, segmentation, lint)
        analysis: LLMAnalysis | None = None
        crosscheck: CrossCheckAnalysis | None = None
        if settings is not None:
            analysis, crosscheck = _run_llm(settings, document, segmentation, graph)
        quality = analyze_quality(document, segmentation, lint, graph)
        if crosscheck is not None:
            quality = quality.with_hypotheses(crosscheck.hypotheses)
        scorecard = assess(
            segmentation,
            lint,
            graph,
            quality,
            analysis=analysis,
            crosscheck=crosscheck,
            mode="rules-only" if args.rules_only else "standard",
        )
        passport = build_passport(
            document,
            settings,
            llm_used=analysis is not None,
            started_at=started_at,
            mode="rules-only" if args.rules_only else "standard",
        )
        if analysis is not None:  # в паспорте — провайдер, который действительно отвечал
            passport = replace(passport, provider=analysis.provider, model=analysis.model)
        report = render_report(
            passport,
            document,
            segmentation,
            lint,
            analysis,
            graph=graph,
            crosscheck=crosscheck,
            quality=quality,
            scorecard=scorecard,
        )
        # Доказательства привязаны к прочитанной версии файла: если файл изменился во время
        # проверки, отчёт был бы о другом документе — он не записывается.
        ensure_source_unchanged(document)
        write_text_atomic(args.output, report)
    except ValidatorError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        if isinstance(exc, ProviderError):
            print(
                "Отчёт не записан: анализ LLM не выполнен. Проверить документ только "
                "детерминированными правилами можно с параметром --rules-only.",
                file=sys.stderr,
            )
        return exc.exit_code
    except OSError as exc:
        print(f"Ошибка записи отчёта {args.output}: {exc.strerror or exc}", file=sys.stderr)
        return InputError.exit_code
    print(
        f"Требований: {len(segmentation.atomic_requirements)}, "
        f"неоднозначных фрагментов: {len(segmentation.ambiguous)}."
    )
    print(
        f"Замечаний: {len(lint.findings)} "
        f"(отклонено проверкой доказательств: {len(lint.rejected)})."
    )
    if analysis is not None:
        print(
            f"Гипотез LLM: {len(analysis.hypotheses)} (не подтверждены), "
            f"отклонено при разборе ответа: {len(analysis.rejected)}."
        )
    _print_graph(graph, crosscheck)
    _print_quality(quality)
    _print_verdict(scorecard)
    if lint.rejected:
        print(
            f"Предупреждение: {len(lint.rejected)} замечаний правил не прошли проверку "
            "доказательств и не включены в отчёт — подробности в отчёте.",
            file=sys.stderr,
        )
    print(f"Отчёт записан: {args.output}")
    return EXIT_OK


def _print_graph(graph: FactGraph, crosscheck: CrossCheckAnalysis | None) -> None:
    selection = graph.candidates
    kept = str(len(selection.pairs))
    if selection.truncated:
        kept += f" (найдено {selection.found}, ограничение отбора {selection.limit})"
    line = f"Связей: подтверждено {len(graph.edges)}"
    if crosscheck is not None:
        line += (
            f", гипотез LLM {len(crosscheck.hypotheses)} (не подтверждены); пар-кандидатов "
            f"{kept}, проверено моделью {crosscheck.requests}."
        )
    else:
        line += f"; пар-кандидатов {kept} (моделью не проверялись)."
    print(line)


def _print_quality(quality: QualityResult) -> None:
    scenarios = sum(1 for finding in quality.findings if finding.rule_id == "ACC-01")
    print(
        f"Противоречий: подтверждено {len(quality.confirmed)}, потенциальных "
        f"{len(quality.potential)}; критериев приёмки: {len(quality.acceptance)}, "
        f"сценариев без ожидаемого результата: {scenarios}."
    )


def _print_verdict(scorecard: Scorecard) -> None:
    major = sum(1 for d in scorecard.defects if d.severity in ("critical", "major"))
    print(
        f"Итог: {VERDICT_TITLES[scorecard.verdict]}. Подтверждённых дефектов: "
        f"{len(scorecard.defects)} (существенных {major}); на проверку человеку: "
        f"{len(scorecard.review)}."
    )


def _run_llm(
    settings: Settings, document: Document, segmentation: Segmentation, graph: FactGraph
) -> tuple[LLMAnalysis, CrossCheckAnalysis]:
    # Ленивый импорт: в режиме --rules-only LLM-слой не загружается вовсе.
    from validator import llm

    provider = llm.create_provider(settings)
    options = llm.LLMOptions.from_settings(settings)
    # Один ограничитель частоты на прогон: обзор и проверка пар делят лимит LLM_RPM.
    limiter = llm.RateLimiter(options.rpm) if options.rpm else None
    host = urlsplit(provider.endpoint).netloc or provider.endpoint
    print(
        f"LLM-обзор: требования ({len(segmentation.atomic_requirements)}) проверяются по одному "
        f"провайдером {provider.name} ({provider.model}, {host}); лимит запросов: "
        f"{options.max_requests}.",
        file=sys.stderr,
    )
    analysis = llm.review_requirements(document, segmentation, provider, options, limiter=limiter)
    print(
        f"Проверка пар: пар-кандидатов {len(graph.candidates.pairs)}, проверяются по одной "
        f"тем же провайдером; лимит запросов LLM_MAX_PAIR_REQUESTS: "
        f"{options.max_pair_requests}"
        + (" (проверка пар выключена)." if options.max_pair_requests == 0 else "."),
        file=sys.stderr,
    )
    crosscheck = llm.crosscheck_pairs(
        document, segmentation, graph, provider, options, limiter=limiter
    )
    return analysis, crosscheck


def _force_utf8_console() -> None:
    """Консоль Windows по умолчанию не UTF-8 — без этого кириллица превращается в «????»."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def run() -> None:
    _force_utf8_console()
    try:
        code = main()
    except Exception:  # непредвиденный сбой: показать трассировку, а не молча упасть
        print("Внутренняя ошибка агента:", file=sys.stderr)
        traceback.print_exc()
        code = EXIT_INTERNAL_ERROR
    sys.exit(code)
