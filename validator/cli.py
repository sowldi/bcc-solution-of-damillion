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
from validator.ingest import SUPPORTED_SUFFIXES, load_document
from validator.lint import run_lint
from validator.model import Document
from validator.passport import build_passport
from validator.report import render_report, write_text_atomic
from validator.segmentation import Segmentation, segment_document

if TYPE_CHECKING:  # LLM-слой импортируется лениво — только в стандартном режиме
    from validator.llm import LLMAnalysis


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
        analysis = None if settings is None else _run_llm(settings, document, segmentation)
        passport = build_passport(
            document,
            settings,
            llm_used=analysis is not None,
            started_at=started_at,
            mode="rules-only" if args.rules_only else "standard",
        )
        if analysis is not None:  # в паспорте — провайдер, который действительно отвечал
            passport = replace(passport, provider=analysis.provider, model=analysis.model)
        report = render_report(passport, document, segmentation, lint, analysis)
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
    if lint.rejected:
        print(
            f"Предупреждение: {len(lint.rejected)} замечаний правил не прошли проверку "
            "доказательств и не включены в отчёт — подробности в отчёте.",
            file=sys.stderr,
        )
    print(f"Отчёт записан: {args.output}")
    return EXIT_OK


def _run_llm(settings: Settings, document: Document, segmentation: Segmentation) -> LLMAnalysis:
    # Ленивый импорт: в режиме --rules-only LLM-слой не загружается вовсе.
    from validator import llm

    provider = llm.create_provider(settings)
    host = urlsplit(provider.endpoint).netloc or provider.endpoint
    print(
        f"LLM: требования ({len(segmentation.atomic_requirements)}) отправляются провайдеру "
        f"{provider.name} ({provider.model}, {host}).",
        file=sys.stderr,
    )
    options = llm.LLMOptions.from_settings(settings)
    return llm.analyze_with_llm(document, segmentation, provider, options)


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
