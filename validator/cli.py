"""Командная строка: python -m validator <документ> <отчёт> [параметры]."""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from validator import __version__
from validator.config import PROVIDERS, load_settings
from validator.errors import EXIT_INTERNAL_ERROR, EXIT_OK, InputError, ValidatorError
from validator.ingest import SUPPORTED_SUFFIXES, load_document
from validator.lint import run_lint
from validator.passport import build_passport
from validator.report import render_report, write_text_atomic
from validator.segmentation import segment_document


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
        passport = build_passport(
            document,
            settings,
            llm_used=False,
            started_at=started_at,
            mode="rules-only" if args.rules_only else "standard",
        )
        write_text_atomic(args.output, render_report(passport, document, segmentation, lint))
    except ValidatorError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
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
    if lint.rejected:
        print(
            f"Предупреждение: {len(lint.rejected)} замечаний правил не прошли проверку "
            "доказательств и не включены в отчёт — подробности в отчёте.",
            file=sys.stderr,
        )
    print(f"Отчёт записан: {args.output}")
    return EXIT_OK


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
