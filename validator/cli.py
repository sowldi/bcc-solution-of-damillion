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
from validator.passport import build_passport
from validator.report import render_report, write_text_atomic

# Форматы, которые агент уже умеет читать. PDF и DOCX добавляются на этапе 1.
SUPPORTED_SUFFIXES = (".md", ".markdown", ".txt")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m validator",
        description="Статическая проверка документа требований с отчётом на русском языке.",
    )
    parser.add_argument("input", type=Path, help="путь к документу требований")
    parser.add_argument("output", type=Path, help="путь к файлу отчёта (Markdown)")
    parser.add_argument("--provider", choices=PROVIDERS, help="LLM-провайдер (иначе из .env)")
    parser.add_argument("--model", help="модель (иначе из .env)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _read_document(path: Path) -> str:
    if not path.exists():
        raise InputError(f"Документ не найден: {path}")
    if not path.is_file():
        raise InputError(f"Ожидался файл, а это папка: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise InputError(
            f"Неподдерживаемый формат «{path.suffix or 'без расширения'}». "
            f"Поддерживаются: {', '.join(SUPPORTED_SUFFIXES)}."
        )
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise InputError(f"Документ не в кодировке UTF-8: {path}") from None
    except OSError as exc:
        raise InputError(f"Не удалось прочитать документ {path}: {exc.strerror}") from None
    if not text.strip():
        raise InputError(f"Документ пуст: {path}")
    return text


def _check_output(output: Path, document: Path) -> None:
    if output.exists() and output.is_dir():
        raise InputError(f"Путь отчёта указывает на папку, нужен файл: {output}")
    if output.resolve() == document.resolve():
        raise InputError("Путь отчёта совпадает с документом — документ был бы перезаписан.")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    started_at = datetime.now().astimezone()
    try:
        settings = load_settings(provider_override=args.provider, model_override=args.model)
        _check_output(args.output, args.input)
        text = _read_document(args.input)
        passport = build_passport(args.input, text, settings, llm_used=False, started_at=started_at)
        write_text_atomic(args.output, render_report(passport))
    except ValidatorError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return exc.exit_code
    except OSError as exc:
        print(f"Ошибка записи отчёта {args.output}: {exc.strerror or exc}", file=sys.stderr)
        return InputError.exit_code
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
