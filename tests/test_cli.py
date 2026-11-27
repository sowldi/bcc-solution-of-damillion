from __future__ import annotations

import hashlib
import socket
from pathlib import Path

import pytest

from tests.builders import (
    SAMPLE,
    Writer,
    write_docx,
    write_markdown,
    write_sample_pdf,
    write_text,
)
from tests.conftest import RunCli
from validator.cli import main
from validator.errors import EXIT_CONFIG_ERROR, EXIT_INPUT_ERROR, EXIT_OK


def test_end_to_end_command_writes_report(run_cli: RunCli, sample_md: Path, tmp_path: Path) -> None:
    report = tmp_path / "reports" / "report.md"

    result = run_cli(str(sample_md), str(report))

    assert result.returncode == EXIT_OK, result.stderr
    assert report.is_file()
    assert str(report) in result.stdout


def test_report_contains_run_passport(sample_md: Path, tmp_path: Path) -> None:
    report = tmp_path / "out" / "report.md"

    assert main([str(sample_md), str(report)]) == EXIT_OK

    text = report.read_text(encoding="utf-8")
    assert "Паспорт прогона" in text
    assert "spec.md" in text
    assert hashlib.sha256(sample_md.read_bytes()).hexdigest() in text
    assert "Версия агента" in text
    assert "Модель" in text
    assert "Дата" in text


def test_creates_missing_output_directories(sample_md: Path, tmp_path: Path) -> None:
    report = tmp_path / "a" / "b" / "c" / "report.md"

    assert main([str(sample_md), str(report)]) == EXIT_OK
    assert report.is_file()


def test_missing_input_file_is_input_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main([str(tmp_path / "nope.md"), str(tmp_path / "report.md")])

    assert code == EXIT_INPUT_ERROR
    err = capsys.readouterr().err
    assert "не найден" in err
    assert not (tmp_path / "report.md").exists()


def test_input_directory_is_input_error(tmp_path: Path) -> None:
    assert main([str(tmp_path), str(tmp_path / "report.md")]) == EXIT_INPUT_ERROR


def test_unsupported_extension_is_input_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "spec.xlsx"
    src.write_bytes(b"data")

    assert main([str(src), str(tmp_path / "report.md")]) == EXIT_INPUT_ERROR
    assert "формат" in capsys.readouterr().err.lower()


def test_empty_document_is_input_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "empty.md"
    src.write_text("   \n\n", encoding="utf-8")

    assert main([str(src), str(tmp_path / "report.md")]) == EXIT_INPUT_ERROR
    assert "пуст" in capsys.readouterr().err


def test_output_same_as_input_is_refused(sample_md: Path) -> None:
    original = sample_md.read_bytes()

    assert main([str(sample_md), str(sample_md)]) == EXIT_INPUT_ERROR
    assert sample_md.read_bytes() == original


def test_output_pointing_to_directory_is_refused(sample_md: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "already_dir"
    out_dir.mkdir()

    assert main([str(sample_md), str(out_dir)]) == EXIT_INPUT_ERROR


def test_invalid_config_value_is_config_error(
    sample_md: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / ".env").write_text("LLM_RPM=много\n", encoding="utf-8")

    assert main([str(sample_md), str(tmp_path / "report.md")]) == EXIT_CONFIG_ERROR
    assert "LLM_RPM" in capsys.readouterr().err


def test_cli_without_arguments_shows_usage(run_cli: RunCli) -> None:
    result = run_cli()

    assert result.returncode == EXIT_INPUT_ERROR
    assert "usage" in result.stderr.lower() or "использование" in result.stderr.lower()


def test_utf8_bom_document_is_read(tmp_path: Path) -> None:
    src = tmp_path / "bom.md"
    src.write_bytes("﻿# Документ\n\nТекст требования.\n".encode())
    report = tmp_path / "report.md"

    assert main([str(src), str(report)]) == EXIT_OK
    assert "bom.md" in report.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("writer", "name", "title"),
    [
        (write_markdown, "spec.md", "Markdown"),
        (write_text, "spec.txt", "Текст (TXT)"),
        (write_docx, "spec.docx", "Word (DOCX)"),
        (write_sample_pdf, "spec.pdf", "PDF"),
    ],
    ids=["md", "txt", "docx", "pdf"],
)
def test_command_reads_every_supported_format(
    run_cli: RunCli, tmp_path: Path, writer: Writer, name: str, title: str
) -> None:
    source = writer(tmp_path / name, SAMPLE)
    report = tmp_path / "out" / f"{name}.report.md"

    result = run_cli(str(source), str(report))

    assert result.returncode == EXIT_OK, result.stderr
    text = report.read_text(encoding="utf-8")
    assert "## 2. Извлечение текста" in text
    assert f"| Формат | {title} |" in text
    assert "## 3. Нарезка требований" in text
    assert "| R-1 | R-1 |" in text


def test_reading_warnings_are_printed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "legacy.txt"
    src.write_bytes("Текст в старой кодировке.\n".encode("cp1251"))

    assert main([str(src), str(tmp_path / "report.md")]) == EXIT_OK

    err = capsys.readouterr().err
    assert "Предупреждение" in err
    assert "Windows-1251" in err


def test_corrupted_pdf_is_input_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "broken.pdf"
    src.write_bytes(b"%PDF-1.7\n" + bytes(range(256)) * 4)

    assert main([str(src), str(tmp_path / "report.md")]) == EXIT_INPUT_ERROR
    assert "PDF повреждён" in capsys.readouterr().err
    assert not (tmp_path / "report.md").exists()


# --- режим --rules-only -----------------------------------------------------------------

LINT_SPEC = (
    "# Сервис бронирования книг\n\n## Функциональные требования\n\n"
    "* Ф-1. Каталог книг должен быть удобным.\n* Ф-2. Срок брони: TBD.\n"
)


def _spec(tmp_path: Path) -> Path:
    path = tmp_path / "spec.md"
    path.write_text(LINT_SPEC, encoding="utf-8")
    return path


def test_rules_only_works_without_credentials(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "report.md"

    assert main([str(_spec(tmp_path)), str(report), "--rules-only"]) == EXIT_OK

    text = report.read_text(encoding="utf-8")
    assert "## 4. Детерминированная проверка" in text
    assert "| Режим | только детерминированные правила (--rules-only) |" in text
    assert "| Модель | не используется (режим --rules-only) |" in text
    assert "TST-01" in text
    assert "CMP-01" in text
    out = capsys.readouterr().out
    assert "Замечаний: 2" in out


def test_rules_only_ignores_invalid_llm_config(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("LLM_PROVIDER=нет-такого\nLLM_RPM=много\n", encoding="utf-8")
    spec = _spec(tmp_path)

    assert main([str(spec), str(tmp_path / "rules.md"), "--rules-only"]) == EXIT_OK
    assert main([str(spec), str(tmp_path / "full.md")]) == EXIT_CONFIG_ERROR


def test_rules_only_never_loads_llm_settings_or_opens_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("в режиме --rules-only нет обращений к настройкам LLM и сети")

    monkeypatch.setattr("validator.cli.load_settings", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)

    assert main([str(_spec(tmp_path)), str(tmp_path / "report.md"), "--rules-only"]) == EXIT_OK


def test_rules_only_conflicts_with_llm_options(run_cli: RunCli, tmp_path: Path) -> None:
    spec = _spec(tmp_path)

    for option in (["--provider", "gemini"], ["--model", "any"]):
        result = run_cli(str(spec), str(tmp_path / "report.md"), "--rules-only", *option)
        assert result.returncode == EXIT_INPUT_ERROR
        assert "--rules-only" in result.stderr
    assert not (tmp_path / "report.md").exists()


def test_rules_only_input_errors_keep_exit_codes(tmp_path: Path) -> None:
    missing = main([str(tmp_path / "nope.md"), str(tmp_path / "r.md"), "--rules-only"])
    empty = tmp_path / "empty.md"
    empty.write_text("\n", encoding="utf-8")

    assert missing == EXIT_INPUT_ERROR
    assert main([str(empty), str(tmp_path / "r.md"), "--rules-only"]) == EXIT_INPUT_ERROR


def test_standard_mode_also_runs_deterministic_rules(tmp_path: Path) -> None:
    report = tmp_path / "report.md"

    assert main([str(_spec(tmp_path)), str(report)]) == EXIT_OK

    text = report.read_text(encoding="utf-8")
    assert "## 4. Детерминированная проверка" in text
    assert "| Режим | стандартный |" in text
    assert "не вызывалась (настроено: gemini" in text


@pytest.mark.parametrize(
    ("writer", "name"),
    [
        (write_markdown, "spec.md"),
        (write_text, "spec.txt"),
        (write_docx, "spec.docx"),
        (write_sample_pdf, "spec.pdf"),
    ],
    ids=["md", "txt", "docx", "pdf"],
)
def test_rules_only_command_reads_every_supported_format(
    run_cli: RunCli, tmp_path: Path, writer: Writer, name: str
) -> None:
    source = writer(tmp_path / name, SAMPLE)
    report = tmp_path / "out" / f"{name}.report.md"

    result = run_cli(str(source), str(report), "--rules-only")

    assert result.returncode == EXIT_OK, result.stderr
    text = report.read_text(encoding="utf-8")
    assert "## 4. Детерминированная проверка" in text
    assert "Замечаний:" in result.stdout
