from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

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
