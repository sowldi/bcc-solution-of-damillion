from __future__ import annotations

import hashlib
import os
import socket
import subprocess
import sys
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
from tests.conftest import AGENT_ENV_VARS, PROJECT_ROOT, RunCli
from tests.graph_helpers import GRAPH_SPEC, relation, relations_json
from tests.llm_helpers import (
    FAKE_KEY,
    StubServer,
    by_task,
    findings_json,
    hypothesis,
    json_reply,
    ollama_reply,
    stub_server,
)
from validator.cli import main
from validator.errors import (
    EXIT_CONFIG_ERROR,
    EXIT_INPUT_ERROR,
    EXIT_OK,
    EXIT_PROVIDER_ERROR,
)
from validator.llm.fake import FakeProvider


@pytest.fixture
def offline_llm(monkeypatch: pytest.MonkeyPatch) -> list[FakeProvider]:
    """Стандартный режим без сети: фабрика провайдеров отдаёт FakeProvider."""
    created: list[FakeProvider] = []

    def factory(settings: object, **_: object) -> FakeProvider:
        created.append(FakeProvider('{"findings": []}'))
        return created[-1]

    monkeypatch.setattr("validator.llm.create_provider", factory)
    monkeypatch.setenv("LLM_RPM", "10000")  # ограничитель частоты не нужен заглушке
    return created


def _ollama_env(server: StubServer, **extra: str) -> dict[str, str]:
    """Настройки для настоящего HTTP-вызова к локальной заглушке, имитирующей Ollama."""
    return {
        "LLM_PROVIDER": "ollama",
        "OLLAMA_BASE_URL": server.url,
        "OLLAMA_MODEL": "test-model",
        "LLM_MAX_RETRIES": "0",
        # По запросу на требование: без высокого лимита частоты тест ждал бы 6 с на запрос.
        "LLM_RPM": "10000",
        # Эти тесты проверяют HTTP-путь обзора (этап 6): заглушка отвечает в формате обзора.
        # Проверку пар (этап 7) включают тесты ниже, с ответом по виду запроса (by_task).
        "LLM_MAX_PAIR_REQUESTS": "0",
        **extra,
    }


def test_end_to_end_command_writes_report(run_cli: RunCli, sample_md: Path, tmp_path: Path) -> None:
    report = tmp_path / "reports" / "report.md"
    reply = ollama_reply(findings_json(hypothesis("R-1", "хранит заказ")))

    with stub_server(reply) as server:
        result = run_cli(str(sample_md), str(report), extra_env=_ollama_env(server))

    assert result.returncode == EXIT_OK, result.stderr
    assert report.is_file()
    assert str(report) in result.stdout
    [received] = server.received
    assert received.path == "/api/chat"
    text = report.read_text(encoding="utf-8")
    assert "| Модель | ollama / test-model |" in text
    assert "## 5. Анализ LLM (гипотезы, не подтверждены)" in text
    assert "«хранит заказ»" in text
    assert "не подтверждено" in text
    assert "Гипотез LLM: 1 (не подтверждены)" in result.stdout


@pytest.mark.usefixtures("offline_llm")
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


@pytest.mark.usefixtures("offline_llm")
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


@pytest.mark.usefixtures("offline_llm")
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
    reply = ollama_reply(findings_json(hypothesis("R-1", "хранит настройку порога")))

    with stub_server(reply) as server:
        result = run_cli(str(source), str(report), extra_env=_ollama_env(server))

    assert result.returncode == EXIT_OK, result.stderr
    text = report.read_text(encoding="utf-8")
    assert "## 2. Извлечение текста" in text
    assert f"| Формат | {title} |" in text
    assert "## 3. Нарезка требований" in text
    assert "| R-1 | R-1 |" in text
    assert "## 4. Детерминированная проверка" in text
    assert "«хранит настройку порога»" in text


@pytest.mark.usefixtures("offline_llm")
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


def test_standard_mode_runs_rules_and_llm(tmp_path: Path, offline_llm: list[FakeProvider]) -> None:
    report = tmp_path / "report.md"

    assert main([str(_spec(tmp_path)), str(report)]) == EXIT_OK

    text = report.read_text(encoding="utf-8")
    assert "## 4. Детерминированная проверка" in text
    assert "| Режим | стандартный |" in text
    assert "| Модель | fake / fake-model |" in text
    assert "## 5. Анализ LLM" in text
    [provider] = offline_llm
    assert len(provider.requests) == 2  # по запросу на каждое из двух требований


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


# --- стандартный режим: LLM-слой ---------------------------------------------------------


def test_rules_only_does_not_import_llm_layer(run_cli: RunCli, tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    script = (
        "import sys; from validator.cli import main; "
        f"code = main([{str(spec)!r}, {str(tmp_path / 'r.md')!r}, '--rules-only']); "
        "print('llm загружен' if 'validator.llm' in sys.modules else 'llm не загружен'); "
        "sys.exit(code)"
    )
    env = {k: v for k, v in os.environ.items() if k not in AGENT_ENV_VARS}
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )

    assert result.returncode == EXIT_OK, result.stderr
    assert "llm не загружен" in result.stdout


def test_standard_mode_without_key_stops_with_config_error(run_cli: RunCli, tmp_path: Path) -> None:
    report = tmp_path / "report.md"

    result = run_cli(str(_spec(tmp_path)), str(report))

    assert result.returncode == EXIT_CONFIG_ERROR
    assert "GEMINI_API_KEY" in result.stderr
    assert "--rules-only" in result.stderr
    assert not report.exists()


def test_provider_failure_is_reported_not_masked(run_cli: RunCli, tmp_path: Path) -> None:
    report = tmp_path / "report.md"

    with stub_server(json_reply(503, {"error": "overloaded"})) as server:
        result = run_cli(str(_spec(tmp_path)), str(report), extra_env=_ollama_env(server))

    assert result.returncode == EXIT_PROVIDER_ERROR
    assert "503" in result.stderr
    assert "--rules-only" in result.stderr
    assert not report.exists()


def test_invalid_model_answer_is_reported_not_masked(run_cli: RunCli, tmp_path: Path) -> None:
    report = tmp_path / "report.md"

    with stub_server(ollama_reply("Я не могу ответить в JSON.")) as server:
        result = run_cli(str(_spec(tmp_path)), str(report), extra_env=_ollama_env(server))

    assert result.returncode == EXIT_PROVIDER_ERROR
    assert "JSON" in result.stderr
    assert not report.exists()


def test_rejected_key_never_appears_in_output(run_cli: RunCli, tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    denied = json_reply(401, {"error": {"message": f"invalid key {FAKE_KEY}"}})

    with stub_server(denied) as server:
        env = {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": FAKE_KEY,
            "OPENAI_MODEL": "test-model",
            "LLM_BASE_URL": f"{server.url}/v1",
            "LLM_MAX_RETRIES": "3",
        }
        result = run_cli(str(_spec(tmp_path)), str(report), extra_env=env)

    assert result.returncode == EXIT_PROVIDER_ERROR
    assert len(server.received) == 1  # авторизация не повторяется
    assert server.received[0].headers["Authorization"] == f"Bearer {FAKE_KEY}"
    assert FAKE_KEY not in result.stderr + result.stdout
    assert "OPENAI_API_KEY" in result.stderr


def test_unknown_model_is_configuration_error(run_cli: RunCli, tmp_path: Path) -> None:
    missing = json_reply(404, {"error": "model 'test-model' not found"})

    with stub_server(missing) as server:
        result = run_cli(
            str(_spec(tmp_path)), str(tmp_path / "r.md"), extra_env=_ollama_env(server)
        )

    assert result.returncode == EXIT_CONFIG_ERROR
    assert "OLLAMA_MODEL" in result.stderr


def test_document_changed_during_run_is_not_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import validator.cli as cli

    spec = _spec(tmp_path)
    real_run_lint = cli.run_lint

    def edit_then_lint(*args: object, **kwargs: object) -> object:
        spec.write_text(LINT_SPEC + "* Ф-3. Добавлено во время проверки.\n", encoding="utf-8")
        return real_run_lint(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cli, "run_lint", edit_then_lint)
    report = tmp_path / "report.md"

    assert main([str(spec), str(report), "--rules-only"]) == EXIT_INPUT_ERROR
    assert "изменён" in capsys.readouterr().err
    assert not report.exists()


def test_partial_llm_failure_is_reported_with_progress(run_cli: RunCli, tmp_path: Path) -> None:
    """Stage 6.1 (E): первое требование проверено, на втором — сбой. Отчёт не пишется,
    код 4 сохраняется, сообщение называет требование и сколько успели проверить."""
    report = tmp_path / "report.md"
    replies = (ollama_reply('{"findings": []}'), json_reply(503, {"error": "overloaded"}))

    with stub_server(*replies) as server:
        result = run_cli(str(_spec(tmp_path)), str(report), extra_env=_ollama_env(server))

    assert result.returncode == EXIT_PROVIDER_ERROR
    assert len(server.received) == 2
    assert "Ф-2" in result.stderr
    assert "проверено 1 из 2" in result.stderr
    assert not report.exists()


# --- этап 7: Fact Graph и проверка пар ----------------------------------------------------

PAIR_CONFLICT = relation(
    "Ф-3", "не хранит бронь книги дольше 3 дней", "Ф-4", "хранит бронь книги 5 дней"
)


def _graph_spec(tmp_path: Path) -> Path:
    path = tmp_path / "graph.md"
    path.write_text(GRAPH_SPEC, encoding="utf-8")
    return path


def test_standard_run_checks_candidate_pairs_and_reports_the_graph(
    run_cli: RunCli, tmp_path: Path
) -> None:
    report = tmp_path / "report.md"
    route = by_task(ollama_reply('{"findings": []}'), ollama_reply(relations_json(PAIR_CONFLICT)))

    with stub_server(route=route) as server:
        env = _ollama_env(server, LLM_MAX_PAIR_REQUESTS="10")
        result = run_cli(str(_graph_spec(tmp_path)), str(report), extra_env=env)

    assert result.returncode == EXIT_OK, result.stderr
    pair_requests = [r for r in server.received if b"relations" in r.body]
    assert len(pair_requests) == 6  # все пары-кандидаты, каждая один раз
    text = report.read_text(encoding="utf-8")
    assert "## 6. Связи между требованиями (Fact Graph)" in text
    assert "`Ф-2` зависит от `Ф-1`" in text
    assert "Возможное противоречие: `Ф-3` и `Ф-4`" in text
    assert "Связей: подтверждено 1, гипотез LLM 1 (не подтверждены)" in result.stdout
    assert "пар-кандидатов 6, проверено моделью 6" in result.stdout
    assert "Проверка пар:" in result.stderr


def test_pair_check_failure_stops_without_a_report(run_cli: RunCli, tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    route = by_task(ollama_reply('{"findings": []}'), json_reply(503, {"error": "overloaded"}))

    with stub_server(route=route) as server:
        env = _ollama_env(server, LLM_MAX_PAIR_REQUESTS="10")
        result = run_cli(str(_graph_spec(tmp_path)), str(report), extra_env=env)

    assert result.returncode == EXIT_PROVIDER_ERROR
    assert "Проверка пар прервана на паре Ф-3 — Ф-4" in result.stderr
    assert not report.exists()


def test_zero_pair_budget_sends_only_review_requests(run_cli: RunCli, tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    route = by_task(ollama_reply('{"findings": []}'), json_reply(500, {"error": "unexpected"}))

    with stub_server(route=route) as server:
        result = run_cli(str(_graph_spec(tmp_path)), str(report), extra_env=_ollama_env(server))

    assert result.returncode == EXIT_OK, result.stderr
    assert not any(b"relations" in r.body for r in server.received)
    assert "Проверка пар моделью отключена (LLM_MAX_PAIR_REQUESTS = 0)" in report.read_text(
        encoding="utf-8"
    )


def test_rules_only_builds_the_deterministic_graph(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "report.md"

    assert main([str(_graph_spec(tmp_path)), str(report), "--rules-only"]) == EXIT_OK

    text = report.read_text(encoding="utf-8")
    assert "`Ф-2` зависит от `Ф-1`" in text
    assert "Пары-кандидаты моделью не проверялись (режим --rules-only)" in text
    assert "Связей: подтверждено 1" in capsys.readouterr().out
