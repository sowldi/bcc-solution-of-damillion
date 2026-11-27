from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone

import pytest

from validator.passport import RunPassport
from validator.report import render_report


def _passport(started_at: datetime) -> RunPassport:
    return RunPassport(
        document_name="spec.md",
        document_path="C:/docs/spec.md",
        document_sha256="0" * 64,
        document_chars=10,
        started_at=started_at,
        agent_version="0.1.0+abc1234",
        provider="gemini",
        model="gemini-3.6-flash",
        llm_used=False,
    )


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (timedelta(hours=5), "2026-11-27 10:20:29 UTC+05:00"),
        (timedelta(0), "2026-11-27 10:20:29 UTC+00:00"),
        (timedelta(hours=-3, minutes=-30), "2026-11-27 10:20:29 UTC-03:30"),
    ],
)
def test_run_time_uses_explicit_utc_offset(offset: timedelta, expected: str) -> None:
    """Время прогона однозначно на любой ОС: смещение от UTC, а не локализованное имя пояса."""
    started = datetime(2026, 11, 27, 10, 20, 29, tzinfo=timezone(offset, "Кызылорда (зима)"))

    text = render_report(_passport(started))

    assert f"| Дата и время прогона | {expected} |" in text
    assert "Кызылорда" not in text


def test_table_cells_are_escaped() -> None:
    passport = replace(_passport(datetime(2026, 1, 1, tzinfo=UTC)), document_name="a|b\nc.md")

    text = render_report(passport)

    assert "a\\|b c.md" in text
