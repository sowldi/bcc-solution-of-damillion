"""Паспорт прогона: что проверяли, когда, какой версией агента, в каком режиме и какой моделью."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from validator.config import Settings
from validator.model import Document
from validator.version import agent_version

# standard — полный прогон (настройки LLM проверяются); rules-only — только детерминированные
# правила: настройки LLM не читаются, ключи и сеть не нужны.
Mode = Literal["standard", "rules-only"]


@dataclass(frozen=True)
class RunPassport:
    document_name: str
    document_path: str
    document_sha256: str
    document_chars: int
    started_at: datetime
    agent_version: str
    provider: str | None  # None — настройки LLM не загружались (режим rules-only)
    model: str | None
    llm_used: bool
    mode: Mode = "standard"


def build_passport(
    document: Document,
    settings: Settings | None,
    *,
    llm_used: bool,
    started_at: datetime,
    mode: Mode = "standard",
) -> RunPassport:
    return RunPassport(
        document_name=document.source.name,
        document_path=str(document.source.resolve()),
        document_sha256=document.sha256,
        document_chars=document.char_count,
        started_at=started_at,
        agent_version=agent_version(),
        provider=settings.provider if settings else None,
        model=settings.model if settings else None,
        llm_used=llm_used,
        mode=mode,
    )
