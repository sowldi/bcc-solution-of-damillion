"""Паспорт прогона: что проверяли, когда, какой версией агента и какой моделью."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from validator.config import Settings
from validator.model import Document
from validator.version import agent_version


@dataclass(frozen=True)
class RunPassport:
    document_name: str
    document_path: str
    document_sha256: str
    document_chars: int
    started_at: datetime
    agent_version: str
    provider: str
    model: str
    llm_used: bool


def build_passport(
    document: Document, settings: Settings, *, llm_used: bool, started_at: datetime
) -> RunPassport:
    return RunPassport(
        document_name=document.source.name,
        document_path=str(document.source.resolve()),
        document_sha256=document.sha256,
        document_chars=document.char_count,
        started_at=started_at,
        agent_version=agent_version(),
        provider=settings.provider,
        model=settings.model,
        llm_used=llm_used,
    )
