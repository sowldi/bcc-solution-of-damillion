"""Паспорт прогона: что проверяли, когда, какой версией агента и какой моделью."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from validator.config import Settings
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
    document: Path, text: str, settings: Settings, *, llm_used: bool, started_at: datetime
) -> RunPassport:
    return RunPassport(
        document_name=document.name,
        document_path=str(document.resolve()),
        document_sha256=hashlib.sha256(document.read_bytes()).hexdigest(),
        document_chars=len(text),
        started_at=started_at,
        agent_version=agent_version(),
        provider=settings.provider,
        model=settings.model,
        llm_used=llm_used,
    )
