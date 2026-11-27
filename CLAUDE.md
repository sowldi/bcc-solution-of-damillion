# CLAUDE.md — инструкции для Claude Code в этом репозитории

Проект: **AI Requirements Validator** — Python CLI-агент статической проверки документа
требований (ISO/IEC/IEEE 29148, IEEE 830) с доказательными замечаниями (Proof-Carrying
Findings). Хакатон BCC-HUB × MNU FinTech.

**Перед любой работой прочитай по порядку:**

1. [HANDOFF.md](HANDOFF.md) — правила работы, проверка окружения, порядок закрытия этапа.
2. [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) — назначение, архитектура, инварианты, история
   Stage 0–8.
3. [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md) — задания Stage 9, 10, 11.

Главное:

- Stage 0–8 завершены. Следующий — **Stage 9**; этапы 9 → 10 → 11 строго по очереди, каждый —
  отдельный проверенный коммит `Stage N: …` и push без force.
- Не переписывать архитектуру и не дублировать существующие механизмы.
- Гипотеза LLM никогда не становится `verified`; Evidence Gate не ослаблять.
- Тесты без сети и ключей (`FakeProvider`, `StubServer`); не читать `.env`.
- Официальный документ задания — вне репозитория, только чтение, не во внешние API без
  разрешения владельца, его дефекты не хардкодить.
- `git add` — явным списком файлов; без `.env`, `.venv/`, временных файлов.

Проверки:

```powershell
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m ruff format --check .
.venv\Scripts\python -m mypy
git diff --check
```
