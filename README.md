# AI Requirements Validator

ИИ-агент статической проверки документа требований (ISO/IEC/IEEE 29148:2018 п. 5.2.5–5.2.6,
IEEE 830-1998 п. 4.3). Каждое замечание в отчёте подкреплено точной цитатой и местом в документе.

> Статус: каркас (этап 0). Полная инструкция появится по мере готовности этапов.

## Быстрый старт (Windows / PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env   # затем впишите ключ в .env
.venv\Scripts\python -m validator путь\к\requirements.md reports\report.md
```

Коды выхода: `0` — успех, `2` — проблема с входным файлом или путём отчёта,
`3` — ошибка настройки (.env), `4` — сбой LLM-провайдера, `1` — внутренняя ошибка.

## Разработка

```powershell
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m mypy
```
