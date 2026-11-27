# HANDOFF — как продолжить проект на другом компьютере

Этот файл — первая точка входа для нового разработчика и его Claude Code. Прежняя переписка
не нужна: всё необходимое — здесь, в [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) (что это и как
устроено), [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md) (что делать на Stage 9, 10, 11) и
[README.md](README.md) (подробности каждого механизма).

**Текущее состояние:** Stage 0–9 завершены и опубликованы (последний — коммит
`Stage 9: …`, 928 тестов). **Следующий шаг — Stage 10 (Evaluation).**

---

## 1. Инструкции для Claude Code (обязательно)

1. Сначала прочитай `HANDOFF.md`, `PROJECT_CONTEXT.md` и `DEVELOPMENT_ROADMAP.md`, затем
   разделы README, относящиеся к задаче.
2. Проверь фактическое состояние проекта: `git log --oneline -5`, `git status`, структуру
   `validator/` и `tests/`, прогон quality gates (раздел 4). Документация описывает коммит
   после Stage 9; если код новее — доверяй коду и обнови документацию.
3. Не начинай с нуля и не переписывай архитектуру. Существующие механизмы (чтение,
   сегментация, `Finding`/`Evidence`, Evidence Gate, правила, провайдеры, Fact Graph, отбор
   кандидатов, разбор ограничений) переиспользуются, а не создаются повторно.
4. Начинай **только со Stage 10** (Stage 9 выполнен).
5. Выполняй Stage 10 и 11 **строго последовательно**.
6. После каждого этапа — независимый аудит (края, ложные срабатывания, пропуски,
   детерминизм), тесты и все quality gates (раздел 5).
7. После успешного завершения этапа — отдельный коммит `Stage N: <кратко>` и push.
8. Не переходи к следующему этапу, пока текущий не завершён и не опубликован; после этапа —
   краткий отчёт и ожидание подтверждения владельца.
9. Никогда не используй force push (`git push --force`, `--force-with-lease`), а также
   `git reset --hard`, `git clean -fd` и другие разрушительные команды без явной просьбы
   владельца.
10. Не включай в Git секреты, `.env`, виртуальное окружение (`.venv/`), кэши, временные файлы,
    официальный документ задания. Файлы добавляй явным списком (`git add <файлы>`), а не
    `git add .`; перед коммитом смотри `git status` и `git diff --cached --stat`.

Дополнительно:

- **TDD.** Сначала тест, убедиться, что он падает, затем код. Найденные дефекты чинить в
  рамках текущего этапа и тоже через тест.
- **Не ослаблять проверки ради зелёных тестов.** Evidence Gate, статусы, бюджеты, повторы,
  таймауты — не трогать без обоснования. Гипотеза LLM никогда не становится `verified`.
- **Честные формулировки.** В отчёте не утверждать «документ корректен», «противоречий нет»,
  «всё проверено», если проверка не полная; непроверенное перечислять.
- **Тесты без сети и без ключей.** Использовать `FakeProvider` (`validator/llm/fake.py`) и
  `StubServer` (`tests/llm_helpers.py`); настоящий ключ в тестах запрещён.
- **Не читать и не печатать содержимое `.env`.**
- **Архитектурный конфликт или неоднозначное требование** — сначала описать и предложить
  минимальное решение, потом действовать.
- **Не трогать другие репозитории владельца** — работа только в этом репозитории.

---

## 2. Получение кода

**Рекомендуемый способ — `git clone`** (нужен Git for Windows):

```powershell
git clone https://github.com/sowldi/bcc-solution-of-damillion.git
```

**Если скачан ZIP-архив** с GitHub, учтите:

- в архиве нет папки `.git`: истории нет, `git status`/`commit`/`push` не работают;
- паспорт отчёта покажет версию агента «0.1.0 (без git)» вместо коммита
  (`validator/version.py` берёт её из `git describe`);
- для работы по этапам (коммиты и push) всё равно сделайте `git clone` в отдельную папку и
  работайте в нём; ZIP годится только для чтения и пробного запуска.

**Доступ на запись.** Push в `sowldi/bcc-solution-of-damillion` возможен только с правами
соавтора (collaborator), которые выдаёт владелец репозитория. Без прав — fork и pull
request, либо передача изменений владельцу.

**Git identity** задайте локально для репозитория своими данными (глобальную настройку не
меняйте без необходимости):

```powershell
git config user.name "Ваше имя"
git config user.email "ваш@адрес"
```

---

## 3. Настройка окружения (Windows, PowerShell)

Требуется Python ≥ 3.11; проект проверялся на **Python 3.12.10**.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` включает `requirements.txt` и добавляет `pytest`, `ruff`, `mypy`.
Активировать окружение не обязательно: все команды вызывают `.venv\Scripts\python` напрямую.

Ключ модели (только для стандартного режима; для разработки и тестов не нужен):

```powershell
Copy-Item .env.example .env
```

Затем впишите ключ в `.env` (например, `GEMINI_API_KEY=...` — ключ из Google AI Studio) или
выберите `LLM_PROVIDER=ollama` для локальной модели без ключа. `.env` в Git не попадает
(`.gitignore`). Ключ никогда не вставляйте в код, тесты, документацию или сообщения коммитов.

---

## 4. Проверка исходного состояния (до любых изменений)

```powershell
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m ruff format --check .
.venv\Scripts\python -m mypy
```

Ожидается: **928 passed** (около 50 с), `All checks passed!`, все файлы отформатированы,
`Success: no issues found`. Если что-то не так — сначала разобраться, не начиная Stage 10.

Пробный запуск CLI на собственном документе (без ключа и сети):

```powershell
@'
# Сервис бронирования

## Функциональные требования

* Ф-1. Срок хранения брони: не более 3 дней.
* Ф-2. Срок хранения брони — 5 дней.
* Ф-3. Система должна быстро подтверждать бронь.
* Ф-4. Срок выдачи: TBD.
'@ | Set-Content -Encoding utf8 sample.md
.venv\Scripts\python -m validator sample.md reports\sample_report.md --rules-only
```

Отчёт `reports\sample_report.md` должен содержать разделы 1–10 (вердикт — «Начинать
разработку и тестирование не рекомендуется»), подтверждённое противоречие
CTR-01 (Ф-1 и Ф-2), замечания TST-01 («быстро») и CMP-01 («TBD»). Файлы `sample.md` и
`reports\` — временные: удалите их и не коммитьте.

---

## 5. Порядок закрытия этапа (чек-лист)

1. Реализация по TDD; README и PROJECT_CONTEXT.md обновлены.
2. `pytest`, `ruff check .`, `ruff format --check .`, `mypy`, `git diff --check` — зелёные.
3. Реальный CLI: собственные документы в MD, TXT, DOCX, PDF (генераторы — `tests/builders.py`),
   режим `--rules-only` и стандартный режим через `FakeProvider` или `StubServer`.
4. Регрессия:
   - разделы отчёта предыдущих этапов не изменились (кроме строки «> Агент на этапе N»);
   - эталон официального документа сохранился (PROJECT_CONTEXT.md, раздел 3);
   - `--rules-only` работает без сети, LLM-слой не загружается.
5. Детерминизм: повторный прогон и разные `PYTHONHASHSEED` дают одинаковый результат.
6. Нет совпадений из 4 слов с официальным документом (раздел 6) и нет переводов строк CRLF.
7. `git status` → явный `git add <файлы>` → `git diff --cached --stat` → коммит
   `Stage N: <кратко>` → `git push origin main` (без force) → проверка
   `git rev-parse HEAD` == `git rev-parse origin/main` и чистого дерева.
8. Отчёт владельцу: что сделано, проверки, число тестов, хеш и ссылка на коммит, ограничения.
   Остановиться.

---

## 6. Официальный документ задания

- Документ требований хакатона выдают организаторы (папка с `README.md` и
  `docs/requirements.md`); получите его у владельца проекта. Храните **вне репозитория**,
  например рядом: `..\ai-requirements-validator-task-main\docs\requirements.md`.
- Только чтение. **Не коммитить** документ и его фрагменты.
- **Не превращать его дефекты в правила, исключения, тестовые данные или разметку оценки.**
  Тестовые документы — только собственные (домены «библиотека», «бронирование» и т. п.).
- **Не отправлять во внешние API** (Gemini, OpenAI, Anthropic) без явного разрешения
  владельца проекта. Для локальных прогонов — `--rules-only` или Ollama.
- Проверка отсутствия совпадений из 4 слов (запускать перед каждым коммитом этапа; скрипт
  держите вне репозитория). Сохраните код ниже, например как `..\tools\overlap.py`:

  ```python
  import re, subprocess, sys
  from pathlib import Path

  WORD = re.compile(r"[^\W_]+")


  def grams(text):
      words = [w.casefold().replace("ё", "е") for w in WORD.findall(text)]
      return {tuple(words[i : i + 4]) for i in range(len(words) - 3)}


  official = grams(Path(sys.argv[1]).read_text(encoding="utf-8"))
  listed = subprocess.run(
      ["git", "ls-files", "-c", "-o", "--exclude-standard"],
      capture_output=True,
      text=True,
      check=True,
  ).stdout.splitlines()
  for name in sorted(set(listed)):
      path = Path(name)
      if path.suffix in {".py", ".md", ".txt", ".toml", ".example", ".json"}:
          hits = grams(path.read_text(encoding="utf-8")) & official
          if hits:
              print(name, len(hits), sorted(hits)[:3])
  ```

  Затем запустите его из корня репозитория:

  ```powershell
  .venv\Scripts\python ..\tools\overlap.py ..\ai-requirements-validator-task-main\docs\requirements.md
  ```

  Допустимые совпадения (общие обороты, не содержание документа):
  - HTTP-заголовок Content-Type со значением application/json — в `tests/llm_helpers.py`,
    `tests/test_llm_transport.py`, `validator/llm/providers.py`;
  - заголовок раздела функциональных требований с первым ID вида «Ф-1» — в нескольких
    тестах и в примере `sample.md` в этом файле.

  Два старых совпадения этапа 2 (JSON-пример в тесте сегментации и пример вариантов выбора
  в правилах сегментации) заменены собственными примерами отдельным коммитом после
  `fa65212`. Других совпадений быть не должно: всё, что найдёт скрипт сверх допустимого
  списка, — убрать до коммита.

---

## 7. Ловушки среды (Windows)

- В Bash-инструменте Claude Code heredoc может схлопнуть `\\` в `\`: строки Python с `\n`
  превращаются в настоящие переводы строк. Для правок с обратными слэшами и для больших
  патчей пишите файл инструментом Write/Edit или сохраняйте скрипт во временную папку и
  запускайте его файлом.
- `Path.write_text` на Windows пишет CRLF. В репозитории `.gitattributes` (`* text=auto eol=lf`);
  пишите через `write_bytes(text.encode("utf-8"))` или `newline="\n"` и перед коммитом
  проверяйте CRLF в изменённых файлах.
- Консоль Windows по умолчанию не UTF-8: для скриптов с кириллицей задавайте
  `PYTHONIOENCODING=utf-8` и `python -X utf8`. CLI сам переключает свои потоки в UTF-8.
- Вывод Python через конвейер (`| sed`) буферизуется — для долгих замеров используйте
  `python -u` или запись в файл.
- `tracemalloc` замедляет код в 3–10 раз: время мерить без него, память — отдельно.
- Ruff 0.17+ проверяет `ruff format --check .` и Python-блоки внутри `.md`-файлов: после
  правки документации с примерами кода запускайте `ruff format <файл>.md`.

---

## 8. Где что лежит

| Нужно | Где |
| :--- | :--- |
| Что это, архитектура, инварианты, история этапов | `PROJECT_CONTEXT.md` |
| ТЗ на Stage 9, 10, 11 | `DEVELOPMENT_ROADMAP.md` |
| Подробности механизмов (правила, Gate, LLM, граф, этапы 8–9) | `README.md` |
| Конвейер CLI | `validator/cli.py` (`main`, `_run_llm`) |
| Отчёт | `validator/report/markdown.py` (`render_report`) |
| Модель замечания и статусы | `validator/findings.py` |
| Evidence Gate | `validator/evidence.py` |
| Тестовые помощники | `tests/conftest.py`, `tests/builders.py`, `tests/llm_helpers.py`, `tests/graph_helpers.py`, `tests/lint_helpers.py` |

---

## 9. Готовый первый запрос для Claude Code

> Прочитай `HANDOFF.md`, `PROJECT_CONTEXT.md` и `DEVELOPMENT_ROADMAP.md`. Проверь
> фактическое состояние репозитория и прогони quality gates. Затем выполни Stage 10 —
> Evaluation — по заданию из `DEVELOPMENT_ROADMAP.md`, соблюдая правила из
> `HANDOFF.md`. После завершения и проверки сделай коммит `Stage 10: …`, push без force,
> отчитайся и остановись. Stage 11 не начинай.
