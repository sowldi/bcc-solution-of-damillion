"""Правила классификации фрагментов.

Роль и тип требования определяются по нескольким сигналам, а не по одному слову:
1. раздел — правило по заголовкам, ближайший заголовок важнее внешних (SECTION_RULES);
2. вид блока — заголовок, шапка таблицы и код — структура, а не требование;
3. содержание — заглушки, ссылка на сам документ, маркеры примеров и пояснений, явный ID,
   нормативные слова («должен», «необходимо»), подлежащее «Система/Сервис/Клиент…»,
   уточняющие признаки производительности, безопасности и доступности.

Имя каждого сработавшего правила попадает в Requirement.detection — классификация объяснима.
Принцип осторожности: нормативный текст вне разделов требований становится «неоднозначным»,
а не требованием.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from validator.model import Block
from validator.segmentation.model import Category, Confidence, Role
from validator.segmentation.text import EXAMPLE_MARKERS, EXPLANATION_MARKERS, Prefix

SectionRole = Literal["requirement", "reference", "example", "explanation"]


@dataclass(frozen=True)
class SectionRule:
    name: str
    pattern: re.Pattern[str]
    role: SectionRole
    category: Category
    specific: bool  # конкретный раздел требований: пункты и строки таблиц — высокая уверенность


def _rule(
    name: str,
    pattern: str,
    role: SectionRole = "requirement",
    category: Category = "unspecified",
    *,
    specific: bool = False,
) -> SectionRule:
    return SectionRule(name, re.compile(pattern), role, category, specific)


# Порядок важен: первое совпавшее правило для заголовка побеждает.
SECTION_RULES: tuple[SectionRule, ...] = (
    _rule(
        "reference_meta",
        r"общ(ие|ая|ее) (сведени|информаци)|паспорт документа|информаци[яи] о документе"
        r"|метаданн|general information|глоссари|glossary|терминолог|\bтермины\b|определени[яй]"
        r"|definitions|сокращени|abbreviation|истори[яи] изменени|change ?log|revision history"
        r"|статус документа|версия документа",
        "reference",
        specific=True,
    ),
    _rule(
        "reference_review",
        r"ревью|review|чек[- ]?лист|checklist|согласовани|согласующ|утверждени|approv|подпис",
        "reference",
        specific=True,
    ),
    _rule(
        "reference_intro",
        r"введени|introduction|обзор|overview|назначение документа|область применения|\bscope\b",
        "reference",
        specific=True,
    ),
    _rule("examples", r"\bпример(ы)?\b|\bexamples?\b", "example", specific=True),
    _rule("notes", r"примечани|\bnotes?\b|комментари|пояснени", "explanation", specific=True),
    _rule(
        "non_functional",
        r"нефункциональн|non[- ]?functional|атрибут\w* качеств|quality attribute",
        category="non_functional",
        specific=True,
    ),
    _rule(
        "functional",
        r"функциональн|functional requirement|функции системы|user stor|пользовательск\w* истори"
        r"|use cases?|сценари\w* использования",
        category="functional",
        specific=True,
    ),
    _rule(
        "business_rule",
        r"бизнес[- ]?правил|business rules?|правила расч[её]та",
        category="business_rule",
        specific=True,
    ),
    _rule(
        "acceptance",
        r"критери\w* при[её]мк|acceptance|при[её]мочн",
        category="acceptance",
        specific=True,
    ),
    _rule(
        "security",
        r"безопасн|security|защит\w* (данных|информации)|аутентификац|авторизац"
        r"|authentication|authorization",
        category="security",
        specific=True,
    ),
    _rule(
        "performance",
        r"производительн|performance|быстродейств|нагрузк|масштабируем|scalab|latency",
        category="performance",
        specific=True,
    ),
    _rule(
        "availability",
        r"доступност|availability|над[её]жност|reliability|отказоустойчив",
        category="availability",
        specific=True,
    ),
    _rule(
        "data",
        r"требовани\w* к данным|\bданн(ые|ых)\b|data requirement|модел\w* данных|data model"
        r"|хранени\w* данных",
        category="data",
        specific=True,
    ),
    _rule(
        "integration",
        r"интеграц|integration|взаимодейств|sequence|диаграмм\w* последовательност"
        r"|обмен\w* данными",
        category="integration",
        specific=True,
    ),
    _rule(
        "interface",
        r"интерфейс|\bui\b|\bux\b|экран|макет|дизайн|user interface|\bgui\b",
        category="interface",
    ),
    _rule(
        "api",
        r"\bapi\b|\brest\b|endpoint|эндпоинт|\bметод\w*|\bзапрос\w*|\bответ(ы|а|ов)?\b"
        r"|\bстатус(ы|ов)?\b|\bпараметр\w*|\brequest\b|\bresponse\b|swagger|openapi",
        category="api",
    ),
    _rule(
        "constraint",
        r"ограничени|constraint|допущени|assumption",
        category="constraint",
        specific=True,
    ),
    _rule("architecture", r"архитектур|architecture", category="constraint"),
    _rule(
        "process",
        r"bpmn|схема процесса|бизнес[- ]?процесс|process flow|workflow",
        category="functional",
    ),
    _rule("goal", r"\bцел[ьи]\b|\bgoals?\b|objectives?|назначение системы", category="goal"),
    _rule("testing", r"тестировани|testing|\btests?\b|верификаци"),
    _rule("generic", r"требовани|requirements?|спецификаци|specification"),
)

MODAL = re.compile(
    r"\b(?:долж(?:ен|на|но|ны)|обязан\w*|необходимо|требуется|следует|не допускается|допускается"
    r"|запрещ\w*|разрешается|shall|must|should|required)\b"
)
_SELF_REFERENCE = re.compile(
    r"(?:(?:настоящ(?:ий|ем|его)|данн(?:ый|ом|ого)|этот|этом|дочерн(?:ий|ем|его)"
    r"|родительск(?:ий|ом|ого))\s+)?документ(?:е|а|ы|ов)?\b"
)
# Вводная фраза раздела описывает документ, а не систему: «В этом разделе описаны…»,
# «Ниже перечислены…», «Данный раздел описывает…». Совпадение — только целой конструкцией
# в начале фрагмента («Ниже порога…», «Далее система…» сюда не попадают).
_SECTION_INTRO = re.compile(
    r"(?:"
    r"(?:в|во) (?:этом|данном|настоящем|следующем) "
    r"(?:разделе|подразделе|пункте|параграфе|главе|документе)\b"
    r"|(?:ниже|далее|выше) "
    r"(?:перечислены|приведены|описаны|представлены|указаны|рассмотрены|изложены)\b"
    r"|(?:этот|данный|настоящий|следующий) (?:раздел|подраздел|пункт|параграф)\b"
    r"|(?:эта|данная|настоящая|следующая) глава\b"
    r"|(?:раздел|подраздел|глава) (?:содержит|описывает|включает|посвящ)"
    r"|(?:this|the following) (?:section|chapter|subsection)\b"
    r"|(?:below|the following) (?:are|is) (?:listed|described)\b"
    r")"
)
_SUBJECT_START = re.compile(
    r"(?:система|сервис|приложение|модуль|компонент|платформа|api|метод|контроллер"
    r"|пользователь|клиент|оператор|администратор|держатель|банк|процессинг)\b"
)
_PLACEHOLDER_EXACT = frozenset(
    {"tbd", "todo", "tba", "n/a", "na", "—", "-", "–", "...", "…", "?", "xxx", "нет данных"}
)
_PLACEHOLDER_START = (
    "заполняется",
    "заполняются",
    "будет заполнен",
    "будет добавлен",
    "будет уточн",
    "уточняется",
    "в разработке",
    "to be defined",
    "to be determined",
    "to be done",
)
_EMPTY_VALUES = frozenset({"", "—", "-", "–", "tbd", "n/a", "?"})
_CODE_START = re.compile(
    r"(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+/|(?:curl|wget)\s|<[A-Za-z?!/]"
)
# Строка JSON-объекта: «"ключ": значение», где значение начинается как значение JSON
# (а не как слово: «"Порог": целое число…» — это текст).
_JSON_MEMBER = re.compile(r'"[^"\n]{1,200}"\s*:\s*(?:["{\[\-\d]|true\b|false\b|null\b)')
_BRACKETS_ONLY = re.compile(r"[\[\]{}(),\s]+")  # «{», «},», «]» — строки структуры JSON
# Отдельное значение JSON («"push",», «1,», «true»): кодом считается только рядом с кодом.
_JSON_VALUE = re.compile(r'(?:"[^"\n]*"|-?\d+(?:[.,]\d+)?|true|false|null)\s*,?')
# Ячейка-вариант выбора: короткие варианты через «/» («Да / Нет / Не требуется», «Dev / Prod»).
_CHOICE_CELL = re.compile(r"[\w ]{1,20}(?:/[\w ]{1,20})+")
_METADATA_KEYS = (
    "author",
    "автор",
    "developer",
    "разработчик",
    "gitlab",
    "github",
    "repository",
    "репозиторий",
    "framework",
    "фреймворк",
    "owner",
    "владелец",
    "аналитик",
    "analyst",
    "jira",
    "задача в jira",
    "версия документа",
    "document version",
)
# Уточнение типа для общих разделов («нефункциональные», «не определён»): первый совпавший признак.
_CUES: tuple[tuple[Category, re.Pattern[str]], ...] = (
    (
        "performance",
        re.compile(
            r"в течение (?:\d|секунд|минут|час|суток)|не (?:более|дольше|позднее) \d+\s*"
            r"(?:мс|миллисекунд|с|сек|секунд|мин|минут|ч|час)|\b\d+\s*(?:мс|ms)\b"
            r"|время (?:отклика|ответа|обработки)|response time|latency|throughput|\brps\b|\btps\b"
            r"|запрос\w* в секунду|одновременн\w* пользовател"
        ),
    ),
    (
        "security",
        re.compile(
            r"авторизац|аутентификац|bearer|токен|token|шифр|парол|прав\w* доступа"
            r"|доступ запрещ|персональн\w* данн|\bpci\b|\btls\b|\bssl\b|oauth"
        ),
    ),
    (
        "availability",
        re.compile(
            r"доступност|uptime|\b99[.,]\d+\s*%|отказоустойчив|резервн\w* коп"
            r"|восстановлен\w* после|\brto\b|\brpo\b|24/7|круглосуточн"
        ),
    ),
)
_WORDS = re.compile(r"\w+")


@dataclass(frozen=True)
class Classification:
    role: Role
    category: Category
    confidence: Confidence
    detection: tuple[str, ...]
    warnings: tuple[str, ...] = ()


W_PLACEHOLDER = "Фрагмент-заглушка: содержание раздела не заполнено."
W_EMPTY_ROW = "Пустая строка таблицы: данные не заполнены."
W_EMPTY_VALUE = "Значение в строке таблицы не заполнено («—» или пусто)."
W_SHORT = "Слишком короткий фрагмент, чтобы считать его требованием."
W_HEADING = "Заголовок сформулирован как требование — проверьте, не пропущено ли требование."
W_MODAL_OUTSIDE = "Нормативная формулировка вне раздела требований — проверьте, требование ли это."
W_SUBJECT_OUTSIDE = "Описание поведения вне раздела требований — проверьте, требование ли это."
W_MODAL_REFERENCE = "Нормативная формулировка в справочном разделе — проверьте, требование ли это."


def match_section(section_path: tuple[str, ...]) -> tuple[SectionRule, str] | None:
    """Правило раздела: ищется от ближайшего заголовка к внешним."""
    for title in reversed(section_path):
        lowered = title.casefold()
        for rule in SECTION_RULES:
            if rule.pattern.search(lowered):
                return rule, title
    return None


def refine_category(category: Category, text: str) -> tuple[Category, tuple[str, ...]]:
    """Уточнить общий тип по содержанию; конкретный тип раздела не переопределяется."""
    if category not in ("non_functional", "unspecified"):
        return category, ()
    lowered = text.casefold()
    for cue_category, pattern in _CUES:
        if pattern.search(lowered):
            return cue_category, (f"cue:{cue_category}",)
    return category, ()


def words(text: str) -> list[str]:
    return _WORDS.findall(text)


def classify(block: Block, prefix: Prefix) -> Classification:
    """Роль, тип, уверенность и объяснение для одного блока."""
    section = match_section(block.location.section_path)
    rule = section[0] if section else None
    base: tuple[str, ...] = (f"section:{rule.name}",) if rule else ()
    category: Category = rule.category if rule and rule.role == "requirement" else "unspecified"
    body = block.text[prefix.body_start :]
    lowered = body.casefold().strip()

    if block.kind == "heading":
        if MODAL.search(lowered) and len(words(body)) >= 3:
            refined, cues = refine_category("unspecified", body)
            return Classification(
                "ambiguous", refined, "low", ("heading_normative", *cues), (W_HEADING,)
            )
        return Classification("heading", "unspecified", "high", ("structure:heading",))
    if block.kind == "table_header":
        return Classification("table_header", category, "high", (*base, "structure:table_header"))
    if block.kind == "code":
        return Classification("code", category, "high", (*base, "structure:code"))
    if block.kind == "table_row":
        if not block.text.strip():
            return Classification(
                "reference", category, "high", (*base, "empty_row"), (W_EMPTY_ROW,)
            )
        if len(block.cells) == 2 and _is_metadata_key(block.cells[0]):
            return Classification("reference", category, "high", (*base, "metadata_row"))
    if block.kind in ("paragraph", "list_item") and _looks_like_code(body):
        return Classification("code", category, "high", (*base, "code_like"))
    if _is_placeholder(lowered):
        return Classification(
            "reference", category, "high", (*base, "placeholder"), (W_PLACEHOLDER,)
        )
    if _SELF_REFERENCE.match(lowered):
        return Classification("reference", category, "high", (*base, "self_reference"))
    if block.kind != "table_row" and not prefix.doc_id and _SECTION_INTRO.match(lowered):
        return Classification("reference", category, "high", (*base, "section_intro"))
    if block.kind != "table_row" and not prefix.doc_id and len(words(body)) < 2:
        return Classification("reference", category, "medium", (*base, "short"), (W_SHORT,))
    if lowered.startswith(EXAMPLE_MARKERS):
        return Classification("example", category, "high", (*base, "example_marker"))
    if lowered.startswith(EXPLANATION_MARKERS):
        return Classification("explanation", category, "high", (*base, "explanation_marker"))
    return _classify_content(block, prefix, rule, base, body, lowered)


def _classify_content(
    block: Block,
    prefix: Prefix,
    rule: SectionRule | None,
    base: tuple[str, ...],
    body: str,
    lowered: str,
) -> Classification:
    modal = bool(MODAL.search(_normative_text(block, lowered)))
    modal_tag: tuple[str, ...] = ("modal",) if modal else ()
    in_requirements = rule is not None and rule.role == "requirement"
    category, cues = refine_category(
        rule.category if rule and in_requirements else "unspecified", body
    )
    empty_value: tuple[str, ...] = ()
    if block.kind == "table_row" and any(
        cell.strip().casefold() in _EMPTY_VALUES for cell in block.cells[1:]
    ):
        empty_value = (W_EMPTY_VALUE,)

    if prefix.doc_id:
        detection = (*base, "explicit_id", *modal_tag, *cues)
        if in_requirements:
            return Classification("requirement", category, "high", detection, empty_value)
        warning = f"Идентификатор требования «{prefix.doc_id}» стоит вне раздела требований."
        return Classification("requirement", category, "medium", detection, (warning, *empty_value))
    if rule is None:
        if modal:
            return Classification(
                "ambiguous", category, "low", ("modal", *cues), (W_MODAL_OUTSIDE,)
            )
        if _SUBJECT_START.match(lowered):
            return Classification(
                "ambiguous", category, "low", ("subject_start", *cues), (W_SUBJECT_OUTSIDE,)
            )
        return Classification("reference", "unspecified", "medium", ("no_section_rule",))
    if in_requirements:
        strong = (rule.specific and block.kind in ("list_item", "table_row")) or modal
        confidence: Confidence = "high" if strong else "medium"
        return Classification(
            "requirement", category, confidence, (*base, *modal_tag, *cues), empty_value
        )
    if rule.role == "reference":
        if modal:
            return Classification(
                "ambiguous", category, "low", (*base, "modal", *cues), (W_MODAL_REFERENCE,)
            )
        return Classification("reference", "unspecified", "high", base)
    return Classification(rule.role, "unspecified", "high", base)


def _looks_like_code(text: str) -> bool:
    """Технический фрагмент без разметки кода (DOCX, PDF, TXT): JSON целиком или построчно
    («{», «"ключ": значение», «},»), строка HTTP-запроса, команда curl/wget, XML/HTML."""
    stripped = text.strip()
    if not stripped:
        return False
    if (stripped[:1], stripped[-1:]) in (("{", "}"), ("[", "]")):
        return True
    if _BRACKETS_ONLY.fullmatch(stripped) or _JSON_MEMBER.match(stripped):
        return True
    if stripped[:1] in "{[" and _JSON_MEMBER.search(stripped):
        return True
    return bool(_CODE_START.match(stripped))


def looks_like_json_value(text: str) -> bool:
    """Отдельное значение JSON в строке («"push",», «1,», «true») — кодом это становится
    только в окружении кода (см. сегментатор), само по себе — нет."""
    return bool(_JSON_VALUE.fullmatch(text.strip()))


def _is_metadata_key(cell: str) -> bool:
    """Ключ метаданных — целым словом: «Gitlab project» да, «Authorization» ≠ «author»."""
    key = cell.strip().casefold()
    return any(
        key == name or (key.startswith(name) and not key[len(name)].isalpha())
        for name in _METADATA_KEYS
    )


def _normative_text(block: Block, lowered: str) -> str:
    """Текст для поиска нормативных слов. У строки таблицы ячейки-варианты выбора
    («Да / Нет / Не требуется») — шаблон, а не формулировка, и не учитываются."""
    if block.kind != "table_row":
        return lowered
    return " | ".join(
        cell.casefold() for cell in block.cells if not _CHOICE_CELL.fullmatch(cell.strip())
    )


def _is_placeholder(lowered: str) -> bool:
    stripped = lowered.strip(" .")
    if stripped in _PLACEHOLDER_EXACT or lowered.strip() in _PLACEHOLDER_EXACT:
        return True
    return placeholder_phrase(lowered)


def placeholder_phrase(text: str) -> bool:
    """Текст — фраза-заглушка («Заполняется позже», «To be defined»), не длиннее 80 символов."""
    stripped = text.casefold().strip(" .")
    return len(stripped) <= 80 and stripped.startswith(_PLACEHOLDER_START)
