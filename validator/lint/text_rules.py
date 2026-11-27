"""Правила формулировок: проверяемость (TST), однозначность (AMB), заглушки и открытые
перечисления (CMP-01, CMP-02).

Объект проверки — атомарные требования (утверждения и неделимые фрагменты с ролью
requirement); CMP-01 смотрит и на справочный текст, потому что заглушка — пробел в документе
в любом разделе. Слова ищутся целиком (с учётом словоформ) в тексте после ID требования.
Контекст учитывается явно: числовой критерий рядом, определение термина в документе,
устойчивые термины («система быстрых платежей», «максимально допустимый»), уточнение
после двоеточия или в скобках.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator

from validator.findings import AbsenceClaim
from validator.ingest.common import one_line
from validator.lint.base import Hit, LintContext, Span
from validator.segmentation import Requirement
from validator.segmentation.rules import placeholder_phrase
from validator.segmentation.text import ID_PATTERN

_I = re.IGNORECASE


def _words(*alternatives: str) -> re.Pattern[str]:
    """Любая из фраз целиком: не часть другого слова и не часть идентификатора."""
    return re.compile(r"(?<![\w-])(?:" + "|".join(alternatives) + r")(?![\w-])", _I)


# --- числа и измеримые критерии --------------------------------------------------------

_NUMBER_WORD = (
    r"од(?:ин|на|но|ного|ной|ну|ним)|дв(?:а|е|ух|ум|умя)|тр(?:и|[её]х|[её]м|емя)"
    r"|четыр(?:е|[её]х|ем|ьмя)|пят(?:ь|и|ью)|шест(?:ь|и|ью)|сем(?:ь|и|ью)|восем(?:ь|ью)|восьми"
    r"|девят(?:ь|и|ью)|десят(?:ь|и|ью)|пятнадцат\w*|двадцат\w*|тридцат\w*|сорок\w*|сто|ста"
    r"|сотн\w*|тысяч\w*|миллион\w*|полтор\w*|полутор\w*"
    r"|one|two|three|four|five|six|seven|eight|nine|ten|twenty|hundred|thousand"
)
_NUMBER = re.compile(rf"\d|(?<![\w-])(?:{_NUMBER_WORD})(?![\w-])", _I)
_ID_TOKEN = re.compile(rf"(?<![\w-]){ID_PATTERN}(?![\w-])")
_TIME_UNIT = (
    r"мс|ms|с|сек\w*|секунд\w*|мин\w*|ч|час\w*|сут\w*|дн\w*|день|дней|недел\w*|месяц\w*|мес"
    r"|год\w*|лет|s|sec\w*|seconds?|min\w*|minutes?|hours?|h|days?|weeks?|months?"
)
_TIME_MEASURE = re.compile(
    rf"(?:\d+(?:[.,]\d+)?|(?<![\w-])(?:{_NUMBER_WORD}))\s*(?:{_TIME_UNIT})(?![\w-])"
    r"|\d{1,2}:\d{2}|(?<![\w-])кажд\w+\s+\d|(?<![\w-])раз\s+в\s+"
    r"|ежедневн|ежечасн|еженедельн|ежемесячн|ежегодн|\bdaily\b|\bhourly\b|\bweekly\b"
    r"|\bmonthly\b|\bevery\s+\d",
    _I,
)


def _without_ids(text: str) -> str:
    """Идентификаторы («Ф-7», «FR-2.1») — не количественные значения; заменяются пробелами,
    чтобы смещения не сдвигались."""
    return _ID_TOKEN.sub(lambda match: " " * len(match.group(0)), text)


def _has_number(text: str) -> bool:
    return bool(_NUMBER.search(_without_ids(text)))


def _number_near(text: str, body: int, start: int, end: int, before: int, after: int) -> bool:
    return _has_number(text[max(body, start - before) : end + after])


# --- общий поиск слов --------------------------------------------------------------------

Keep = Callable[[str, int, re.Match[str]], bool]  # (текст, начало содержания, совпадение)


def _word_hits(
    context: LintContext,
    pattern: re.Pattern[str],
    explain: Callable[[str], str],
    *,
    keep: Keep | None = None,
    fragments: Iterable[Requirement] | None = None,
) -> Iterator[Hit]:
    for fragment in context.targets if fragments is None else fragments:
        text, body = fragment.text, context.body_start(fragment)
        for match in pattern.finditer(text, body):
            if context.definitions.covers(text, match.start(), match.end()):
                continue
            if keep is not None and not keep(text, body, match):
                continue
            span = Span(fragment, match.start(), match.end())
            yield Hit((span,), explain(one_line(match.group(0))))


def _inside(pattern: re.Pattern[str], text: str, match: re.Match[str]) -> bool:
    """Совпадение — часть устойчивого термина (pattern), а не самостоятельная оценка."""
    return any(
        term.start() <= match.start() and match.end() <= term.end()
        for term in pattern.finditer(text)
    )


# --- A. Проверяемость ------------------------------------------------------------------

_SUBJECTIVE = _words(
    r"удобн\w*",
    r"удобств\w*",
    r"интуитивн\w*",
    r"понятн\w*",
    r"быстр(?:о|ый|ая|ое|ые|ого|ой|ую|ым|ыми|ых|ее|ейш\w*)",
    r"л[её]гк(?:о|ий|ая|ое|ие|ого|ой|ую|им|их)",
    r"эффективн\w*",
    r"над[её]жн(?:о|ый|ая|ое|ые|ого|ой|ую|ым|ыми|ых|ее)",
    r"современн\w*",
    r"красив\w*",
    r"привлекательн\w*",
    r"дружелюбн\w*",
    r"дружественн\w*",
    r"гибк(?:о|ий|ая|ое|ие|ого|ой|ую|им|их)",
    r"оптимальн\w*",
    r"качественн(?:о|ый|ая|ое|ые|ого|ой|ую|ым|ых)",
    r"приемлем\w*",
    r"достаточн(?:ый|ая|ое|ые|ого|ой|ую|ым|ых)",
    r"адекватн\w*",
    r"разумн\w*",
    r"максимально",
    r"минимально",
    r"user[- ]friendly",
    r"easy",
    r"easily",
    r"intuitive(?:ly)?",
    r"fast",
    r"quick(?:ly)?",
    r"efficient(?:ly)?",
    r"robust",
    r"flexible",
    r"seamless(?:ly)?",
    r"convenient(?:ly)?",
    r"simple",
    r"modern",
    r"adequate(?:ly)?",
    r"sufficient(?:ly)?",
    r"reasonabl[ey]",
)
# Устойчивые термины, где слово не оценка: «система быстрых платежей», «максимально
# допустимая сумма», «эффективная процентная ставка».
_ESTABLISHED = re.compile(
    r"(?:максимально|минимально)\s+допустим\w*|быстр\w*\s+(?:платеж|платёж|перевод)\w*"
    r"|эффективн\w*\s+(?:годов\w*\s+)?(?:процентн\w*\s+)?ставк\w*",
    _I,
)


def tst01(context: LintContext) -> Iterator[Hit]:
    def keep(text: str, body: int, match: re.Match[str]) -> bool:
        measured = _number_near(text, body, match.start(), match.end(), before=30, after=60)
        return not measured and not _inside(_ESTABLISHED, text, match)

    yield from _word_hits(
        context,
        _SUBJECTIVE,
        lambda word: (
            f"«{word}» — субъективная оценка: без измеримого критерия нельзя проверить, "
            "выполнено ли требование."
        ),
        keep=keep,
    )


_OPTIONAL = _words(
    r"по\s+(?:мере\s+)?возможности",
    r"при\s+(?:наличии\s+)?возможности",
    r"если\s+(?:это\s+)?возможно",
    r"желательно",
    r"по\s+(?:своему\s+)?усмотрению",
    r"в\s+идеале",
    r"if\s+possible",
    r"as\s+far\s+as\s+possible",
    r"where(?:ver)?\s+possible",
    r"preferably",
    r"ideally",
    r"if\s+feasible",
)


def tst02(context: LintContext) -> Iterator[Hit]:
    yield from _word_hits(
        context,
        _OPTIONAL,
        lambda phrase: (
            f"Оговорка «{phrase}» делает требование необязательным: нельзя определить, "
            "когда оно нарушено."
        ),
    )


_CLAUSE_END = r"(?=\s*(?:[,.;:)]|$))"
_CONDITION = _words(
    r"при\s+необходимости",
    r"в\s+случае\s+необходимости",
    r"по\s+(?:мере\s+)?необходимости",
    rf"если\s+(?:это\s+)?(?:необходимо|нужно|требуется){_CLAUSE_END}",
    r"если\s+потребуется",
    r"когда\s+потребуется",
    r"при\s+потребности",
    r"как\s+правило",
    r"обычно",
    r"в\s+большинстве\s+случаев",
    r"if\s+(?:necessary|needed|required)",
    r"as\s+(?:needed|required|appropriate)",
    r"where\s+appropriate",
    r"when\s+necessary",
    r"usually",
    r"typically",
    r"normally",
    r"in\s+most\s+cases",
)


def tst03(context: LintContext) -> Iterator[Hit]:
    yield from _word_hits(
        context,
        _CONDITION,
        lambda phrase: (
            f"Условие «{phrase}» не определено: неясно, кто и по какому признаку решает, "
            "что оно наступило."
        ),
    )


_GOAL = _words(
    r"минимиз\w*",
    r"максимиз\w*",
    r"оптимиз\w*",
    r"стрем(?:ится|иться|ятся|ление\w*)",
    r"minimi[sz]\w*",
    r"maximi[sz]\w*",
    r"optimi[sz]\w*",
)


def tst04(context: LintContext) -> Iterator[Hit]:
    yield from _word_hits(
        context,
        _GOAL,
        lambda word: (
            f"«{word}» задаёт направление улучшения без целевого значения: результат нельзя "
            "принять или отклонить."
        ),
        keep=lambda text, body, match: not _has_number(text[body:]),
    )


_ATTRIBUTE = _words(
    r"время\s+(?:отклика|ответа|обработки|выполнения|загрузки)",
    r"производительност\w*",
    r"пропускн\w*\s+способност\w*",
    r"нагрузк\w*",
    r"отказоустойчивост\w*",
    r"доступност\w*",
    r"масштабируемост\w*",
    r"uptime",
    r"response\s+time",
    r"latency",
    r"throughput",
    r"availability",
    r"scalability",
    r"performance",
)
_QUALITY_CATEGORIES = frozenset({"performance", "availability", "non_functional"})


def tst05(context: LintContext) -> Iterator[Hit]:
    for fragment in context.targets:
        text, body = fragment.text, context.body_start(fragment)
        if fragment.category not in _QUALITY_CATEGORIES or _has_number(text[body:]):
            continue
        match = _ATTRIBUTE.search(text, body)
        if match:
            yield Hit(
                (Span(fragment, match.start(), match.end()),),
                f"Требование к атрибуту качества «{one_line(match.group(0))}» не содержит "
                "числового критерия: порога, единицы измерения, условий замера.",
                absence=AbsenceClaim("requirement", fragment.id, "number"),
            )


# --- B. Однозначность ------------------------------------------------------------------

_QUANTITY = _words(
    r"несколько",
    r"нескольк(?:их|им|ими)",
    r"много",
    r"многи(?:е|х|м|ми)",
    r"некотор\w*",
    r"большинств\w*",
    r"значительн\w*",
    r"незначительн\w*",
    r"небольш\w*",
    r"больш(?:ое|ой|ие|их|ого|ую|ом|им|ими)\s+(?:количеств|объ[её]м|числ|нагрузк)\w*",
    r"мал(?:ое|ый|ую|ом|ого|ым)\s+(?:количеств|объ[её]м|числ)\w*",
    r"ограниченн\w*\s+(?:количеств|числ|объ[её]м)\w*",
    r"определ[её]нн\w*\s+(?:количеств|числ|объ[её]м)\w*",
    r"several",
    r"many",
    r"various",
    r"numerous",
    r"a\s+few",
    r"a\s+(?:number|lot)\s+of",
    r"(?:large|small)\s+(?:number|amount)\s+of",
)
_CLARIFIED = re.compile(r"[:(]")  # дальше идёт перечень или уточнение в скобках


def amb01(context: LintContext) -> Iterator[Hit]:
    def keep(text: str, body: int, match: re.Match[str]) -> bool:
        if _number_near(text, body, match.start(), match.end(), before=15, after=25):
            return False
        return not _CLARIFIED.search(text, match.end())

    yield from _word_hits(
        context,
        _QUANTITY,
        lambda phrase: f"«{phrase}» не задаёт количество: разные читатели поймут его по-разному.",
        keep=keep,
    )


_APPROXIMATE = _words(
    r"(?:около|порядка)\s+\d[\d\s]*(?:[.,]\d+)?",
    r"примерно",
    r"приблизительно",
    r"ориентировочно",
    r"(?:about|approximately|around|roughly)\s+\d[\d\s]*(?:[.,]\d+)?",
    r"approx\.?",
    r"[~≈]\s*\d+(?:[.,]\d+)?",
)
_TOLERANCE = re.compile(
    r"±|\+/-|погрешност|допуск|отклонени\w*\s+не\s+более|tolerance|deviation", _I
)


def amb02(context: LintContext) -> Iterator[Hit]:
    yield from _word_hits(
        context,
        _APPROXIMATE,
        lambda phrase: (
            f"«{phrase}» — приблизительное значение без допустимого отклонения: неясно, какой "
            "результат считать выполнением."
        ),
        keep=lambda text, body, match: not _TOLERANCE.search(text, body),
    )


_VAGUE_TIME = _words(
    r"своевременн\w*",
    r"оперативно",
    r"немедленн\w*",
    r"незамедлительн\w*",
    r"безотлагательн\w*",
    r"мгновенн\w*",
    r"в\s+кратчайш\w+\s+срок\w*",
    r"в\s+(?:короткие|сжатые|разумные|минимальные)\s+срок\w*",
    r"в\s+ближайшее\s+время",
    r"как\s+можно\s+(?:скорее|быстрее|раньше)",
    r"в\s+(?:режиме\s+)?реальн\w+\s+времен\w*",
    r"периодическ\w*",
    r"регулярн\w*",
    r"в\s+течение\s+(?:нескольких|некоторого|определ[её]нного)\s+\w+",
    r"через\s+(?:некоторое|определ[её]нное)\s+время",
    r"с\s+(?:небольшой|минимальной)\s+задержк\w*",
    r"без\s+задерж\w*",
    r"as\s+soon\s+as\s+possible",
    r"asap",
    r"promptly",
    r"immediately",
    r"in\s+a\s+timely\s+manner",
    r"timely",
    r"real[- ]time",
    r"periodically",
    r"regularly",
    r"without\s+delay",
)
# «регулярный платёж», «регулярное выражение» — названия, а не срок
_TIME_TERMS = re.compile(r"(?:регулярн|периодическ)\w*\s+(?:платеж|платёж|выражени)\w*", _I)


def amb03(context: LintContext) -> Iterator[Hit]:
    def keep(text: str, body: int, match: re.Match[str]) -> bool:
        measured = _TIME_MEASURE.search(_without_ids(text), body)
        return not measured and not _inside(_TIME_TERMS, text, match)

    yield from _word_hits(
        context,
        _VAGUE_TIME,
        lambda phrase: (
            f"«{phrase}» не задаёт срок или периодичность в единицах времени — выполнение "
            "нельзя проверить."
        ),
        keep=keep,
    )


_PRONOUN_START = re.compile(
    r"(?:он|она|оно|они|это|этот|эта|эти|этого|этой|этих|этим|этому|эту|этими"
    r"|так(?:ой|ая|ое|ие|ого|ую|им|их)|данн(?:ый|ая|ое|ого|ой|ому|ую|ым)"
    r"|указанн\w*|вышеуказанн\w*|вышеупомянут\w*|упомянут\w*"
    r"|it|they|this|these|that|those|such)(?![\w-])",
    _I,
)


def amb04(context: LintContext) -> Iterator[Hit]:
    for fragment in context.targets:
        if context.kind(fragment) not in ("paragraph", "list_item"):
            continue
        match = _PRONOUN_START.match(fragment.text, context.body_start(fragment))
        if match:
            yield Hit(
                (Span(fragment, match.start(), match.end()),),
                f"Требование начинается со слова «{match.group(0)}»: о чём оно, понятно только "
                "из соседнего текста, а при переносе или ссылке связь теряется.",
            )


_ABBREVIATION = re.compile(r"(?<![\w-])[А-ЯЁ]{2,6}(?![\w-])")
# Общеизвестные сокращения, которые не требуют расшифровки в документе.
_COMMON_ABBREVIATIONS = frozenset(
    {
        "РФ", "РК", "РБ", "США", "ЕС", "ООН", "СНГ", "ЕАЭС", "ФИО", "ИИН", "БИН", "ИНН",
        "КПП", "ОГРН", "НДС", "ТОО", "АО", "ООО", "ПАО", "ОАО", "ЗАО", "ИП", "ГОСТ", "СМС",
        "ПК", "ПО", "ОС", "ИТ", "ЭЦП", "ВУЗ", "МСК", "КБ", "МБ", "ГБ", "ТБ", "ГГГГ", "ММ",
        "ДД", "ЧЧ",
    }
)  # fmt: skip
# Слова, набранные прописными для выделения, — не сокращения.
_CAPS_WORDS = frozenset(
    {
        "НЕ", "НИ", "ДА", "НЕТ", "ИЛИ", "ЛИБО", "ВСЕ", "ВСЁ", "ТОЛЬКО", "ВАЖНО", "КРОМЕ",
        "ЕСЛИ", "ТО", "ДО", "ОТ", "ПО", "НА", "ЗА", "ИЗ", "ПРИ", "ДЛЯ", "БЕЗ", "ЧЕМ", "КАК",
        "ЭТО", "ОН", "ОНА", "ОНИ", "МЫ", "ВЫ", "ИХ", "ЕГО", "ЕЁ", "ЕЕ", "ЛИ", "ЖЕ", "УЖЕ",
        "ЕЩЁ", "ЕЩЕ", "ВСЕГДА", "НИКОГДА", "ВСЕХ", "ОДИН", "ДВА", "ТРИ",
    }
)  # fmt: skip


def amb05(context: LintContext) -> Iterator[Hit]:
    first: dict[str, Span] = {}
    uses: Counter[str] = Counter()
    for fragment in context.targets:
        for match in _ABBREVIATION.finditer(fragment.text, context.body_start(fragment)):
            abbreviation = match.group(0)
            if (
                abbreviation in _COMMON_ABBREVIATIONS
                or abbreviation in _CAPS_WORDS
                or context.definitions.defines_abbreviation(abbreviation)
            ):
                continue
            uses[abbreviation] += 1
            first.setdefault(abbreviation, Span(fragment, match.start(), match.end()))
    for abbreviation, span in first.items():
        yield Hit(
            (span,),
            f"Сокращение «{abbreviation}» не расшифровано в документе: его нет в глоссарии и "
            f"нет расшифровки в скобках. Употреблений в требованиях: {uses[abbreviation]}.",
        )


# --- C. Полнота: заглушки и открытые перечисления -----------------------------------------

_PLACEHOLDER = re.compile(
    r"(?<![\w-])(?:TBD|TBA|TODO|FIXME)(?![\w-])|\?{3,}"
    r"|\[\s*(?:уточнить|заполнить|добавить|tbd|todo)\s*\]"
    r"|(?<!\w)будет\s+(?:уточн|определ|добавл|заполн|описан|согласован)\w*"
    rf"(?:\s+(?:позже|позднее|дополнительно|отдельно|в\s+дальнейшем))?{_CLAUSE_END}"
    rf"|(?<!\w)подлежит\s+уточнению|(?<!\w)уточняется{_CLAUSE_END}"
    rf"|(?<!\w)(?:нужно|необходимо|требуется|надо)\s+уточнить{_CLAUSE_END}"
    r"|(?<!\w)to\s+be\s+(?:defined|determined|confirmed|done|added|clarified)(?!\w)",
    _I,
)
_NOT_TEXT = frozenset({"code", "heading", "table_header"})


_LINE = re.compile(r"[^\n]+")


def cmp01(context: LintContext) -> Iterator[Hit]:
    for fragment in context.leaves:
        if fragment.role in _NOT_TEXT or not fragment.text.strip():
            continue
        text, body = fragment.text, context.body_start(fragment)
        if "placeholder" in fragment.detection:
            yield Hit(
                (Span(fragment, body, len(text)),),
                f"Вместо содержания стоит заглушка «{one_line(text[body:])}».",
            )
            continue
        # Строка-заглушка внутри фрагмента: например, в TXT неоформленный заголовок
        # склеивается со следующей строкой «заполняется позже».
        for line in _LINE.finditer(text, body):
            start = line.start() + len(line.group(0)) - len(line.group(0).lstrip())
            end = line.start() + len(line.group(0).rstrip())
            if start < end and placeholder_phrase(text[start:end]):
                yield Hit(
                    (Span(fragment, start, end),),
                    f"Вместо содержания стоит заглушка «{one_line(text[start:end])}».",
                )
        for match in _PLACEHOLDER.finditer(text, body):
            yield Hit(
                (Span(fragment, match.start(), match.end()),),
                f"Вместо значения стоит заглушка «{one_line(match.group(0))}»: требование "
                "не дописано.",
            )


_OPEN_LIST = re.compile(
    r"(?<!\w)и\s+т\.\s?[дп]\.?"
    r"|(?<!\w)и\s+др\.?(?!\w)"
    r"|(?<!\w)и\s+проч(?:ее|ие|их|ая)(?!\w)"
    r"|(?<!\w)и\s+так\s+далее(?!\w)"
    r"|(?<!\w)и\s+друг(?:ие|их|им|ими|ое|ого)(?!\w)"
    r"|(?<!\w)и\s+далее(?=\s*(?:[.;,:)\]—–]|-\s|$))"
    r"|,\s*(?:\.{3}|…)"
    r"|(?<!\w)etc\.?"
    r"|(?<!\w)and\s+so\s+on(?!\w)"
    r"|(?<!\w)and\s+others(?!\w)"
    r"|(?<!\w)(?:including|включая)\s*,?\s*(?:but|но)\s+(?:not|не)\s+(?:limited|ограничива)\w*"
    r"|(?<!\w)в\s+том\s+числе,?\s+но\s+не\s+только(?!\w)",
    _I,
)


def cmp02(context: LintContext) -> Iterator[Hit]:
    yield from _word_hits(
        context,
        _OPEN_LIST,
        lambda phrase: (
            f"Перечисление открыто («{phrase}»): неизвестно, какие ещё элементы входят "
            "в требование, — полноту реализации нельзя проверить."
        ),
    )
