"""Числовые ограничения и их сравнение — основа машинного контрпримера (этап 8).

parse_constraint разбирает значение параметра целиком («не более 3 дней», «от 1 до 5 минут»,
«10 MB»): лишние слова («не позднее чем через 3 дня после оплаты», «3 дня или 5 дней») —
не ограничение, значение не угадывается. find_constraint ищет одно ограничение в свободном
тексте цитаты и отказывается, если ограничений несколько или в цитате есть отрицание вне
оборота сравнения («не хранит … дольше 3 дней»).

Сравнение: единицы одного семейства пересчитываются (секунды…недели, байты…терабайты по
1024, месяцы и годы — отдельно: месяц не равен 30 дням); проценты и валюты сравниваются
только с собой; прочие единицы («попыток») — по форме слова. Ограничения «не пересекаются»
(disjoint) — нет значения, которое удовлетворяет обоим: это и есть контрпример. «Совместимы»
(compatible) — такое значение есть. «Несравнимы» (incomparable) — разные единицы.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from validator.graph.candidates import word_cores

Op = Literal["le", "lt", "ge", "gt", "eq", "range"]
Verdict = Literal["disjoint", "compatible", "incomparable"]

_NUMBER = r"\d+(?:[   ]\d{3})*(?:[.,]\d+)?"
_ADJECTIVE = re.compile(r"(?:ых|их|ого|его|ые|ие|ый|ий|ой|ая|яя|ое|ее)$")
_COMPARATORS: tuple[tuple[str, Op], ...] = (
    ("не более", "le"), ("не больше", "le"), ("не выше", "le"), ("не дольше", "le"),
    ("не позднее", "le"), ("не позже", "le"), ("не превышает", "le"), ("максимум", "le"),
    ("не менее", "ge"), ("не меньше", "ge"), ("не ниже", "ge"), ("не короче", "ge"),
    ("не ранее", "ge"), ("минимум", "ge"),
    ("менее", "lt"), ("меньше", "lt"), ("ниже", "lt"),
    ("более", "gt"), ("больше", "gt"), ("свыше", "gt"), ("выше", "gt"), ("дольше", "gt"),
    ("ровно", "eq"), ("до", "le"), ("от", "ge"),
    ("≤", "le"), ("<=", "le"), ("≥", "ge"), (">=", "ge"), ("<", "lt"), (">", "gt"), ("=", "eq"),
)  # fmt: skip
_CMP = "|".join(re.escape(phrase) for phrase, _ in sorted(_COMPARATORS, key=lambda c: -len(c[0])))
_OPS = dict(_COMPARATORS)
# Единица — знак, слово или «прилагательное + слово» («рабочих дней»); «дня или» — не единица.
_UNIT = (
    r"(?:%|₸|\$|€|₽|[^\W\d_]+(?:ых|их|ого|его|ые|ие|ый|ий|ой|ая|яя|ое|ее)\s+[^\W\d_]+\.?"
    r"|[^\W\d_]+\.?)"
)
_RANGE_FROM = re.compile(
    rf"от\s+(?P<a>{_NUMBER})\s*(?P<ua>{_UNIT})?\s+до\s+(?P<b>{_NUMBER})\s*(?P<ub>{_UNIT})?"
)
_RANGE_DASH = re.compile(
    rf"(?P<a>{_NUMBER})\s*(?P<ua>{_UNIT})?\s*[-–—]\s*(?P<b>{_NUMBER})\s*(?P<ub>{_UNIT})?"
)
_SINGLE = re.compile(rf"(?:(?P<cmp>{_CMP})\s*)?(?P<n>{_NUMBER})\s*(?P<u>{_UNIT})?")
_NEGATION = re.compile(r"(?<![\w-])(?:не|нет|без|not|no|never)(?![\w-])")

# Единица → (семейство, множитель к базовой единице семейства).
_UNITS: tuple[tuple[str, str, float], ...] = (
    (r"мс|миллисекунд\w*|ms", "time", 0.001),
    (r"с|сек|секунд\w*|s|sec|seconds?", "time", 1),
    (r"мин|минут\w*|min|minutes?", "time", 60),
    (r"ч|час|часа|часов|h|hours?", "time", 3600),
    (r"сут|сутки|суток|дн|день|дня|дней|d|days?", "time", 86_400),
    (r"нед|недел\w*|weeks?", "time", 604_800),
    (r"мес|месяц\w*|months?", "calendar", 1),
    (r"г|год|года|лет|years?", "calendar", 12),
    (r"б|байт\w*|b|bytes?", "size", 1),
    (r"кб|кбайт|килобайт\w*|kb|kib", "size", 1024),
    (r"мб|мбайт|мегабайт\w*|mb|mib", "size", 1024**2),
    (r"гб|гбайт|гигабайт\w*|gb|gib", "size", 1024**3),
    (r"тб|тбайт|терабайт\w*|tb|tib", "size", 1024**4),
    (r"%|процент\w*", "percent", 1),
    (r"₸|тенге|тг|kzt", "kzt", 1),
    (r"₽|руб\w*|rub", "rub", 1),
    (r"\$|usd|доллар\w*", "usd", 1),
    (r"€|eur|евро", "eur", 1),
)
_BASE_UNITS = {"time": "с", "calendar": "мес.", "size": "Б"}


@dataclass(frozen=True)
class Constraint:
    """Ограничение значения: интервал в базовой единице семейства и исходная запись."""

    op: Op
    first: float  # число в исходной единице
    second: float | None  # верхняя граница диапазона
    family: str  # time, calendar, size, percent, валюта, word (прочие), "" — без единицы
    factor: float
    unit: str
    text: str
    word: frozenset[str] = frozenset()  # варианты основы для единицы-слова

    @property
    def interval(self) -> tuple[float, bool, float, bool]:
        """(нижняя граница, включена, верхняя граница, включена) в базовой единице."""
        a = self.first * self.factor
        if self.op == "le":
            return -math.inf, False, a, True
        if self.op == "lt":
            return -math.inf, False, a, False
        if self.op == "ge":
            return a, True, math.inf, False
        if self.op == "gt":
            return a, False, math.inf, False
        if self.op == "range":
            assert self.second is not None
            return a, True, self.second * self.factor, True
        return a, True, a, True

    def describe(self, factor: float | None = None) -> str:
        """Запись без единицы: «≤ 3», «от 1 до 5»; factor — пересчёт в другую единицу."""
        scale = (self.factor / factor) if factor else 1.0
        first = _number(self.first * scale)
        if self.op == "range":
            assert self.second is not None
            return f"от {first} до {_number(self.second * scale)}"
        sign = {"le": "≤", "lt": "<", "ge": "≥", "gt": ">", "eq": "="}[self.op]
        return f"{sign} {first}"


@dataclass(frozen=True)
class NumericCheck:
    verdict: Verdict
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"verdict": self.verdict, "detail": self.detail}


def _number(value: float) -> str:
    return f"{value:g}"


def _to_float(text: str) -> float:
    return float(re.sub(r"[   ]", "", text).replace(",", "."))


def _unit(text: str | None) -> tuple[str, float, str, frozenset[str]] | None:
    """(семейство, множитель, единица, основы) или None, если это не единица."""
    if not text:
        return "", 1.0, "", frozenset()
    unit = text.casefold().replace("ё", "е").strip()
    words = unit.split()
    if len(words) == 2 and not _ADJECTIVE.search(words[0].rstrip(".")):
        return None  # «дня или» — не единица
    plain = unit.rstrip(".")
    for pattern, family, factor in _UNITS:
        if re.fullmatch(pattern, plain):
            return family, factor, text.strip(), frozenset()
    if len(words) == 1 and len(plain) >= 3 and plain.isalpha():
        return "word", 1.0, text.strip(), word_cores(plain)
    if len(words) == 2:
        return "word", 1.0, text.strip(), frozenset({plain})
    return None


def _build(
    op: Op, first: str, second: str | None, unit: str | None, text: str
) -> Constraint | None:
    info = _unit(unit)
    if info is None:
        return None
    family, factor, label, word = info
    a, b = _to_float(first), (_to_float(second) if second is not None else None)
    if b is not None and b < a:
        return None
    return Constraint(op, a, b, family, factor, label, text.strip(), word)


def _match(match: re.Match[str], text: str) -> Constraint | None:
    groups = match.groupdict()
    if "a" in groups:
        unit = groups["ub"] or groups["ua"]
        if groups["ua"] and groups["ub"] and _unit(groups["ua"]) != _unit(groups["ub"]):
            return None
        return _build("range", groups["a"], groups["b"], unit, text)
    op: Op = _OPS[groups["cmp"]] if groups["cmp"] else "eq"
    return _build(op, groups["n"], None, groups["u"], text)


def parse_constraint(value: str) -> Constraint | None:
    """Значение параметра целиком как ограничение; иначе None."""
    text = value.strip().rstrip(".;").strip()
    lowered = text.casefold()
    for pattern in (_RANGE_FROM, _RANGE_DASH, _SINGLE):
        match = pattern.fullmatch(lowered)
        if match is not None:
            return _match(match, text)
    return None


def find_constraint(text: str) -> Constraint | None:
    """Единственное ограничение в свободном тексте цитаты; иначе None (не угадывать)."""
    lowered = text.casefold()
    found: list[tuple[int, int, Constraint]] = []
    for pattern in (_RANGE_FROM, _RANGE_DASH, _SINGLE):
        for match in pattern.finditer(lowered):
            start, end = match.span()
            if start > 0 and (lowered[start - 1].isalnum() or lowered[start - 1] in "-_"):
                continue  # часть ID или слова («Ф-3»)
            if end < len(lowered) and lowered[end].isalnum():
                continue
            if any(start < e and s < end for s, e, _ in found):
                continue
            constraint = _match(match, text[start:end])
            if constraint is None:
                return None
            found.append((start, end, constraint))
    if len(found) != 1:
        return None
    start, end, constraint = found[0]
    if _NEGATION.search(lowered[:start] + " " + lowered[end:]):
        return None  # отрицание вне оборота сравнения меняет смысл
    return constraint


def compare_constraints(a: Constraint, b: Constraint) -> NumericCheck:
    if a.family != b.family or (a.family == "word" and not a.word & b.word):
        return NumericCheck(
            "incomparable",
            f"единицы несравнимы: «{a.unit or 'без единицы'}» и «{b.unit or 'без единицы'}»",
        )
    lo_a, lo_a_in, hi_a, hi_a_in = a.interval
    lo_b, lo_b_in, hi_b, hi_b_in = b.interval
    lower = max((lo_a, lo_a_in), (lo_b, lo_b_in), key=lambda x: (x[0], not x[1]))
    upper = min((hi_a, hi_a_in), (hi_b, hi_b_in), key=lambda x: (x[0], x[1]))
    disjoint = lower[0] > upper[0] or (lower[0] == upper[0] and not (lower[1] and upper[1]))
    same_unit = a.factor == b.factor
    unit = a.unit or ""
    shown = (
        f"{a.describe()} и {b.describe(a.factor)} {unit}".rstrip()
        if not same_unit
        else f"{a.describe()} и {b.describe()} {unit}".rstrip()
    )
    if disjoint:
        return NumericCheck(
            "disjoint",
            f"«{a.text}» и «{b.text}» не пересекаются ({shown}): нет значения, которое "
            "удовлетворяет обоим ограничениям",
        )
    return NumericCheck(
        "compatible",
        f"«{a.text}» и «{b.text}» совместимы ({shown}): есть значение, которое удовлетворяет "
        "обоим ограничениям",
    )
