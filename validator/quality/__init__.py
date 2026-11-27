"""Этап 8: противоречия между требованиями, критерии приёмки, полнота.

Детерминированные правила этапа (CTR-01, CTR-02, ACC-01) — вне каталога lint и проходят тот
же Evidence Gate со своими записями каталога. Новых запросов к модели этап не делает:
семантику дают обзор требований (этап 6) и проверка пар (этап 7), а этап 8 сводит их
результаты, не повышая статус гипотез. Пакет не зависит от LLM-слоя.
"""

from validator.quality.analysis import ACC_01, CTR_01, CTR_02, STAGE8_RULES, analyze_quality
from validator.quality.constraints import (
    Constraint,
    NumericCheck,
    compare_constraints,
    find_constraint,
    parse_constraint,
)
from validator.quality.model import Contradiction, QualityResult

__all__ = [
    "ACC_01",
    "CTR_01",
    "CTR_02",
    "STAGE8_RULES",
    "Constraint",
    "Contradiction",
    "NumericCheck",
    "QualityResult",
    "analyze_quality",
    "compare_constraints",
    "find_constraint",
    "parse_constraint",
]
