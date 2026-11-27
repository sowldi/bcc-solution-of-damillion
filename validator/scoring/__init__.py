"""Этап 9: оценка требований, сводка и итог по результатам этапов 3–8.

Ничего не проверяет заново и не обращается к модели. Подтверждённое (verified-замечания,
подтверждённые противоречия), гипотезы (на проверку человеку) и непроверенное разделены;
вердикт — категория с явными правилами, без баллов и процентов.
"""

from validator.scoring.assess import CATEGORY_PRIORITY, CRITICAL_CATEGORIES, assess
from validator.scoring.model import (
    STATUS_TITLES,
    VERDICT_TITLES,
    Coverage,
    DefectRef,
    RequirementAssessment,
    ReviewRef,
    Scorecard,
)

__all__ = [
    "CATEGORY_PRIORITY",
    "CRITICAL_CATEGORIES",
    "STATUS_TITLES",
    "VERDICT_TITLES",
    "Coverage",
    "DefectRef",
    "RequirementAssessment",
    "ReviewRef",
    "Scorecard",
    "assess",
]
