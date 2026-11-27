"""Детерминированные правила (lint): каталог, контекст, движок.

Не зависит от CLI, отчёта и LLM: на входе документ и сегментация, на выходе LintResult —
замечания, прошедшие Evidence Gate.
"""

from validator.lint.base import Hit, LintContext, Rule, Span
from validator.lint.catalog import CATALOG, CATALOG_VERSION, rule_by_id
from validator.lint.context import build_context
from validator.lint.engine import SCHEMA, LintResult, run_lint, run_rule

__all__ = [
    "CATALOG",
    "CATALOG_VERSION",
    "SCHEMA",
    "Hit",
    "LintContext",
    "LintResult",
    "Rule",
    "Span",
    "build_context",
    "rule_by_id",
    "run_lint",
    "run_rule",
]
