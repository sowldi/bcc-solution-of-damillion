"""Формирование и запись отчёта."""

from validator.report.markdown import render_report
from validator.report.writer import write_text_atomic

__all__ = ["render_report", "write_text_atomic"]
