"""Сегментация требований: документ этапа 1 → фрагменты с ролями, типами, стабильными ID
и доказательными ссылками на источник."""

from validator.segmentation.matching import RequirementMatch, match_segmentations
from validator.segmentation.model import (
    CATEGORY_TITLES,
    ROLE_TITLES,
    SCHEMA,
    Category,
    Confidence,
    Requirement,
    Role,
    Segmentation,
    SourceRef,
)
from validator.segmentation.segmenter import SegmentationError, segment_document
from validator.segmentation.sources import source_location, span_sources
from validator.segmentation.text import Statement
from validator.segmentation.verify import verify_segmentation

__all__ = [
    "CATEGORY_TITLES",
    "ROLE_TITLES",
    "SCHEMA",
    "Category",
    "Confidence",
    "Requirement",
    "RequirementMatch",
    "Role",
    "Segmentation",
    "SegmentationError",
    "SourceRef",
    "Statement",
    "match_segmentations",
    "segment_document",
    "source_location",
    "span_sources",
    "verify_segmentation",
]
