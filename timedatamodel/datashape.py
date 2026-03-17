"""
Shared DataShape enum and column constants.

Pure-Python, no third-party dependencies — safe to import from both the NumPy
backend (which must work without polars) and the Polars backend.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List


class DataShape(str, Enum):
    """Which temporal columns are present in the underlying data store."""

    SIMPLE     = "SIMPLE"     # valid_time + value
    VERSIONED  = "VERSIONED"  # knowledge_time + valid_time + value
    AUDIT      = "AUDIT"      # knowledge_time + change_time + valid_time + value
    CORRECTED  = "CORRECTED"  # valid_time + change_time + value


#: Required columns per shape.
_REQUIRED_COLUMNS: Dict[DataShape, List[str]] = {
    DataShape.SIMPLE:     ["valid_time", "value"],
    DataShape.VERSIONED:  ["knowledge_time", "valid_time", "value"],
    DataShape.AUDIT:      ["knowledge_time", "change_time", "valid_time", "value"],
    DataShape.CORRECTED:  ["valid_time", "change_time", "value"],
}

_TIME_COLS: frozenset = frozenset(
    {"valid_time", "valid_time_end", "knowledge_time", "change_time"}
)
