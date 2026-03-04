from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest


@pytest.fixture
def utc_timestamps():
    """Five hourly UTC timestamps starting 2024-01-01."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return [base + timedelta(hours=i) for i in range(5)]


@pytest.fixture
def sample_values():
    """Simple float values with one None (missing)."""
    return [1.0, 2.0, 3.0, None, 5.0]
