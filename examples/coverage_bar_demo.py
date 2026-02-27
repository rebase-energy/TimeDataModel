"""Demo: TimeSeries objects and coverage_bar() usage."""

from datetime import datetime, timedelta, timezone

import numpy as np

from timedatamodel import (
    DataType,
    Frequency,
    MultivariateTimeSeries,
    Resolution,
    TimeSeries,
)

resolution = Resolution(frequency=Frequency.PT1H, timezone="Europe/Oslo")
base = datetime(2024, 6, 1, tzinfo=timezone.utc)

# ── TimeSeries with full coverage ────────────────────────────────────────

timestamps = [base + timedelta(hours=i) for i in range(168)]  # one week
values = [20.0 + 5 * np.sin(i / 12 * np.pi) for i in range(168)]

ts_full = TimeSeries(
    resolution,
    timestamps=timestamps,
    values=values,
    name="temperature",
    unit="°C",
    data_type=DataType.ACTUAL,
)

print("═" * 60)
print("  TimeSeries — full coverage (168 hourly points)")
print("═" * 60)
print(ts_full)
print()
print(ts_full.coverage_bar())

# ── TimeSeries with gaps ─────────────────────────────────────────────────

values_with_gaps = [
    v if not (40 <= i < 65 or 120 <= i < 140) else None
    for i, v in enumerate(values)
]

ts_gaps = TimeSeries(
    resolution,
    timestamps=timestamps,
    values=values_with_gaps,
    name="temperature",
    unit="°C",
    data_type=DataType.ACTUAL,
)

print()
print("═" * 60)
print("  TimeSeries — with two gaps")
print("═" * 60)
print(ts_gaps)
print()
print(ts_gaps.coverage_bar())

# ── MultivariateTimeSeries with per-column gaps ──────────────────────────

n = 168
power = np.array([100 + 20 * np.sin(i / 24 * np.pi) for i in range(n)])
temp = np.array([22 + 4 * np.cos(i / 12 * np.pi) for i in range(n)])
wind = np.array([8 + 3 * np.sin(i / 6 * np.pi) for i in range(n)])

# Punch holes in different columns at different times
power[30:50] = np.nan
temp[80:110] = np.nan
wind[50:60] = np.nan
wind[130:155] = np.nan

vals = np.column_stack([power, temp, wind])

mts = MultivariateTimeSeries(
    resolution,
    timestamps=timestamps,
    values=vals,
    names=["power", "temperature", "wind_speed"],
    units=["MW", "°C", "m/s"],
)

print()
print("═" * 60)
print("  MultivariateTimeSeries — three columns with gaps")
print("═" * 60)
print(mts)
print()
print(mts.coverage_bar())
