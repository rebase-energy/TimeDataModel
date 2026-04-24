"""Tests for the DataPoint class."""

from __future__ import annotations

from datetime import datetime

import pytest
from timedatamodel.datapoint import DataPoint

TS = datetime(2024, 1, 1, 12, 0, 0)


class TestInit:
    def test_basic(self):
        dp = DataPoint(TS, 42.0)
        assert dp.timestamp == TS
        assert dp.value == 42.0

    def test_none_value(self):
        dp = DataPoint(TS, None)
        assert dp.value is None


class TestTupleCompat:
    """Unpacking, indexing, and length — tuple-style API."""

    def test_iter_unpacking(self):
        dp = DataPoint(TS, 3.14)
        ts, val = dp
        assert ts == TS
        assert val == 3.14

    @pytest.mark.parametrize(
        ("index", "expected"),
        [(0, TS), (1, 5.0)],
        ids=["timestamp", "value"],
    )
    def test_getitem(self, index, expected):
        dp = DataPoint(TS, 5.0)
        assert dp[index] == expected

    @pytest.mark.parametrize("index", [-1, 2, 100], ids=["neg", "two", "large"])
    def test_getitem_out_of_range(self, index):
        dp = DataPoint(TS, 1.0)
        with pytest.raises(IndexError, match="out of range"):
            dp[index]

    def test_len(self):
        assert len(DataPoint(TS, 0.0)) == 2


class TestEquality:
    def test_eq_datapoint(self):
        assert DataPoint(TS, 1.0) == DataPoint(TS, 1.0)

    def test_neq_datapoint(self):
        assert DataPoint(TS, 1.0) != DataPoint(TS, 2.0)

    def test_eq_tuple(self):
        assert DataPoint(TS, 1.0) == (TS, 1.0)

    def test_neq_tuple(self):
        assert DataPoint(TS, 1.0) != (TS, 9.0)

    def test_eq_unrelated_type(self):
        assert DataPoint(TS, 1.0).__eq__("hello") is NotImplemented


class TestHash:
    def test_consistency(self):
        dp = DataPoint(TS, 7.0)
        assert hash(dp) == hash(dp)

    def test_equal_objects_same_hash(self):
        a = DataPoint(TS, 7.0)
        b = DataPoint(TS, 7.0)
        assert hash(a) == hash(b)

    def test_dict_key(self):
        dp = DataPoint(TS, 7.0)
        d = {dp: "found"}
        assert d[DataPoint(TS, 7.0)] == "found"


class TestRepr:
    def test_repr_returns_string(self):
        dp = DataPoint(TS, 1.0)
        assert isinstance(repr(dp), str)

    def test_repr_html_returns_string(self):
        dp = DataPoint(TS, 1.0)
        assert isinstance(dp._repr_html_(), str)
