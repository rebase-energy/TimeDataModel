import pytest
from datetime import datetime

from timedatamodel import (
    AggregationMethod,
    Frequency,
    HierarchicalTimeSeries,
    HierarchyNode,
    TimeSeries,
    TimeSeriesCollection,
    TimeSeriesTable,
)


# ---- fixtures --------------------------------------------------------------


def _make_ts(name, values, timestamps=None):
    if timestamps is None:
        timestamps = [datetime(2024, 1, i + 1) for i in range(len(values))]
    return TimeSeries(
        Frequency.P1D,
        timestamps=timestamps,
        values=values,
        name=name,
    )


@pytest.fixture
def norway_hierarchy():
    """Two-level hierarchy: country -> city."""
    ts_oslo = _make_ts("Oslo", [10.0, 20.0, 30.0])
    ts_bergen = _make_ts("Bergen", [5.0, 10.0, 15.0])
    ts_trondheim = _make_ts("Trondheim", [3.0, 6.0, 9.0])

    oslo = HierarchyNode(key="Oslo", level="city", timeseries=ts_oslo)
    bergen = HierarchyNode(key="Bergen", level="city", timeseries=ts_bergen)
    trondheim = HierarchyNode(key="Trondheim", level="city", timeseries=ts_trondheim)

    norway = HierarchyNode(
        key="Norway",
        level="country",
        children=[oslo, bergen, trondheim],
    )
    return HierarchicalTimeSeries(
        norway,
        name="Norway Energy",
        levels=["country", "city"],
    )


@pytest.fixture
def two_country_hierarchy():
    """Three-level hierarchy: region -> country -> city."""
    ts_oslo = _make_ts("Oslo", [10.0, 20.0])
    ts_bergen = _make_ts("Bergen", [5.0, 10.0])
    ts_stockholm = _make_ts("Stockholm", [8.0, 16.0])

    oslo = HierarchyNode(key="Oslo", level="city", timeseries=ts_oslo)
    bergen = HierarchyNode(key="Bergen", level="city", timeseries=ts_bergen)
    stockholm = HierarchyNode(key="Stockholm", level="city", timeseries=ts_stockholm)

    norway = HierarchyNode(key="Norway", level="country", children=[oslo, bergen])
    sweden = HierarchyNode(key="Sweden", level="country", children=[stockholm])

    nordics = HierarchyNode(
        key="Nordics", level="region", children=[norway, sweden]
    )
    return HierarchicalTimeSeries(
        nordics,
        name="Nordic Energy",
        levels=["region", "country", "city"],
    )


# ---- construction ----------------------------------------------------------


class TestConstruction:
    def test_manual_tree(self, norway_hierarchy):
        h = norway_hierarchy
        assert h.name == "Norway Energy"
        assert h.n_leaves == 3
        assert h.n_levels == 2
        assert h.levels == ["country", "city"]

    def test_from_dict(self):
        ts_oslo = _make_ts("Oslo", [1.0, 2.0])
        ts_bergen = _make_ts("Bergen", [3.0, 4.0])
        tree = {"Norway": {"Oslo": "oslo_ts", "Bergen": "bergen_ts"}}
        series_map = {"oslo_ts": ts_oslo, "bergen_ts": ts_bergen}
        h = HierarchicalTimeSeries.from_dict(
            tree, series_map, name="test", levels=["country", "city"]
        )
        assert h.n_leaves == 2
        assert "Oslo" in h
        assert "Bergen" in h

    def test_from_dict_infers_levels(self):
        ts_a = _make_ts("A", [1.0])
        tree = {"Root": {"A": "a_ts"}}
        h = HierarchicalTimeSeries.from_dict(tree, {"a_ts": ts_a})
        assert h.n_levels >= 1

    def test_from_dataframe(self):
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        df = pd.DataFrame({
            "country": ["Norway", "Norway", "Sweden", "Sweden"],
            "city": ["Oslo", "Bergen", "Stockholm", "Gothenburg"],
            "value": [10.0, 5.0, 8.0, 3.0],
            "timestamp": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
            ],
        })
        h = HierarchicalTimeSeries.from_dataframe(
            df,
            level_columns=["country", "city"],
            value_column="value",
            timestamp_column="timestamp",
            name="test",
        )
        assert h.n_leaves == 4


# ---- properties -----------------------------------------------------------


class TestProperties:
    def test_root(self, norway_hierarchy):
        assert norway_hierarchy.root.key == "Norway"

    def test_n_nodes(self, norway_hierarchy):
        # 1 root + 3 leaves = 4
        assert norway_hierarchy.n_nodes == 4

    def test_n_leaves(self, norway_hierarchy):
        assert norway_hierarchy.n_leaves == 3

    def test_n_levels(self, two_country_hierarchy):
        assert two_country_hierarchy.n_levels == 3


# ---- node properties -------------------------------------------------------


class TestNodeProperties:
    def test_is_leaf(self, norway_hierarchy):
        leaves = norway_hierarchy.leaves()
        for leaf in leaves:
            assert leaf.is_leaf is True
        assert norway_hierarchy.root.is_leaf is False

    def test_parent(self, norway_hierarchy):
        oslo = norway_hierarchy.get_node("Norway", "Oslo")
        assert oslo.parent is not None
        assert oslo.parent.key == "Norway"

    def test_root_parent_is_none(self, norway_hierarchy):
        assert norway_hierarchy.root.parent is None

    def test_siblings(self, norway_hierarchy):
        oslo = norway_hierarchy.get_node("Norway", "Oslo")
        sibling_keys = {s.key for s in oslo.siblings}
        assert "Bergen" in sibling_keys
        assert "Trondheim" in sibling_keys
        assert "Oslo" not in sibling_keys

    def test_depth(self, two_country_hierarchy):
        assert two_country_hierarchy.root.depth == 0
        norway = two_country_hierarchy.get_node("Nordics", "Norway")
        assert norway.depth == 1
        oslo = two_country_hierarchy.get_node("Nordics", "Norway", "Oslo")
        assert oslo.depth == 2

    def test_path(self, two_country_hierarchy):
        oslo = two_country_hierarchy.get_node("Nordics", "Norway", "Oslo")
        assert oslo.path == ["Nordics", "Norway", "Oslo"]

    def test_leaf_count(self, two_country_hierarchy):
        norway = two_country_hierarchy.get_node("Nordics", "Norway")
        assert norway.leaf_count == 2


# ---- traversal -------------------------------------------------------------


class TestTraversal:
    def test_get_node(self, norway_hierarchy):
        oslo = norway_hierarchy.get_node("Norway", "Oslo")
        assert oslo.key == "Oslo"
        assert oslo.timeseries is not None

    def test_get_node_invalid(self, norway_hierarchy):
        with pytest.raises(KeyError):
            norway_hierarchy.get_node("Norway", "Helsinki")

    def test_get_level_by_name(self, two_country_hierarchy):
        countries = two_country_hierarchy.get_level("country")
        keys = {n.key for n in countries}
        assert keys == {"Norway", "Sweden"}

    def test_get_level_by_index(self, two_country_hierarchy):
        level_0 = two_country_hierarchy.get_level(0)
        assert len(level_0) == 1
        assert level_0[0].key == "Nordics"

    def test_leaves(self, norway_hierarchy):
        leaves = norway_hierarchy.leaves()
        assert len(leaves) == 3
        keys = {l.key for l in leaves}
        assert keys == {"Oslo", "Bergen", "Trondheim"}

    def test_walk_pre(self, norway_hierarchy):
        nodes = list(norway_hierarchy.walk("pre"))
        assert nodes[0].key == "Norway"  # root first in pre-order
        assert len(nodes) == 4

    def test_walk_post(self, norway_hierarchy):
        nodes = list(norway_hierarchy.walk("post"))
        assert nodes[-1].key == "Norway"  # root last in post-order
        assert len(nodes) == 4

    def test_walk_invalid_order(self, norway_hierarchy):
        with pytest.raises(ValueError):
            list(norway_hierarchy.walk("invalid"))

    def test_subtree(self, two_country_hierarchy):
        sub = two_country_hierarchy.subtree("Nordics", "Norway")
        assert sub.root.key == "Norway"
        assert sub.n_leaves == 2
        assert "Oslo" in sub


# ---- aggregation -----------------------------------------------------------


class TestAggregation:
    def test_sum(self, norway_hierarchy):
        result = norway_hierarchy.aggregate(method=AggregationMethod.SUM)
        # 10+5+3=18, 20+10+6=36, 30+15+9=54
        assert result.values == [18.0, 36.0, 54.0]

    def test_mean(self, norway_hierarchy):
        result = norway_hierarchy.aggregate(method=AggregationMethod.MEAN)
        assert abs(result.values[0] - 6.0) < 0.01  # (10+5+3)/3
        assert abs(result.values[1] - 12.0) < 0.01  # (20+10+6)/3

    def test_min(self, norway_hierarchy):
        result = norway_hierarchy.aggregate(method=AggregationMethod.MIN)
        assert result.values == [3.0, 6.0, 9.0]

    def test_max(self, norway_hierarchy):
        result = norway_hierarchy.aggregate(method=AggregationMethod.MAX)
        assert result.values == [10.0, 20.0, 30.0]

    def test_aggregate_node(self, two_country_hierarchy):
        norway = two_country_hierarchy.get_node("Nordics", "Norway")
        result = two_country_hierarchy.aggregate(norway, AggregationMethod.SUM)
        assert result.values == [15.0, 30.0]  # 10+5, 20+10

    def test_aggregate_leaf(self, norway_hierarchy):
        oslo = norway_hierarchy.get_node("Norway", "Oslo")
        result = norway_hierarchy.aggregate(oslo)
        assert result.values == [10.0, 20.0, 30.0]

    def test_multi_level_sum(self, two_country_hierarchy):
        result = two_country_hierarchy.aggregate(method=AggregationMethod.SUM)
        # (10+5+8)=23, (20+10+16)=46
        assert result.values == [23.0, 46.0]

    def test_mismatched_timestamps_raises(self):
        ts_a = _make_ts("A", [1.0, 2.0], [datetime(2024, 1, 1), datetime(2024, 1, 2)])
        ts_b = _make_ts("B", [3.0, 4.0], [datetime(2024, 2, 1), datetime(2024, 2, 2)])
        a = HierarchyNode(key="A", level="leaf", timeseries=ts_a)
        b = HierarchyNode(key="B", level="leaf", timeseries=ts_b)
        root = HierarchyNode(key="Root", level="root", children=[a, b])
        h = HierarchicalTimeSeries(root, levels=["root", "leaf"])
        with pytest.raises(ValueError, match="timestamps mismatch"):
            h.aggregate()

    def test_auto_align(self):
        ts_a = _make_ts("A", [1.0, 2.0], [datetime(2024, 1, 1), datetime(2024, 1, 2)])
        ts_b = _make_ts("B", [3.0], [datetime(2024, 1, 2)])
        a = HierarchyNode(key="A", level="leaf", timeseries=ts_a)
        b = HierarchyNode(key="B", level="leaf", timeseries=ts_b)
        root = HierarchyNode(key="Root", level="root", children=[a, b])
        h = HierarchicalTimeSeries(root, levels=["root", "leaf"])
        result = h.aggregate(auto_align=True)
        # Jan 1: 1.0 + NaN = 1.0 (nansum), Jan 2: 2.0 + 3.0 = 5.0
        assert result.values[0] == 1.0
        assert result.values[1] == 5.0

    def test_aggregate_level(self, two_country_hierarchy):
        result = two_country_hierarchy.aggregate_level("country")
        assert "Norway" in result
        assert "Sweden" in result
        assert result["Norway"].values == [15.0, 30.0]
        assert result["Sweden"].values == [8.0, 16.0]


# ---- conversion -----------------------------------------------------------


class TestConversion:
    def test_to_collection_leaves(self, norway_hierarchy):
        coll = norway_hierarchy.to_collection()
        assert isinstance(coll, TimeSeriesCollection)
        assert len(coll) == 3

    def test_to_collection_level(self, two_country_hierarchy):
        coll = two_country_hierarchy.to_collection("country")
        assert isinstance(coll, TimeSeriesCollection)
        assert len(coll) == 2

    def test_to_table(self, norway_hierarchy):
        tbl = norway_hierarchy.to_table()
        assert isinstance(tbl, TimeSeriesTable)
        assert tbl.n_columns == 3


# ---- display ---------------------------------------------------------------


class TestDisplay:
    def test_repr(self, norway_hierarchy):
        r = repr(norway_hierarchy)
        assert "Norway" in r
        assert "Oslo" in r
        assert len(r) > 0

    def test_repr_html(self, norway_hierarchy):
        html = norway_hierarchy._repr_html_()
        assert "<" in html
        assert "Norway" in html
        assert len(html) > 0


# ---- sequence protocol ----------------------------------------------------


class TestSequenceProtocol:
    def test_len(self, norway_hierarchy):
        assert len(norway_hierarchy) == 4

    def test_contains(self, norway_hierarchy):
        assert "Oslo" in norway_hierarchy
        assert "Helsinki" not in norway_hierarchy

    def test_getitem_string(self, norway_hierarchy):
        node = norway_hierarchy["Norway/Oslo"]
        assert node.key == "Oslo"

    def test_getitem_tuple(self, norway_hierarchy):
        node = norway_hierarchy[("Norway", "Oslo")]
        assert node.key == "Oslo"

    def test_iter(self, norway_hierarchy):
        keys = [n.key for n in norway_hierarchy]
        assert "Norway" in keys
        assert "Oslo" in keys


# ---- edge cases ------------------------------------------------------------


class TestEdgeCases:
    def test_single_node(self):
        ts = _make_ts("solo", [1.0, 2.0])
        root = HierarchyNode(key="solo", level="root", timeseries=ts)
        h = HierarchicalTimeSeries(root, levels=["root"])
        assert h.n_nodes == 1
        assert h.n_leaves == 1
        result = h.aggregate()
        assert result.values == [1.0, 2.0]

    def test_leaf_only_hierarchy(self):
        ts = _make_ts("leaf", [5.0])
        root = HierarchyNode(key="leaf", level="leaf", timeseries=ts)
        h = HierarchicalTimeSeries(root)
        assert len(h.leaves()) == 1
        assert h.to_collection() is not None

    def test_leaf_without_timeseries(self):
        root = HierarchyNode(key="empty_leaf", level="leaf")
        h = HierarchicalTimeSeries(root)
        with pytest.raises(ValueError, match="no timeseries"):
            h.aggregate()
