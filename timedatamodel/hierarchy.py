from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from html import escape
from typing import TYPE_CHECKING, Iterator

import numpy as np

from .location import Location
from .timeseries import TimeSeries

if TYPE_CHECKING:
    from .collection import TimeSeriesCollection
    from .table import TimeSeriesTable


class AggregationMethod(StrEnum):
    SUM = "sum"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"


_AGG_FUNCS = {
    AggregationMethod.SUM: np.nansum,
    AggregationMethod.MEAN: np.nanmean,
    AggregationMethod.MIN: np.nanmin,
    AggregationMethod.MAX: np.nanmax,
}


@dataclass(slots=True)
class HierarchyNode:
    key: str
    level: str
    children: list[HierarchyNode] = field(default_factory=list)
    timeseries: TimeSeries | None = None
    location: Location | None = None
    _parent: HierarchyNode | None = field(default=None, repr=False)

    # ---- properties -------------------------------------------------------

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def parent(self) -> HierarchyNode | None:
        return self._parent

    @property
    def siblings(self) -> list[HierarchyNode]:
        if self._parent is None:
            return []
        return [c for c in self._parent.children if c is not self]

    @property
    def depth(self) -> int:
        d = 0
        node = self._parent
        while node is not None:
            d += 1
            node = node._parent
        return d

    @property
    def path(self) -> list[str]:
        parts: list[str] = []
        node: HierarchyNode | None = self
        while node is not None:
            parts.append(node.key)
            node = node._parent
        parts.reverse()
        return parts

    @property
    def leaf_count(self) -> int:
        if self.is_leaf:
            return 1
        return sum(c.leaf_count for c in self.children)


def _set_parents(node: HierarchyNode, parent: HierarchyNode | None) -> None:
    """Recursively set _parent back-references."""
    node._parent = parent
    for child in node.children:
        _set_parents(child, node)


class HierarchicalTimeSeries:
    """A tree of time series organised into named hierarchy levels."""

    __slots__ = ("_root", "_name", "_description", "_aggregation", "_levels")

    def __init__(
        self,
        root: HierarchyNode,
        *,
        name: str | None = None,
        description: str | None = None,
        aggregation: AggregationMethod = AggregationMethod.SUM,
        levels: list[str] | None = None,
    ) -> None:
        self._root = root
        self._name = name
        self._description = description
        self._aggregation = aggregation
        _set_parents(root, None)
        if levels is not None:
            self._levels = list(levels)
        else:
            self._levels = self._infer_levels()

    def _infer_levels(self) -> list[str]:
        """Collect unique level names in BFS order."""
        seen: set[str] = set()
        order: list[str] = []
        queue: list[HierarchyNode] = [self._root]
        while queue:
            node = queue.pop(0)
            if node.level not in seen:
                seen.add(node.level)
                order.append(node.level)
            queue.extend(node.children)
        return order

    # ---- classmethods -----------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        tree: dict,
        series_map: dict[str, TimeSeries],
        *,
        levels: list[str] | None = None,
        name: str | None = None,
        description: str | None = None,
        aggregation: AggregationMethod = AggregationMethod.SUM,
    ) -> HierarchicalTimeSeries:
        """Build from a nested dict and a flat series mapping.

        Example::

            tree = {"Norway": {"Bergen": "bergen_ts", "Oslo": "oslo_ts"}}
            series_map = {"bergen_ts": ts_bergen, "oslo_ts": ts_oslo}
        """
        if levels is None:
            depth = cls._dict_depth(tree)
            levels = [f"level_{i}" for i in range(depth)]

        def _build(d: dict | str, depth_idx: int) -> HierarchyNode:
            if isinstance(d, str):
                ts = series_map.get(d)
                level_name = levels[depth_idx] if depth_idx < len(levels) else f"level_{depth_idx}"
                return HierarchyNode(key=d, level=level_name, timeseries=ts)
            level_name = levels[depth_idx] if depth_idx < len(levels) else f"level_{depth_idx}"
            children: list[HierarchyNode] = []
            for key, value in d.items():
                if isinstance(value, dict):
                    child = HierarchyNode(
                        key=key,
                        level=level_name,
                        children=[],
                    )
                    child.children = [_build({k: v}, depth_idx) for k, v in value.items()]
                    # Flatten: if the child has a single child at same level, unwrap
                    children.append(child)
                elif isinstance(value, str):
                    children.append(_build(value, depth_idx + 1))
                    children[-1].key = key
                else:
                    raise TypeError(f"unexpected value type {type(value)}")
            if len(d) == 1:
                key = next(iter(d))
                node = HierarchyNode(key=key, level=level_name, children=children)
                return node
            # Multiple keys at top level — create a synthetic root
            node = HierarchyNode(key="root", level=level_name, children=children)
            return node

        root = _build(tree, 0)
        return cls(root, name=name, description=description, aggregation=aggregation, levels=levels)

    @staticmethod
    def _dict_depth(d: dict) -> int:
        if not isinstance(d, dict) or not d:
            return 0
        return 1 + max(
            HierarchicalTimeSeries._dict_depth(v) if isinstance(v, dict) else 1
            for v in d.values()
        )

    @classmethod
    def from_dataframe(
        cls,
        df,
        level_columns: list[str],
        value_column: str,
        timestamp_column: str | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        aggregation: AggregationMethod = AggregationMethod.SUM,
        frequency=None,
        timezone: str = "UTC",
    ) -> HierarchicalTimeSeries:
        """Build from a long-format pandas DataFrame with hierarchy columns."""
        from .enums import Frequency

        if frequency is None:
            frequency = Frequency.NONE

        # Group by all level columns to find unique leaf paths
        grouped = df.groupby(level_columns)

        # Build series for each leaf
        series_map: dict[str, TimeSeries] = {}
        paths: list[tuple[str, ...]] = []
        for group_key, group_df in grouped:
            if isinstance(group_key, str):
                group_key = (group_key,)
            path = tuple(str(k) for k in group_key)
            paths.append(path)
            leaf_key = "/".join(path)

            if timestamp_column is not None:
                from datetime import datetime as _dt
                timestamps = group_df[timestamp_column].tolist()
            else:
                timestamps = group_df.index.tolist()

            values_list = group_df[value_column].tolist()
            ts = TimeSeries(
                frequency,
                timezone=timezone,
                timestamps=timestamps,
                values=[float(v) if v == v else None for v in values_list],
                name=leaf_key,
            )
            series_map[leaf_key] = ts

        # Build tree from paths using nested dicts keyed by node key.
        # Each entry is (HierarchyNode, children_dict).
        levels = list(level_columns)
        tree_root: dict[str, tuple[HierarchyNode, dict]] = {}

        for path in paths:
            leaf_key = "/".join(path)
            current = tree_root
            for depth, key in enumerate(path[:-1]):
                if key not in current:
                    node = HierarchyNode(key=key, level=levels[depth], children=[])
                    current[key] = (node, {})
                current = current[key][1]

            leaf_name = path[-1]
            leaf_level = levels[len(path) - 1] if len(path) - 1 < len(levels) else f"level_{len(path)-1}"
            leaf_node = HierarchyNode(
                key=leaf_name, level=leaf_level, timeseries=series_map[leaf_key]
            )
            current[leaf_name] = (leaf_node, {})

        def _resolve(d: dict[str, tuple[HierarchyNode, dict]]) -> list[HierarchyNode]:
            result: list[HierarchyNode] = []
            for node, children_dict in d.values():
                node.children = _resolve(children_dict)
                result.append(node)
            return result

        top_nodes = _resolve(tree_root)

        if len(top_nodes) == 1:
            root = top_nodes[0]
        else:
            root = HierarchyNode(key="root", level="root", children=top_nodes)
            levels = ["root"] + levels

        return cls(root, name=name, description=description, aggregation=aggregation, levels=levels)

    # ---- properties -------------------------------------------------------

    @property
    def root(self) -> HierarchyNode:
        return self._root

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def description(self) -> str | None:
        return self._description

    @property
    def levels(self) -> list[str]:
        return list(self._levels)

    @property
    def n_levels(self) -> int:
        return len(self._levels)

    @property
    def n_leaves(self) -> int:
        return self._root.leaf_count

    @property
    def n_nodes(self) -> int:
        return sum(1 for _ in self.walk())

    # ---- traversal --------------------------------------------------------

    def get_node(self, *path: str) -> HierarchyNode:
        """Navigate to a node by key path."""
        node = self._root
        for key in path:
            if key == node.key:
                continue
            found = False
            for child in node.children:
                if child.key == key:
                    node = child
                    found = True
                    break
            if not found:
                raise KeyError(f"key {key!r} not found under {node.key!r}")
        return node

    def get_level(self, level: str | int) -> list[HierarchyNode]:
        """All nodes at a given level name or depth index."""
        if isinstance(level, int):
            return [n for n in self.walk() if n.depth == level]
        return [n for n in self.walk() if n.level == level]

    def leaves(self) -> list[HierarchyNode]:
        """All leaf nodes."""
        return [n for n in self.walk() if n.is_leaf]

    def walk(self, order: str = "pre") -> Iterator[HierarchyNode]:
        """Yield nodes in pre-order or post-order."""
        if order == "pre":
            yield from self._walk_pre(self._root)
        elif order == "post":
            yield from self._walk_post(self._root)
        else:
            raise ValueError(f"order must be 'pre' or 'post', got {order!r}")

    @staticmethod
    def _walk_pre(node: HierarchyNode) -> Iterator[HierarchyNode]:
        yield node
        for child in node.children:
            yield from HierarchicalTimeSeries._walk_pre(child)

    @staticmethod
    def _walk_post(node: HierarchyNode) -> Iterator[HierarchyNode]:
        for child in node.children:
            yield from HierarchicalTimeSeries._walk_post(child)
        yield node

    def subtree(self, *path: str) -> HierarchicalTimeSeries:
        """Extract sub-hierarchy rooted at the given path."""
        import copy
        node = self.get_node(*path)
        new_root = copy.deepcopy(node)
        remaining_levels = [l for l in self._levels if l in {n.level for n in self._walk_pre(new_root)}]
        return HierarchicalTimeSeries(
            new_root,
            name=self._name,
            description=self._description,
            aggregation=self._aggregation,
            levels=remaining_levels or None,
        )

    # ---- aggregation ------------------------------------------------------

    def aggregate(
        self,
        node: HierarchyNode | None = None,
        method: AggregationMethod | None = None,
        auto_align: bool = False,
    ) -> TimeSeries:
        """Recursive bottom-up aggregation."""
        if node is None:
            node = self._root
        if method is None:
            method = self._aggregation

        if node.is_leaf:
            if node.timeseries is None:
                raise ValueError(f"leaf node {node.key!r} has no timeseries")
            return node.timeseries

        child_series = [self.aggregate(c, method, auto_align) for c in node.children]

        if auto_align:
            child_series = self._align_series(child_series)
        else:
            ref_ts = child_series[0]._timestamps
            for i, cs in enumerate(child_series[1:], 1):
                if cs._timestamps != ref_ts:
                    raise ValueError(
                        f"timestamps mismatch between children of {node.key!r}: "
                        f"child 0 vs child {i}. Use auto_align=True to align."
                    )

        arrays = [s._to_float_array() for s in child_series]
        stacked = np.column_stack(arrays)
        agg_func = _AGG_FUNCS[method]
        result_arr = agg_func(stacked, axis=1)

        return TimeSeries(
            child_series[0].frequency,
            timezone=child_series[0].timezone,
            timestamps=list(child_series[0]._timestamps),
            values=child_series[0]._from_float_array(result_arr),
            name=node.key,
        )

    @staticmethod
    def _align_series(series_list: list[TimeSeries]) -> list[TimeSeries]:
        """Align series to the union of all timestamps, filling NaN where missing."""
        all_ts: set = set()
        for s in series_list:
            all_ts.update(s._timestamps)
        union_ts = sorted(all_ts)

        aligned: list[TimeSeries] = []
        for s in series_list:
            ts_set = dict(zip(s._timestamps, s._values))
            new_values = [ts_set.get(t) for t in union_ts]
            aligned.append(TimeSeries(
                s.frequency,
                timezone=s.timezone,
                timestamps=union_ts,
                values=new_values,
                name=s.name,
            ))
        return aligned

    def aggregate_level(
        self,
        level: str | int,
        method: AggregationMethod | None = None,
        auto_align: bool = False,
    ) -> dict[str, TimeSeries]:
        """Aggregate every node at *level*."""
        nodes = self.get_level(level)
        return {n.key: self.aggregate(n, method, auto_align) for n in nodes}

    # ---- conversion -------------------------------------------------------

    def to_collection(
        self, level: str | int | None = None
    ) -> TimeSeriesCollection:
        """Flatten to a TimeSeriesCollection."""
        from .collection import TimeSeriesCollection

        if level is None:
            items = {n.key: n.timeseries for n in self.leaves() if n.timeseries is not None}
            return TimeSeriesCollection(items, name=self._name)
        agg = self.aggregate_level(level)
        return TimeSeriesCollection(agg, name=self._name)

    def to_table(
        self, level: str | int | None = None
    ) -> TimeSeriesTable:
        """Flatten to a TimeSeriesTable (requires shared timestamps)."""
        collection = self.to_collection(level)
        series_list = [v for v in collection.values() if isinstance(v, TimeSeries)]
        if not series_list:
            raise ValueError("no TimeSeries found to build table")
        return TimeSeries.merge(series_list)

    # ---- sequence protocol ------------------------------------------------

    def __len__(self) -> int:
        return self.n_nodes

    def __contains__(self, key: str) -> bool:
        return any(n.key == key for n in self.walk())

    def __getitem__(self, path: str | tuple[str, ...]) -> HierarchyNode:
        if isinstance(path, str):
            parts = path.split("/")
        else:
            parts = list(path)
        return self.get_node(*parts)

    def __iter__(self) -> Iterator[HierarchyNode]:
        return self.walk()

    # ---- display ----------------------------------------------------------

    def __repr__(self) -> str:
        class_name = type(self).__name__
        title = class_name
        if self._name:
            title = f"{class_name}: {self._name}"

        lines: list[str] = [title]
        self._repr_tree(self._root, "", True, lines)
        info = f"{self.n_nodes} nodes, {self.n_leaves} leaves, {self.n_levels} levels"
        lines.append(f"({info})")
        return "\n".join(lines)

    @staticmethod
    def _repr_tree(
        node: HierarchyNode,
        prefix: str,
        is_last: bool,
        lines: list[str],
    ) -> None:
        connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
        label = node.key
        if node.is_leaf and node.timeseries is not None:
            n = len(node.timeseries)
            label += f" [{n} pts]"
        elif not node.is_leaf:
            label += f" ({node.level})"
        lines.append(f"{prefix}{connector}{label}")
        new_prefix = prefix + ("    " if is_last else "\u2502   ")
        for i, child in enumerate(node.children):
            HierarchicalTimeSeries._repr_tree(
                child, new_prefix, i == len(node.children) - 1, lines
            )

    def _repr_html_(self) -> str:
        title = type(self).__name__
        if self._name:
            title = f"{title}: {escape(self._name)}"
        info = f"{self.n_nodes} nodes, {self.n_leaves} leaves, {self.n_levels} levels"

        html = [
            '<div style="font-family: monospace; font-size: 13px;">',
            f"<b>{escape(title)}</b><br>",
            self._html_node(self._root),
            f"<i>{escape(info)}</i>",
            "</div>",
        ]
        return "\n".join(html)

    @staticmethod
    def _html_node(node: HierarchyNode) -> str:
        if node.is_leaf:
            label = escape(node.key)
            if node.timeseries is not None:
                label += f" [{len(node.timeseries)} pts]"
            return f"<div style='margin-left:16px'>{label}</div>\n"
        label = f"{escape(node.key)} ({escape(node.level)})"
        children_html = "".join(
            HierarchicalTimeSeries._html_node(c) for c in node.children
        )
        return (
            f"<details open style='margin-left:16px'>"
            f"<summary>{label}</summary>\n"
            f"{children_html}"
            f"</details>\n"
        )
