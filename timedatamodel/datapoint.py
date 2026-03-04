from __future__ import annotations

from datetime import datetime
from html import escape
from typing import NamedTuple

import numpy as np

from ._base import _get_repr_css, _fmt_short_date, _fmt_tz_with_offset, _render_box


class DataPoint(NamedTuple):
    timestamp: datetime
    value: float | None

    @staticmethod
    def _fmt_value(v: float | None) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "NaN"
        if v == int(v):
            return f"{v:.1f}"
        return f"{v:g}"

    def __repr__(self) -> str:
        meta_lines: list[str] = []

        ts_str = _fmt_short_date(self.timestamp)
        meta_lines.append(f"Timestamp:  {ts_str}")

        if hasattr(self.timestamp, "utcoffset") and self.timestamp.utcoffset() is not None:
            tz_str = str(self.timestamp.tzinfo)
            tz_display = _fmt_tz_with_offset(tz_str, [self.timestamp])
            meta_lines.append(f"Timezone:   {tz_display}")

        meta_lines.append(f"Value:      {self._fmt_value(self.value)}")

        return _render_box("DataPoint", meta_lines)

    def _repr_html_(self) -> str:
        ts_str = escape(_fmt_short_date(self.timestamp))
        val_str = escape(self._fmt_value(self.value))

        meta_rows = [("Timestamp", ts_str)]

        if hasattr(self.timestamp, "utcoffset") and self.timestamp.utcoffset() is not None:
            tz_str = str(self.timestamp.tzinfo)
            tz_display = escape(_fmt_tz_with_offset(tz_str, [self.timestamp]))
            meta_rows.append(("Timezone", tz_display))

        meta_rows.append(("Value", val_str))

        html = [_get_repr_css(), '<div class="ts-repr">']
        html.append('<div class="ts-header">DataPoint</div>')
        html.append('<div class="ts-meta"><table>')
        for label, value in meta_rows:
            html.append(f"<tr><td>{label}</td><td>{value}</td></tr>")
        html.append("</table></div>")
        html.append("</div>")
        return "\n".join(html)
