from __future__ import annotations

from datetime import datetime
from html import escape

from ._base import _fmt_short_date
from ._theme import THEME


class CoverageBar:
    """Displayable coverage bar for TimeSeries objects."""

    _TERM_BINS = 40
    _SVG_BINS = 60

    def __init__(
        self,
        masks: list[tuple[str, list[bool]]],
        begin: datetime | None,
        end: datetime | None,
    ) -> None:
        self._masks = masks
        self._begin = begin
        self._end = end

    @staticmethod
    def _bin_coverage(mask: list[bool], n_bins: int) -> list[bool]:
        n = len(mask)
        if n == 0:
            return [False] * n_bins
        actual_bins = min(n_bins, n)
        bins: list[bool] = []
        for i in range(actual_bins):
            lo = i * n // actual_bins
            hi = (i + 1) * n // actual_bins
            bins.append(any(mask[lo:hi]))
        return bins

    def __repr__(self) -> str:
        if not self._masks:
            return ""
        n_bins = self._TERM_BINS
        label_w = max(len(name) for name, _ in self._masks) + 2

        lines: list[str] = []
        for name, mask in self._masks:
            binned = self._bin_coverage(mask, n_bins)
            bar = "".join("\u2588" if b else "\u2591" for b in binned)
            lines.append(f"{name:<{label_w}}{bar}")
        bar_len = len(self._bin_coverage(self._masks[0][1], n_bins))

        start_str = _fmt_short_date(self._begin) if self._begin else ""
        end_str = _fmt_short_date(self._end) if self._end else ""
        date_line = f"{start_str:<{bar_len // 2}}{end_str:>{bar_len - bar_len // 2}}"
        lines.append(f"{'':<{label_w}}{date_line}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        if not self._masks:
            return ""
        n_bins = self._SVG_BINS
        # Use actual bin count (may be less than n_bins for short series)
        max_mask_len = max(len(m) for _, m in self._masks) if self._masks else 0
        actual_bins = min(n_bins, max_mask_len) if max_mask_len > 0 else n_bins
        label_w = 120  # px reserved for labels
        bar_w = 480  # px for the bar area
        row_h = 22
        n_rows = len(self._masks)
        date_h = 18
        total_h = n_rows * row_h + date_h + 4

        parts: list[str] = []
        parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {label_w + bar_w} {total_h}" '
            f'width="100%" style="max-width:{label_w + bar_w}px;'
            f'font-family:monospace;font-size:12px;">'
        )

        lt = THEME["light"]
        for row_idx, (name, mask) in enumerate(self._masks):
            y = row_idx * row_h
            # label
            parts.append(
                f'<text x="{label_w - 6}" y="{y + 15}" '
                f'text-anchor="end" fill="{lt["coverage_label"]}">{escape(name)}</text>'
            )
            # bar segments
            binned = self._bin_coverage(mask, n_bins)
            seg_w = bar_w / len(binned) if binned else bar_w
            for i, b in enumerate(binned):
                color = lt["coverage_present"] if b else lt["coverage_absent"]
                x = label_w + i * seg_w
                parts.append(
                    f'<rect x="{x:.1f}" y="{y + 2}" '
                    f'width="{seg_w:.2f}" height="{row_h - 4}" '
                    f'fill="{color}" />'
                )

        # date labels
        date_y = n_rows * row_h + date_h
        if self._begin:
            parts.append(
                f'<text x="{label_w}" y="{date_y}" '
                f'text-anchor="start" fill="{lt["coverage_date"]}">'
                f'{escape(_fmt_short_date(self._begin))}</text>'
            )
        if self._end:
            parts.append(
                f'<text x="{label_w + bar_w}" y="{date_y}" '
                f'text-anchor="end" fill="{lt["coverage_date"]}">'
                f'{escape(_fmt_short_date(self._end))}</text>'
            )

        parts.append("</svg>")
        return "\n".join(parts)
