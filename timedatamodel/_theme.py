"""Centralized theme management for timedatamodel HTML reprs.

Provides three public functions:

- ``set_theme(overrides)`` — partial deep-merge on top of defaults + config file
- ``get_theme()`` — returns a deep copy of the resolved theme
- ``reset_theme()`` — restores defaults and re-enables config file discovery
"""

from __future__ import annotations

import copy
import json
import re
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults (immutable, loaded once from theme.json)
# ---------------------------------------------------------------------------

_THEME_PATH = Path(__file__).parent / "theme.json"

with open(_THEME_PATH) as _f:
    _DEFAULT_THEME: dict[str, dict[str, str]] = json.load(_f)

_VALID_MODES = frozenset(_DEFAULT_THEME.keys())
_VALID_KEYS: dict[str, frozenset[str]] = {
    mode: frozenset(keys) for mode, keys in _DEFAULT_THEME.items()
}
_HEX_RE = re.compile(r"^#[0-9a-fA-F]{3,8}$")

# ---------------------------------------------------------------------------
# Mutable state
# ---------------------------------------------------------------------------

_programmatic_overrides: dict[str, dict[str, str]] = {}
_config_file_overrides: dict[str, dict[str, str]] | None = None  # None = not yet discovered
_theme_version: int = 0
_resolved_cache: dict[str, dict[str, str]] | None = None
_resolved_cache_version: int = -1


def get_theme_version() -> int:
    """Return the current theme version counter (for CSS caching)."""
    return _theme_version


# ---------------------------------------------------------------------------
# Config file discovery
# ---------------------------------------------------------------------------

_CONFIG_FILENAME = "timedatamodel_theme.json"


def _discover_config_file() -> dict[str, dict[str, str]]:
    """Walk CWD upward looking for timedatamodel_theme.json.

    Returns the parsed overrides (possibly empty dict).  Invalid values
    are warned and skipped.
    """
    try:
        cwd = Path.cwd()
    except OSError:
        return {}

    for directory in (cwd, *cwd.parents):
        candidate = directory / _CONFIG_FILENAME
        if candidate.is_file():
            try:
                with open(candidate) as f:
                    raw = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                warnings.warn(
                    f"Failed to read {candidate}: {exc}",
                    stacklevel=3,
                )
                return {}
            return _validate_overrides(raw, source=str(candidate), strict=False)
    return {}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_overrides(
    overrides: dict,
    *,
    source: str = "set_theme()",
    strict: bool = True,
) -> dict[str, dict[str, str]]:
    """Validate and return a cleaned copy of *overrides*.

    *strict=True* (programmatic API) raises on bad hex values.
    *strict=False* (config file) warns and skips bad entries.
    """
    cleaned: dict[str, dict[str, str]] = {}
    for mode, entries in overrides.items():
        if mode not in _VALID_MODES:
            warnings.warn(
                f"{source}: unknown mode {mode!r}, expected one of {sorted(_VALID_MODES)}",
                stacklevel=3,
            )
            continue
        if not isinstance(entries, dict):
            msg = f"{source}: value for mode {mode!r} must be a dict"
            if strict:
                raise TypeError(msg)
            warnings.warn(msg, stacklevel=3)
            continue
        cleaned_entries: dict[str, str] = {}
        for key, value in entries.items():
            if key not in _VALID_KEYS[mode]:
                warnings.warn(
                    f"{source}: unknown key {key!r} in mode {mode!r}",
                    stacklevel=3,
                )
                continue
            if not isinstance(value, str) or not _HEX_RE.match(value):
                msg = f"{source}: invalid hex color {value!r} for {mode}.{key}"
                if strict:
                    raise ValueError(msg)
                warnings.warn(msg, stacklevel=3)
                continue
            cleaned_entries[key] = value
        if cleaned_entries:
            cleaned[mode] = cleaned_entries
    return cleaned


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _resolve_theme() -> dict[str, dict[str, str]]:
    """Merge: defaults ← config file ← programmatic overrides."""
    global _config_file_overrides, _resolved_cache, _resolved_cache_version

    # Lazy config file discovery
    if _config_file_overrides is None:
        _config_file_overrides = _discover_config_file()

    if _resolved_cache is not None and _resolved_cache_version == _theme_version:
        return _resolved_cache

    merged = copy.deepcopy(_DEFAULT_THEME)

    for layer in (_config_file_overrides, _programmatic_overrides):
        for mode, entries in layer.items():
            if mode in merged:
                merged[mode].update(entries)

    _resolved_cache = merged
    _resolved_cache_version = _theme_version
    return merged


# ---------------------------------------------------------------------------
# ThemeProxy — backward-compatible dict-like access
# ---------------------------------------------------------------------------


class _ThemeProxy:
    """Dict-like object so ``THEME["light"]`` works and always returns
    the resolved/merged theme."""

    def __getitem__(self, key: str) -> dict[str, str]:
        return _resolve_theme()[key]

    def __contains__(self, key: object) -> bool:
        return key in _resolve_theme()

    def __iter__(self):
        return iter(_resolve_theme())

    def keys(self):
        return _resolve_theme().keys()

    def values(self):
        return _resolve_theme().values()

    def items(self):
        return _resolve_theme().items()

    def __repr__(self) -> str:
        return repr(_resolve_theme())


THEME: _ThemeProxy = _ThemeProxy()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def set_theme(overrides: dict[str, dict[str, str]]) -> None:
    """Apply partial color overrides (deep-merged on top of current overrides).

    Parameters
    ----------
    overrides : dict
        E.g. ``{"light": {"header_bg": "#fff"}}``

    Raises
    ------
    ValueError
        If a color value is not a valid hex string.
    """
    global _theme_version
    cleaned = _validate_overrides(overrides, source="set_theme()", strict=True)
    for mode, entries in cleaned.items():
        _programmatic_overrides.setdefault(mode, {}).update(entries)
    _theme_version += 1


def get_theme() -> dict[str, dict[str, str]]:
    """Return a deep copy of the fully resolved theme."""
    return copy.deepcopy(_resolve_theme())


def reset_theme() -> None:
    """Restore default theme, clear programmatic overrides, re-discover config."""
    global _theme_version, _config_file_overrides, _resolved_cache
    _programmatic_overrides.clear()
    _config_file_overrides = None  # will re-discover on next access
    _resolved_cache = None
    _theme_version += 1
