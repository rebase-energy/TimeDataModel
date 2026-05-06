from __future__ import annotations

import warnings

import pytest
from timedatamodel._repr import _get_repr_css
from timedatamodel._theme import (
    _DEFAULT_THEME,
    get_theme,
    get_theme_version,
    reset_theme,
    set_theme,
)


@pytest.fixture(autouse=True)
def _clean_theme():
    """Reset theme state before and after every test."""
    reset_theme()
    yield
    reset_theme()


# ---------------------------------------------------------------------------
# set_theme / get_theme basics
# ---------------------------------------------------------------------------


class TestSetTheme:
    def test_partial_override_merges(self):
        set_theme({"light": {"header_bg": "#ff0000"}})
        resolved = get_theme()
        assert resolved["light"]["header_bg"] == "#ff0000"
        # Other keys unchanged
        assert resolved["light"]["header_text"] == _DEFAULT_THEME["light"]["header_text"]

    def test_accumulates_across_calls(self):
        set_theme({"light": {"header_bg": "#111111"}})
        set_theme({"light": {"header_text": "#222222"}})
        resolved = get_theme()
        assert resolved["light"]["header_bg"] == "#111111"
        assert resolved["light"]["header_text"] == "#222222"

    def test_later_call_overwrites_same_key(self):
        set_theme({"light": {"header_bg": "#aaa"}})
        set_theme({"light": {"header_bg": "#bbb"}})
        assert get_theme()["light"]["header_bg"] == "#bbb"

    def test_dark_mode_override(self):
        set_theme({"dark": {"meta_bg": "#001122"}})
        assert get_theme()["dark"]["meta_bg"] == "#001122"
        # Light untouched
        assert get_theme()["light"]["meta_bg"] == _DEFAULT_THEME["light"]["meta_bg"]


# ---------------------------------------------------------------------------
# reset_theme
# ---------------------------------------------------------------------------


class TestResetTheme:
    def test_restores_defaults(self):
        set_theme({"light": {"header_bg": "#ff0000"}})
        reset_theme()
        assert get_theme()["light"]["header_bg"] == _DEFAULT_THEME["light"]["header_bg"]

    def test_version_increments_on_reset(self):
        v0 = get_theme_version()
        reset_theme()
        assert get_theme_version() > v0


# ---------------------------------------------------------------------------
# get_theme returns deep copy
# ---------------------------------------------------------------------------


class TestGetTheme:
    def test_deep_copy(self):
        t = get_theme()
        t["light"]["header_bg"] = "#000000"
        # Internal state unaffected
        assert get_theme()["light"]["header_bg"] == _DEFAULT_THEME["light"]["header_bg"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_hex_raises(self):
        with pytest.raises(ValueError, match="invalid hex color"):
            set_theme({"light": {"header_bg": "not-a-color"}})

    def test_invalid_hex_no_hash(self):
        with pytest.raises(ValueError, match="invalid hex color"):
            set_theme({"light": {"header_bg": "ff0000"}})

    def test_unknown_key_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            set_theme({"light": {"nonexistent_key": "#fff"}})
        assert any("unknown key" in str(x.message) for x in w)

    def test_unknown_mode_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            set_theme({"sepia": {"header_bg": "#fff"}})
        assert any("unknown mode" in str(x.message) for x in w)

    def test_valid_hex_formats(self):
        # 3, 4, 6, 8 hex digits should all pass
        set_theme({"light": {"header_bg": "#fff"}})
        set_theme({"light": {"header_bg": "#ffff"}})
        set_theme({"light": {"header_bg": "#ffffff"}})
        set_theme({"light": {"header_bg": "#ffffffff"}})


# ---------------------------------------------------------------------------
# CSS caching
# ---------------------------------------------------------------------------


class TestCSSCaching:
    def test_css_regenerates_after_set_theme(self):
        css_before = _get_repr_css()
        set_theme({"light": {"header_bg": "#abcdef"}})
        css_after = _get_repr_css()
        assert "#abcdef" in css_after
        assert css_before != css_after

    def test_css_cached_when_unchanged(self):
        css1 = _get_repr_css()
        css2 = _get_repr_css()
        assert css1 is css2  # same object identity


# ---------------------------------------------------------------------------
# ThemeProxy backward compat
# ---------------------------------------------------------------------------


class TestThemeProxy:
    def test_getitem(self):
        from timedatamodel._theme import THEME

        assert THEME["light"]["header_bg"] == _DEFAULT_THEME["light"]["header_bg"]

    def test_contains(self):
        from timedatamodel._theme import THEME

        assert "light" in THEME
        assert "dark" in THEME
        assert "sepia" not in THEME

    def test_reflects_set_theme(self):
        from timedatamodel._theme import THEME

        set_theme({"light": {"header_bg": "#123456"}})
        assert THEME["light"]["header_bg"] == "#123456"
