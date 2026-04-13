"""Optional unit-string validation. Requires the ``[pint]`` extra."""

from __future__ import annotations

_UREG = None


def _get_registry():
    """Return a shared, lazily-constructed ``pint.UnitRegistry``.

    Raises ``ImportError`` if the ``[pint]`` extra is not installed.
    """
    global _UREG
    if _UREG is None:
        import pint  # raises ImportError when the [pint] extra is not installed

        _UREG = pint.UnitRegistry()
    return _UREG


def validate_unit(unit: str) -> bool:
    """Return ``True`` if *unit* parses as a valid pint unit string.

    Examples
    --------
    >>> from timedatamodel.units import validate_unit
    >>> validate_unit("MW")
    True
    >>> validate_unit("not a unit")
    False

    Raises
    ------
    ImportError
        If pint is not installed.  Install with
        ``pip install timedatamodel[pint]``.

    Notes
    -----
    Pint is required.  Use this only on code paths where you actually want
    validation; if pint may be absent, gate the call on a try/except for
    ``ImportError`` at the call site.
    """
    ureg = _get_registry()
    try:
        ureg(unit)
        return True
    except Exception:
        return False
