import pytest

pytest.importorskip("pint")

from timedatamodel.units import _get_registry, validate_unit


class TestValidateUnit:
    def test_valid_simple(self):
        assert validate_unit("MW") is True

    def test_valid_dimensionless(self):
        assert validate_unit("dimensionless") is True

    def test_valid_compound_caret(self):
        assert validate_unit("kg*m/s^2") is True

    def test_valid_compound_double_star(self):
        # Pint accepts Python-style ** exponent notation as well.
        assert validate_unit("kg*m/s**2") is True

    def test_invalid(self):
        assert validate_unit("not a unit") is False

    def test_invalid_gibberish(self):
        assert validate_unit("zzzzzzz_not_a_unit") is False


class TestRegistryCache:
    def test_get_registry_returns_same_instance(self):
        assert _get_registry() is _get_registry()
