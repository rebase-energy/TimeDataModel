from __future__ import annotations

from dataclasses import dataclass, field

from .enums import DataType, StorageType
from .location import Location


@dataclass(frozen=True, slots=True)
class Metadata:
    unit: str | None = None
    data_type: DataType | None = None
    location: Location | None = None
    name: str | None = None
    description: str | None = None
    storage_type: StorageType = StorageType.FLAT
    attributes: dict[str, str] = field(default_factory=dict)

    @property
    def pint_unit(self):
        """Resolve the unit string to a pint.Unit object."""
        if self.unit is None:
            raise ValueError("unit is not set")
        import pint

        ureg = pint.UnitRegistry()
        try:
            return ureg.Unit(self.unit)
        except pint.errors.UndefinedUnitError as e:
            raise ValueError(f"invalid unit string: {self.unit!r}") from e
