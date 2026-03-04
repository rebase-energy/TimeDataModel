from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .enums import DataType, Frequency, TimeSeriesType
from .location import Location, _location_from_json, _location_to_json


class _TimeSeriesListIOMixin:
    """JSON and CSV I/O methods for TimeSeriesList."""

    def to_json(self) -> str:
        """Serialize to a JSON string (timestamps as ISO-8601 strings)."""
        if self.is_multi_index:
            ts_json = [
                [dt.isoformat() for dt in tup] for tup in self._timestamps
            ]
        else:
            ts_json = [t.isoformat() for t in self._timestamps]

        val_json = (
            self._values
            if isinstance(self._values, list)
            else self._values.tolist()
        )

        payload: dict = {
            "timestamps": ts_json,
            "values": val_json,
            "frequency": str(self.frequency),
            "timezone": self.timezone,
        }
        if self._index_names is not None:
            payload["index_names"] = self._index_names
        if self.name is not None:
            payload["name"] = self.name
        if self.unit is not None:
            payload["unit"] = self.unit
        if self.description is not None:
            payload["description"] = self.description
        if self.data_type is not None:
            payload["data_type"] = str(self.data_type)
        if self.timeseries_type != TimeSeriesType.FLAT:
            payload["timeseries_type"] = str(self.timeseries_type)
        if self.attributes:
            payload["attributes"] = self.attributes
        if self.labels:
            payload["labels"] = self.labels
        if self.location is not None:
            payload["location"] = _location_to_json(self.location)
        return json.dumps(payload)

    @classmethod
    def from_json(
        cls,
        s: str,
        frequency: Frequency | None = None,
        *,
        timezone: str | None = None,
        name: str | None = None,
        unit: str | None = None,
        description: str | None = None,
        data_type: DataType | None = None,
        location: Location | None = None,
        timeseries_type: TimeSeriesType | None = None,
        attributes: dict[str, str] | None = None,
        labels: dict[str, str] | None = None,
    ):
        """Reconstruct a TimeSeriesList from a JSON string produced by to_json()."""
        data = json.loads(s)
        raw_ts = data["timestamps"]

        if raw_ts and isinstance(raw_ts[0], list):
            timestamps = [
                tuple(datetime.fromisoformat(dt) for dt in row)
                for row in raw_ts
            ]
        else:
            timestamps = [datetime.fromisoformat(t) for t in raw_ts]

        values: list[float | None] = data["values"]
        index_names = data.get("index_names")

        if frequency is not None:
            freq = frequency
        elif "frequency" in data:
            freq = Frequency(data["frequency"])
        else:
            raise ValueError("frequency must be provided either in JSON or as argument")

        tz = timezone if timezone is not None else data.get("timezone", "UTC")
        nm = name if name is not None else data.get("name")
        un = unit if unit is not None else data.get("unit")
        desc = description if description is not None else data.get("description")
        dt_ = data_type if data_type is not None else (
            DataType(data["data_type"]) if "data_type" in data else None
        )
        tst = timeseries_type if timeseries_type is not None else (
            TimeSeriesType(data["timeseries_type"]) if "timeseries_type" in data else TimeSeriesType.FLAT
        )
        attrs = attributes if attributes is not None else data.get("attributes")
        lbls = labels if labels is not None else data.get("labels")
        loc = location if location is not None else _location_from_json(
            data.get("location")
        )

        return cls(
            freq,
            timezone=tz,
            timestamps=timestamps,
            values=values,
            name=nm,
            unit=un,
            description=desc,
            data_type=dt_,
            location=loc,
            timeseries_type=tst,
            attributes=attrs,
            labels=lbls,
            index_names=index_names,
        )

    def to_csv(self, path: str | Path) -> None:
        """Write timestamps and values to a CSV file."""
        idx_names = list(self.index_names)
        col_names = list(self.column_names)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(idx_names + col_names)
            for i, t in enumerate(self._timestamps):
                if isinstance(t, tuple):
                    ts_cells = [dt.isoformat() for dt in t]
                else:
                    ts_cells = [t.isoformat()]
                v = self._values[i]
                val_cells = ["" if v is None else v]
                writer.writerow(ts_cells + val_cells)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        frequency: Frequency,
        *,
        timezone: str = "UTC",
        name: str | None = None,
        unit: str | None = None,
        description: str | None = None,
        data_type: DataType | None = None,
        location: Location | None = None,
        timeseries_type: TimeSeriesType = TimeSeriesType.FLAT,
        attributes: dict[str, str] | None = None,
        labels: dict[str, str] | None = None,
    ):
        """Read a TimeSeriesList from a CSV file produced by to_csv()."""
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)

            idx_cols: list[int] = []
            val_cols: list[int] = []
            for i, hname in enumerate(header):
                if hname in ("timestamp",) or hname.endswith("_time"):
                    idx_cols.append(i)
                else:
                    val_cols.append(i)
            if not idx_cols:
                idx_cols = [0]
                val_cols = list(range(1, len(header)))

            multi_index = len(idx_cols) > 1

            timestamps: list = []
            rows: list = []
            for row in reader:
                if multi_index:
                    timestamps.append(
                        tuple(
                            datetime.fromisoformat(row[i]) for i in idx_cols
                        )
                    )
                else:
                    timestamps.append(
                        datetime.fromisoformat(row[idx_cols[0]])
                    )

                raw = row[val_cols[0]] if val_cols else ""
                rows.append(None if raw == "" else float(raw))

        index_names = (
            [header[i] for i in idx_cols] if multi_index else None
        )
        if name is None and val_cols:
            name = header[val_cols[0]]

        return cls(
            frequency,
            timezone=timezone,
            timestamps=timestamps,
            values=rows,
            name=name,
            unit=unit,
            description=description,
            data_type=data_type,
            location=location,
            timeseries_type=timeseries_type,
            attributes=attributes,
            labels=labels,
            index_names=index_names,
        )


class _TimeSeriesTableIOMixin:
    """JSON and CSV I/O methods for TimeSeriesTable."""

    def to_json(self) -> str:
        """Serialize to a JSON string (timestamps as ISO-8601 strings)."""
        if self.is_multi_index:
            ts_json = [
                [dt.isoformat() for dt in tup] for tup in self._timestamps
            ]
        else:
            ts_json = [t.isoformat() for t in self._timestamps]

        payload: dict = {
            "timestamps": ts_json,
            "values": self._values.tolist(),
            "column_names": list(self.column_names),
            "frequency": str(self.frequency),
            "timezone": self.timezone,
        }
        if any(n is not None for n in self.names):
            payload["names"] = self.names
        if any(u is not None for u in self.units):
            payload["units"] = self.units
        if any(d is not None for d in self.descriptions):
            payload["descriptions"] = self.descriptions
        if any(d is not None for d in self.data_types):
            payload["data_types"] = [str(d) if d else None for d in self.data_types]
        if any(t != TimeSeriesType.FLAT for t in self.timeseries_types):
            payload["timeseries_types"] = [str(t) for t in self.timeseries_types]
        if any(a for a in self.attributes):
            payload["attributes"] = self.attributes
        if any(lbl for lbl in self.labels):
            payload["labels"] = self.labels
        if any(loc is not None for loc in self.locations):
            payload["locations"] = [
                _location_to_json(loc) if loc is not None else None
                for loc in self.locations
            ]
        if self._index_names is not None:
            payload["index_names"] = self._index_names
        return json.dumps(payload)

    @classmethod
    def from_json(
        cls,
        s: str,
        frequency: Frequency | None = None,
        *,
        timezone: str | None = None,
        names: list[str | None] | None = None,
        units: list[str | None] | None = None,
        descriptions: list[str | None] | None = None,
        data_types: list[DataType | None] | None = None,
        locations: list[Location | None] | None = None,
        timeseries_types: list[TimeSeriesType] | None = None,
        attributes: list[dict[str, str]] | None = None,
        labels: list[dict[str, str]] | None = None,
    ):
        """Reconstruct from a JSON string produced by to_json()."""
        data = json.loads(s)
        raw_ts = data["timestamps"]

        if raw_ts and isinstance(raw_ts[0], list):
            timestamps = [
                tuple(datetime.fromisoformat(dt) for dt in row)
                for row in raw_ts
            ]
        else:
            timestamps = [datetime.fromisoformat(t) for t in raw_ts]

        values = np.array(data["values"], dtype=np.float64)
        index_names = data.get("index_names")

        if frequency is not None:
            freq = frequency
        elif "frequency" in data:
            freq = Frequency(data["frequency"])
        else:
            raise ValueError("frequency must be provided either in JSON or as argument")

        tz = timezone if timezone is not None else data.get("timezone", "UTC")

        if names is None:
            names = data.get("names") or data.get("column_names")
        if units is None:
            units = data.get("units")
        if descriptions is None:
            descriptions = data.get("descriptions")
        if locations is None:
            raw_locations = data.get("locations")
            if raw_locations is not None:
                locations = [
                    _location_from_json(loc) if loc is not None else None
                    for loc in raw_locations
                ]
        if data_types is None:
            raw_dt = data.get("data_types")
            if raw_dt:
                data_types = [DataType(d) if d else None for d in raw_dt]
        if timeseries_types is None:
            raw_tst = data.get("timeseries_types")
            if raw_tst:
                timeseries_types = [TimeSeriesType(t) for t in raw_tst]
        if attributes is None:
            attributes = data.get("attributes")
        if labels is None:
            labels = data.get("labels")

        return cls(
            freq,
            timezone=tz,
            timestamps=timestamps,
            values=values,
            names=names,
            units=units,
            descriptions=descriptions,
            data_types=data_types,
            locations=locations,
            timeseries_types=timeseries_types,
            attributes=attributes,
            labels=labels,
            index_names=index_names,
        )

    def to_csv(self, path: str | Path) -> None:
        """Write timestamps and values to a CSV file."""
        idx_names = list(self.index_names)
        col_names = list(self.column_names)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(idx_names + col_names)
            for i, t in enumerate(self._timestamps):
                if isinstance(t, tuple):
                    ts_cells = [dt.isoformat() for dt in t]
                else:
                    ts_cells = [t.isoformat()]
                val_cells = [
                    "" if np.isnan(v) else v for v in self._values[i]
                ]
                writer.writerow(ts_cells + val_cells)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        frequency: Frequency,
        *,
        timezone: str = "UTC",
        names: list[str | None] | None = None,
        units: list[str | None] | None = None,
        descriptions: list[str | None] | None = None,
        data_types: list[DataType | None] | None = None,
        locations: list[Location | None] | None = None,
        timeseries_types: list[TimeSeriesType] | None = None,
        attributes: list[dict[str, str]] | None = None,
        labels: list[dict[str, str]] | None = None,
    ):
        """Read a TimeSeriesTable from a CSV file produced by to_csv()."""
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)

            idx_cols: list[int] = []
            val_cols: list[int] = []
            for i, hname in enumerate(header):
                if hname in ("timestamp",) or hname.endswith("_time"):
                    idx_cols.append(i)
                else:
                    val_cols.append(i)
            if not idx_cols:
                idx_cols = [0]
                val_cols = list(range(1, len(header)))

            multi_index = len(idx_cols) > 1

            timestamps: list = []
            rows: list = []
            for row in reader:
                if multi_index:
                    timestamps.append(
                        tuple(
                            datetime.fromisoformat(row[i]) for i in idx_cols
                        )
                    )
                else:
                    timestamps.append(
                        datetime.fromisoformat(row[idx_cols[0]])
                    )

                rows.append([
                    np.nan if row[i] == "" else float(row[i])
                    for i in val_cols
                ])

        values = np.array(rows, dtype=np.float64)
        index_names = (
            [header[i] for i in idx_cols] if multi_index else None
        )

        if names is None:
            names = [header[i] for i in val_cols]

        return cls(
            frequency,
            timezone=timezone,
            timestamps=timestamps,
            values=values,
            names=names,
            units=units,
            descriptions=descriptions,
            data_types=data_types,
            locations=locations,
            timeseries_types=timeseries_types,
            attributes=attributes,
            labels=labels,
            index_names=index_names,
        )
