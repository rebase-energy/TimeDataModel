from datetime import datetime, timedelta, timezone

from timedatamodel import (
    DataPoint,
    DataType,
    Frequency,
    GeoLocation,
    Metadata,
    Resolution,
    TimeSeries,
)


def main():
    # Define resolution and metadata
    resolution = Resolution(frequency=Frequency.PT1H, timezone="Europe/Oslo")
    location = GeoLocation(latitude=59.91, longitude=10.75)
    metadata = Metadata(
        unit="MW",
        data_type=DataType.ACTUAL,
        location=location,
        name="power",
        description="Hourly power generation",
        attributes={"source": "example"},
    )

    # Create a time series
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = [base + timedelta(hours=i) for i in range(24)]
    values = [100.0 + i * 5.0 for i in range(24)]
    ts = TimeSeries(resolution, metadata, timestamps=timestamps, values=values)

    print(f"TimeSeries: {len(ts)} points")
    print(f"Resolution: {resolution.frequency} ({resolution.timezone})")
    print(f"Unit: {metadata.unit} -> pint: {metadata.pint_unit}")
    print()

    # Iterate over first 3 data points
    print("First 3 data points:")
    for dp in ts[:3]:
        print(f"  {dp.timestamp} -> {dp.value}")
    print()

    # Validate
    warnings = ts.validate()
    print(f"Validation: {'OK' if not warnings else warnings}")
    print()

    # Convert to numpy
    arr = ts.to_numpy()
    print(f"NumPy array: shape={arr.shape}, dtype={arr.dtype}")
    print()

    # Convert to pandas
    df_pd = ts.to_pandas_dataframe()
    print("Pandas DataFrame:")
    print(df_pd.head())
    print()

    # Convert to polars
    try:
        df_pl = ts.to_polars_dataframe()
        print("Polars DataFrame:")
        print(df_pl.head())
    except ImportError:
        print("Polars not installed, skipping polars conversion")
    print()

    # Round-trip via pandas
    ts2 = TimeSeries.from_pandas(df_pd, resolution, metadata)
    print(f"Round-trip via pandas: {len(ts2)} points, first={ts2[0]}")

    # Construction via DataPoint
    data = [DataPoint(base + timedelta(hours=i), float(i)) for i in range(3)]
    ts3 = TimeSeries(resolution, data=data)
    print(f"From DataPoints: {list(ts3)}")


if __name__ == "__main__":
    main()
