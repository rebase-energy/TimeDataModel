from datetime import datetime, timedelta, timezone

import timedatamodel as tdm


def main():
    # Define frequency and timezone
    frequency = tdm.Frequency.PT1H
    tz = "Europe/Oslo"
    location = tdm.GeoLocation(latitude=59.91, longitude=10.75)

    # Create a time series with scalar metadata
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = [base + timedelta(hours=i) for i in range(24)]
    values = [100.0 + i * 5.0 for i in range(24)]
    ts = tdm.TimeSeries(
        frequency,
        timezone=tz,
        timestamps=timestamps,
        values=values,
        name="power",
        unit="MW",
        data_type=tdm.DataType.ACTUAL,
        location=location,
        description="Hourly power generation",
        attributes={"source": "example"},
    )

    print(f"TimeSeries: {len(ts)} points")
    print(f"Frequency: {ts.frequency} (Timezone: {ts.timezone})")
    print(f"Unit: {ts.unit} -> pint: {ts.pint_unit}")
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
    ts2 = tdm.TimeSeries.from_pandas(df_pd, frequency, timezone=tz)
    print(f"Round-trip via pandas: {len(ts2)} points, first={ts2[0]}")

    # Construction via DataPoint
    data = [tdm.DataPoint(base + timedelta(hours=i), float(i)) for i in range(3)]
    ts3 = tdm.TimeSeries(frequency, timezone=tz, data=data)
    print(f"From DataPoints: {list(ts3)}")


if __name__ == "__main__":
    main()
