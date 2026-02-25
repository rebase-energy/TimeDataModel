from enum import StrEnum


class Frequency(StrEnum):
    P1Y = "P1Y"
    P3M = "P3M"
    P1M = "P1M"
    P1W = "P1W"
    P1D = "P1D"
    PT1H = "PT1H"
    PT30M = "PT30M"
    PT15M = "PT15M"
    PT10M = "PT10M"
    PT5M = "PT5M"
    PT1M = "PT1M"
    PT1S = "PT1S"
    NONE = "NONE"


class DataType(StrEnum):
    MEASUREMENT = "MEASUREMENT"
    ESTIMATION = "ESTIMATION"
    FORECAST = "FORECAST"
    SCENARIO = "SCENARIO"
    SYNTHETIC = "SYNTHETIC"
    CLIMATE = "CLIMATE"
    ACTUAL = "ACTUAL"


class StorageType(StrEnum):
    FLAT = "FLAT"
    OVERLAPPING = "OVERLAPPING"
