# AI-assisted: generated with Claude Code, reviewed and run by the author
"""Cleaning, feature engineering and the time-based split for the Sri Lanka weather data.

Every feature is built from day t (or earlier) inside each city's own series, and every
target is a day t+1 value, so a row never sees the day it is trying to predict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRAIN_END = "2020-12-31"
VAL_START, VAL_END = "2021-01-01", "2021-12-31"
TEST_START, TEST_END = "2022-01-01", "2023-06-17"

DROP_COLS = ["rain_sum", "snowfall_sum", "country", "sunrise", "sunset"]

# day-t weather values that are used as-is
DAY_T_COLS = [
    "weathercode", "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
    "shortwave_radiation_sum", "precipitation_sum", "precipitation_hours",
    "windspeed_10m_max", "windgusts_10m_max", "wind_dir_sin", "wind_dir_cos",
    "et0_fao_evapotranspiration", "day_length_h",
]
LAG_COLS = [
    "precipitation_sum", "precipitation_hours", "temperature_2m_max", "temperature_2m_min",
    "temperature_2m_mean", "windspeed_10m_max", "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]
LAGS = [1, 2, 3, 7]
ROLL_COLS = [
    "precipitation_sum", "precipitation_hours", "temperature_2m_max",
    "temperature_2m_mean", "shortwave_radiation_sum",
]
ROLLS = [7, 30]
CALENDAR_COLS = ["year", "month", "doy_sin", "doy_cos", "monsoon_sw", "monsoon_ne"]
STATIC_COLS = ["latitude", "longitude", "elevation", "city_code"]

TARGETS = {
    "rain_amount": "y_rain_next",   # log1p(next-day precipitation, mm), regression
    "rain_flag": "y_rain_flag",     # next-day precipitation >= 1 mm, binary
    "tmax": "y_tmax_next",          # next-day max temperature (deg C), regression
}


def load_raw(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values(["city", "time"]).reset_index(drop=True)


def monsoon_season(month: pd.Series) -> pd.Series:
    """SW monsoon May-Sep, NE monsoon Dec-Feb, inter-monsoon otherwise."""
    return np.select(
        [month.between(5, 9), month.isin([12, 1, 2])], ["SW", "NE"], default="Inter"
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rise = pd.to_datetime(df["sunrise"])
    sett = pd.to_datetime(df["sunset"])
    df["day_length_h"] = (sett - rise).dt.total_seconds() / 3600
    df = df.drop(columns=DROP_COLS)

    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    doy = df["time"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["season"] = monsoon_season(df["month"])
    df["monsoon_sw"] = (df["season"] == "SW").astype(int)
    df["monsoon_ne"] = (df["season"] == "NE").astype(int)

    rad = np.deg2rad(df["winddirection_10m_dominant"])
    df["wind_dir_sin"] = np.sin(rad)
    df["wind_dir_cos"] = np.cos(rad)
    df = df.drop(columns=["winddirection_10m_dominant"])

    df["city_code"] = df["city"].astype("category").cat.codes.astype(int)
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("city", sort=False)
    next_rain = g["precipitation_sum"].shift(-1)
    df["rain_next_mm"] = next_rain
    df["y_rain_next"] = np.log1p(next_rain)
    df["y_rain_flag"] = (next_rain >= 1).astype(float).where(next_rain.notna())
    df["y_tmax_next"] = g["temperature_2m_max"].shift(-1)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("city", sort=False)
    new = {}
    for col in LAG_COLS:
        for k in LAGS:
            new[f"{col}_lag{k}"] = g[col].shift(k)
    for col in ROLL_COLS:
        for w in ROLLS:
            # window ends on day t, so it only uses days <= t
            new[f"{col}_roll{w}"] = g[col].transform(
                lambda s, w=w: s.rolling(w, min_periods=w).mean()
            )
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def feature_columns() -> list[str]:
    lagged = [f"{c}_lag{k}" for c in LAG_COLS for k in LAGS]
    rolled = [f"{c}_roll{w}" for c in ROLL_COLS for w in ROLLS]
    return DAY_T_COLS + lagged + rolled + CALENDAR_COLS + STATIC_COLS


def build_dataset(path) -> pd.DataFrame:
    """Raw CSV -> one modelling table (features + targets), NaN warm-up rows removed."""
    df = clean(load_raw(path))
    df = add_targets(df)
    df = add_lag_features(df)
    needed = feature_columns() + list(TARGETS.values())
    df = df.dropna(subset=needed)
    # order by date (then city) so TimeSeriesSplit / KFold(shuffle=False) see real time blocks
    return df.sort_values(["time", "city"]).reset_index(drop=True)


def time_split(df: pd.DataFrame, date_col: str = "time"):
    """Boolean masks for train / validation / test, by calendar date of day t."""
    d = df[date_col]
    train = d <= TRAIN_END
    val = d.between(VAL_START, VAL_END)
    test = d.between(TEST_START, TEST_END)
    return train, val, test
