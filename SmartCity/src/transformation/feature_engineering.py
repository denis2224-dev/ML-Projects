from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["weather_timestamp"] = pd.to_datetime(result["weather_timestamp"], errors="coerce")

    result["hour"] = result["weather_timestamp"].dt.hour
    result["day_of_week"] = result["weather_timestamp"].dt.dayofweek
    result["month"] = result["weather_timestamp"].dt.month
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)

    return result


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["temp_humidity_interaction"] = result["temperature"] * result["humidity"]
    result["temp_wind_interaction"] = result["temperature"] * result["wind_speed"]

    return result


def add_pollution_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["pm_ratio"] = result["pm2_5"] / result["pm10"].replace(0, pd.NA)
    result["pollution_load"] = result[["co", "no2", "o3", "so2", "pm2_5", "pm10"]].sum(axis=1)

    return result


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result = add_time_features(result)
    result = add_interaction_features(result)
    result = add_pollution_features(result)
    return result