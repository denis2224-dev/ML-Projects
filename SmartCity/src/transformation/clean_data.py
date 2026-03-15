from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def ensure_processed_dir(path: Path = PROCESSED_DIR) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def build_combined_record(weather_record: dict[str, Any], air_record: dict[str, Any]) -> dict[str, Any]:
    record = {
        "city": clean_text(first_not_none(weather_record.get("city"), air_record.get("city"))),
        "country": clean_text(weather_record.get("country")),
        "lat": clean_numeric(first_not_none(weather_record.get("lat"), air_record.get("lat"))),
        "lon": clean_numeric(first_not_none(weather_record.get("lon"), air_record.get("lon"))),

        "weather_main": clean_text(weather_record.get("weather_main")),
        "weather_description": clean_text(weather_record.get("weather_description")),

        "temperature": clean_numeric(weather_record.get("temperature")),
        "feels_like": clean_numeric(weather_record.get("feels_like")),
        "temp_min": clean_numeric(weather_record.get("temp_min")),
        "temp_max": clean_numeric(weather_record.get("temp_max")),
        "pressure": clean_numeric(weather_record.get("pressure")),
        "humidity": clean_numeric(weather_record.get("humidity")),
        "visibility": clean_numeric(weather_record.get("visibility")),
        "wind_speed": clean_numeric(weather_record.get("wind_speed")),
        "wind_deg": clean_numeric(weather_record.get("wind_deg")),
        "clouds_pct": clean_numeric(weather_record.get("clouds_pct")),

        "aqi": clean_numeric(air_record.get("aqi")),
        "co": clean_numeric(air_record.get("co")),
        "no": clean_numeric(air_record.get("no")),
        "no2": clean_numeric(air_record.get("no2")),
        "o3": clean_numeric(air_record.get("o3")),
        "so2": clean_numeric(air_record.get("so2")),
        "pm2_5": clean_numeric(air_record.get("pm2_5")),
        "pm10": clean_numeric(air_record.get("pm10")),
        "nh3": clean_numeric(air_record.get("nh3")),

        "weather_timestamp": clean_text(weather_record.get("weather_timestamp")),
        "air_quality_timestamp": clean_text(air_record.get("air_quality_timestamp")),
        "weather_source": clean_text(weather_record.get("source")),
        "air_quality_source": clean_text(air_record.get("source")),
        "weather_raw_file": clean_text(weather_record.get("raw_file")),
        "air_quality_raw_file": clean_text(air_record.get("raw_file")),

        "pipeline_run_at": datetime.now(timezone.utc).isoformat(),
    }

    return record


def to_dataframe(record: dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([record])

    datetime_cols = [
        "weather_timestamp",
        "air_quality_timestamp",
        "pipeline_run_at",
    ]
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def save_processed_data(df: pd.DataFrame, filename: str = "environment_data.csv") -> Path:
    ensure_processed_dir()
    output_path = PROCESSED_DIR / filename

    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        df_to_save = pd.concat([existing_df, df], ignore_index=True)
    else:
        df_to_save = df.copy()

    datetime_cols = ["weather_timestamp", "air_quality_timestamp", "pipeline_run_at"]
    for col in datetime_cols:
        df_to_save[col] = pd.to_datetime(df_to_save[col], errors="coerce")

    df_to_save.drop_duplicates(
        subset=["weather_timestamp", "air_quality_timestamp", "city"],
        keep="last",
        inplace=True,
    )

    df_to_save.to_csv(output_path, index=False)
    return output_path


def clean_and_save(weather_record: dict[str, Any], air_record: dict[str, Any]) -> tuple[pd.DataFrame, Path]:
    combined_record = build_combined_record(weather_record, air_record)
    df = to_dataframe(combined_record)
    output_path = save_processed_data(df)
    return df, output_path