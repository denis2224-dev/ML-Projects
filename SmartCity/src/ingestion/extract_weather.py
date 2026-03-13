from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def ensure_raw_dir(path: Path = RAW_DIR) -> None:
    path.mkdir(parents=True, exist_ok=True)

def unix_to_iso(ts: int | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

def save_raw_json(payload: dict[str, Any], prefix: str = "weather") -> Path:
    ensure_raw_dir()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = RAW_DIR / f"{prefix}_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    return output_path

def extract_weather(config: dict[str, Any]) -> dict[str, Any]:
    api_key = config["api"]["weather_api_key"]
    base_url = config["api"]["weather_base_url"]
    units = config["api"].get("units", "metric")

    lat = config["location"]["lat"]
    lon = config["location"]["lon"]
    city = config["location"].get("city")

    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": units,
    }

    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()

    payload = response.json()

    payload["main"]["temp"]
    payload["wind"]["speed"]

    raw_file = save_raw_json(payload)

    weather_info = payload.get("weather", [])
    first_weather = weather_info[0] if weather_info else {}

    record = {
        "city": payload.get("name", city),
        "country": payload.get("sys", {}).get("country"),
        "lat": payload.get("coord", {}).get("lat", lat),
        "lon": payload.get("coord", {}).get("lon", lon),
        "weather_main": first_weather.get("main"),
        "weather_description": first_weather.get("description"),
        "temperature": payload.get("main", {}).get("temp"),
        "feels_like": payload.get("main", {}).get("feels_like"),
        "temp_min": payload.get("main", {}).get("temp_min"),
        "temp_max": payload.get("main", {}).get("temp_max"),
        "pressure": payload.get("main", {}).get("pressure"),
        "humidity": payload.get("main", {}).get("humidity"),
        "visibility": payload.get("visibility"),
        "wind_speed": payload.get("wind", {}).get("speed"),
        "wind_deg": payload.get("wind", {}).get("deg"),
        "clouds_pct": payload.get("clouds", {}).get("all"),
        "weather_timestamp": unix_to_iso(payload.get("dt")),
        "sunrise": unix_to_iso(payload.get("sys", {}).get("sunrise")),
        "sunset": unix_to_iso(payload.get("sys", {}).get("sunset")),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "source": "openweather_current",
        "raw_file": str(raw_file),
    }

    return record

def main() -> None:
    config = load_config()
    record = extract_weather(config)
    print(json.dumps(record, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()