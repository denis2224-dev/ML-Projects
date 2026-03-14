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


def save_raw_json(payload: dict[str, Any], prefix: str = "air_quality") -> Path:
    ensure_raw_dir()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = RAW_DIR / f"{prefix}_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    return output_path


def extract_air_quality(config: dict[str, Any]) -> dict[str, Any]:
    api_key = config["api"]["openweather_api_key"].strip()
    base_url = config["api"]["air_quality_base_url"].strip()

    lat = config["location"]["lat"]
    lon = config["location"]["lon"]
    city = config["location"].get("city")

    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
    }

    response = requests.get(base_url, params=params, timeout=30)

    print("Status:", response.status_code)
    print("Content-Type:", response.headers.get("Content-Type"))
    print("Response preview:", response.text[:300])

    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "application/json" not in content_type:
        raise ValueError(
            f"Expected JSON response but got {content_type}. "
            f"Check air_quality_base_url in config.yaml."
        )

    payload = response.json()
    raw_file = save_raw_json(payload)

    air_quality_info = payload.get("list", [])
    first_item = air_quality_info[0] if air_quality_info else {}

    main = first_item.get("main", {})
    components = first_item.get("components", {})

    record = {
        "city": city,
        "lat": lat,
        "lon": lon,
        "aqi": main.get("aqi"),
        "co": components.get("co"),
        "no": components.get("no"),
        "no2": components.get("no2"),
        "o3": components.get("o3"),
        "so2": components.get("so2"),
        "pm2_5": components.get("pm2_5"),
        "pm10": components.get("pm10"),
        "nh3": components.get("nh3"),
        "air_quality_timestamp": unix_to_iso(first_item.get("dt")),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "source": "openweather_air_pollution",
        "raw_file": str(raw_file),
    }

    return record


def main() -> None:
    try:
        config = load_config()
        record = extract_air_quality(config)
        print(json.dumps(record, indent=2, ensure_ascii=False))
    except FileNotFoundError:
        print(f"Config file not found: {CONFIG_PATH}")
    except KeyError as error:
        print(f"Missing config key: {error}")
    except requests.exceptions.HTTPError as error:
        print(f"HTTP error: {error}")
    except requests.exceptions.RequestException as error:
        print(f"Request failed: {error}")
    except Exception as error:
        print(f"Unexpected error: {error}")


if __name__ == "__main__":
    main()