"""Weather data utilities for fetching from Open-Meteo API."""

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
REQUEST_DELAY = 0.1  # 100ms between requests
DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "data" / "training" / "weather"

# Weather variables to fetch — comprehensive set for mountaineering
DAILY_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]


def fetch_weather_window(
    latitude: float,
    longitude: float,
    target_date: date,
    window_days: int = 90,
) -> pd.DataFrame:
    """
    Fetch historical weather data from Open-Meteo Archive API.

    Args:
        latitude: Peak latitude
        longitude: Peak longitude
        target_date: End date for weather window (typically summit date)
        window_days: Number of days before target_date to fetch

    Returns:
        DataFrame with daily weather data, or empty DataFrame on error.
    """
    start_date = target_date - timedelta(days=window_days - 1)

    try:
        response = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date.isoformat(),
                "end_date": target_date.isoformat(),
                "daily": ",".join(DAILY_VARIABLES),
                "timezone": "UTC",
            },
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        daily = data.get("daily", {})

        if not daily or "time" not in daily:
            logger.warning(f"No weather data returned for ({latitude}, {longitude})")
            return pd.DataFrame()

        df = pd.DataFrame(daily)
        df = df.rename(columns={"time": "date"})
        df["date"] = pd.to_datetime(df["date"])

        time.sleep(REQUEST_DELAY)
        return df

    except requests.RequestException as e:
        logger.error(f"Weather API request failed: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error processing weather data: {e}")
        return pd.DataFrame()


def build_multiscale_windows(
    raw_weather: pd.DataFrame,
    target_date: date,
) -> dict[str, pd.DataFrame]:
    """
    Build multi-scale weather windows from raw daily data.

    Creates three temporal resolutions:
    - 7d: Last 7 days at full daily resolution (7 rows)
    - 30d: Last 30 days aggregated to 3-day buckets (10 rows)
    - 90d: Last 90 days aggregated to 10-day buckets (9 rows)

    Args:
        raw_weather: DataFrame with daily weather data and 'date' column
        target_date: The reference date (typically summit date)

    Returns:
        Dict with keys '7d', '30d', '90d', each containing a DataFrame.
    """
    if raw_weather.empty:
        return {
            "7d": pd.DataFrame(),
            "30d": pd.DataFrame(),
            "90d": pd.DataFrame(),
        }

    df = raw_weather.copy()
    df["date"] = pd.to_datetime(df["date"])
    target_dt = pd.Timestamp(target_date)

    # Numeric columns only for aggregation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 7-day window: full resolution
    start_7d = target_dt - timedelta(days=6)
    df_7d = df[(df["date"] >= start_7d) & (df["date"] <= target_dt)].copy()
    df_7d["scale"] = "7d"
    df_7d["bucket"] = range(len(df_7d))

    # 30-day window: 3-day aggregates (10 buckets)
    start_30d = target_dt - timedelta(days=29)
    df_30d_raw = df[(df["date"] >= start_30d) & (df["date"] <= target_dt)].copy()
    df_30d_raw["bucket"] = (df_30d_raw["date"] - start_30d).dt.days // 3
    df_30d = df_30d_raw.groupby("bucket")[numeric_cols].mean().reset_index()
    df_30d["scale"] = "30d"

    # 90-day window: 10-day aggregates (9 buckets)
    start_90d = target_dt - timedelta(days=89)
    df_90d_raw = df[(df["date"] >= start_90d) & (df["date"] <= target_dt)].copy()
    df_90d_raw["bucket"] = (df_90d_raw["date"] - start_90d).dt.days // 10
    df_90d = df_90d_raw.groupby("bucket")[numeric_cols].mean().reset_index()
    df_90d["scale"] = "90d"

    return {
        "7d": df_7d,
        "30d": df_30d,
        "90d": df_90d,
    }


def get_weather_cache_path(
    peakid: str,
    target_date: date,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Get the cache file path for a peak+date combination."""
    return cache_dir / f"{peakid}_{target_date.isoformat()}.csv"


def get_cached_weather(
    peakid: str,
    target_date: date,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame | None:
    """Load cached weather data if available."""
    cache_path = get_weather_cache_path(peakid, target_date, cache_dir)
    if cache_path.exists():
        try:
            return pd.read_csv(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path}: {e}")
    return None


def save_weather_cache(
    weather_df: pd.DataFrame,
    peakid: str,
    target_date: date,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> None:
    """Save weather data to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = get_weather_cache_path(peakid, target_date, cache_dir)
    weather_df.to_csv(cache_path, index=False)
    logger.debug(f"Cached weather to {cache_path}")
