"""Weather data utilities for fetching from Open-Meteo API."""

import logging
import time
from datetime import date, timedelta
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import requests

if TYPE_CHECKING:
    from utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
REQUEST_DELAY = 0.5  # 500ms between requests
MAX_RETRIES = 5  # Retry attempts
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "training"

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
    retry_count: int = 0,
    rate_limiter: "RateLimiter | None" = None,
) -> pd.DataFrame:
    """
    Fetch historical weather data from Open-Meteo Archive API.

    Implements exponential backoff retry on HTTP 429 (rate limit) errors.

    Args:
        latitude: Peak latitude
        longitude: Peak longitude
        target_date: End date for weather window (typically summit date)
        window_days: Number of days before target_date to fetch
        retry_count: Current retry attempt (internal use for recursion)
        rate_limiter: Optional RateLimiter to coordinate requests across workers

    Returns:
        DataFrame with daily weather data, or empty DataFrame on error.
    """
    start_date = target_date - timedelta(days=window_days - 1)

    # Acquire rate limit token before making request
    if rate_limiter:
        rate_limiter.acquire()

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

        # Handle rate limiting specifically
        if response.status_code == 429:
            if retry_count >= MAX_RETRIES:
                 logger.error(f"Max retries exceeded for ({latitude}, {longitude})")
                 return pd.DataFrame()

            # Exponential backoff: 2^retry_count seconds
            wait_time = 2 ** retry_count
            logger.warning(
                f"Rate limit hit (429). Waiting {wait_time}s before retry "
                f"(attempt {retry_count + 1}/{MAX_RETRIES})"
            )
            time.sleep(wait_time)

            # Recursive retry
            return fetch_weather_window(
                latitude, longitude, target_date, window_days, retry_count + 1, rate_limiter
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

        # Only sleep if not using rate limiter (backward compatibility)
        if not rate_limiter:
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


def make_weather_id(peakid: str, target_date: date | str) -> str:
    """
    Create a unique weather ID from peak and date.
    """
    if isinstance(target_date, date):
        date_str = target_date.isoformat()
    else:
        date_str = str(target_date)[:10]
    return f"{peakid}_{date_str}"


def flatten_multiscale_windows(
    windows: dict[str, pd.DataFrame],
    peakid: str,
    target_date: date,
) -> dict[str, float | str]:
    """
    Flatten multi-scale weather windows into a single row (wide format).

    Creates columns like: 7d_b0_temperature_2m_mean, 30d_b5_wind_speed_10m_max, etc.

    Args:
        windows: Dict with '7d', '30d', '90d' DataFrames from build_multiscale_windows
        peakid: Peak identifier
        target_date: Target date for this weather record

    Returns:
        Dict representing a single row with all weather variables flattened.
    """
    row: dict[str, float | str] = {
        "weather_id": make_weather_id(peakid, target_date),
        "peakid": peakid,
        "smtdate": target_date.isoformat() if isinstance(target_date, date) else str(target_date)[:10],
    }

    for scale, df in windows.items():
        if df.empty:
            continue

        # Get numeric columns (weather variables)
        numeric_cols = [c for c in df.columns if c not in ("date", "scale", "bucket")]

        for _, bucket_row in df.iterrows():
            bucket_idx = int(bucket_row.get("bucket", 0))
            for col in numeric_cols:
                col_name = f"{scale}_b{bucket_idx}_{col}"
                value = bucket_row[col]
                row[col_name] = float(value) if pd.notna(value) else np.nan

    return row


class WeatherDataCollector:
    """
    Thread-safe collector for accumulating weather data into a single CSV.

    Usage:
        collector = WeatherDataCollector()

        # In parallel workers:
        collector.add(windows, peakid, target_date)

        # After all fetches complete:
        collector.save()
    """

    def __init__(self, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
        self.output_dir = output_dir
        self.rows: list[dict[str, float | str]] = []
        self.lock = Lock()
        self.weather_ids: set[str] = set()

    def load_existing(self) -> None:
        """
        Load existing weather.csv if present (for resume support).
        """
        weather_path = self.output_dir / "weather.csv"
        if weather_path.exists():
            try:
                df = pd.read_csv(weather_path)
                self.rows = df.to_dict("records")
                self.weather_ids = set(df["weather_id"].tolist())
                logger.info(f"Loaded {len(self.rows)} existing weather records")
            except Exception as e:
                logger.warning(f"Failed to load existing weather.csv: {e}")

    def has_weather(self, peakid: str, target_date: date | str) -> bool:
        """
        Check if weather data already exists for this peak+date.
        """
        weather_id = make_weather_id(peakid, target_date)
        return weather_id in self.weather_ids

    def add(
        self,
        windows: dict[str, pd.DataFrame],
        peakid: str,
        target_date: date,
    ) -> str:
        """
        Add weather data to the collector.

        Returns:
            The weather_id for this record.
        """
        row = flatten_multiscale_windows(windows, peakid, target_date)
        weather_id = row["weather_id"]

        with self.lock:
            if weather_id not in self.weather_ids:
                self.rows.append(row)
                self.weather_ids.add(weather_id)

        return weather_id

    def save(self) -> Path:
        """
        Save all collected weather data to weather.csv.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "weather.csv"

        with self.lock:
            if not self.rows:
                logger.warning("No weather data to save")
                return output_path

            df = pd.DataFrame(self.rows)

            # Sort columns: identifiers first, then weather variables
            id_cols = ["weather_id", "peakid", "smtdate"]
            weather_cols = sorted([c for c in df.columns if c not in id_cols])
            df = df[id_cols + weather_cols]

            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(df)} weather records to {output_path}")

        return output_path

    def __len__(self) -> int:
        return len(self.rows)
