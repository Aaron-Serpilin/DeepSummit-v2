"""Geocoding utilities for peak coordinate lookup via Nominatim/OSM."""

import hashlib
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
CACHE_DIR = Path.home() / ".cache" / "deepsummit" / "geocode"
REQUEST_DELAY = 1.0  # Nominatim requires 1 req/sec max


def _get_cache_path(peak_name: str, himal: str) -> Path:
    """
    Generate cache file path from peak name and himal.
    """
    key = f"{peak_name}_{himal}".lower()
    slug = hashlib.md5(key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{slug}.json"


def _load_from_cache(cache_path: Path) -> tuple[float, float] | None:
    """
    Load cached coordinates if available.
    """
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            return (data["lat"], data["lon"])
        except (json.JSONDecodeError, KeyError):
            return None
    return None


def _save_to_cache(cache_path: Path, lat: float, lon: float) -> None:
    """
    Save coordinates to cache.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"lat": lat, "lon": lon}))


def geocode_peak(
    peak_name: str,
    himal: str,
    use_cache: bool = True,
) -> tuple[float, float] | None:
    """
    Look up coordinates for a Himalayan peak using Nominatim.

    Args:
        peak_name: Name of the peak (e.g., "Everest", "Annapurna I")
        himal: Mountain range (e.g., "Khumbu Himal", "Annapurna Himal")
        use_cache: Whether to use disk cache (default True)

    Returns:
        Tuple of (latitude, longitude) or None if not found.
    """
    cache_path = _get_cache_path(peak_name, himal)

    if use_cache:
        cached = _load_from_cache(cache_path)
        if cached is not None:
            logger.debug(f"Cache hit for {peak_name}")
            return cached

    # Build query — try peak name with region context
    queries = [
        f"{peak_name}, {himal}, Nepal",
        f"{peak_name}, Nepal",
        f"{peak_name}, Himalayas",
    ]

    for query in queries:
        try:
            response = requests.get(
                NOMINATIM_URL,
                params={
                    "q": query,
                    "format": "json",
                    "limit": 1,
                },
                headers={"User-Agent": "DeepSummit/1.0 (research project)"},
                timeout=10,
            )
            response.raise_for_status()

            results = response.json()
            if results:
                lat = float(results[0]["lat"])
                lon = float(results[0]["lon"])

                if use_cache:
                    _save_to_cache(cache_path, lat, lon)

                logger.info(f"Geocoded {peak_name}: ({lat}, {lon})")
                time.sleep(REQUEST_DELAY)  # Rate limit
                return (lat, lon)

            time.sleep(REQUEST_DELAY)

        except requests.RequestException as e:
            logger.warning(f"Geocoding request failed for {peak_name}: {e}")
            continue

    logger.warning(f"Could not geocode {peak_name}")
    return None


def geocode_peaks_batch(
    peaks_df: pd.DataFrame,
    name_col: str = "pkname",
    himal_col: str = "himal",
    skip_existing: bool = True,
) -> pd.DataFrame:
    """
    Add latitude/longitude columns to a peaks DataFrame.

    Args:
        peaks_df: DataFrame with peak names and himal info
        name_col: Column containing peak names
        himal_col: Column containing mountain range names
        skip_existing: Skip rows that already have coords

    Returns:
        DataFrame with latitude and longitude columns added.
    """
    df = peaks_df.copy()

    if "latitude" not in df.columns:
        df["latitude"] = pd.NA
    if "longitude" not in df.columns:
        df["longitude"] = pd.NA

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Geocoding peaks"):
        # Skip if already has coordinates
        if skip_existing and pd.notna(row.get("latitude")):
            continue

        coords = geocode_peak(row[name_col], row.get(himal_col, ""))

        if coords:
            df.at[idx, "latitude"] = coords[0]
            df.at[idx, "longitude"] = coords[1]

    return df
