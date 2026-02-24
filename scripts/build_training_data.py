#!/usr/bin/env python3
"""
Build training data from HDB CSVs and Open-Meteo weather.

Joins members + expeditions + peaks, fetches weather for each unique
(peak, date) combination, engineers features, and outputs features.csv.

Usage:
    python scripts/build_training_data.py
"""

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.rate_limiter import RateLimiter
from utils.weather import (
    build_multiscale_windows,
    fetch_weather_window,
    get_cached_weather,
    save_weather_cache,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "himalayas"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "training"

SEASON_MAP = {"Spring": 0, "Summer": 1, "Autumn": 2, "Winter": 3}

def load_csvs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three HDB CSVs.
    """
    members = pd.read_csv(DATA_DIR / "members_clean.csv")
    expeditions = pd.read_csv(DATA_DIR / "expeditions_clean.csv")
    peaks = pd.read_csv(DATA_DIR / "peaks_clean.csv")
    return members, expeditions, peaks


def join_data(
    members: pd.DataFrame,
    expeditions: pd.DataFrame,
    peaks: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join members with expeditions and peaks.
    """
    # Join members → expeditions on expid
    df = members.merge(
        expeditions,
        on="expid",
        how="left",
        suffixes=("", "_exp"),
    )

    # Join with peaks on peakid — handle both peakid columns
    # Members and expeditions both have peakid, use the one from members
    peak_cols = ["peakid", "pkname", "heightm", "himal"]
    if "latitude" in peaks.columns:
        peak_cols.extend(["latitude", "longitude"])

    df = df.merge(
        peaks[peak_cols],
        on="peakid",
        how="left",
    )

    return df


def filter_valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to rows with valid smtdate and coordinates.
    """
    initial_count = len(df)

    # Filter out rows without summit date
    df = df[df["smtdate"].notna()].copy()
    after_date = len(df)
    logger.info(f"Filtered {initial_count - after_date} rows without smtdate")

    # Filter out rows without coordinates (if latitude column exists)
    if "latitude" in df.columns:
        df = df[df["latitude"].notna() & df["longitude"].notna()].copy()
        after_coords = len(df)
        logger.info(f"Filtered {after_date - after_coords} rows without coordinates")

    return df


def compute_experience_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute prior expedition/summit counts per climber."""
    df = df.copy()

    # Parse smtdate for sorting
    df["smtdate_parsed"] = pd.to_datetime(df["smtdate"], errors="coerce")

    # Create climber identifier (fname + lname + nationality)
    df["climber_id"] = (
        df["fname"].fillna("").str.lower()
        + "_"
        + df["lname"].fillna("").str.lower()
        + "_"
        + df["nationality"].fillna("").str.lower()
    )

    # Sort by climber and date
    df = df.sort_values(["climber_id", "smtdate_parsed"])

    # Compute cumulative stats per climber (excluding current row)
    df["prior_expeditions"] = df.groupby("climber_id").cumcount()

    # Prior summits: cumulative sum of summit_reached, shifted by 1
    df["prior_summits"] = (
        df.groupby("climber_id")["summit_reached"]
        .apply(lambda x: x.shift(1).cumsum().fillna(0))
        .reset_index(level=0, drop=True)
    )

    # Highest previous altitude
    df["highest_prev_altitude_m"] = (
        df.groupby("climber_id")["highpt_m"]
        .apply(lambda x: x.shift(1).cummax().fillna(0))
        .reset_index(level=0, drop=True)
    )

    return df


def fetch_weather_for_expeditions(df: pd.DataFrame, max_workers: int = 10) -> pd.DataFrame:
    """
    Fetch weather for unique (peakid, smtdate) pairs with parallel workers.

    Args:
        df: DataFrame with expedition data
        max_workers: Number of parallel workers for API calls (default 10)

    Returns:
        DataFrame with weather_path column added
    """
    df = df.copy()

    # Get unique peak+date combinations
    unique_weather = df[["peakid", "smtdate", "latitude", "longitude"]].drop_duplicates(
        subset=["peakid", "smtdate"]
    )
    logger.info(f"Need weather for {len(unique_weather)} unique peak+date combinations")
    logger.info(f"Using {max_workers} parallel workers")

    # Create shared rate limiter for all workers
    # Use conservative rate to avoid hitting API limits
    rate_limiter = RateLimiter(requests_per_minute=500)
    logger.info(f"Rate limiter: {rate_limiter.requests_per_minute} req/min, ~{rate_limiter.min_interval*1000:.0f}ms between requests")

    # Track weather paths
    weather_paths: dict[tuple[str, str], str] = {}
    fetched = 0
    cached = 0
    failed = 0

    # Progress tracking file
    progress_file = OUTPUT_DIR / "weather_fetch_progress.txt"
    completed_keys = set()

    # Thread-safe lock for writing progress
    progress_lock = Lock()

    if progress_file.exists():
        with open(progress_file) as f:
            completed_keys = set(line.strip() for line in f)
        logger.info(f"Resuming: {len(completed_keys)} already completed")

    def process_weather_row(row_data):
        """Worker function to process a single weather fetch."""
        peakid, smtdate_str, lat, lon = row_data
        progress_key = f"{peakid}_{smtdate_str}"

        # Skip if already completed
        if progress_key in completed_keys:
            return ("cached", peakid, smtdate_str, None)

        try:
            smtdate = datetime.strptime(smtdate_str, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            logger.warning(f"Invalid date format: {smtdate_str}")
            return ("failed", peakid, smtdate_str, "Invalid date")

        # Check cache first
        cached_data = get_cached_weather(peakid, smtdate)
        if cached_data is not None:
            # Mark as completed
            with progress_lock:
                with open(progress_file, "a") as f:
                    f.write(f"{progress_key}\n")
                completed_keys.add(progress_key)
            return ("cached", peakid, smtdate_str, None)

        # Fetch from API (with retry logic built in)
        raw_weather = fetch_weather_window(
            lat, lon, smtdate, window_days=90, rate_limiter=rate_limiter
        )

        if not raw_weather.empty:
            # Build multi-scale windows and concatenate
            windows = build_multiscale_windows(raw_weather, smtdate)
            combined = pd.concat(
                [windows["7d"], windows["30d"], windows["90d"]], ignore_index=True
            )

            save_weather_cache(combined, peakid, smtdate)

            # Mark as completed
            with progress_lock:
                with open(progress_file, "a") as f:
                    f.write(f"{progress_key}\n")
                completed_keys.add(progress_key)

            return ("fetched", peakid, smtdate_str, None)
        else:
            return ("failed", peakid, smtdate_str, "Empty response")

    # Prepare work items
    work_items = [
        (row["peakid"], row["smtdate"], row["latitude"], row["longitude"])
        for _, row in unique_weather.iterrows()
    ]

    # Process in parallel with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_weather_row, item): item for item in work_items}

        # Process results as they complete
        with tqdm(total=len(futures), desc="Fetching weather") as pbar:
            for future in as_completed(futures):
                try:
                    status, peakid, smtdate_str, error = future.result()

                    if status == "cached":
                        cached += 1
                        weather_paths[(peakid, smtdate_str)] = f"weather/{peakid}_{smtdate_str[:10]}.csv"
                    elif status == "fetched":
                        fetched += 1
                        weather_paths[(peakid, smtdate_str)] = f"weather/{peakid}_{smtdate_str[:10]}.csv"
                    elif status == "failed":
                        failed += 1
                        if error:
                            logger.debug(f"Failed {peakid}_{smtdate_str}: {error}")

                except Exception as e:
                    logger.error(f"Worker error: {e}")
                    failed += 1

                pbar.update(1)

    logger.info(f"Weather fetch complete: {cached} cached, {fetched} fetched, {failed} failed")

    # Log rate limiter statistics
    stats = rate_limiter.get_stats()
    logger.info(
        f"Rate limiter stats: {stats['total_requests']} requests, "
        f"actual rate: {stats['actual_rate_per_min']}/min, "
        f"elapsed: {stats['elapsed_minutes']:.1f} min"
    )

    # Add weather_path column to main df
    df["weather_path"] = df.apply(
        lambda row: weather_paths.get((row["peakid"], row["smtdate"]), ""),
        axis=1,
    )

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer final feature set."""
    df = df.copy()

    # Parse smtdate for temporal features
    df["smtdate_parsed"] = pd.to_datetime(df["smtdate"], errors="coerce")
    df["day_of_year"] = df["smtdate_parsed"].dt.dayofyear

    # Encode season
    df["season_encoded"] = df["season"].map(SEASON_MAP).fillna(-1).astype(int)

    # Binary encodings
    df["sex_encoded"] = (df["sex"] == "F").astype(int)
    df["is_sherpa"] = df["sherpa"].fillna(False).astype(int)
    df["oxygen_planned"] = df["oxygen_used"].fillna(False).astype(int)
    df["is_hired"] = df["hired"].fillna(False).astype(int)
    df["is_commercial"] = df["commercial"].fillna(False).astype(int)
    df["o2_available"] = df["o2used"].fillna(False).astype(int)

    # Compute historical success rates per route
    route_success = df.groupby("route1")["summit_reached"].mean().to_dict()
    df["route_historical_success_rate"] = df["route1"].map(route_success).fillna(0.5)

    # Historical success rate per peak
    peak_success = df.groupby("peakid")["summit_reached"].mean().to_dict()
    df["peak_historical_success_rate"] = df["peakid"].map(peak_success).fillna(0.5)

    return df


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select and rename columns for final output."""
    output_cols = [
        # Identifiers
        "expid",
        "membid",
        "peakid",
        "smtdate",
        "weather_path",
        # Target
        "summit_reached",
        # Member features
        "age",
        "sex",
        "sex_encoded",
        "nationality",
        "oxygen_planned",
        "is_hired",
        "is_sherpa",
        "prior_expeditions",
        "prior_summits",
        "highest_prev_altitude_m",
        # Expedition features
        "totmembers",
        "smtmembers",
        "is_commercial",
        "o2_available",
        "camps",
        "style",
        "route1",
        # Peak features
        "heightm",
        "himal",
        # Temporal features
        "year",
        "season",
        "season_encoded",
        "day_of_year",
        # Derived features
        "route_historical_success_rate",
        "peak_historical_success_rate",
    ]

    # Keep only columns that exist
    existing_cols = [c for c in output_cols if c in df.columns]
    return df[existing_cols]


def main(request_delay: float = 0.5) -> None:
    """Build training data pipeline."""

    # Override global delay if specified
    import utils.weather as weather_module
    weather_module.REQUEST_DELAY = request_delay


    logger.info("Starting training data build")

    # Load data
    logger.info("Loading CSVs...")
    members, expeditions, peaks = load_csvs()
    logger.info(
        f"Loaded: {len(members)} members, {len(expeditions)} expeditions, {len(peaks)} peaks"
    )

    # Check if peaks have coordinates
    if "latitude" not in peaks.columns or peaks["latitude"].isna().all():
        logger.error("Peaks file has no coordinates. Run enrich_peak_coordinates.py first.")
        sys.exit(1)

    # Join data
    logger.info("Joining data...")
    df = join_data(members, expeditions, peaks)
    logger.info(f"Joined data: {len(df)} rows")

    # Filter valid rows
    logger.info("Filtering valid rows...")
    df = filter_valid_rows(df)
    logger.info(f"After filtering: {len(df)} rows")

    if len(df) == 0:
        logger.error("No valid rows after filtering")
        sys.exit(1)

    # Compute experience features
    logger.info("Computing experience features...")
    df = compute_experience_features(df)

    # Fetch weather
    logger.info("Fetching weather data...")
    df = fetch_weather_for_expeditions(df)

    # Engineer features
    logger.info("Engineering features...")
    df = engineer_features(df)

    # Select output columns
    df = select_output_columns(df)

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "features.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} rows to {output_path}")

    # Summary stats
    logger.info("=== Summary ===")
    logger.info(f"Total training samples: {len(df)}")
    logger.info(f"Summit success rate: {df['summit_reached'].mean():.2%}")
    logger.info(f"Unique peaks: {df['peakid'].nunique()}")
    logger.info(f"Year range: {df['year'].min()} - {df['year'].max()}")
    logger.info(f"Weather files: {(df['weather_path'] != '').sum()}")


if __name__ == "__main__":
    import sys
    delay = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    main(request_delay=delay)
