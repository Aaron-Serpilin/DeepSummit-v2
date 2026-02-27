#!/usr/bin/env python3
"""
Enrich peaks_clean.csv with latitude/longitude coordinates.

Run once to geocode all peaks. Results are saved back to the CSV.
Subsequent runs skip already-geocoded peaks.

Usage:
    python scripts/enrich_peak_coordinates.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.geocoding import geocode_peaks_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "himalayas"
PEAKS_FILE = DATA_DIR / "peaks_clean.csv"


def main() -> None:
    """Enrich peaks with coordinates and save."""
    logger.info(f"Loading peaks from {PEAKS_FILE}")

    if not PEAKS_FILE.exists():
        logger.error(f"Peaks file not found: {PEAKS_FILE}")
        sys.exit(1)

    peaks_df = pd.read_csv(PEAKS_FILE)
    logger.info(f"Loaded {len(peaks_df)} peaks")

    # Check how many already have coordinates
    if "latitude" in peaks_df.columns:
        existing = peaks_df["latitude"].notna().sum()
        logger.info(f"{existing} peaks already have coordinates")
    else:
        existing = 0

    # Geocode
    enriched_df = geocode_peaks_batch(peaks_df)

    # Report results
    new_coords = enriched_df["latitude"].notna().sum()
    logger.info(f"Geocoding complete: {new_coords} peaks have coordinates")
    logger.info(f"Added coordinates for {new_coords - existing} new peaks")

    # Count failures
    missing = enriched_df["latitude"].isna().sum()
    if missing > 0:
        logger.warning(f"{missing} peaks could not be geocoded:")
        for _, row in enriched_df[enriched_df["latitude"].isna()].iterrows():
            logger.warning(f"  - {row['pkname']} ({row['peakid']})")

    # Save
    enriched_df.to_csv(PEAKS_FILE, index=False)
    logger.info(f"Saved enriched peaks to {PEAKS_FILE}")


if __name__ == "__main__":
    main()
