#!/usr/bin/env python3
"""
Convert Himalayan Database XLS files to clean CSVs.

Filters to all peaks >= MIN_HEIGHT_M (default 6000m), selects relevant
columns, and applies type conversions.

Note: HDB covers Nepal/Himalaya only. Karakoram peaks (K2, G1, G2, Broad
Peak, Nanga Parbat) and Shishapangma are not in this database.

Output:
    data/himalayas/peaks_clean.csv
    data/himalayas/expeditions_clean.csv
    data/himalayas/members_clean.csv

Usage:
    python scripts/preprocess_hdb.py
"""

import warnings
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import xlrd
from python_calamine import CalamineWorkbook

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "himalayas"

MIN_HEIGHT_M = 6000

SEASON_MAP = {1.0: "Spring", 2.0: "Summer", 3.0: "Autumn", 4.0: "Winter", 0.0: None}

# Loaders

def load_xlrd(path: Path) -> pd.DataFrame:
    wb = xlrd.open_workbook(str(path), ignore_workbook_corruption=True)
    ws = wb.sheet_by_index(0)
    headers = [ws.cell_value(0, c) for c in range(ws.ncols)]
    rows = [[ws.cell_value(r, c) for c in range(ws.ncols)] for r in range(1, ws.nrows)]
    return pd.DataFrame(rows, columns=headers)


def load_calamine(path: Path) -> pd.DataFrame:
    wb = CalamineWorkbook.from_path(str(path))
    rows = list(wb.get_sheet_by_index(0).to_python())
    return pd.DataFrame(rows[1:], columns=rows[0])


# Converters

def convert_excel_date(value) -> str | None:
    """
    Convert an Excel serial date to ISO string, or None if not a real date.
    """
    if value == "" or value is None:
        return None
    if isinstance(value, (int, float)) and value > 0:
        try:
            t = xlrd.xldate_as_tuple(float(value), datemode=0)
            return date(*t[:3]).isoformat()
        except Exception:
            return None
    if isinstance(value, (date, datetime)):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    return None


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    return False


def clean_age(value) -> float | None:
    """
    Return None for age=0 (not recorded) or clearly invalid values.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v <= 0 or v > 100:
        return None
    return v


def clean_sex(value) -> str | None:
    """
    Map HDB sex codes to our schema values. 'X' → None.
    """
    if value in ("M", "F"):
        return value
    return None


# Processors

def process_peaks(df: pd.DataFrame, peak_ids: set) -> pd.DataFrame:
    filtered = df[df["peakid"].isin(peak_ids)].copy()

    out = pd.DataFrame({
        "peakid":   filtered["peakid"],
        "pkname":   filtered["pkname"],
        "heightm":  filtered["heightm"].apply(lambda x: int(float(x))),
        "himal":    filtered["himal"].apply(lambda x: int(float(x)) if x not in ("", 0, 0.0) else None),
        "location": filtered["location"],
    })
    return out.reset_index(drop=True)


def process_expeditions(df: pd.DataFrame, peak_ids: set) -> pd.DataFrame:
    filtered = df[df["peakid"].isin(peak_ids)].copy()

    # Derive style: solo when team is 1 person; alpine/expedition can't be distinguished from HDB data — set to None.
    def derive_style(totmembers) -> str | None:
        try:
            n = int(float(totmembers))
        except (TypeError, ValueError):
            return None
        return "solo" if n == 1 else None

    out = pd.DataFrame({
        "expid":        filtered["expid"],
        "peakid":       filtered["peakid"],
        "year":         filtered["year"].apply(lambda x: int(float(x)) if x != "" else None),
        "season":       filtered["season"].apply(lambda x: SEASON_MAP.get(float(x) if x != "" else 0.0)),
        "route1":       filtered["route1"].apply(lambda x: x if x != "" else None),
        "totmembers":   filtered["totmembers"].apply(lambda x: int(float(x)) if x != "" else None),
        "smtmembers":   filtered["smtmembers"].apply(lambda x: int(float(x)) if x != "" else None),
        "smtdate":      filtered["smtdate"].apply(convert_excel_date),
        "success":      filtered["success1"].apply(to_bool),
        "termreason":   filtered["termreason"].apply(lambda x: int(float(x)) if x != "" else None),
        "highpoint":    filtered["highpoint"].apply(lambda x: int(float(x)) if x not in ("", 0, 0.0) else None),
        "o2used":       filtered["o2used"].apply(to_bool),
        "commercial":   filtered["comrte"].apply(to_bool),
        "camps":        filtered["camps"].apply(lambda x: int(float(x)) if x not in ("", 0, 0.0) else None),
        "style":        filtered["totmembers"].apply(derive_style),
    })
    return out.reset_index(drop=True)


def process_members(df: pd.DataFrame, valid_expids: set) -> pd.DataFrame:
    filtered = df[df["expid"].isin(valid_expids)].copy()

    out = pd.DataFrame({
        "expid":          filtered["expid"],
        "membid":         filtered["membid"],
        "peakid":         filtered["peakid"],
        "myear":          filtered["myear"].apply(lambda x: int(float(x)) if x not in ("", None) else None),
        "fname":          filtered["fname"].apply(lambda x: x if x not in ("", None) else None),
        "lname":          filtered["lname"].apply(lambda x: x if x not in ("", None) else None),
        "sex":            filtered["sex"].apply(clean_sex),
        "age":            filtered["calcage"].apply(clean_age),
        "nationality":    filtered["citizen"].apply(lambda x: x if x not in ("", None) else None),
        "oxygen_used":    filtered["mo2used"].apply(to_bool),
        "summit_reached": filtered["msuccess"].apply(to_bool),
        "died":           filtered["death"].apply(to_bool),
        "highpt_m":       filtered["mperhighpt"].apply(lambda x: int(float(x)) if x not in ("", None, 0, 0.0) else None),
        "hired":          filtered["hired"].apply(to_bool),
        "sherpa":         filtered["sherpa"].apply(to_bool),
    })
    return out.reset_index(drop=True)


def main() -> None:
    print("Loading source files...")

    peaks_raw = load_xlrd(DATA_DIR / "peaks.xls")
    exped_raw = load_xlrd(DATA_DIR / "expeditions.xls")
    mem1_raw  = load_calamine(DATA_DIR / "members-1940-2010.xls")
    mem2_raw  = load_calamine(DATA_DIR / "members-2011-2025.xls")
    members_raw = pd.concat([mem1_raw, mem2_raw], ignore_index=True)

    peaks_raw["heightm"] = pd.to_numeric(peaks_raw["heightm"], errors="coerce")
    peak_ids = set(peaks_raw[peaks_raw["heightm"] >= MIN_HEIGHT_M]["peakid"])
    print(f"Peaks >= {MIN_HEIGHT_M}m: {len(peak_ids)}")

    print("Processing peaks...")
    peaks_clean = process_peaks(peaks_raw, peak_ids)

    print("Processing expeditions...")
    exped_clean = process_expeditions(exped_raw, peak_ids)
    valid_expids = set(exped_clean["expid"])

    print("Processing members...")
    members_clean = process_members(members_raw, valid_expids)

    # Write CSVs
    peaks_path   = DATA_DIR / "peaks_clean.csv"
    exped_path   = DATA_DIR / "expeditions_clean.csv"
    members_path = DATA_DIR / "members_clean.csv"

    peaks_clean.to_csv(peaks_path, index=False)
    exped_clean.to_csv(exped_path, index=False)
    members_clean.to_csv(members_path, index=False)

    # Summary
    print("\n=== Output Summary ===")
    print(f"peaks_clean.csv:        {len(peaks_clean):>6} rows  →  {peaks_path}")
    print(f"expeditions_clean.csv:  {len(exped_clean):>6} rows  →  {exped_path}")
    print(f"members_clean.csv:      {len(members_clean):>6} rows  →  {members_path}")
    print("\nDone.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()