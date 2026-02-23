# Training Data Pipeline — Workflow Guide

This document explains the complete workflow for building training data from raw Himalayan Database CSVs to a model-ready feature matrix with weather data.

---

## Overview

The pipeline has two phases:

```
Phase 1: Coordinate Enrichment (run once)
    peaks_clean.csv → enrich_peak_coordinates.py → peaks_clean.csv (with lat/lng)

Phase 2: Training Data Build (run after Phase 1)
    members_clean.csv  ─┐
    expeditions_clean.csv ─┼─→ build_training_data.py ─→ features.csv
    peaks_clean.csv    ─┘                               weather/*.csv
```

---

## Phase 1: Coordinate Enrichment

### Script: `scripts/enrich_peak_coordinates.py`

**Purpose:** Add latitude/longitude coordinates to `peaks_clean.csv` by geocoding peak names via OpenStreetMap's Nominatim API.

**When to run:** Once, before the first training data build. Subsequent runs skip already-geocoded peaks.

**Command:**
```bash
python scripts/enrich_peak_coordinates.py
```

### Step-by-Step Execution Flow

```
┌────────────────────────────────────────────────────────────────────────────┐
│  1. Load peaks_clean.csv                                                   │
│     └─→ Reads ~470 peaks with columns: peakid, pkname, heightm, himal,     │
│         location                                                           │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  2. Call geocode_peaks_batch(peaks_df)                                     │
│     └─→ From utils/geocoding.py                                            │
│     └─→ Iterates through each peak row with a progress bar                 │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  3. For each peak, call geocode_peak(peak_name, himal)                     │
│     └─→ From utils/geocoding.py                                            │
│                                                                            │
│     3a. Check disk cache first                                             │
│         └─→ Cache location: ~/.cache/deepsummit/geocode/{hash}.json        │
│         └─→ Hash is MD5 of "{peak_name}_{himal}".lower()                   │
│         └─→ If cache exists, return cached (lat, lon) immediately          │
│                                                                            │
│     3b. If cache miss, query Nominatim API                                 │
│         └─→ Tries three query formats in order:                            │
│             1. "{peak_name}, {himal}, Nepal"                               │
│             2. "{peak_name}, Nepal"                                        │
│             3. "{peak_name}, Himalayas"                                    │
│         └─→ Uses User-Agent: "DeepSummit/1.0 (research project)"           │
│         └─→ Waits 1 second between requests (Nominatim rate limit)         │
│                                                                            │
│     3c. If found, cache result and return (lat, lon)                       │
│         If not found, return None                                          │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  4. Update DataFrame with latitude/longitude columns                       │
│     └─→ Peaks that couldn't be geocoded have NaN values                    │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  5. Save enriched DataFrame back to peaks_clean.csv                        │
│     └─→ Overwrites the original file with new columns                      │
│     └─→ Logs count of successful geocodes and failures                     │
└────────────────────────────────────────────────────────────────────────────┘
```

### Module: `utils/geocoding.py`

| Function | Purpose |
|----------|---------|
| `geocode_peak(peak_name, himal)` | Look up single peak coordinates via Nominatim |
| `geocode_peaks_batch(peaks_df)` | Process entire DataFrame, adding lat/lng columns |
| `_get_cache_path(peak_name, himal)` | Generate cache file path from peak info |
| `_load_from_cache(cache_path)` | Load coordinates from disk cache |
| `_save_to_cache(cache_path, lat, lon)` | Save coordinates to disk cache |

### Output

After running, `peaks_clean.csv` gains two new columns:
```csv
peakid,pkname,heightm,himal,location,latitude,longitude
EVER,Everest,8849,12,Khumbu Himal,27.9881,86.9250
ANN1,Annapurna I,8091,1,Annapurna Himal,28.5966,83.8203
...
```

---

## Phase 2: Training Data Build

### Script: `scripts/build_training_data.py`

**Purpose:** Join HDB CSVs, fetch weather data from Open-Meteo, engineer features, and produce the final training dataset.

**Prerequisites:**
- `peaks_clean.csv` must have latitude/longitude columns (run Phase 1 first)

**Command:**
```bash
python scripts/build_training_data.py
```

### Step-by-Step Execution Flow

```
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Load CSVs                                                         │
│  ═══════════════════                                                       │
│                                                                            │
│  load_csvs() reads three files from data/himalayas/:                       │
│                                                                            │
│  members_clean.csv (~80,000 rows)                                          │
│  ├─ expid      — links to expedition                                       │
│  ├─ membid     — member ID within expedition                               │
│  ├─ peakid     — which peak                                                │
│  ├─ fname, lname, nationality — climber identity                           │
│  ├─ age, sex   — demographics                                              │
│  ├─ oxygen_used, summit_reached, died — outcomes                           │
│  ├─ highpt_m   — highest point reached                                     │
│  └─ hired, sherpa — role flags                                             │
│                                                                            │
│  expeditions_clean.csv (~12,000 rows)                                      │
│  ├─ expid      — unique expedition ID                                      │
│  ├─ peakid, year, season — when/where                                      │
│  ├─ route1     — climbing route                                            │
│  ├─ totmembers, smtmembers — team size/success count                       │
│  ├─ smtdate    — summit attempt date (key for weather!)                    │
│  ├─ success    — expedition outcome                                        │
│  └─ o2used, commercial, camps, style — expedition characteristics          │
│                                                                            │
│  peaks_clean.csv (~470 rows, enriched with coords)                         │
│  ├─ peakid, pkname, heightm, himal                                         │
│  └─ latitude, longitude — from Phase 1                                     │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Join Data                                                         │
│  ═════════════════                                                         │
│                                                                            │
│  join_data(members, expeditions, peaks)                                    │
│                                                                            │
│  members ──┬── JOIN on expid ──→ member_expedition rows                    │
│  expeditions─┘                                                             │
│                     │                                                      │
│                     └── JOIN on peakid ──→ full dataset with coords        │
│  peaks (with lat/lng)                                                      │
│                                                                            │
│  Result: ~80,000 rows with all columns from all three sources              │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Filter Valid Rows                                                 │
│  ═════════════════════════                                                 │
│                                                                            │
│  filter_valid_rows(df) removes:                                            │
│                                                                            │
│  ✗ Rows without smtdate (no date = can't fetch weather)                    │
│  ✗ Rows without coordinates (no coords = can't locate for weather)         │
│                                                                            │
│  Logs how many rows were filtered at each step                             │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: Compute Experience Features                                       │
│  ═══════════════════════════════════                                       │
│                                                                            │
│  compute_experience_features(df)                                           │
│                                                                            │
│  For each climber (identified by fname + lname + nationality):             │
│                                                                            │
│  1. Sort all their expeditions by smtdate                                  │
│                                                                            │
│  2. For each expedition row, compute (looking only at PRIOR expeditions):  │
│     ├─ prior_expeditions — count of previous expeditions                   │
│     ├─ prior_summits     — count of previous successful summits            │
│     └─ highest_prev_altitude_m — max altitude reached before this trip     │
│                                                                            │
│  This creates "leak-free" features: we only use information available      │
│  BEFORE the current expedition when predicting its outcome.                │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: Fetch Weather Data                                                │
│  ═══════════════════════════                                               │
│                                                                            │
│  fetch_weather_for_expeditions(df)                                         │
│                                                                            │
│  5a. Deduplicate weather requests                                          │
│      └─→ Many members share the same (peakid, smtdate)                     │
│      └─→ ~80,000 member rows → ~8,000 unique (peak, date) pairs            │
│                                                                            │
│  5b. For each unique (peakid, smtdate) pair:                               │
│      │                                                                     │
│      ├─→ Check cache: get_cached_weather(peakid, smtdate)                  │
│      │   └─→ From utils/weather.py                                         │
│      │   └─→ Looks for data/training/weather/{peakid}_{date}.csv           │
│      │   └─→ If exists, skip to next pair                                  │
│      │                                                                     │
│      ├─→ Cache miss: fetch_weather_window(lat, lon, smtdate, 90)           │
│      │   └─→ From utils/weather.py                                         │
│      │   └─→ Calls Open-Meteo Archive API                                  │
│      │   └─→ Fetches 90 days of weather ending on smtdate                  │
│      │   └─→ Returns DataFrame with 15 daily weather variables             │
│      │                                                                     │
│      ├─→ Build multi-scale windows: build_multiscale_windows(raw, date)    │
│      │   └─→ From utils/weather.py                                         │
│      │   └─→ Creates three temporal views:                                 │
│      │       ├─ 7d:  7 rows (days -6 to 0, full resolution)                │
│      │       ├─ 30d: 10 rows (3-day aggregates)                            │
│      │       └─ 90d: 9 rows (10-day aggregates)                            │
│      │                                                                     │
│      └─→ Save to cache: save_weather_cache(combined, peakid, smtdate)      │
│          └─→ From utils/weather.py                                         │
│          └─→ Writes to data/training/weather/{peakid}_{date}.csv           │
│                                                                            │
│  5c. Add weather_path column to DataFrame                                  │
│      └─→ Each row gets path like "weather/EVER_1953-05-29.csv"             │
│      └─→ Training DataLoader will join weather at runtime                  │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: Engineer Features                                                 │
│  ═══════════════════════════                                               │
│                                                                            │
│  engineer_features(df)                                                     │
│                                                                            │
│  Temporal features:                                                        │
│  ├─ day_of_year     — 1-366, captures seasonal position                    │
│  └─ season_encoded  — Spring=0, Summer=1, Autumn=2, Winter=3               │
│                                                                            │
│  Binary encodings:                                                         │
│  ├─ sex_encoded     — F=1, M=0                                             │
│  ├─ is_sherpa       — hired Sherpa flag                                    │
│  ├─ oxygen_planned  — whether climber used oxygen                          │
│  ├─ is_hired        — hired support staff                                  │
│  ├─ is_commercial   — commercial expedition flag                           │
│  └─ o2_available    — expedition provided oxygen                           │
│                                                                            │
│  Historical success rates (derived features):                              │
│  ├─ route_historical_success_rate  — mean(summit_reached) for this route   │
│  └─ peak_historical_success_rate   — mean(summit_reached) for this peak    │
│                                                                            │
│  Note: These rates use ALL data (including future), so they're slightly    │
│  leaky. For production, you'd compute them only from training set.         │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 7: Select Output Columns                                             │
│  ═════════════════════════════                                             │
│                                                                            │
│  select_output_columns(df)                                                 │
│                                                                            │
│  Keeps only the columns needed for training:                               │
│                                                                            │
│  Identifiers (for traceability, not features):                             │
│  └─ expid, membid, peakid, smtdate, weather_path                           │
│                                                                            │
│  Target:                                                                   │
│  └─ summit_reached (boolean)                                               │
│                                                                            │
│  Member features:                                                          │
│  └─ age, sex, nationality, oxygen_planned, is_hired, is_sherpa,            │
│     prior_expeditions, prior_summits, highest_prev_altitude_m              │
│                                                                            │
│  Expedition features:                                                      │
│  └─ totmembers, smtmembers, is_commercial, o2_available, camps,            │
│     style, route1                                                          │
│                                                                            │
│  Peak features:                                                            │
│  └─ heightm, himal                                                         │
│                                                                            │
│  Temporal features:                                                        │
│  └─ year, season, season_encoded, day_of_year                              │
│                                                                            │
│  Derived features:                                                         │
│  └─ route_historical_success_rate, peak_historical_success_rate            │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 8: Save Output                                                       │
│  ═══════════════════                                                       │
│                                                                            │
│  Writes data/training/features.csv                                         │
│  └─→ One row per member-expedition                                         │
│  └─→ ~50,000-70,000 rows (after filtering)                                 │
│  └─→ ~30 columns                                                           │
│                                                                            │
│  Logs summary statistics:                                                  │
│  └─ Total training samples                                                 │
│  └─ Summit success rate                                                    │
│  └─ Unique peaks                                                           │
│  └─ Year range                                                             │
│  └─ Weather files generated                                                │
└────────────────────────────────────────────────────────────────────────────┘
```

### Module: `utils/weather.py`

| Function | Purpose |
|----------|---------|
| `fetch_weather_window(lat, lon, date, days)` | Fetch historical weather from Open-Meteo API |
| `build_multiscale_windows(raw_df, date)` | Create 7d/30d/90d temporal aggregations |
| `get_weather_cache_path(peakid, date)` | Generate cache file path |
| `get_cached_weather(peakid, date)` | Load cached weather CSV if available |
| `save_weather_cache(df, peakid, date)` | Save weather data to CSV cache |

### Open-Meteo Weather Variables

The API fetches these 15 daily variables:
```
temperature_2m_mean, temperature_2m_max, temperature_2m_min
apparent_temperature_mean, apparent_temperature_max, apparent_temperature_min
precipitation_sum, rain_sum, snowfall_sum, precipitation_hours
wind_speed_10m_max, wind_gusts_10m_max, wind_direction_10m_dominant
shortwave_radiation_sum, et0_fao_evapotranspiration
```

---

## Output Files

After running both scripts:

```
data/
├── himalayas/
│   ├── expeditions_clean.csv    # Unchanged
│   ├── members_clean.csv        # Unchanged
│   └── peaks_clean.csv          # Now has latitude, longitude columns
└── training/
    ├── features.csv             # Main training data (~30 columns)
    └── weather/
        ├── EVER_1953-05-29.csv  # Weather for Everest, May 29 1953
        ├── EVER_1960-05-25.csv
        ├── ANN1_1950-06-03.csv
        └── ...                   # One file per unique (peak, date)
```

### features.csv Schema

```csv
# Sample row (abbreviated)
expid,membid,peakid,smtdate,weather_path,summit_reached,age,sex,nationality,...
EVER53101,01,EVER,1953-05-29,weather/EVER_1953-05-29.csv,True,33,M,New Zealand,...
```

### weather/*.csv Schema

```csv
# Each file contains concatenated multi-scale windows
date,temperature_2m_mean,wind_speed_10m_max,...,scale,bucket
2023-05-09,10.5,25.0,...,7d,0
2023-05-10,11.2,30.0,...,7d,1
...
,12.1,28.0,...,30d,0      # Aggregated rows don't have date
,11.8,26.0,...,30d,1
...
,10.9,24.0,...,90d,0
```

---

## Execution Timeline

| Step | Duration (first run) | Duration (subsequent) |
|------|---------------------|----------------------|
| Coordinate enrichment | ~5-8 minutes | ~10 seconds (cached) |
| Load & join CSVs | ~2 seconds | ~2 seconds |
| Filter rows | <1 second | <1 second |
| Compute experience | ~10 seconds | ~10 seconds |
| Fetch weather | ~2-4 hours | ~30 seconds (cached) |
| Engineer features | ~5 seconds | ~5 seconds |
| Save output | ~2 seconds | ~2 seconds |

The weather fetching dominates first-run time because:
- ~8,000 unique (peak, date) pairs
- 100ms delay between API calls
- Some API calls take 200-500ms

After the first run, weather data is cached as CSVs, so subsequent runs are fast.

---

## Common Issues

### "Peaks file has no coordinates"
Run `enrich_peak_coordinates.py` first. The build script checks for the `latitude` column.

### Some peaks couldn't be geocoded
The enrichment script logs which peaks failed. You can:
1. Manually add coordinates for important peaks
2. Accept that expeditions to those peaks will be filtered out

### Weather API timeouts
The script will log failures and continue. Re-run and it will only fetch missing data (cache check happens first).

### Memory issues with large datasets
The entire dataset fits in memory on a modern laptop (~500MB peak). If needed, process in chunks by year.
