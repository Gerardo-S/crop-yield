"""
fetch_usda_corn.py
--------------------
Fetches corn data from the USDA NASS QuickStats API for 2021–present.

Data retrieved (both state-level and county-level):
  - Yield (BU / ACRE)
  - Acres planted
  - Acres harvested
  - Production (BU)

Safeguards:
  - count_records()  : checks how many rows a query would return BEFORE fetching.
  - Batching by year : if a query exceeds BATCH_THRESHOLD rows, it is automatically
                       split year-by-year and each batch is appended to the CSV
                       incrementally, keeping memory usage low.

Usage:
    1. Get a free API key at: https://quickstats.nass.usda.gov/api
    2. Set your key:  export NASS_API_KEY="your_key_here"
       OR paste it directly into API_KEY below (not recommended for shared code).
    3. Run:  python fetch_usda_corn.py

Output:
    corn_state.csv   — state-level records
    corn_county.csv  — county-level records
"""

import os
import requests
import pandas as pd
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────

API_KEY = os.environ.get("NASS_API_KEY", "YOUR_KEY_HERE")

BASE_URL        = "https://quickstats.nass.usda.gov/api/api_GET/"
COUNT_URL       = "https://quickstats.nass.usda.gov/api/get_counts/"
API_ROW_LIMIT   = 50_000   # NASS hard limit — requests above this will error
BATCH_THRESHOLD = 5_000    # If a query would exceed this, batch it year-by-year

COMMODITY       = "CORN"
CLASS           = "ALL CLASSES"
STAT_CATEGORIES = ["YIELD", "AREA PLANTED", "AREA HARVESTED", "PRODUCTION"]
REF_PERIOD      = "YEAR"
PROD_PRACT      = "ALL PRODUCTION PRACTICES"
START_YEAR      = 2021
END_YEAR        = datetime.now().year   # fetch up to the current calendar year

# ── Core API helpers ──────────────────────────────────────────────────────────

def _base_params(stat: str, agg_level: str) -> dict:
    """Build the shared query parameters for a given stat + aggregation level."""
    return {
        "sector_desc": "CROPS",
        "commodity_desc": COMMODITY,
        "class_desc": CLASS,
        "statisticcat_desc": stat,
        "reference_period_desc": REF_PERIOD,
        "prodn_practice_desc": PROD_PRACT,
        "agg_level_desc": agg_level,
        "source_desc": "SURVEY",
    }


def count_records(params: dict) -> int:
    """
    Safeguard #1: Query the NASS get_counts endpoint to find out how many rows
    a given set of parameters would return, WITHOUT downloading the actual data.

    Returns the record count as an integer.
    """
    full_params = {**params, "key": API_KEY}
    response = requests.get(COUNT_URL, params=full_params, timeout=30)
    response.raise_for_status()
    data = response.json()
    # Response looks like: {"count": "1234"}
    return int(data.get("count", 0))


def fetch_records(params: dict) -> list[dict]:
    """
    Fetch actual records from the NASS api_GET endpoint.
    Assumes the caller has already verified the count is within API_ROW_LIMIT.
    Returns a list of raw record dicts.
    """
    full_params = {**params, "key": API_KEY, "format": "JSON"}
    response = requests.get(BASE_URL, params=full_params, timeout=60)
    response.raise_for_status()
    data = response.json()
    if "data" not in data:
        print(f"    Warning: unexpected response — {data}")
        return []
    return data["data"]


# ── Count-check safeguard ─────────────────────────────────────────────────────

def check_all_counts() -> dict:
    """
    Before fetching anything, print a full table of expected record counts
    broken down by stat category and aggregation level. Flags anything that
    will be batched or that would exceed the hard API limit.

    Returns a nested dict: counts[agg_level][stat] = count
    """
    print("\n── Pre-fetch record count check ──")
    print(f"  {'Stat category':<20} {'STATE':>8} {'COUNTY':>8}  Notes")
    print("  " + "-" * 58)

    counts = {"STATE": {}, "COUNTY": {}}
    for stat in STAT_CATEGORIES:
        notes = []
        for agg in ("STATE", "COUNTY"):
            params = {**_base_params(stat, agg), "year__GE": START_YEAR}
            n = count_records(params)
            counts[agg][stat] = n
            if n > API_ROW_LIMIT:
                notes.append(f"{agg}: EXCEEDS API LIMIT ({n:,})")
            elif n > BATCH_THRESHOLD:
                notes.append(f"{agg}: will batch ({n:,})")

        state_n  = counts["STATE"][stat]
        county_n = counts["COUNTY"][stat]
        note_str = " | ".join(notes) if notes else "ok"
        print(f"  {stat:<20} {state_n:>8,} {county_n:>8,}  {note_str}")

    total_state  = sum(counts["STATE"].values())
    total_county = sum(counts["COUNTY"].values())
    print("  " + "-" * 58)
    print(f"  {'TOTAL':<20} {total_state:>8,} {total_county:>8,}")
    print()

    return counts


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_records(records: list[dict]) -> pd.DataFrame:
    """
    Convert raw NASS records to a clean DataFrame with selected columns.
    Suppressed values ('(D)', '(Z)', etc.) become NaN.
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    keep_cols = {
        "year": "year",
        "reference_period_desc": "reference_period_desc",
        "prodn_practice_desc": "prodn_practice_desc",
        "state_name": "state",
        "state_alpha": "state_abbr",
        "state_fips_code": "state_fips",
        "county_name": "county",
        "county_ansi": "county_fips",
        "agg_level_desc": "agg_level",
        "commodity_desc": "commodity",
        "class_desc": "class",
        "util_practice_desc":"util_practice_desc",
        "statisticcat_desc": "statistic_category",
        "short_desc": "description",
        "unit_desc": "unit",
        "Value": "value",
        "CV (%)": "coeff_variation_pct",
    }
    existing = {k: v for k, v in keep_cols.items() if k in df.columns}
    df = df[list(existing.keys())].rename(columns=existing)

    df["value"] = (
        df["value"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .apply(pd.to_numeric, errors="coerce")
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df


# ── Batched fetching ──────────────────────────────────────────────────────────

def fetch_stat_to_csv(stat: str, agg_level: str, out_path: str,
                      known_count: int, write_header: bool) -> int:
    """
    Fetch one stat category at one aggregation level, batching by year if the
    total row count exceeds BATCH_THRESHOLD. Appends cleaned rows to out_path
    as it goes to keep memory usage low.

    Returns total rows written.
    Raises RuntimeError if any single year-batch would exceed the API row limit.
    """
    total_written = 0

    if known_count <= BATCH_THRESHOLD:
        # ── Small enough to fetch in one shot ──
        params = {**_base_params(stat, agg_level), "year__GE": START_YEAR}
        records = fetch_records(params)
        df = clean_records(records)
        if not df.empty:
            df.to_csv(out_path, mode="a", index=False, header=write_header)
            total_written += len(df)

    else:
        # ── Safeguard #2: batch year-by-year ──
        print(f"    Batching by year ({START_YEAR}–{END_YEAR})...")
        for year in range(START_YEAR, END_YEAR + 1):
            params = {**_base_params(stat, agg_level), "year": year}

            # Count this individual year before fetching
            year_count = count_records(params)
            if year_count == 0:
                print(f"      {year}: no data, skipping")
                continue
            if year_count > API_ROW_LIMIT:
                raise RuntimeError(
                    f"Year {year} / {stat} / {agg_level} has {year_count:,} rows — "
                    f"exceeds the API limit of {API_ROW_LIMIT:,}. "
                    "Consider adding state-level batching as well."
                )

            records = fetch_records(params)
            df = clean_records(records)
            if not df.empty:
                df.to_csv(out_path, mode="a", index=False, header=write_header)
                write_header = False   # only the very first batch gets the header
                total_written += len(df)
                print(f"      {year}: {len(df):,} rows appended")

    return total_written


def fetch_all_to_csv(agg_level: str, out_path: str, counts: dict) -> int:
    """
    Fetch all stat categories for a given aggregation level, streaming
    results to out_path. Returns the total number of rows written.
    """
    # Start fresh
    open(out_path, "w").close()

    total = 0
    write_header = True  # first batch written to the file gets the CSV header
    for stat in STAT_CATEGORIES:
        known_count = counts[agg_level].get(stat, 0)
        print(f"  [{stat}] — {known_count:,} expected rows")
        if known_count == 0:
            print("    No data, skipping.")
            continue

        rows = fetch_stat_to_csv(stat, agg_level, out_path,
                                 known_count, write_header)
        write_header = False  # subsequent stats append without a header
        total += rows
        print(f"    → {rows:,} rows written")

    return total


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if API_KEY == "YOUR_API_KEY_HERE":
        raise ValueError(
            "Please set your NASS API key.\n"
            "  Option 1 (recommended): export NASS_API_KEY='your_key'\n"
            "  Option 2: edit the API_KEY variable in this script.\n"
            "  Get a key at: https://quickstats.nass.usda.gov/api"
        )

    print("=" * 60)
    print("USDA NASS QuickStats — Corn Data Fetch")
    print(f"Commodity  : {COMMODITY} / {CLASS}")
    print(f"Years      : {START_YEAR} – {END_YEAR}")
    print(f"Batch limit: {BATCH_THRESHOLD:,} rows per request")
    print("=" * 60)

    # ── Safeguard #1: check all counts before fetching any real data ──
    counts = check_all_counts()

    # Abort early if anything would blow past the API hard limit
    for agg in ("STATE", "COUNTY"):
        for stat, n in counts[agg].items():
            if n > API_ROW_LIMIT:
                print(f"  ABORT: {stat} / {agg} has {n:,} rows — exceeds the "
                      f"API hard limit of {API_ROW_LIMIT:,}.")
                return

    # ── State-level ──
    print("[1/2] Fetching state-level data → corn_state.csv")
    state_total = fetch_all_to_csv("STATE", "corn_state.csv", counts)
    print(f"  Done. {state_total:,} total rows saved.\n")

    # ── County-level ──
    print("[2/2] Fetching county-level data → corn_county.csv")
    county_total = fetch_all_to_csv("COUNTY", "corn_county.csv", counts)

    county_df = pd.read_csv("corn_county.csv")
    suppressed = county_df["value"].isna().sum()
    print(f"  Done. {county_total:,} total rows saved.")
    print(f"  Note: {suppressed:,} suppressed county values (privacy rules) → NaN\n")

    print("=" * 60)
    print(f"All done!  corn_state.csv  ({state_total:,} rows)")
    print(f"           corn_county.csv ({county_total:,} rows)")
    print("=" * 60)


if __name__ == "__main__":
    main()
