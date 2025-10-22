#!/usr/bin/env python3
"""
PM_clean_pov_total.py

Cleans PolicyMap TOTAL poverty for:
- PMP1 (2014–2018) -> raw_data/PMP1_pov_total_raw.csv
- PMP2 (2019–2023) -> raw_data/PMP2_pov_total_raw.csv

Outputs (in clean_data/):
- PM_pov_total_full_clean.csv               (stacked total poverty)
Optionally (prompted in terminal):
- PMP1_pov_total_clean.csv, PMP2_pov_total_clean.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Folders relative to this script
BASE_DIR  = Path(__file__).resolve().parent
RAW_DIR   = BASE_DIR / "raw_data"
CLEAN_DIR = BASE_DIR / "clean_data"
CLEAN_DIR.mkdir(exist_ok=True)

SENTINELS = {-9999, -6666, -2222, 9999, 99999}
NA_LIKE = {"", ".", "NA", "N/A", "na", "n/a", "Null", "NULL", "null", "-"}

# Common header maps (PolicyMap long/short variants)
LONG_COMMON = {
    "Geography Type Description": "geography_type",
    "Geography Name": "geography_name",
    "Sits in State": "state",
    "GeoID": "geoid",
    "Formatted GeoID": "geoid_formatted",
    "Data Time Period": "time_period",
    "Geographic Vintage": "geo_vintage",
    "Data Source": "source",
    "Selected Location": "selected_location",
}
SHORT_COMMON = {
    "GeoID_Description": "geography_type",
    "GeoID_Name": "geography_name",
    "SitsinState": "state",
    "GeoID": "geoid",
    "GeoID_Formatted": "geoid_formatted",
    "TimeFrame": "time_period",
    "GeoVintage": "geo_vintage",
    "Source": "source",
    "Location": "selected_location",
}

# Likely TOTAL poverty column names (script also auto-detects via keywords)
TOTAL_POV_CANDIDATES = {
    "Percent in Poverty",
    "Percent Population in Poverty",
    "Percent of Individuals in Poverty",
    "Percent People in Poverty",
    "ppov",  # if PolicyMap uses a short code
}

def ask_yes_no(question: str, default: bool = False) -> bool:
    prompt = " [Y/n] " if default else " [y/N] "
    try:
        ans = input(question + prompt).strip().lower()
    except EOFError:
        return default
    if not ans:
        return default
    return ans in {"y","yes"}

def pick_common_map(cols):
    cols = set(cols)
    if set(LONG_COMMON).issubset(cols):  return LONG_COMMON
    if set(SHORT_COMMON).issubset(cols): return SHORT_COMMON
    # fallback: whichever overlaps more
    return LONG_COMMON if len(cols & set(LONG_COMMON)) >= len(cols & set(SHORT_COMMON)) else SHORT_COMMON

def find_total_pov_col(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    # exact knowns first
    for c in TOTAL_POV_CANDIDATES:
        if c in cols:
            return c
    # fallback: keyword scan
    low = {c: c.lower() for c in df.columns}
    def matches(name: str) -> bool:
        n = low[name]
        return ("percent" in n and "poverty" in n) or ("percent" in n and "in poverty" in n)
    candidates = [c for c in df.columns if matches(c)]
    if candidates:
        candidates.sort(key=len)
        return candidates[0]
    raise ValueError(f"Could not find TOTAL poverty measure column. Saw headers like: {list(df.columns)[:12]} ...")

def coerce_percent(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.where(~s.isin(NA_LIKE), np.nan)
    x = pd.to_numeric(s, errors="coerce")
    bad = (x < 0) | (x > 100) | x.isin(list(SENTINELS))
    x[bad] = np.nan
    return x

def parse_period(tp: pd.Series):
    tp = tp.astype(str).str.strip()
    years = tp.str.findall(r"(?:19|20)\d{2}")
    start = years.apply(lambda ys: int(min(ys)) if ys else np.nan)
    end   = years.apply(lambda ys: int(max(ys)) if ys else np.nan)
    return tp, start, end

def clean_total_pov(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df = df.rename(columns=pick_common_map(df.columns))
    pov_col = find_total_pov_col(df)
    df["pov_total"] = coerce_percent(df[pov_col])
    if "time_period" in df.columns:
        df["time_period"], df["period_start"], df["period_end"] = parse_period(df["time_period"])
    keep = ["geoid","geography_name","state","geoid_formatted",
            "pov_total","time_period","period_start","period_end"]
    return df[[c for c in keep if c in df.columns]]

def main():
    print("\n=== PolicyMap TOTAL Poverty Cleaner (PMP1 & PMP2) ===")
    print(f"- Raw input dir:    {RAW_DIR}")
    print(f"- Clean output dir: {CLEAN_DIR}\n")
    print("This will produce:")
    print("  • PM_pov_total_full_clean.csv  (stacked total poverty across 2014–18 & 2019–23)")
    save_intermediate = ask_yes_no("\nSave intermediate per-period cleaned files? (PMP1_pov_total_clean.csv, PMP2_pov_total_clean.csv)?", default=False)

    files = {
        "PMP1": RAW_DIR / "PMP1_pov_total_raw.csv",
        "PMP2": RAW_DIR / "PMP2_pov_total_raw.csv",
    }

    print("\nLooking for raw files:")
    for tag, p in files.items():
        print(f"  - {tag}: {'FOUND' if p.exists() else 'MISSING'}  ({p.name})")

    cleaned = {}
    for tag, p in files.items():
        if not p.exists():
            continue
        try:
            df = clean_total_pov(p)
            cleaned[tag] = df
            if save_intermediate:
                out = CLEAN_DIR / f"{tag}_pov_total_clean.csv"
                df.to_csv(out, index=False)
                print(f"[write] clean_data/{out.name}")
        except Exception as e:
            print(f"[ERROR] {p.name}: {e}")

    frames = [cleaned[k] for k in ["PMP1","PMP2"] if k in cleaned]
    if frames:
        full = (pd.concat(frames, ignore_index=True)
                  .drop_duplicates(subset=["geoid","period_start","period_end"]))
        full.to_csv(CLEAN_DIR / "PM_pov_total_full_clean.csv", index=False)
        print("\n[write] clean_data/PM_pov_total_full_clean.csv")
    else:
        print("\n[skip] No total-poverty inputs found; nothing written.")

    print("\nDone.")

if __name__ == "__main__":
    main()
