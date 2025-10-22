#!/usr/bin/env python3
"""
PM_P1P2_cleaner.py

What this script does:
1) Reads raw PolicyMap CSVs from:   snap_gap/raw_data/
   - PMP1_pov_fam_raw.csv, PMP2_pov_fam_raw.csv
   - PMP1_snap_fam_raw.csv, PMP2_snap_fam_raw.csv

2) Cleans & stacks them into final outputs in: snap_gap/clean_data/
   - PM_pov_fam_full_clean.csv   (all poverty, all periods)
   - PM_snap_fam_full_clean.csv  (all SNAP,   all periods)
   - PM_all_fam_clean.csv        (FAMILY poverty + family SNAP joined by period, includes gap)

3) Optionally saves the 4 intermediate per-period cleaned files:
   - PMP1_pov_fam_clean.csv, PMP2_pov_fam_clean.csv
   - PMP1_snap_fam_clean.csv, PMP2_snap_fam_clean.csv
You’ll be asked in the terminal: “Save intermediate files? [y/N]”
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Use folders relative to THIS script
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_data"
CLEAN_DIR = BASE_DIR / "clean_data"
CLEAN_DIR.mkdir(exist_ok=True)

SENTINELS = {-9999, -6666, -2222, 9999, 99999}
NA_LIKE = {"", ".", "NA", "N/A", "na", "n/a", "Null", "NULL", "null", "-"}

# Header maps
LONG_MAP_POV = {
    "Geography Type Description": "geography_type",
    "Geography Name": "geography_name",
    "Sits in State": "state",
    "GeoID": "geoid",
    "Formatted GeoID": "geoid_formatted",
    "Percent Families in Poverty": "pov_fam_raw",
    "Data Time Period": "time_period",
    "Geographic Vintage": "geo_vintage",
    "Data Source": "source",
    "Selected Location": "selected_location",
}
SHORT_MAP_POV = {
    "GeoID_Description": "geography_type",
    "GeoID_Name": "geography_name",
    "SitsinState": "state",
    "GeoID": "geoid",
    "GeoID_Formatted": "geoid_formatted",
    "pfampov": "pov_fam_raw",
    "TimeFrame": "time_period",
    "GeoVintage": "geo_vintage",
    "Source": "source",
    "Location": "selected_location",
}
LONG_MAP_SNAP = {
    "Geography Type Description": "geography_type",
    "Geography Name": "geography_name",
    "Sits in State": "state",
    "GeoID": "geoid",
    "Formatted GeoID": "geoid_formatted",
    "Percent Families Receiving Food Stamp/SNAP Benefits": "snap_fam_raw",
    "Data Time Period": "time_period",
    "Geographic Vintage": "geo_vintage",
    "Data Source": "source",
    "Selected Location": "selected_location",
}
SHORT_MAP_SNAP = {
    "GeoID_Description": "geography_type",
    "GeoID_Name": "geography_name",
    "SitsinState": "state",
    "GeoID": "geoid",
    "GeoID_Formatted": "geoid_formatted",
    "pfamsnap": "snap_fam_raw",
    "TimeFrame": "time_period",
    "GeoVintage": "geo_vintage",
    "Source": "source",
    "Location": "selected_location",
}

def pick_map(cols, long_map, short_map):
    cols = set(cols)
    if set(long_map).issubset(cols): return long_map
    if set(short_map).issubset(cols): return short_map
    return long_map if len(cols & set(long_map)) >= len(cols & set(short_map)) else short_map

def coerce_percent(s: pd.Series, zero_to_nan: bool = False) -> pd.Series:
    """
    Parse a percent-like series into numeric [0,100] with quality guards.
    - Strips '%'
    - Maps NA-like and sentinels to NaN
    - If zero_to_nan=True, coerces exact 0 to NaN (used for SNAP to 'get rid of 0 uptake')
    """
    s = s.astype(str).str.strip()
    s = s.where(~s.isin(NA_LIKE), np.nan)
    # remove literal percent signs like "12.3%"
    s = s.str.replace("%", "", regex=False)
    x = pd.to_numeric(s, errors="coerce")
    # sentinels & out-of-range to NaN
    x[(x < 0) | (x > 100) | x.isin(list(SENTINELS))] = np.nan
    if zero_to_nan:
        x = x.mask(x == 0)
    return x

def parse_period(tp: pd.Series):
    tp = tp.astype(str).str.strip()
    years = tp.str.findall(r"(?:19|20)\d{2}")
    start = years.apply(lambda ys: int(min(ys)) if ys else np.nan)
    end   = years.apply(lambda ys: int(max(ys)) if ys else np.nan)
    return tp, start, end

def clean_pov(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df = df.rename(columns=pick_map(df.columns, LONG_MAP_POV, SHORT_MAP_POV))
    if "pov_fam_raw" not in df.columns:
        raise ValueError(f"{path.name}: missing poverty column (expect 'Percent Families in Poverty' or 'pfampov').")
    df["pov_fam"] = coerce_percent(df["pov_fam_raw"], zero_to_nan=False)
    if "time_period" in df.columns:
        df["time_period"], df["period_start"], df["period_end"] = parse_period(df["time_period"])
    keep = ["geoid","geography_name","state","geoid_formatted","pov_fam","time_period","period_start","period_end"]
    return df[[c for c in keep if c in df.columns]]

def clean_snap(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df = df.rename(columns=pick_map(df.columns, LONG_MAP_SNAP, SHORT_MAP_SNAP))
    if "snap_fam_raw" not in df.columns:
        raise ValueError(f"{path.name}: missing SNAP column (expect long header or 'pfamsnap').")
    # Crucial change: treat 0% uptake as missing/unreliable
    df["snap_fam"] = coerce_percent(df["snap_fam_raw"], zero_to_nan=True)
    if "time_period" in df.columns:
        df["time_period"], df["period_start"], df["period_end"] = parse_period(df["time_period"])
    keep = ["geoid","geography_name","state","geoid_formatted","snap_fam","time_period","period_start","period_end"]
    return df[[c for c in keep if c in df.columns]]

def ask_yes_no(question: str, default: bool = False) -> bool:
    """
    Ask a yes/no question via input() and return True for yes, False for no.
    Default shown as [Y/n] or [y/N]. If input is empty/EOF, returns default.
    """
    prompt = " [Y/n] " if default else " [y/N] "
    try:
        ans = input(question + prompt).strip().lower()
    except EOFError:
        return default
    if not ans:
        return default
    return ans in {"y", "yes"}

def main():
    print("\n=== PolicyMap P1/P2 Cleaner ===")
    print(f"- Raw input dir:   {RAW_DIR}")
    print(f"- Clean output dir:{CLEAN_DIR}\n")
    print("This will produce:")
    print("  • PM_pov_fam_full_clean.csv   (all poverty, stacked)")
    print("  • PM_snap_fam_full_clean.csv  (all SNAP, stacked)")
    print("  • PM_all_fam_clean.csv        (poverty + SNAP joined by period, includes gap)")
    print("Optionally, it can ALSO write these intermediate files:")
    print("  • PMP1_pov_fam_clean.csv, PMP2_pov_fam_clean.csv")
    print("  • PMP1_snap_fam_clean.csv, PMP2_snap_fam_clean.csv\n")

    save_intermediate = ask_yes_no("Save intermediate per-period cleaned files?", default=False)

    files = {
        "PMP1_pov": RAW_DIR / "PMP1_pov_fam_raw.csv",
        "PMP2_pov": RAW_DIR / "PMP2_pov_fam_raw.csv",
        "PMP1_snap": RAW_DIR / "PMP1_snap_fam_raw.csv",
        "PMP2_snap": RAW_DIR / "PMP2_snap_fam_raw.csv",
    }

    print("\nLooking for raw files:")
    for k, p in files.items():
        print(f"  - {k}: {'FOUND' if p.exists() else 'MISSING'} ({p.name})")

    cleaned = {}
    if files["PMP1_pov"].exists(): cleaned["PMP1_pov"] = clean_pov(files["PMP1_pov"])
    if files["PMP2_pov"].exists(): cleaned["PMP2_pov"] = clean_pov(files["PMP2_pov"])
    if files["PMP1_snap"].exists(): cleaned["PMP1_snap"] = clean_snap(files["PMP1_snap"])
    if files["PMP2_snap"].exists(): cleaned["PMP2_snap"] = clean_snap(files["PMP2_snap"])

    # Optionally write the four intermediate per-period files
    if save_intermediate:
        if "PMP1_pov" in cleaned:
            cleaned["PMP1_pov"].to_csv(CLEAN_DIR / "PMP1_pov_fam_clean.csv", index=False)
            print("[write] clean_data/PMP1_pov_fam_clean.csv")
        if "PMP2_pov" in cleaned:
            cleaned["PMP2_pov"].to_csv(CLEAN_DIR / "PMP2_pov_fam_clean.csv", index=False)
            print("[write] clean_data/PMP2_pov_fam_clean.csv")
        if "PMP1_snap" in cleaned:
            cleaned["PMP1_snap"].to_csv(CLEAN_DIR / "PMP1_snap_fam_clean.csv", index=False)
            print("[write] clean_data/PMP1_snap_fam_clean.csv")
        if "PMP2_snap" in cleaned:
            cleaned["PMP2_snap"].to_csv(CLEAN_DIR / "PMP2_snap_fam_clean.csv", index=False)
            print("[write] clean_data/PMP2_snap_fam_clean.csv")

    # Stack POV
    pov_frames = [cleaned[k] for k in ["PMP1_pov","PMP2_pov"] if k in cleaned]
    pov_all = None
    if pov_frames:
        pov_all = (pd.concat(pov_frames, ignore_index=True)
                   .drop_duplicates(subset=["geoid","period_start","period_end"]))
        pov_all.to_csv(CLEAN_DIR / "PM_pov_fam_full_clean.csv", index=False)
        print("[write] clean_data/PM_pov_fam_full_clean.csv")

    # Stack SNAP
    snap_frames = [cleaned[k] for k in ["PMP1_snap","PMP2_snap"] if k in cleaned]
    snap_all = None
    if snap_frames:
        snap_all = (pd.concat(snap_frames, ignore_index=True)
                    .drop_duplicates(subset=["geoid","period_start","period_end"]))
        snap_all.to_csv(CLEAN_DIR / "PM_snap_fam_full_clean.csv", index=False)
        print("[write] clean_data/PM_snap_fam_full_clean.csv")

    # Join by period
    if (pov_all is not None) and (snap_all is not None):
        key = ["geoid","period_start","period_end"]
        extras = ["geography_name","state","geoid_formatted","time_period"]
        for c in extras:
            if c not in pov_all.columns: pov_all[c] = np.nan
            if c not in snap_all.columns: snap_all[c] = np.nan
        merged = pd.merge(
            pov_all[key + extras + ["pov_fam"]],
            snap_all[key + ["snap_fam"]],
            on=key, how="outer", validate="one_to_one"
        )
        merged["gap"] = merged["pov_fam"] - merged["snap_fam"]
        cols = ["geoid","geography_name","state","geoid_formatted","time_period",
                "period_start","period_end","pov_fam","snap_fam","gap"]
        merged = merged[cols].sort_values(["geoid","period_start","period_end"])
        merged.to_csv(CLEAN_DIR / "PM_all_fam_clean.csv", index=False)
        print("[write] clean_data/PM_all_fam_clean.csv")
    else:
        print("[skip] join: need at least one stacked POV and one stacked SNAP.")

    print("\nDone.")

if __name__ == "__main__":
    main()
