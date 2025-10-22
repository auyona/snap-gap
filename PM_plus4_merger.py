#!/usr/bin/env python3
# scripts/PM_plus4_merger.py
"""
Cleans & merges four new PolicyMap ZIP-level variables into your base file.

Inputs (must exist):
- clean_data/PM_merged.csv   (has geoid + period + your existing variables)
- raw_data/PMP1_no_vehicle_housing_unit.csv
- raw_data/PMP2_no_vehicle_housing_unit.csv
- raw_data/PMP1_house_no_internet.csv
- raw_data/PMP2_house_no_internet.csv
- raw_data/PMP1_no_computer_pct_households.csv
- raw_data/PMP2_no_computer_pct_households.csv
- raw_data/PMP1_hs_diploma_no_college.csv
- raw_data/PMP2_hs_diploma_no_college.csv

Output:
- clean_data/PM_plus4.csv
"""

import os, re, sys, glob
import pandas as pd
import numpy as np

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
RAW_DIR   = os.path.join(BASE_DIR, "raw_data")
CLEAN_DIR = os.path.join(BASE_DIR, "clean_data")
BASE_FILE = os.path.join(CLEAN_DIR, "PM_merged.csv")
OUT_FILE  = os.path.join(CLEAN_DIR, "PM_plus4.csv")

# Map filename fragments -> clean column names
VAR_MAP = {
    "no_vehicle_housing_unit":      "pct_no_vehicle",
    "house_no_internet":            "pct_no_internet",
    "no_computer_pct_households":   "pct_no_computer",
    "hs_diploma_no_college":        "pct_hs_diploma_only",
}

# Strict list of required raw files (both periods)
REQUIRED = [
    "PMP1_no_vehicle_housing_unit.csv",
    "PMP2_no_vehicle_housing_unit.csv",
    "PMP1_house_no_internet.csv",
    "PMP2_house_no_internet.csv",
    "PMP1_no_computer_pct_households.csv",
    "PMP2_no_computer_pct_households.csv",
    "PMP1_hs_diploma_no_college.csv",
    "PMP2_hs_diploma_no_college.csv",
]

def _assert_files():
    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(RAW_DIR, f))]
    if missing:
        raise FileNotFoundError(
            "Missing required raw files in 'raw_data/':\n  " + "\n  ".join(missing)
        )
    if not os.path.exists(BASE_FILE):
        raise FileNotFoundError(f"Missing base file: {BASE_FILE}")

def _infer_var_from_fname(fname: str) -> str:
    # find the key in VAR_MAP that appears in the filename
    for key, clean in VAR_MAP.items():
        if key in fname:
            return clean
    raise ValueError(f"Unrecognized filename for variable mapping: {fname}")

def _infer_period_from_fname(fname: str) -> str:
    if "PMP1" in fname:
        return "P1"
    if "PMP2" in fname:
        return "P2"
    raise ValueError(f"Filename must contain PMP1 or PMP2: {fname}")

def _coerce_numeric(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    # hard cap to [0, 100] for % variables; silently clip outliers
    return s.clip(lower=0, upper=100)

def _period_sort_key(value) -> float:
    if isinstance(value, str):
        match = re.search(r"\d{4}", value)
        if match:
            return float(match.group())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")

def _standardize_columns(df: pd.DataFrame, var_name: str) -> pd.DataFrame:
    # Keep only geoid + var; coerce geoid -> str; dedupe
    if "geoid" not in df.columns:
        geoid_candidates = [c for c in df.columns if c.strip().lower() == "geoid"]
        if not geoid_candidates:
            geoid_candidates = [c for c in df.columns if "geoid" in c.strip().lower()]
        if geoid_candidates:
            df = df.rename(columns={geoid_candidates[0]: "geoid"})
        else:
            df = df.rename(columns={df.columns[0]: "geoid"})
    df["geoid"] = df["geoid"].astype(str).str.strip()

    # Find metric column: prefer 'value', else the first non-geoid column
    metric_col = None
    if "value" in df.columns:
        metric_col = "value"
    else:
        excluded = {
            "geoid",
            "geography",
            "geography type description",
            "geography name",
            "name",
            "sits in state",
            "formatted geoid",
            "selected location",
            "data time period",
            "data source",
            "geographic vintage",
            "geographic vintage description",
            "timeframe",
            "geo vintage",
            "source",
            "location",
        }
        numeric_candidates = []
        for col in df.columns:
            if col == "geoid" or col.strip().lower() in excluded:
                continue
            sample = pd.to_numeric(df[col], errors="coerce")
            if sample.notna().any():
                numeric_candidates.append(col)
        if not numeric_candidates:
            raise ValueError("Could not find a metric column.")
        metric_col = numeric_candidates[0]

    df = df[["geoid", metric_col]].rename(columns={metric_col: var_name})
    df[var_name] = _coerce_numeric(df[var_name])

    # Some PolicyMap pulls can include duplicates; collapse them (mean)
    df = (
        df.groupby("geoid", as_index=False, dropna=False)
          .agg({var_name: "mean"})
    )
    return df

def _ensure_period_column(df: pd.DataFrame) -> pd.DataFrame:
    if "period" in df.columns:
        df["period"] = df["period"].astype(str).str.strip()
        return df

    if "time_period" in df.columns:
        unique_periods = sorted(df["time_period"].dropna().unique(), key=_period_sort_key)
        if unique_periods:
            period_map = {period: f"P{idx + 1}" for idx, period in enumerate(unique_periods)}
            df["period"] = df["time_period"].map(period_map)
            if df["period"].isna().any():
                missing_values = df[df["period"].isna()]["time_period"].unique()
                raise KeyError(
                    "Unable to infer 'period' from time_period values: "
                    + ", ".join(map(str, missing_values))
                )
            return df

    if {"period_start", "period_end"}.issubset(df.columns):
        starts = sorted(df["period_start"].dropna().unique())
        if starts:
            period_map = {start: f"P{idx + 1}" for idx, start in enumerate(starts)}
            df["period"] = df["period_start"].map(period_map)
            if df["period"].isna().any():
                missing_values = df[df["period"].isna()]["period_start"].unique()
                raise KeyError(
                    "Unable to infer 'period' from period_start values: "
                    + ", ".join(map(str, missing_values))
                )
            return df

    raise KeyError(
        "Expected 'period' column in PM_merged.csv with values 'P1' or 'P2'. "
        "Please add/propagate period before merging."
    )

def _load_one(path: str) -> pd.DataFrame:
    fname   = os.path.basename(path)
    period  = _infer_period_from_fname(fname)
    varname = _infer_var_from_fname(fname)

    df = pd.read_csv(path, dtype={"geoid": str}, low_memory=False)
    df = _standardize_columns(df, varname)
    df["period"] = period

    # final order
    return df[["geoid", "period", varname]]

def main():
    _assert_files()

    # 1) Load base file
    base = pd.read_csv(BASE_FILE, dtype={"geoid": str}, low_memory=False)
    if "geoid" not in base.columns:
        # tolerate older pipelines: assume first column is geoid
        base = base.rename(columns={base.columns[0]: "geoid"})
    base["geoid"] = base["geoid"].astype(str).str.strip()

    base = _ensure_period_column(base)

    # 2) Load & concatenate all four variables for both periods
    pieces = []
    for req in REQUIRED:
        path = os.path.join(RAW_DIR, req)
        pieces.append(_load_one(path))

    add_df = pieces[0]
    for p in pieces[1:]:
        # outer join on (geoid, period) to accumulate columns
        add_df = add_df.merge(p, on=["geoid", "period"], how="outer")

    # 3) Optional: quick sanity NA cleanup (keep NA if truly missing)
    # nothing aggressive here; just ensure dtypes are numeric
    for c in VAR_MAP.values():
        if c in add_df.columns:
            add_df[c] = _coerce_numeric(add_df[c])

    # 4) Merge with base on (geoid, period)
    merged = base.merge(add_df, on=["geoid", "period"], how="left")

    # 5) Write out
    os.makedirs(CLEAN_DIR, exist_ok=True)
    merged.to_csv(OUT_FILE, index=False)
    print(f"[write] {OUT_FILE}  shape={merged.shape}")
    print("Columns added:", [c for c in VAR_MAP.values() if c in merged.columns])

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
