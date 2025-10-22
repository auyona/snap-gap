#!/usr/bin/env python3
"""
PM_merger.py

Merges the three main cleaned PolicyMap datasets:
- PM_pov_fam_full_clean.csv      (family poverty)
- PM_pov_total_full_clean.csv (total poverty)
- PM_snap_fam_full_clean.csv     (family SNAP)

Output:
- clean_data/PM_merged.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent
CLEAN_DIR = BASE_DIR / "clean_data"

def load_csv(name):
    path = CLEAN_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, dtype={"geoid":str})

def main():
    print("\n=== PolicyMap Merger ===")

    # Load with your actual filenames
    pov_fam   = load_csv("PM_pov_fam_full_clean.csv")
    pov_total = load_csv("PM_pov_total_full_clean.csv")
    snap_fam  = load_csv("PM_snap_fam_full_clean.csv")

    key = ["geoid","period_start","period_end"]
    extras = ["geography_name","state","geoid_formatted","time_period"]

    # Ensure extras exist
    for df in [pov_fam, pov_total, snap_fam]:
        for c in extras:
            if c not in df.columns:
                df[c] = np.nan

    # Merge step 1: total poverty with family poverty
    merged = pd.merge(
        pov_total[key + extras + ["pov_total"]],
        pov_fam[key + ["pov_fam"]],
        on=key, how="outer"
    )

    # Merge step 2: add family SNAP
    merged = pd.merge(
        merged,
        snap_fam[key + ["snap_fam"]],
        on=key, how="outer"
    )

    # Compute gaps
    merged["gap_total_minus_snapfam"] = merged["pov_total"] - merged["snap_fam"]
    merged["gap_fam_minus_snapfam"]   = merged["pov_fam"]   - merged["snap_fam"]

    # Reorder
    cols = ["geoid","geography_name","state","geoid_formatted","time_period",
            "period_start","period_end",
            "pov_total","pov_fam","snap_fam",
            "gap_total_minus_snapfam","gap_fam_minus_snapfam"]
    merged = merged[cols].sort_values(["geoid","period_start","period_end"])

    out = CLEAN_DIR / "PM_merged.csv"
    merged.to_csv(out, index=False)
    print(f"[write] {out}")
    print("\nDone.")

if __name__ == "__main__":
    main()
