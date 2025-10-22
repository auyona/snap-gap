#!/usr/bin/env python3
"""
Area-wise backtester: runs the legacy univariate + multivariate routines
separately for Urban, Rural, and All ZIPs (based on 2019 designations).

Outputs:
  analysis/backtester_area_<area>_<tag>.csv/.xlsx
  analysis/backtester_area_rankings.csv  (P2 rankings per area)
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# ========= CONFIG =========
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "clean_data" / "PM_plus4.csv"
AREA_PATH = BASE_DIR / "clean_data" / "PM_zip_area_designation.csv"
OUT_DIR = BASE_DIR / "analysis"

POV_COL = "pov_fam"
SNAP_COL = "snap_fam"
HI_Q = 0.70
LO_Q = 0.10
POVERTY_FLOOR = 0.15
CANDIDATES = [
    "pct_no_vehicle",
    "pct_no_internet",
    "pct_no_computer",
    "pct_hs_diploma_only",
]
RANDOM_STATE = 42
PERIOD_SUFFIX_RE = re.compile(r"^(?P<base>.+)_P(?P<period>\d+)$", re.IGNORECASE)
AREAS = ["All", "Urban", "Rural", "Mixed"]
RANK_ROWS: List[Dict[str, object]] = []
TREND_DATA: Dict[Tuple[str, str], pd.DataFrame] = {}
# ==========================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reshape_period_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    id_cols: List[str] = []
    period_map: Dict[str, Dict[str, str]] = {}
    for col in df.columns:
        match = PERIOD_SUFFIX_RE.match(col)
        if match:
            base = match.group("base")
            period = match.group("period").upper()
            key = f"P{period}"
            period_map.setdefault(key, {})[base] = col
        else:
            id_cols.append(col)
    if not period_map:
        return df
    frames = []
    for period, base_map in sorted(period_map.items()):
        frame = df[id_cols].copy()
        for base, col in base_map.items():
            frame[base] = df[col]
        frame["period"] = period
        frames.append(frame)
    long_df = pd.concat(frames, ignore_index=True)
    long_df.replace({"": np.nan, " ": np.nan}, inplace=True)
    measures = [POV_COL, SNAP_COL] + [c for c in CANDIDATES if c in long_df.columns]
    existing = [c for c in measures if c in long_df.columns]
    if existing:
        mask = long_df[existing].notna().any(axis=1)
        long_df = long_df.loc[mask].reset_index(drop=True)
    return long_df


def coalesce(df: pd.DataFrame, base: str) -> pd.Series:
    a = df.get(base + "_x")
    b = df.get(base + "_y")
    if a is None and b is None:
        return df[base] if base in df.columns else pd.Series([np.nan] * len(df))
    if a is None:
        return b
    if b is None:
        return a
    return a.where(~a.isna(), b)


def normalize_pct(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if (s > 1).mean() > 0.1:
        s = s / 100.0
    return s.clip(lower=0, upper=1)


def build_target(df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, float]]:
    pov = normalize_pct(df[POV_COL])
    snap = normalize_pct(df[SNAP_COL])
    uptake = (snap / pov.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    eligible = (pov >= POVERTY_FLOOR) & uptake.notna() & (uptake > 0)

    hi_thr = np.nanquantile(pov[eligible], HI_Q) if eligible.any() else np.nan
    lo_thr = np.nanquantile(uptake[eligible], LO_Q) if eligible.any() else np.nan

    y = (eligible & (pov >= hi_thr) & (uptake <= lo_thr)).astype(float)
    y = y.where(~(pov.isna() | uptake.isna()), np.nan)

    meta = {
        "hi_poverty_quantile": HI_Q,
        "hi_poverty_threshold": float(hi_thr) if np.isfinite(hi_thr) else np.nan,
        "low_uptake_quantile": LO_Q,
        "low_uptake_threshold": float(lo_thr) if np.isfinite(lo_thr) else np.nan,
        "poverty_floor": POVERTY_FLOOR,
        "n_total": int(len(df)),
        "n_eligible": int(eligible.sum()),
        "n_valid": int(y.notna().sum()),
        "pos_rate": float(np.nanmean(y)),
    }
    return y, meta


def youden_threshold(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    idx = np.argmax(j)
    return thr[idx] if idx is not None else 0.5


def evaluate_univariate(X: pd.Series, y: pd.Series):
    mask = ~(X.isna() | y.isna())
    if mask.sum() < 30:
        return {"n": int(mask.sum()), "auc": np.nan, "ap": np.nan, "f1": np.nan, "acc": np.nan, "thr": np.nan}

    x = X[mask].values.reshape(-1, 1)
    yy = y[mask].values.astype(int)

    if len(np.unique(x)) < 2 or len(np.unique(yy)) < 2:
        return {"n": int(mask.sum()), "auc": np.nan, "ap": np.nan, "f1": np.nan, "acc": np.nan, "thr": np.nan}

    Xtr, Xte, ytr, yte = train_test_split(x, yy, test_size=0.3, random_state=RANDOM_STATE, stratify=yy)

    auc = roc_auc_score(yte, Xte.ravel())
    ap = average_precision_score(yte, Xte.ravel())
    thr = youden_threshold(ytr, Xtr.ravel())
    pred = (Xte.ravel() >= thr).astype(int)
    f1 = f1_score(yte, pred)
    acc = accuracy_score(yte, pred)

    return {"n": int(mask.sum()), "auc": float(auc), "ap": float(ap), "f1": float(f1), "acc": float(acc), "thr": float(thr)}


def evaluate_multivariate(X: pd.DataFrame, y: pd.Series):
    mask = ~(X.isna().any(axis=1) | y.isna())
    X_ = X.loc[mask].copy()
    y_ = y.loc[mask].astype(int).copy()

    if len(np.unique(y_)) < 2 or X_.shape[0] < 50:
        return {"logit_auc_cv": np.nan, "rf_auc_cv": np.nan, "n": int(X_.shape[0]),
                "logit_coefs": None, "perm_importance": None}

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    logit = LogisticRegression(max_iter=300, random_state=RANDOM_STATE)
    aucs = []
    for tr, te in skf.split(Xs, y_):
        logit.fit(Xs[tr], y_.iloc[tr])
        prob = logit.predict_proba(Xs[te])[:, 1]
        aucs.append(roc_auc_score(y_.iloc[te], prob))
    logit_auc_cv = float(np.mean(aucs))

    rf = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=1
    )
    aucs_rf = []
    for tr, te in skf.split(X_, y_):
        rf.fit(X_.iloc[tr], y_.iloc[tr])
        prob = rf.predict_proba(X_.iloc[te])[:, 1]
        aucs_rf.append(roc_auc_score(y_.iloc[te], prob))
    rf_auc_cv = float(np.mean(aucs_rf))

    logit.fit(Xs, y_)
    coefs = dict(zip(X_.columns, logit.coef_[0].tolist()))

    rf.fit(X_, y_)
    perm = permutation_importance(rf, X_, y_, n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
    perm_imp = {feat: float(imp) for feat, imp in zip(X_.columns, perm.importances_mean)}

    return {
        "logit_auc_cv": logit_auc_cv,
        "rf_auc_cv": rf_auc_cv,
        "n": int(X_.shape[0]),
        "logit_coefs": coefs,
        "perm_importance": perm_imp
    }


def run_suite(df: pd.DataFrame, tag: str, area: str):
    area_suffix = area.lower()
    for c in CANDIDATES:
        df[c] = coalesce(df, c)
    X = pd.DataFrame({c: normalize_pct(df[c]) for c in CANDIDATES if c in df.columns})
    y, meta = build_target(df)

    uni_rows = []
    for feat in X.columns:
        res = evaluate_univariate(X[feat], y)
        row = {"feature": feat}
        row.update(res)
        uni_rows.append(row)
    uni = pd.DataFrame(uni_rows).sort_values(by="auc", ascending=False)

    for _, row in uni.iterrows():
        RANK_ROWS.append({
            "area": area,
            "tag": tag,
            "feature": row["feature"],
            "auc": row["auc"],
            "ap": row["ap"],
        })

    if tag in {"P1", "P2"}:
        TREND_DATA[(area, tag)] = uni[["feature", "auc", "ap"]].copy()

    multi = evaluate_multivariate(X, y)

    base_name = f"backtester_area_{area_suffix}_{tag}"
    out_csv = OUT_DIR / f"{base_name}.csv"
    out_xlsx = OUT_DIR / f"{base_name}.xlsx"

    uni.to_csv(out_csv, index=False)
    with pd.ExcelWriter(out_xlsx) as w:
        uni.to_excel(w, sheet_name="univariate", index=False)
        pd.DataFrame([meta]).to_excel(w, sheet_name="meta", index=False)
        if multi.get("logit_coefs"):
            (
                pd.DataFrame([multi["logit_coefs"]])
                .T.reset_index()
                .rename(columns={"index": "feature", 0: "std_logit_coef"})
            ).to_excel(w, sheet_name="logit_coefs", index=False)
        if multi.get("perm_importance"):
            (
                pd.DataFrame([multi["perm_importance"]])
                .T.reset_index()
                .rename(columns={"index": "feature", 0: "rf_perm_importance"})
            ).to_excel(w, sheet_name="rf_perm_importance", index=False)
        pd.DataFrame(
            [
                {
                    "logit_auc_cv": multi.get("logit_auc_cv"),
                    "rf_auc_cv": multi.get("rf_auc_cv"),
                    "n": multi.get("n"),
                }
            ]
        ).to_excel(w, sheet_name="cv_summary", index=False)

    print(f"\n=== {area} – {tag} ===")
    print("Target meta:", meta)
    print("Top univariate by AUC:")
    print(uni.head(4).to_string(index=False))
    print(f"[write] {out_csv}")
    print(f"[write] {out_xlsx}")


def main() -> None:
    ensure_dir(OUT_DIR)
    df = pd.read_csv(CSV_PATH, dtype={"geoid": str}, low_memory=False)
    df.replace({"": np.nan, " ": np.nan}, inplace=True)
    if "period" not in df.columns:
        df = reshape_period_wide_to_long(df)

    area_map = pd.read_csv(AREA_PATH, dtype={"zip": str})
    area_map["zip"] = area_map["zip"].str.zfill(5)
    area_map = area_map.rename(
        columns={
            "area_designation_2019": "area_designation",
            "mix_category_2019": "area_mix",
        }
    )
    df["geoid"] = df["geoid"].astype(str).str.zfill(5)
    df = df.merge(area_map[["zip", "area_designation", "area_mix"]], left_on="geoid", right_on="zip", how="left")
    df["area_designation"] = df["area_designation"].fillna("Unknown")
    df["area_mix"] = df["area_mix"].fillna("Unknown")

    for area in AREAS:
        if area == "All":
            df_area = df[df["area_mix"].isin(["Urban", "Rural", "Mixed"])]
        else:
            df_area = df[df["area_mix"].str.upper() == area.upper()]
        if df_area.empty:
            print(f"\n[warn] No rows for area={area}")
            continue
        run_suite(df_area.copy(), "overall", area)
        if "period" in df_area.columns:
            p1 = df_area[df_area["period"].astype(str).str.upper().eq("P1")].copy()
            if len(p1) > 0:
                run_suite(p1, "P1", area)
            else:
                print(f"[warn] No P1 rows for area={area}")

            p2 = df_area[df_area["period"].astype(str).str.upper().eq("P2")].copy()
            if len(p2) > 0:
                run_suite(p2, "P2", area)
            else:
                print(f"[warn] No P2 rows for area={area}")

    if RANK_ROWS:
        rank_df = pd.DataFrame(RANK_ROWS)
        mask = rank_df["tag"] == "P2"
        rank_df = rank_df[mask].sort_values(["area", "auc"], ascending=[True, False])
        rank_df.to_csv(OUT_DIR / "backtester_area_rankings.csv", index=False)
        print(f"[write] {OUT_DIR / 'backtester_area_rankings.csv'}")
        print("\nTop indicators by area (P2 AUC):")
        for area in AREAS:
            sub = rank_df[rank_df["area"] == area]
            if sub.empty:
                continue
            print(f"  {area}:")
            print(
                sub[["feature", "auc", "ap"]]
                .head(5)
                .to_string(index=False)
            )

    trend_rows: List[pd.DataFrame] = []
    for area in AREAS:
        key_p1 = (area, "P1")
        key_p2 = (area, "P2")
        if key_p1 not in TREND_DATA or key_p2 not in TREND_DATA:
            continue
        p1_df = TREND_DATA[key_p1].rename(columns={"auc": "auc_P1", "ap": "ap_P1"})
        p2_df = TREND_DATA[key_p2].rename(columns={"auc": "auc_P2", "ap": "ap_P2"})
        merged = p1_df.merge(p2_df, on="feature", how="outer")
        merged["area"] = area
        merged["auc_delta"] = merged["auc_P2"] - merged["auc_P1"]
        merged["ap_delta"] = merged["ap_P2"] - merged["ap_P1"]
        trend_rows.append(merged)

    if trend_rows:
        trend_df = pd.concat(trend_rows, ignore_index=True)
        trend_path = OUT_DIR / "backtester_area_trend_summary.csv"
        trend_df.to_csv(trend_path, index=False)
        print(f"[write] {trend_path}")

    print("\nArea-wise backtesting complete.")


if __name__ == "__main__":
    main()
