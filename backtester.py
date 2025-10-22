#!/usr/bin/env python3
"""
Univariate backtester aligned with multivariate protocol.

Train on P1 (2014–2018), evaluate on P2 (2019–2023) with
1-D logistic models (class_weight balanced, C tuned) plus
isotonic calibration and prevalence-aligned thresholding.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ====== CONFIG ======
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "clean_data" / "PM_plus4.csv"
OUT_DIR = BASE_DIR / "analysis"

POV_COL = "pov_fam"
SNAP_COL = "snap_fam"
CANDIDATES = [
    "pct_no_vehicle",
    "pct_no_internet",
    "pct_no_computer",
    "pct_hs_diploma_only",
]

HI_Q = 0.70
LO_Q = 0.10
POVERTY_FLOOR = 0.15
LOGISTIC_C_VALUES = [0.25, 0.5, 1.0, 2.0]
PRK_FRACS = [0.01, 0.05]
RANDOM_STATE = 42
PERIOD_SUFFIX_RE = re.compile(r"^(?P<base>.+)_P(?P<period>\d+)$", re.IGNORECASE)
# =====================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P1→P2 univariate backtester.")
    parser.add_argument("--data", default=str(CSV_PATH), help="Input CSV (default: %(default)s).")
    parser.add_argument(
        "--outdir",
        default=str(OUT_DIR),
        help="Output directory (default: %(default)s).",
    )
    return parser.parse_args()


def normalize_pct(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return s
    if (s > 1).mean(skipna=True) > 0.1:
        s = s / 100.0
    return s.clip(lower=0, upper=1)


def coalesce_columns(df: pd.DataFrame, base: str) -> pd.Series:
    pattern = re.compile(rf"^{re.escape(base)}(?:[\W_]*(?:x|y))?$", flags=re.IGNORECASE)
    matches = [c for c in df.columns if pattern.match(c) or c.lower() == base.lower()]
    if not matches:
        return pd.Series([np.nan] * len(df))
    result = pd.to_numeric(df[matches[0]], errors="coerce")
    for col in matches[1:]:
        result = result.where(result.notna(), pd.to_numeric(df[col], errors="coerce"))
    return result


def _split_period_columns(df: pd.DataFrame) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    id_cols: List[str] = []
    period_map: Dict[str, Dict[str, str]] = {}
    for col in df.columns:
        match = PERIOD_SUFFIX_RE.match(col)
        if match:
            base = match.group("base")
            period = f"P{match.group('period').upper()}"
            period_map.setdefault(period, {})[base] = col
        else:
            id_cols.append(col)
    return list(dict.fromkeys(id_cols)), period_map


def reshape_period_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    id_cols, period_map = _split_period_columns(df)
    if not period_map:
        return df.copy()
    frames: List[pd.DataFrame] = []
    for period, mapping in sorted(period_map.items()):
        frame = df[id_cols].copy()
        for base, col in mapping.items():
            frame[base] = pd.to_numeric(df[col], errors="coerce")
        frame["period"] = period
        frames.append(frame)
    long_df = pd.concat(frames, ignore_index=True)
    long_df.replace({"": np.nan, " ": np.nan}, inplace=True)
    return long_df


def ensure_long_format(df: pd.DataFrame) -> pd.DataFrame:
    if "period" in df.columns:
        long_df = df.copy()
    else:
        long_df = reshape_period_wide_to_long(df)
    long_df["period"] = long_df["period"].astype(str).str.upper()
    for base in [POV_COL, SNAP_COL] + CANDIDATES:
        if base not in long_df.columns:
            long_df[base] = coalesce_columns(long_df, base)
    long_df[POV_COL] = normalize_pct(long_df[POV_COL])
    long_df[SNAP_COL] = normalize_pct(long_df[SNAP_COL])
    for feat in CANDIDATES:
        long_df[feat] = normalize_pct(long_df[feat])
    return long_df


def compute_target(
    df: pd.DataFrame,
    low_uptake_q: float = LO_Q,
    hi_q: float = HI_Q,
    poverty_floor: float = POVERTY_FLOOR,
) -> Tuple[pd.Series, Dict[str, float]]:
    pov = df[POV_COL]
    snap = df[SNAP_COL]
    uptake = snap / pov.replace(0, np.nan)
    eligible = pov.notna() & snap.notna() & (pov >= poverty_floor) & uptake.notna() & (uptake > 0)
    if eligible.any():
        hi_thr = float(np.nanquantile(pov[eligible], hi_q))
        lo_thr = float(np.nanquantile(uptake[eligible], low_uptake_q))
    else:
        hi_thr = math.nan
        lo_thr = math.nan
    y = pd.Series(np.nan, index=df.index)
    y.loc[eligible] = 0.0
    positives = eligible & (pov >= hi_thr) & (uptake <= lo_thr)
    y.loc[positives] = 1.0
    meta = {
        "n_total": int(len(df)),
        "n_eligible": int(eligible.sum()),
        "n_valid": int(y.notna().sum()),
        "pos_rate": float(y.dropna().mean()) if y.notna().any() else math.nan,
        "hi_poverty_threshold": hi_thr,
        "low_uptake_threshold": lo_thr,
    }
    return y, meta


def prevalence_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    prevalence = y_true.mean()
    if prevalence <= 0:
        return 1.0
    sorted_scores = np.sort(y_proba)[::-1]
    k = int(len(sorted_scores) * prevalence)
    k = min(max(k, 1), len(sorted_scores)) - 1
    return float(sorted_scores[k])


def precision_at_k(y_true: np.ndarray, y_proba: np.ndarray, frac: float) -> float:
    k = max(int(len(y_true) * frac), 1)
    order = np.argsort(y_proba)[::-1][:k]
    return float(y_true[order].mean())


def safe_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return math.nan
        return float(roc_auc_score(y_true, y_proba))
    except ValueError:
        return math.nan


def safe_ap(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_proba))
    except ValueError:
        return math.nan


def youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    idx = np.argmax(j)
    return float(thr[idx]) if idx is not None else 0.5


def evaluate_univariate(X: pd.Series, y: pd.Series) -> Dict[str, float]:
    mask = ~(X.isna() | y.isna())
    if mask.sum() < 30:
        return {"n": int(mask.sum()), "auc": np.nan, "ap": np.nan, "f1": np.nan, "acc": np.nan, "thr": np.nan}

    x = X[mask].values.reshape(-1, 1)
    yy = y[mask].astype(int).values

    if len(np.unique(x)) < 2 or len(np.unique(yy)) < 2:
        return {"n": int(mask.sum()), "auc": np.nan, "ap": np.nan, "f1": np.nan, "acc": np.nan, "thr": np.nan}

    Xtr, Xte, ytr, yte = train_test_split(
        x, yy, test_size=0.3, random_state=RANDOM_STATE, stratify=yy
    )

    auc = roc_auc_score(yte, Xte.ravel())
    ap = average_precision_score(yte, Xte.ravel())
    thr = youden_threshold(ytr, Xtr.ravel())
    pred = (Xte.ravel() >= thr).astype(int)
    f1 = f1_score(yte, pred)
    acc = accuracy_score(yte, pred)

    return {
        "n": int(mask.sum()),
        "auc": float(auc),
        "ap": float(ap),
        "f1": float(f1),
        "acc": float(acc),
        "thr": float(thr),
    }


def base_mask(df: pd.DataFrame, y: pd.Series) -> pd.Series:
    mask = y.notna()
    for feat in CANDIDATES:
        mask &= df[feat].notna()
    return mask


def train_feature_model(
    feature: str,
    p1_df: pd.DataFrame,
    p2_df: pd.DataFrame,
    y_p1: pd.Series,
    y_p2: pd.Series,
) -> Tuple[Dict[str, object], Optional[Dict[str, object]]]:
    mask_p1 = base_mask(p1_df, y_p1)
    mask_p2 = base_mask(p2_df, y_p2)

    X_train = p1_df.loc[mask_p1, feature].to_numpy().reshape(-1, 1)
    y_train = y_p1.loc[mask_p1].astype(int).to_numpy()
    X_test = p2_df.loc[mask_p2, feature].to_numpy().reshape(-1, 1)
    y_test = y_p2.loc[mask_p2].astype(int).to_numpy()

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {
            "feature": feature,
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "p2_auc": math.nan,
            "p2_ap": math.nan,
        }, None

    best_c = None
    best_ap = -math.inf
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for C in LOGISTIC_C_VALUES:
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        ap_scores: List[float] = []
        for tr_idx, val_idx in cv.split(X_train, y_train):
            estimator = pipeline.fit(X_train[tr_idx], y_train[tr_idx])
            val_proba = estimator.predict_proba(X_train[val_idx])[:, 1]
            ap_scores.append(safe_ap(y_train[val_idx], val_proba))
        mean_ap = float(np.nanmean(ap_scores))
        if mean_ap > best_ap:
            best_ap = mean_ap
            best_c = C

    final_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=best_c,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    final_pipeline.fit(X_train, y_train)
    coefs = final_pipeline.named_steps["clf"].coef_.ravel()
    intercept = float(final_pipeline.named_steps["clf"].intercept_[0])

    calibrator = CalibratedClassifierCV(
        final_pipeline,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        method="isotonic",
    )
    calibrator.fit(X_train, y_train)

    train_proba = calibrator.predict_proba(X_train)[:, 1]
    test_proba = calibrator.predict_proba(X_test)[:, 1]

    threshold = prevalence_threshold(y_train, train_proba)
    train_pred = (train_proba >= threshold).astype(int)
    test_pred = (test_proba >= threshold).astype(int)

    train_precision = float(precision_score(y_train, train_pred, zero_division=0))
    train_recall = float(recall_score(y_train, train_pred, zero_division=0))
    train_f1 = float(f1_score(y_train, train_pred, zero_division=0))

    p2_auc = safe_auc(y_test, test_proba)
    p2_ap = safe_ap(y_test, test_proba)
    p2_precision = float(precision_score(y_test, test_pred, zero_division=0))
    p2_recall = float(recall_score(y_test, test_pred, zero_division=0))
    p2_f1 = float(f1_score(y_test, test_pred, zero_division=0))
    p2_accuracy = float(accuracy_score(y_test, test_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()

    pr_at_k = {
        f"p2_precision_at_{int(frac*100)}pct": precision_at_k(y_test, test_proba, frac)
        for frac in PRK_FRACS
    }

    result: Dict[str, object] = {
        "feature": feature,
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "best_C": best_c,
        "threshold_prevalence": threshold,
        "train_precision": train_precision,
        "train_recall": train_recall,
        "train_f1": train_f1,
        "p2_auc": p2_auc,
        "p2_ap": p2_ap,
        "p2_precision": p2_precision,
        "p2_recall": p2_recall,
        "p2_f1": p2_f1,
        "p2_accuracy": p2_accuracy,
        "p2_tn": int(tn),
        "p2_fp": int(fp),
        "p2_fn": int(fn),
        "p2_tp": int(tp),
    }
    result.update(pr_at_k)

    coef_record = {
        "feature": feature,
        "best_C": best_c,
        "intercept": intercept,
        "coef": float(coefs[0]) if coefs.size else math.nan,
    }
    return result, coef_record


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).expanduser()
    out_dir = Path(args.outdir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, low_memory=False)
    df.replace({"": np.nan, " ": np.nan}, inplace=True)
    df_long = ensure_long_format(df)

    p1_df = df_long[df_long["period"] == "P1"].reset_index(drop=True)
    p2_df = df_long[df_long["period"] == "P2"].reset_index(drop=True)
    if p1_df.empty or p2_df.empty:
        raise ValueError("Could not locate both P1 and P2 rows in the dataset.")

    y_p1, meta_p1 = compute_target(p1_df)
    y_p2, meta_p2 = compute_target(p2_df)

    print(f"P1 rows: {meta_p1['n_valid']}  pos_rate={meta_p1['pos_rate']:.4f}")
    print(f"P2 rows: {meta_p2['n_valid']}  pos_rate={meta_p2['pos_rate']:.4f}")

    results: List[Dict[str, object]] = []
    coef_rows: List[Dict[str, object]] = []

    for feature in CANDIDATES:
        print(f"Training univariate logistic for {feature} ...")
        res, coef = train_feature_model(feature, p1_df, p2_df, y_p1, y_p2)
        results.append(res)
        if coef is not None:
            coef_rows.append(coef)

    summary_df = pd.DataFrame(results)
    summary_df["hi_poverty_quantile"] = HI_Q
    summary_df["low_uptake_quantile"] = LO_Q
    summary_df["poverty_floor"] = POVERTY_FLOOR
    summary_df["p1_pos_rate"] = meta_p1["pos_rate"]
    summary_df["p2_pos_rate"] = meta_p2["pos_rate"]
    summary_df["calibrated"] = True
    summary_df["threshold_policy"] = "prevalence"
    summary_df.sort_values(by="p2_auc", ascending=False, inplace=True)

    summary_path = out_dir / "backtester_univariate_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[write] {summary_path}")

    if coef_rows:
        coef_df = pd.DataFrame(coef_rows)
        coef_path = out_dir / "backtester_univariate_lr_coefficients.csv"
        coef_df.to_csv(coef_path, index=False)
        print(f"[write] {coef_path}")

    print("\nTop features by P2 AUC:")
    display_cols = [
        "feature",
        "p2_auc",
        "p2_ap",
        "p2_precision",
        "p2_recall",
        "p2_f1",
    ]
    print(summary_df[display_cols].head(4).to_string(index=False))

    metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "random_state": RANDOM_STATE,
        "data_path": str(data_path),
        "hi_q": HI_Q,
        "lo_q": LO_Q,
        "poverty_floor": POVERTY_FLOOR,
        "features": CANDIDATES,
    }
    meta_path = out_dir / "backtester_univariate_run_metadata.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"[write] {meta_path}")

    print("\nUnivariate P1→P2 backtesting complete.")


if __name__ == "__main__":
    main()
