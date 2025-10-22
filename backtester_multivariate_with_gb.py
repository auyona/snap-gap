#!/usr/bin/env python3
"""
Multivariate backtester with gradient boosting enhancements.

- Trains on P1 (2014–2018), evaluates on P2 (2019–2023)
- Tests every logistic feature combination + tuned random forest
- Adds gradient boosting (hand-tuned grid) for non-linear benchmarking
"""

import argparse
import itertools
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ====== CONFIG ======
RANDOM_STATE = 42
HI_POVERTY_Q = 0.70
LOW_UPTAKE_Q = 0.10
POVERTY_FLOOR = 0.15
BASE_FEATURES = [
    "pct_no_vehicle",
    "pct_no_internet",
    "pct_no_computer",
    "pct_hs_diploma_only",
]
P1_PERIOD = "P1"
P2_PERIOD = "P2"
POV_COL = "pov_fam"
SNAP_COL = "snap_fam"
LOGISTIC_C_VALUES = [0.01, 0.1, 1.0, 10.0]
GB_PARAM_GRID = [
    {"learning_rate": 0.05, "n_estimators": 400, "max_depth": 3, "subsample": 0.9},
    {"learning_rate": 0.05, "n_estimators": 600, "max_depth": 3, "subsample": 0.8},
    {"learning_rate": 0.1, "n_estimators": 400, "max_depth": 2, "subsample": 0.9},
]
PRK_FRACS = [0.01, 0.05]
# =====================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base_dir = Path(__file__).resolve().parent
    default_data = base_dir / "clean_data" / "PM_plus4.csv"
    default_out = base_dir / "analysis" / "visuals"
    parser.add_argument("--data", default=str(default_data), help="Path to PM_plus4.csv")
    parser.add_argument(
        "--outdir",
        default=str(default_out),
        help="Output directory (default: %(default)s)",
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
    candidates = [c for c in df.columns if pattern.match(c) or c.lower() == base.lower()]
    if not candidates:
        return pd.Series([np.nan] * len(df))
    result = pd.to_numeric(df[candidates[0]], errors="coerce")
    for col in candidates[1:]:
        result = result.where(result.notna(), pd.to_numeric(df[col], errors="coerce"))
    return result


def reshape_period_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    pattern = re.compile(r"^(?P<base>.+)_P(?P<period>\d+)$", flags=re.IGNORECASE)
    period_map: Dict[str, Dict[str, str]] = {}
    id_cols: List[str] = []
    for col in df.columns:
        match = pattern.match(col)
        if match:
            base = match.group("base")
            period = f"P{match.group('period').upper()}"
            period_map.setdefault(period, {})[base] = col
        else:
            id_cols.append(col)
    if not period_map:
        return df.copy()
    frames = []
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
    for base in [POV_COL, SNAP_COL] + BASE_FEATURES:
        if base not in long_df.columns:
            long_df[base] = coalesce_columns(long_df, base)
    long_df[POV_COL] = normalize_pct(long_df[POV_COL])
    long_df[SNAP_COL] = normalize_pct(long_df[SNAP_COL])
    for feat in BASE_FEATURES:
        long_df[feat] = normalize_pct(long_df[feat])
    return long_df


def compute_target(
    df: pd.DataFrame,
    low_uptake_q: float = LOW_UPTAKE_Q,
    hi_q: float = HI_POVERTY_Q,
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


def prevalence_threshold(y_true, y_proba):
    prevalence = y_true.mean()
    sorted_proba = np.sort(y_proba)[::-1]
    n_positive = int(len(y_true) * prevalence)
    if n_positive >= len(sorted_proba):
        return 0.0
    return float(sorted_proba[n_positive])


def precision_at_k(y_true, y_proba, frac):
    k = int(len(y_true) * frac)
    if k == 0:
        return np.nan
    top_idx = np.argsort(y_proba)[::-1][:k]
    return float(y_true[top_idx].mean())


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return math.nan
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return math.nan


def safe_ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, scores))
    except ValueError:
        return math.nan


def train_and_evaluate_combo(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_combo: Tuple[str, ...],
    model_type: str,
) -> Dict[str, object]:
    train_mask = y_train.notna()
    test_mask = y_test.notna()
    for feat in feature_combo:
        train_mask &= train_df[feat].notna()
        test_mask &= test_df[feat].notna()

    if train_mask.sum() < 30 or len(np.unique(y_train[train_mask].astype(int))) < 2:
        return {
            "model": model_type,
            "feature_set": "+".join(feature_combo),
            "n_features": len(feature_combo),
            "p1_train_size": int(train_mask.sum()),
            "p2_test_size": int(test_mask.sum()),
            "p1_train_auc": math.nan,
            "p2_auc": math.nan,
        }

    X_train = train_df.loc[train_mask, list(feature_combo)].values
    y_train_arr = y_train.loc[train_mask].astype(int).values
    X_test = test_df.loc[test_mask, list(feature_combo)].values
    y_test_arr = y_test.loc[test_mask].astype(int).values

    test_valid = test_mask.sum() >= 10 and len(np.unique(y_test_arr)) >= 2

    if model_type == "logistic":
        best_c, best_ap = None, -np.inf
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
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
            ap_scores = []
            for tr_idx, val_idx in cv.split(X_train, y_train_arr):
                estimator = pipeline.fit(X_train[tr_idx], y_train_arr[tr_idx])
                val_scores = estimator.predict_proba(X_train[val_idx])[:, 1]
                ap_scores.append(safe_ap(y_train_arr[val_idx], val_scores))
            mean_ap = np.nanmean(ap_scores)
            if mean_ap > best_ap:
                best_ap, best_c = mean_ap, C

        pipeline = Pipeline(
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
        pipeline.fit(X_train, y_train_arr)
        model = CalibratedClassifierCV(pipeline, cv=3, method="isotonic")
        model.fit(X_train, y_train_arr)
        coefs = pipeline.named_steps["clf"].coef_.ravel()
        coef_dict = {f"coef_{feat}": val for feat, val in zip(feature_combo, coefs)}

    elif model_type == "random_forest":
        rf = RandomForestClassifier(
            n_estimators=1000,
            min_samples_leaf=5,
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train_arr)
        model = CalibratedClassifierCV(rf, cv=3, method="isotonic")
        model.fit(X_train, y_train_arr)
        coef_dict = {}

    else:  # gradient boosting
        pos = (y_train_arr == 1).sum()
        neg = len(y_train_arr) - pos
        weight_pos = neg / pos if pos else 1.0
        sample_weight = np.where(y_train_arr == 1, weight_pos, 1.0)
        best_params, best_ap = None, -np.inf
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        for params in GB_PARAM_GRID:
            gb = GradientBoostingClassifier(random_state=RANDOM_STATE, **params)
            ap_scores = []
            for tr_idx, val_idx in cv.split(X_train, y_train_arr):
                gb.fit(X_train[tr_idx], y_train_arr[tr_idx], sample_weight=sample_weight[tr_idx])
                val_scores = gb.predict_proba(X_train[val_idx])[:, 1]
                ap_scores.append(safe_ap(y_train_arr[val_idx], val_scores))
            mean_ap = np.nanmean(ap_scores)
            if mean_ap > best_ap:
                best_ap, best_params = mean_ap, params

        gb = GradientBoostingClassifier(random_state=RANDOM_STATE, **best_params)
        gb.fit(X_train, y_train_arr, sample_weight=sample_weight)
        model = CalibratedClassifierCV(gb, cv=3, method="isotonic")
        model.fit(X_train, y_train_arr)
        coef_dict = {f"gb_param_{k}": v for k, v in best_params.items()}

    train_proba = model.predict_proba(X_train)[:, 1]
    train_auc = safe_auc(y_train_arr, train_proba)
    train_ap = safe_ap(y_train_arr, train_proba)

    threshold = prevalence_threshold(y_train_arr, train_proba)
    train_pred = (train_proba >= threshold).astype(int)
    train_precision = float(precision_score(y_train_arr, train_pred, zero_division=0))
    train_recall = float(recall_score(y_train_arr, train_pred, zero_division=0))
    train_f1 = float(f1_score(y_train_arr, train_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_train_arr, train_pred).ravel()
    train_pr_at_1pct = precision_at_k(y_train_arr, train_proba, 0.01)
    train_pr_at_5pct = precision_at_k(y_train_arr, train_proba, 0.05)

    result = {
        "model": model_type,
        "feature_set": "+".join(feature_combo),
        "n_features": len(feature_combo),
        "p1_train_size": int(train_mask.sum()),
        "p1_train_auc": train_auc,
        "p1_train_ap": train_ap,
        "p1_train_precision": train_precision,
        "p1_train_recall": train_recall,
        "p1_train_f1": train_f1,
        "p1_train_pr_at_1pct": train_pr_at_1pct,
        "p1_train_pr_at_5pct": train_pr_at_5pct,
        "p1_threshold": threshold,
        **coef_dict,
    }

    if not test_valid:
        result.update(
            {
                "p2_test_size": int(test_mask.sum()),
                "p2_auc": math.nan,
                "p2_ap": math.nan,
                "p2_precision": math.nan,
                "p2_recall": math.nan,
                "p2_f1": math.nan,
                "p2_accuracy": math.nan,
                "p2_pr_at_1pct": math.nan,
                "p2_pr_at_5pct": math.nan,
            }
        )
        return result

    test_proba = model.predict_proba(X_test)[:, 1]
    test_auc = safe_auc(y_test_arr, test_proba)
    test_ap = safe_ap(y_test_arr, test_proba)

    test_pred = (test_proba >= threshold).astype(int)
    result.update(
        {
            "p2_test_size": int(test_mask.sum()),
            "p2_auc": test_auc,
            "p2_ap": test_ap,
            "p2_precision": float(precision_score(y_test_arr, test_pred, zero_division=0)),
            "p2_recall": float(recall_score(y_test_arr, test_pred, zero_division=0)),
            "p2_f1": float(f1_score(y_test_arr, test_pred, zero_division=0)),
            "p2_accuracy": float(accuracy_score(y_test_arr, test_pred)),
            "p2_pr_at_1pct": precision_at_k(y_test_arr, test_proba, 0.01),
            "p2_pr_at_5pct": precision_at_k(y_test_arr, test_proba, 0.05),
        }
    )

    if len(feature_combo) == len(BASE_FEATURES):
        result["_plot_data"] = (y_test_arr, test_proba)
    return result


def test_all_combinations(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    results: List[Dict[str, object]] = []

    for r in range(1, len(BASE_FEATURES) + 1):
        for combo in itertools.combinations(BASE_FEATURES, r):
            print(f"  Testing LR: {' + '.join(combo)}")
            results.append(
                train_and_evaluate_combo(train_df, test_df, y_train, y_test, combo, "logistic")
            )

    print("  Testing RF: all features")
    results.append(
        train_and_evaluate_combo(
            train_df, test_df, y_train, y_test, tuple(BASE_FEATURES), "random_forest"
        )
    )

    print("  Testing GB: all features")
    results.append(
        train_and_evaluate_combo(
            train_df, test_df, y_train, y_test, tuple(BASE_FEATURES), "gradient_boosting"
        )
    )

    return pd.DataFrame(results)


def plot_curves(plot_data: Dict[str, Tuple[np.ndarray, np.ndarray]], out_dir: Path) -> None:
    lr_data = plot_data.get("logistic")
    rf_data = plot_data.get("random_forest")
    gb_data = plot_data.get("gradient_boosting")

    if all(d is None for d in (lr_data, rf_data, gb_data)):
        return

    plt.figure(figsize=(8, 6))
    if lr_data is not None and len(np.unique(lr_data[0])) > 1:
        fpr, tpr, _ = roc_curve(lr_data[0], lr_data[1])
        plt.plot(fpr, tpr, label=f"Logistic (AUC={roc_auc_score(lr_data[0], lr_data[1]):.3f})")
    if rf_data is not None and len(np.unique(rf_data[0])) > 1:
        fpr, tpr, _ = roc_curve(rf_data[0], rf_data[1])
        plt.plot(fpr, tpr, label=f"Random Forest (AUC={roc_auc_score(rf_data[0], rf_data[1]):.3f})")
    if gb_data is not None and len(np.unique(gb_data[0])) > 1:
        fpr, tpr, _ = roc_curve(gb_data[0], gb_data[1])
        plt.plot(fpr, tpr, label=f"Gradient Boosting (AUC={roc_auc_score(gb_data[0], gb_data[1]):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("P2 ROC Curves – All Features")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "p2_roc_curves_with_gb.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    if lr_data is not None:
        precision, recall, _ = precision_recall_curve(lr_data[0], lr_data[1])
        plt.plot(recall, precision, label=f"Logistic (AP={average_precision_score(lr_data[0], lr_data[1]):.3f})")
    if rf_data is not None:
        precision, recall, _ = precision_recall_curve(rf_data[0], rf_data[1])
        plt.plot(recall, precision, label=f"Random Forest (AP={average_precision_score(rf_data[0], rf_data[1]):.3f})")
    if gb_data is not None:
        precision, recall, _ = precision_recall_curve(gb_data[0], gb_data[1])
        plt.plot(recall, precision, label=f"Gradient Boosting (AP={average_precision_score(gb_data[0], gb_data[1]):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("P2 Precision-Recall Curves – All Features")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "p2_pr_curves_with_gb.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).expanduser()
    out_dir = Path(args.outdir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(data_path, low_memory=False)
    df.replace({"": np.nan, " ": np.nan}, inplace=True)
    df_long = ensure_long_format(df)

    missing_rates = {feat: float(df_long[feat].isna().mean()) for feat in BASE_FEATURES}
    print("Missing data rates per feature:", missing_rates)

    p1_df = df_long[df_long["period"] == P1_PERIOD].reset_index(drop=True)
    p2_df = df_long[df_long["period"] == P2_PERIOD].reset_index(drop=True)
    if p1_df.empty or p2_df.empty:
        raise ValueError("Could not locate both P1 and P2 rows.")

    mask_p1 = p1_df[BASE_FEATURES].notna().all(axis=1)
    mask_p2 = p2_df[BASE_FEATURES].notna().all(axis=1)
    print(f"P1 rows before mask: {len(p1_df)}  after mask: {mask_p1.sum()}")
    print(f"P2 rows before mask: {len(p2_df)}  after mask: {mask_p2.sum()}")
    p1_df = p1_df.loc[mask_p1].reset_index(drop=True)
    p2_df = p2_df.loc[mask_p2].reset_index(drop=True)

    print("Computing targets...")
    y_p1, meta_p1 = compute_target(p1_df)
    y_p2, meta_p2 = compute_target(p2_df)
    print(f"P1 valid={meta_p1['n_valid']} pos_rate={meta_p1['pos_rate']:.3%}")
    print(f"P2 valid={meta_p2['n_valid']} pos_rate={meta_p2['pos_rate']:.3%}")

    print("\nTesting all feature combinations...")
    results_df = test_all_combinations(p1_df, p2_df, y_p1, y_p2)

    results_df["hi_poverty_quantile"] = HI_POVERTY_Q
    results_df["low_uptake_quantile"] = LOW_UPTAKE_Q
    results_df["poverty_floor"] = POVERTY_FLOOR
    results_df["p1_pos_rate"] = meta_p1["pos_rate"]
    results_df["p2_pos_rate"] = meta_p2["pos_rate"]
    results_df.sort_values(by="p2_auc", ascending=False, inplace=True)

    plot_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if "_plot_data" in results_df.columns:
        for _, row in results_df.dropna(subset=["_plot_data"]).iterrows():
            plot_data[row["model"]] = row["_plot_data"]
        results_df = results_df.drop(columns=["_plot_data"])

    auc_by_features = (
        results_df.dropna(subset=["p2_auc"])
        .groupby("n_features")["p2_auc"]
        .max()
        .reset_index()
        .sort_values("n_features")
    )
    if not auc_by_features.empty:
        plt.figure(figsize=(6, 4))
        plt.bar(auc_by_features["n_features"], auc_by_features["p2_auc"])
        plt.xlabel("Number of features")
        plt.ylabel("Best P2 AUC")
        plt.title("P2 AUC by Feature Count")
        plt.tight_layout()
        plt.savefig(out_dir / "combo_auc_by_feature_count_with_gb.png", dpi=200)
        plt.close()

    best_logistic = (
        results_df[results_df["model"] == "logistic"]
        .dropna(subset=["p2_auc"])
        .sort_values("p2_auc", ascending=False)
        .head(1)
    )
    if not best_logistic.empty:
        row = best_logistic.iloc[0]
        coef_cols = [c for c in results_df.columns if c.startswith("coef_")]
        coefs = row[coef_cols].dropna()
        if not coefs.empty:
            plt.figure(figsize=(6, 4))
            plt.barh([c.replace("coef_", "") for c in coefs.index], coefs.values.astype(float))
            plt.xlabel("Coefficient")
            plt.title(f"Logistic coefficients – best combo ({row['feature_set']})")
            plt.tight_layout()
            plt.savefig(out_dir / "combo_best_logistic_coefficients_with_gb.png", dpi=200)
            plt.close()

    summary_path = out_dir / "combo_backtest_summary_with_gb.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"[write] {summary_path}")

    rankings = []
    for scope_name in ["Top_by_AUC", "Top_by_AP", "Top_by_F1"]:
        if scope_name == "Top_by_AUC":
            sorted_df = results_df.sort_values(by="p2_auc", ascending=False)
        elif scope_name == "Top_by_AP":
            sorted_df = results_df.sort_values(by="p2_ap", ascending=False)
        else:
            sorted_df = results_df.sort_values(by="p2_f1", ascending=False)
        for rank, (_, row) in enumerate(sorted_df.head(10).iterrows(), 1):
            record = row.to_dict()
            record["ranking_scope"] = scope_name
            record["rank"] = rank
            rankings.append(record)
    rankings_path = out_dir / "combo_backtest_rankings_with_gb.csv"
    pd.DataFrame(rankings).to_csv(rankings_path, index=False)
    print(f"[write] {rankings_path}")

    plot_curves(plot_data, out_dir)

    print("\n=== SUMMARY ===\n")
    print("Top 5 by P2 AUC:")
    cols = ["model", "feature_set", "n_features", "p2_auc", "p2_ap", "p2_precision", "p2_recall", "p2_f1"]
    print(results_df[cols].head(5).to_string(index=False))

    print("\nMultivariate + GB backtesting complete.")


if __name__ == "__main__":
    main()
