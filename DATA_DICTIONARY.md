# SNAP Gap Project – Data Dictionary

This document describes the purpose and schema of the key datasets produced or consumed
by the `snap_gap` analysis codebase. File paths are relative to the project root.

---

## 1. Raw Inputs

### `raw_data/PMP1_tract_types_2015_raw.csv`
| Column | Description |
| --- | --- |
| `GeoID` / `GeoID_Formatted` | Census tract identifier (string) |
| `urban` | USDA/PolicyMap urban‐rural classification for 2015 (`Urban`, `Rural`, etc.) |
| Additional columns | Metadata such as tract name, state, source, etc. |

### `raw_data/PMP2_tract_types_2019_raw.csv`
*Same schema as the 2015 file; reflects the 2019 tract classifications.*

### `raw_data/ZIP_TRACT_062025.xlsx`
| Column | Description |
| --- | --- |
| `ZIP` | 5‑digit ZIP Code Tabulation Area |
| `TRACT` | 11‑digit Census tract ID |
| `RES_RATIO` | Residential ratio weight of tract within ZIP |
| `BUS_RATIO`, `OTH_RATIO`, `TOT_RATIO` | Additional weighting ratios (business, other, total) |
| `USPS_ZIP_PREF_CITY`, `USPS_ZIP_PREF_STATE` | Postal reference fields |

---

## 2. Cleaned Datasets

### `clean_data/PM_plus4.csv`
Merged PolicyMap ZIP‑level dataset used for modelling.
| Column | Description |
| --- | --- |
| `geoid` | ZIP code identifier (string) |
| `geography_name`, `state` | Postal reference information |
| `time_period`, `period_start`, `period_end` | Source time span (e.g., `2014-2018`) |
| `pov_total`, `pov_fam` | Total/Family poverty percentages |
| `snap_fam` | SNAP participation share (family) |
| `gap_total_minus_snapfam`, `gap_fam_minus_snapfam` | Difference between poverty and SNAP share |
| `period` | `P1` (2014–2018) or `P2` (2019–2023) |
| `%` indicators (`pct_no_vehicle_x`, `pct_no_vehicle_y`, etc.) | PolicyMap proxy variables. `_x` versions correspond to primary extraction, `_y` to alternate pull. |

### `clean_data/PM_zip_area_designation.csv`
Output of `scripts/build_zip_area_designation.py`.
| Column | Description |
| --- | --- |
| `zip` | 5‑digit ZIP code |
| `urban_share_2015`, `rural_share_2015` | Residential shares derived from 2015 tract labels |
| `area_designation_2015` | `Urban` / `Rural` based on 80/20 rule |
| `total_weight_2015` | Sum of RES_RATIO used in weighting |
| `urban_share_2019`, `rural_share_2019` | Same for 2019 tract labels |
| `area_designation_2019` | Majority classification (Urban/Rural) |
| `mix_category_2019` | `Urban`, `Rural`, `Mixed` (20–80%), or `Unknown` |
| `total_weight_2019` | Sum of 2019 weights |

---

## 3. Derived Analytical Tables

### `analysis/low_snap_uptake_area_breakdown.csv`
Counts of flagged ZIPs by area type and low‑uptake quantile.
| Column | Description |
| --- | --- |
| `low_uptake_quantile` | Uptake quantile evaluated (0.10, 0.20, 0.30) |
| `area_mix_2019` | Area class (`Urban`, `Rural`, `Mixed`, `Unknown`) |
| `count` | Number of ZIPs meeting the criteria |
| `percentage` | Share of the quantile group represented by the area class |

### `analysis/backtester_univariate_summary.csv`
P1→P2 logistic results per single feature.
| Column | Description |
| --- | --- |
| `feature` | Proxy indicator name |
| `train_size`, `test_size` | Number of ZIPs used in train/test |
| `best_C` | Logistic regularisation parameter selected via CV |
| `threshold_prevalence` | Prevalence-aligned cut determined on P1 |
| `train_precision`, `train_recall`, `train_f1` | Training metrics at prevalence threshold |
| `p2_auc`, `p2_ap` | P2 ROC AUC and average precision |
| `p2_precision`, `p2_recall`, `p2_f1`, `p2_accuracy` | P2 metrics at prevalence threshold |
| `p2_tn`, `p2_fp`, `p2_fn`, `p2_tp` | Confusion matrix counts |
| `p2_precision_at_1pct`, `p2_precision_at_5pct` | PR@1%, PR@5% |

### `analysis/backtester_univariate_lr_coefficients.csv`
Coefficients (`coef`) and intercept per logistic model / feature; same row order as the summary table.

### `analysis/backtester_univariate_run_metadata.json`
JSON metadata: timestamp, random state, CV parameters, and the target definition used.

### `analysis/visuals/combo_backtest_summary.csv`
Full results for every logistic combination and random forest model.
Key columns mirror those in the univariate table with additional fields:
- `model`: `logistic` or `random_forest`
- `feature_set`: concatenated feature names
- `n_features`
- `coef_*`: Logistic coefficients (for combo models)
- Rankings and P1 metrics (train precision/recall etc.)

### `analysis/visuals/combo_backtest_summary_with_gb.csv`
Same schema as above, with an additional `gradient_boosting` row (and tuning metadata `gb_param_*` where available).

### `analysis/visuals/combo_backtest_rankings.csv`
Top models ordered by `Top_by_AUC`, `Top_by_AP`, `Top_by_F1`. Includes the same metrics as the summary file plus `ranking_scope` and `rank`.

### `analysis/backtester_area_rankings.csv`
Legacy runner’s P2 rankings per area (`All`, `Urban`, `Rural`, `Mixed`).  
Columns: `area`, `tag` (overall/P1/P2), `feature`, `auc`, `ap`.

### `analysis/backtester_area_trend_summary.csv`
P1→P2 drift for each area:
| Column | Description |
| --- | --- |
| `area` | Area class |
| `feature` | Proxy feature |
| `auc_P1`, `ap_P1` | P1 metrics |
| `auc_P2`, `ap_P2` | P2 metrics |
| `auc_delta`, `ap_delta` | Differences (P2 − P1) |

### `analysis/visuals/yearwise_univariate_metrics.csv`
Per-year univariate diagnostics using prevalence-aligned thresholds.
- `year`, `period`: temporal grouping
- `feature`: proxy indicator
- `n`: ZIP count used in that year
- `auc`, `ap`, `precision`, `recall`, `f1`, `acc`: metrics at the prevalence threshold
- `thr`: prevalence-aligned cut
- `hi_poverty_threshold`, `low_uptake_threshold`: yearly quantile values
- `n_total`, `n_valid`, `n_eligible`, `pos_rate`: yearly counts

### `analysis/backtest_report_overall.csv` (legacy)
Univariate ROC/AUC summary from the legacy runner. Shows features evaluated on the pooled dataset (random split).

### `analysis/pca_proxy_loadings.csv`
Principal component loadings for the four proxy indicators.
- Rows: `PC1`, `PC2`, `PC3`, `PC4`
- Columns: `pct_no_vehicle`, `pct_no_internet`, `pct_no_computer`, `pct_hs_diploma_only`
- Derived from P2 complete cases with StandardScaler normalisation.

---

## 4. Visual Artifacts
All figures are stored in `analysis/visuals/` as PNG files. Key examples:
- `snap_gap_poverty_vs_uptake.png`: poverty vs. SNAP uptake scatter (low uptake outliers highlighted)
- `low_snap_uptake_pie_bottom30.png`: area mix pie chart for the bottom 30% uptake segment
- `top_predictors_performance.png`: P2 performance comparison (AUC/AP) across top models
- `p2_roc_all3.png`, `p2_pr_all3.png`: ROC/PR curves for logistic and random forest
- `p2_roc_curves_with_gb.png`, `p2_pr_curves_with_gb.png`: same with gradient boosting included
- `auc_over_time.png`, `ap_over_time.png`: yearly feature performance trends
- `combo_auc_by_feature_count.png`, `combo_best_logistic_coefficients.png`: multivariate diagnostics

---

## 5. Notes on Missing Data
- Proxy variables are coalesced from `_x` and `_y` columns where available.
- P1→P2 analyses use complete cases across all four proxies.
- `scripts/build_zip_area_designation.py` logs missing-data rates in the console.
- No imputation is currently applied; missing rows are excluded.

---

For additional code details, refer to the respective Python scripts under `snap_gap/`. Each script logs key parameters and writes outputs to the paths documented above.
