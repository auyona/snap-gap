# snap-gap

ZIP-level modelling of Supplemental Nutrition Assistance Program (SNAP) uptake
versus poverty, built around PolicyMap data for 2014–2023. The toolkit prepares
aligned poverty and SNAP indicators, enriches them with neighbourhood context,
and evaluates how well a compact feature set can surface under-reached
communities.

---

## Project Goals
- Align family poverty and family SNAP definitions so coverage gaps are measured
  on a common denominator.
- Build reproducible cleaning scripts that turn raw PolicyMap extracts into
  modelling-ready tables.
- Benchmark how well tract-proxy features (internet access, vehicle access,
  computer access, education) predict low SNAP uptake.
- Produce diagnostics (rankings, charts, quality checks, mapped classifications)
  that highlight ZIP codes most likely to be underserved.

---

## Directory Layout
```
snap_gap/
├── raw_data/                  # Unmodified PolicyMap downloads and crosswalks
├── clean_data/                # Canonical cleaned tables (see below)
├── analysis/                  # Model outputs, reports, plots
│   └── visuals/               # Figures and CSV summaries used in presentations
├── scripts/                   # Helper scripts (ZIP area designations, etc.)
├── backtester*.py             # Modelling and evaluation routines
├── PM_*_cleaner / merger      # Data-prep scripts (family SNAP/poverty pipeline)
├── yearwise_auc_plot.py       # Temporal diagnostics for key predictors
├── snap_gap_analysis          # Residual-based OLS diagnostics
├── DATA_DICTIONARY.md         # Field-level reference for major tables
└── README.md                  # This document
```

Key cleaned datasets (`clean_data/`):
- `PM_pov_fam_full_clean.csv` – family poverty (% households in poverty)
- `PM_snap_fam_full_clean.csv` – family SNAP usage (% households on SNAP)
- `PM_pov_total_full_clean.csv` – total population in poverty (for context)
- `PM_all_fam_clean.csv` – family poverty + SNAP stacked long by period
- `PM_merged.csv` – base table (pov total / pov fam / SNAP fam + gaps)
- `PM_plus4.csv` – base table + four PolicyMap “Plus 4” metrics
- `PM_zip_area_designation.csv` – 2019 PolicyMap area mix at the ZIP level

See `DATA_DICTIONARY.md` for detailed schema notes.

---

## Data Sources
- **PolicyMap ZIP code extracts (2014–2018 “P1”, 2019–2023 “P2”)**
  - Family poverty (`PMP?_pov_fam_raw.csv`)
  - Family SNAP (`PMP?_snap_fam_raw.csv`)
  - Total poverty (`PMP?_pov_total_raw.csv`)
  - Plus-4 contextual variables:
    - households without vehicles
    - households without internet
    - households without computers
    - adults with high school diploma only
- **USPS HUD ZIP-tract crosswalk** (`raw_data/ZIP_TRACT_062025.xlsx`) for area
  mix computations.
- **PolicyMap tract classification files** (2015 and 2019 vintages) to derive
  urban / rural / mixed labels.
- **2010 ZCTA shapefile** (`2010 zip shapefile/`) for optional mapping.

---

## Toolchain & Environment
Target Python ≥ 3.9. Create a virtual environment and install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy scikit-learn statsmodels matplotlib seaborn \
            geopandas pyogrio openpyxl pyarrow tqdm
```

Only the packages you actively use are required; the list above covers every
script in this folder (data cleaning, statsmodels regressions, geospatial
exports, charts).

---

## Processing Pipeline

1. **Clean family poverty + SNAP (`PM_full_fam_cleaner`)**
   ```bash
   python PM_full_fam_cleaner.py
   ```
   - Reads `PMP?_pov_fam_raw.csv` and `PMP?_snap_fam_raw.csv`.
   - Handles sentinel values (`-9999`, etc.), normalises percentages, parses
     `time_period` into numeric start/end years.
   - Outputs:
     - `clean_data/PM_pov_fam_full_clean.csv`
     - `clean_data/PM_snap_fam_full_clean.csv`
     - `clean_data/PM_all_fam_clean.csv`

2. **Clean total poverty (`PM_total_pov_cleaner.py`)**
   ```bash
   python PM_total_pov_cleaner.py
   ```
   Produces `clean_data/PM_pov_total_full_clean.csv` (stacked long by period).

3. **Merge aligned datasets (`PM_merger.py`)**
   ```bash
   python PM_merger.py
   ```
   - Combines family poverty, total poverty, and family SNAP into
     `clean_data/PM_merged.csv`.
   - Adds gap columns:
     - `gap_fam_minus_snapfam`
     - `gap_total_minus_snapfam`

4. **Append “Plus 4” tract proxies (`PM_plus4_merger`)**
   ```bash
   python PM_plus4_merger.py
   ```
   - Adds four household context metrics to each ZIP.
   - Outputs `clean_data/PM_plus4.csv`, the primary modelling dataset.

5. **Derive ZIP area designations (`scripts/build_zip_area_designation.py`)**
   ```bash
   python scripts/build_zip_area_designation.py
   ```
   - Translates tract-level urban/rural labels to ZIPs using the HUD crosswalk.
   - Outputs:
     - `clean_data/PM_zip_area_designation.csv`
     - `analysis/low_snap_uptake_area_breakdown.csv`

6. **Baseline regression diagnostics (`snap_gap_analysis`)**
   ```bash
   python snap_gap_analysis
   ```
   - Fits OLS on `snap_fam` vs `pov_fam` (overall and within state-period).
   - Flags impossible records (SNAP > poverty), exports residual leaderboards,
     and writes `analysis/snap_gap_summary.txt`.

7. **Optional map + export (`../make_choropleth_snap_gap.py`)**
   Run from the project root to produce:
   - `analysis/visuals/choropleth_low_uptake_by_area.png`
   - `analysis/low_uptake_classifications.csv`
   - `analysis/all_zip_classifications.csv` (all ZIPs with uptake tier + area)

---

## Modelling & Evaluation Suite

### Univariate Logistic Backtester (`backtester.py`)
- Trains balanced logistic regression (with isotonic calibration) on P1, scores
  P2, and reports AUC/AP, thresholded precision/recall, and calibration stats.
- Produces summary files in `analysis/`:
  - `backtester_univariate_summary.csv`
  - `backtester_univariate_lr_coefficients.csv`
  - `backtester_univariate_run_metadata.json`

### Area-Specific Backtesting (`backtester_area_wise.py`)
- Re-runs the univariate pipeline separately for Urban, Rural, Mixed, and All ZIPs.
- Ranks predictors for each geography and exports:
  - `analysis/backtester_area_*_{P1,P2,overall}.{csv,xlsx}`
  - `analysis/backtester_area_rankings.csv`
  - `analysis/backtester_area_trend_summary.csv`

### Exhaustive Combination Tests (`backtester_multivariate.py`)
- Enumerates every subset of the four plus-4 features.
- Trains calibrated logistic regression and tuned random forest for each combo.
- Writes summaries and P2 ROC/PR plots to `analysis/visuals/`:
  - `combo_backtest_summary.csv`
  - `combo_backtest_rankings.csv`
  - `p2_roc_all3.png`, `p2_pr_all3.png`

### Gradient Boosting Benchmark (`backtester_multivariate_with_gb.py`)
- Extends the combination search by adding an informed grid of gradient boosting
  models as a non-linear reference.
- Outputs mirrored `*_with_gb` CSVs and plots in `analysis/visuals/`.

### Year-wise Diagnostics (`yearwise_auc_plot.py`)
- Computes univariate AUC/AP for selected predictors in each calendar year
  between 2014 and 2023.
- Saves `analysis/visuals/yearwise_univariate_metrics.csv`,
  `auc_over_time.png`, and `ap_over_time.png`.

### Quality Checks
- `analysis/data_quality_flags_snap_gt_pov.csv` lists ZIP-periods where SNAP
  exceeds poverty by >5 percentage points (potential data anomalies).
- `analysis/hidden_fragility_baseline.csv` and
  `analysis/hidden_fragility_within_state_period.csv` contain the bottom-decile
  residuals from the OLS workflow (candidate under-reached ZIPs).

---

## Visual Outputs
`analysis/visuals/` aggregates charts ready for decks:
- `snap_gap_poverty_vs_uptake.png` – scatter of SNAP vs poverty with regression fit.
- `top_predictors_performance.png` – feature ranking from the backtester suite.
- `combo_best_logistic_coefficients*.png` – coefficients for best logistic models.
- `p2_pr_curves_with_gb.png`, `p2_roc_curves_with_gb.png` – comparison plots.
- `low_snap_uptake_pie_bottom30.png` – share of low-uptake ZIPs by area type.

---

## Reproducing the Analysis

1. Place all required PolicyMap CSV exports and the HUD crosswalk into
   `raw_data/` (do not modify filenames).
2. Run the cleaning + merging pipeline (Steps 1–5 above).
3. Activate your virtual environment and execute the modelling scripts that you
   need. Each script has `--help` flags with optional parameters such as custom
   data or output paths.
4. Inspect `analysis/` for CSV/XLSX tables and `analysis/visuals/` for figures.
5. (Optional) Generate map-ready classifications with
   `python ../make_choropleth_snap_gap.py --skip-map --skip-geojson`.

---

## Data Dictionary & Further Reading
- Refer to `DATA_DICTIONARY.md` for column-level descriptions, units, and data
  lineage.
- `analysis/snap_gap_summary.txt` captures model fit statistics that help
  contextualise the size of the coverage gap.

---

## Next Steps
- Incorporate additional explanatory variables (e.g., unemployment, eviction)
  and rerun the combination backtester to test incremental lift.
- Extend the mapping workflow to publish a filtered GeoJSON of the bottom-30%
  ZIPs for partners.
- Automate data refresh by wrapping the cleaning scripts into a single CLI or
  notebook for future PolicyMap pulls.
- Add continuous integration (CI) to automatically test cleaners and backtesters
  on each commit.

For questions or collaboration ideas, open an issue or reach out directly. Happy
digging!

---

## License
Licensed under the MIT License — see LICENSE for details.

## Citation / Credit
If you use this code or build on this research, please credit:  
Auyona Ray (2025), "SNAP Gap Analysis / Household & Regional Fragility Index."

