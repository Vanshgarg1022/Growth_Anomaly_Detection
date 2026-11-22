# Paediatric Growth Anomaly Early Warning (0–5 years)

Decision-support **prototype**. For each clinic visit it computes WHO z-scores, applies the conventional WHO cut-off rule, and runs a
trajectory model that tries to flag deterioration earlier than the rule. The final status is the more severe of the two.

> **Trained and evaluated on synthetic data only. Not clinically validated. Not a diagnosis.**

<img width="944" height="452" alt="Screenshot 2026-11-22 234027" src="https://github.com/user-attachments/assets/23788731-2050-43b7-9a14-d6de235fece1" />

Figure: System Architecture

---

## 1. Results At A Glance

Held-out **test set: 275 children** (seed 42). The alert threshold was chosen on a separate validation split so that the model alerts on
non-deteriorating visits at about the same rate as the WHO rule (0.171 vs 0.198). The comparison is therefore at a similar alert burden.

| Metric | WHO rule | Random Forest | Change |
|---|---|---|---|
| Precision (visit level) | 0.384 | 0.578 | +0.194 |
| Recall (visit level) | 0.383 | 0.728 | +0.345 |
| F1 | 0.384 | 0.644 | +0.261 |
| Deteriorating children flagged at some point (n = 90) | 60.0% | 92.2% | +32.2 pp |
| Median delay from true onset to first flag | 3 visits | 1 visit | 2 visits earlier |
| Pre-onset alarms (deteriorating children) | 11.1% | 23.3% | +12.2 pp |


### Model Comparison (Test Set)
Random Forest is the best model (ROC-AUC 0.860, average precision 0.776), ahead of Logistic Regression (0.811, 0.701) and
Isolation Forest (0.647, 0.447, unsupervised). The app uses the Random Forest at threshold 0.292 (precision 0.578, recall 0.728, F1 0.644).
The no-skill level for average precision is 0.24, the share of positive visits.

The model flags 1 visit earlier than the rule at a median (95% CI 1–2). It flagged 30 children the rule missed, against 1 the other way.
Both methods alert on non-deteriorating visits at a similar rate (0.171 model, 0.198 rule).

---

## 2. Recall By Trajectory Type

The model beats the rule on every trajectory type. The largest gains are progressive stunting (recall 0.19 → 0.65, +0.46),
head-circumference deceleration (0.40 → 0.81, +0.41) and head-circumference acceleration (0.39 → 0.76, +0.37).
The smallest are multi-indicator failure (0.56 → 0.76, +0.20) and rapid weight loss (0.58 → 0.79, +0.21), which the rule already catches fairly well.

---

## 3. Early Warning: Detection & Delay

Across the 90 deteriorating test children, the model flagged 92.2% and the rule 60.0%. The median first flag came 1 visit after onset
for the model and 3 visits after for the rule. The model is earlier or equal for every type. The biggest detection gaps are progressive
stunting (93.3% vs 40.0%) and gradual weight faltering (85.7% vs 47.6%). Multi-indicator failure is flagged at the same median delay
(1 visit), although the model still catches more children (100% vs 66.7%). The model also raises pre-onset alarms for 23.3% of deteriorating children (rule 11.1%).

<img width="1684" height="689" alt="early_warning" src="https://github.com/user-attachments/assets/c88b8ef3-093b-4e48-aabc-1121cdf1e2f1" />

Per deteriorating test child. Delay is counted in visits from the true onset visit. Children never flagged are counted at the last visit + 1.

---

## 4. What Drives The Model

The model relies most on how much weight and head circumference z-scores vary over the child's history (importance 0.0179 and 0.0169).
Next come overall change in length/height (0.0095), head circumference (0.0087) and weight (0.0085), then minimum weight z-score and the slopes.
Variability and change features matter more than any single current z-score, which is the point of using trajectories.

Each alert in the app comes with a SHAP explanation for that visit:

<img width="1432" height="604" alt="shap_example" src="https://github.com/user-attachments/assets/f8959159-f041-44d1-948d-32b2bfc74366" />


---

## 5. WHO Engine Validation (WHO Anthro Survey Analyser)

The z-score engine was compared with WHO's online **Anthro Survey Analyser** on **300 synthetic visits** (both sexes, 0–5 years, recumbent
and standing, including the first weeks of life). Anthro flagged **0%** of visits as implausible. The reference output is in
`tests/reference/` and the comparison runs in `tests/test_who_zscore.py`.

| Indicator | Max difference, monthly tables only | Max difference, now |
|---|---|---|
| Weight-for-age | 0.185 | **0.005** |
| BMI-for-age | 0.446 | **0.005** |
| Length/height-for-age | 0.012 | 0.012 |

Where the old differences were (weight and BMI, largest per age band):
Before the fix, all the larger differences were in the first weeks of life: up to 0.446 z at 0–13 days, 0.116 at 14–29 days and 0.054 at 30–38 days.
From 39 days to 5 years the difference was at most 0.022.

| Age | Visits | Max difference before |
|---|---|---|
| 0–13 days | 20 | 0.446 |
| 14–29 days | 6 | 0.116 |
| 30–38 days | 10 | 0.054 |
| 39 days – 5 years | 264 | 0.022 |

The check exposed an error in the first weeks of life. The engine now uses the WHO **daily (expanded) tables** for weight-for-age and
BMI-for-age with linear interpolation between days. Length/height and head circumference remain on monthly tables with spline
interpolation. Anthro has no head-circumference output, so that indicator is not covered by this check.

<img width="1112" height="592" alt="anthro_by_age" src="https://github.com/user-attachments/assets/a34bffa9-fe56-4185-8613-c93d3448e838" />


---

## 6. Dataset & Evaluation Protocol

| Item | Value |
|---|---|
| Synthetic children | 2,000 |
| Visits | 16,505 |
| Trajectory types | 17 (7 deteriorating) |
| As-of feature rows (visit ≥ 3) | 12,575 |
| Positive visits | 3,022 (24%) |
| Features per row | 52 |
| Split by child, train / validation / test | 1,297 / 280 / 275 |
| Random Forest tuning | grouped 5-fold CV, scoring average precision |
| Alert threshold | 0.292, chosen on validation |
| Watch band | 0.6 × alert threshold (project-defined) |
| Automated tests | 115 passing |

---

## 7. What Changed In This Version

- **Labels:** fixed a stale trajectory list; labels are now cross-checked against the generator.
- **Evaluation:** child-level split, grouped CV, threshold on validation only, test used once.
- **Early warning:** delay measured from true onset, with censored misses, pre-onset alarms and a bootstrap CI.
- **Status logic:** final status is the more severe of rule and model; the model never downgrades a rule alert.
- **WHO engine:** daily tables for weight and BMI, ±0.7 cm adjustment, Anthro-verified (max difference 0.005).
- **Input validation:** duplicate dates, height drops, WHO limits; nothing is saved on failure.
- **Database:** `analysis_results` keyed by visit, method and model version.
- **UI:** new header, status cards, colour-coded visit table.

---

## 8. App

<!-- ADD IMAGES: app_entry.png, app_risk.png, app_chart.png (optional, from your own run if you prefer) -->

Risk summary
<img width="1280" height="900" alt="app_risk" src="https://github.com/user-attachments/assets/91e8fca9-cf1c-4dd3-bab1-534272d0ad1b" />

---

## 9. Repo Layout

| Path | Purpose |
|---|---|
| `app.py`, `db.py`, `growth/ui.py` | Streamlit app on MySQL |
| `who_zscore.py`, `who_tables/` | WHO LMS engine, z = ((y/M)^L − 1)/(L·S) |
| `growth/features.py`, `trajectory.py` | as-of trajectory features (no look-ahead) |
| `growth/baseline.py` | WHO cut-off rule (R1–R4 anomaly, R5–R6 watch) |
| `growth/models.py`, `early_warning.py`, `explain.py`, `inference.py` | models, delay evaluation, SHAP, app inference |
| `growth/validation.py`, `growth/report.py` | entry checks, PDF report |
| `scripts/` | `make_synthetic`, `run_phase2/3/4`, `make_figures`, `seed_demo`, `load_synthetic_to_db`, `check_db_connection` |
| `sql/analysis_results.sql` | results table |
| `tests/` | 115 tests including the WHO Anthro reference |

## 10. Reproduce

```
pip install -r requirements.txt
python scripts/make_synthetic.py --children 2000 --seed 42
python scripts/run_phase2.py
python scripts/run_phase3.py        # models/rf_v2.joblib, results/phase3_*.csv (~4 min)
python scripts/run_phase4.py        # results/phase4_*.csv
python scripts/make_figures.py      # regenerates docs/figures/*.png
pytest -q
```

Run the app: add credentials to `.streamlit/secrets.toml` (`[database]` user, password, host, port, database), run
`sql/analysis_results.sql` once, optionally `python scripts/seed_demo.py`, then `streamlit run app.py`.

## 11. Limitations

* Synthetic data with assumed trajectory shapes. These numbers show technical feasibility, not clinical performance.
* 30% of children deteriorate and many are borderline, so alert rates are high for both methods. The matched-alert-rate comparison is the fair one.
* Gains are measured in visits after the true onset and depend on the generator's visit schedule.
* Model alarms before the true onset are not necessarily false, since the onset is a modelling choice.
* SHAP explains the model's score, not the child's biology.
* Head circumference is not covered by the Anthro cross-check.
