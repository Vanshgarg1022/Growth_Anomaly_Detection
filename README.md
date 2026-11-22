# Paediatric Growth Anomaly Early Warning (0–5 years)

Decision-support **prototype**. For each clinic visit it computes WHO z-scores, applies the conventional WHO cut-off rule, and runs a
trajectory model that tries to flag deterioration earlier than the rule. The final status is the more severe of the two.

> **Trained and evaluated on synthetic data only. Not clinically validated. Not a diagnosis.**

<img width="944" height="452" alt="Screenshot 2026-11-22 234027" src="https://github.com/user-attachments/assets/23788731-2050-43b7-9a14-d6de235fece1" />

Figure: System Architecture

---

## 1. Results at a glance

Held-out **test set: 275 children** (seed 42). The alert threshold was chosen on a separate validation split so that the model alerts on
non-deteriorating visits at about the same rate as the WHO rule (0.171 vs 0.198). The comparison is therefore at a similar alert burden.

| Metric | WHO rule | Random Forest | Change |
|---|---|---|---|
| Precision (visit level) | 0.384 | 0.578 | +0.194 |
| Recall (visit level) | 0.383 | 0.728 | +0.345 |
| F1 | 0.384 | 0.644 | +0.261 |
| Deteriorating children flagged at some point (n = 90) | 60.0% | 92.2% | +32.2 pp |
| Median delay from true onset to first flag | 3 visits | 1 visit | 2 visits earlier |
| Median gain per child | – | 1 visit (95% CI 1–2) | – |
| Children flagged only by one method | 1 | 30 | – |
| Pre-onset alarms (deteriorating children) | 11.1% | 23.3% | +12.2 pp |
| Alert rate, non-deteriorating visits | 0.198 | 0.171 | −0.027 |
| Non-deteriorating children ever alerted | 40.5% | 41.6% | +1.1 pp |


### Model comparison (test set)

| Model | ROC-AUC | Avg. precision | Precision | Recall | Note |
|---|---|---|---|---|---|
| Logistic Regression | 0.8107 | 0.7009 | 0.5543 | 0.6592 | threshold 0.5 |
| Random Forest | 0.8595 | 0.7757 | 0.8403 | 0.5857 | threshold 0.5 |
| Random Forest (used in app) | 0.8595 | 0.7757 | 0.5777 | 0.7283 | threshold 0.292, F1 0.6443 |
| Isolation Forest | 0.6468 | 0.4473 | – | – | unsupervised |

Share of positive visits in the test data is 0.24, which is the no-skill level for average precision.

---

## 2. Recall by trajectory type

Share of deteriorating visits flagged.

| Trajectory type | WHO rule | Random Forest | Change |
|---|---|---|---|
| hc_deceleration | 0.40 | 0.81 | +0.41 |
| rapid_weight_loss | 0.58 | 0.79 | +0.21 |
| rising_bmi | 0.41 | 0.76 | +0.35 |
| hc_acceleration | 0.39 | 0.76 | +0.37 |
| multi_indicator_failure | 0.56 | 0.76 | +0.20 |
| progressive_stunting | 0.19 | 0.65 | +0.46 |
| gradual_weight_faltering | 0.31 | 0.65 | +0.34 |

---

## 3. Early warning: detection and delay

Per deteriorating test child. Delay is counted in visits from the true onset visit. Children never flagged are counted at the last visit + 1.

| Trajectory type | Children | Detected, rule | Detected, model | Median delay, rule | Median delay, model |
|---|---|---|---|---|---|
| gradual_weight_faltering | 21 | 47.6% | 85.7% | 3 | 1 |
| hc_acceleration | 9 | 66.7% | 100% | 3 | 1 |
| hc_deceleration | 9 | 77.8% | 100% | 3 | 1 |
| multi_indicator_failure | 6 | 66.7% | 100% | 1 | 1 |
| progressive_stunting | 15 | 40.0% | 93.3% | 3 | 2 |
| rapid_weight_loss | 12 | 91.7% | 100% | 2 | 1 |
| rising_bmi | 18 | 55.6% | 83.3% | 3 | 1 |
| **All deteriorating** | **90** | **60.0%** | **92.2%** | **3** | **1** |

<img width="1684" height="689" alt="early_warning" src="https://github.com/user-attachments/assets/c88b8ef3-093b-4e48-aabc-1121cdf1e2f1" />


---

## 4. What drives the model

Permutation importance on test children (drop in average precision when the feature is shuffled).

| Rank | Feature | Importance |
|---|---|---|
| 1 | Weight: variability (SD of z-scores) | 0.0179 |
| 2 | Head circumference: variability | 0.0169 |
| 3 | Length/height: overall change | 0.0095 |
| 4 | Head circumference: overall change | 0.0087 |
| 5 | Weight: overall change | 0.0085 |
| 6 | Weight: minimum z-score | 0.0082 |
| 7 | Weight: slope (z per month) | 0.0076 |
| 8 | Length/height: slope (z per month) | 0.0072 |

Each alert in the app comes with a SHAP explanation for that visit:

<img width="1432" height="604" alt="shap_example" src="https://github.com/user-attachments/assets/f8959159-f041-44d1-948d-32b2bfc74366" />


---

## 5. WHO engine validation (WHO Anthro Survey Analyser)

The z-score engine was compared with WHO's online **Anthro Survey Analyser** on **300 synthetic visits** (both sexes, 0–5 years, recumbent
and standing, including the first weeks of life). Anthro flagged **0%** of visits as implausible. The reference output is in
`tests/reference/` and the comparison runs in `tests/test_who_zscore.py`.

| Indicator | Max difference, monthly tables only | Max difference, now |
|---|---|---|
| Weight-for-age | 0.185 | **0.005** |
| BMI-for-age | 0.446 | **0.005** |
| Length/height-for-age | 0.012 | 0.012 |

Where the old differences were (weight and BMI, largest per age band):

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

## 6. Dataset and evaluation protocol

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

## 7. What changed in this version

| Area | Change |
|---|---|
| Labels | Fixed a stale list of deteriorating types; labels are derived from the 7 true types and cross-checked against the generator, which raises an error on mismatch |
| Evaluation | Child-level split with leakage check, grouped CV, threshold on validation only, test used once |
| Early warning | Delay from true onset, censored misses, pre-onset alarms, bootstrap CI over children |
| Status logic | Final status is the more severe of rule and model; the model never downgrades a rule alert; missing model shows a banner and falls back to the rule |
| WHO engine | Daily tables for weight and BMI, ±0.7 cm recumbent/standing adjustment, Anthro-verified (max difference 0.005) |
| Input validation | Duplicate dates, height drops, WHO plausibility limits, new-patient confirmation; nothing is saved on failure |
| Database | `analysis_results` keyed by `(visit_id, method, model_version)`, explanation stored as JSON |
| UI | New header, status cards, colour-coded visit table (amber beyond ±2 SD, red beyond ±3 SD), light theme |
| Tests | 115 tests, no database needed |

---

## 8. App

<!-- ADD IMAGES: app_entry.png, app_risk.png, app_chart.png (optional, from your own run if you prefer) -->

Risk summary
<img width="1280" height="900" alt="app_risk" src="https://github.com/user-attachments/assets/91e8fca9-cf1c-4dd3-bab1-534272d0ad1b" />

---

## 9. Repo layout

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
