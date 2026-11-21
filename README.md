# Paediatric Growth Anomaly Early Warning (0–5 years)

Paediatric growth anomaly early-warning system (0–5 yrs) using WHO daily LMS references &amp; Random Forest (AUC 0.86, 92% detection rate). Engineers 52 longitudinal features to flag stunting and faltering early. Built with Python, Streamlit, MySQL, and ReportLab.


> **Prototype only.** The model is trained and evaluated on **synthetic data**. It is not clinically validated and is not a
> diagnosis. All findings require review by a qualified paediatric clinician.

## Features
- **Visit entry** (Streamlit) with validation: dates, age scope (0–5 years), WHO age- and sex-specific plausibility limits,
  duplicate-visit and previous-visit checks.
- **WHO z-scores**: weight-, length/height-, BMI- and head-circumference-for-age from the official WHO LMS tables, including the
  recumbent/standing (±0.7 cm) adjustment.
- **MySQL storage** of patients, visits and analysis results.
- **Trajectory features** built "as of" each visit (no look-ahead): current z-score, change, slope, variability, consecutive declines.
- **Conventional baseline** (WHO cut-off rules) and **ML models** (Logistic Regression, Random Forest, Isolation Forest).
- **Risk status** = the more severe of the WHO rule and the model, so the model can add alerts but never hide a rule-based alert.
- **Explanations** with SHAP values, and a downloadable **PDF report** per patient.

## Results (synthetic test set: 275 children, seed 42)
| Method | Precision | Recall | ROC-AUC |
|---|---|---|---|
| WHO cut-off rule alone | 0.38 | 0.38 | n/a (binary rule) |
| Logistic Regression (threshold 0.5) | 0.55 | 0.68 | 0.81 |
| Random Forest (threshold 0.5) | 0.85 | 0.60 | 0.86 |
| Random Forest (validation-matched threshold) | 0.59 | 0.72 | 0.86 |

Early warning across the 90 deteriorating test children: the model flags 92% of them versus 60% for the WHO rule. The median
detection delay from the true onset is 1 visit (model) versus 3 visits (rule); the median gain is 1 visit (95% CI 1 to 2).
Full tables are written to `results/` by `scripts/run_phase3.py` and `scripts/run_phase4.py`.

## Project structure
| Path | Purpose |
|---|---|
| `app.py`, `db.py` | Streamlit app and MySQL connection |
| `who_zscore.py`, `who_tables/` | WHO z-score engine and the official WHO tables |
| `trajectory.py`, `growth/features.py` | Trajectory features and the as-of feature matrix |
| `growth/synth.py` | Synthetic longitudinal data generator (17 trajectory types) |
| `growth/baseline.py` | Conventional WHO rule baseline |
| `growth/models.py`, `growth/early_warning.py`, `growth/explain.py` | Models, early-warning evaluation, SHAP |
| `growth/inference.py`, `growth/validation.py`, `growth/report.py` | App inference, input limits, PDF report |
| `scripts/` | Data generation, training, evaluation, demo seeding |
| `sql/analysis_results.sql` | Table for stored analysis results |
| `tests/` | Automated tests (no database required) |

## Setup
1. Python 3.10+ and a MySQL 8 server.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create the tables `patients`, `visits` and `analysis_results` in MySQL (`sql/analysis_results.sql` for the last one).
4. Add your database credentials in `.streamlit/secrets.toml` (never commit this file):
   ```toml
   [database]
   user = "your_user"
   password = "your_password"
   host = "localhost"
   port = 3306
   database = "your_database"
   ```

## Reproduce the data and model
```
python scripts/make_synthetic.py --children 2000 --seed 42   # synthetic dataset -> data/
python scripts/run_phase2.py                                  # as-of features and labels
python scripts/run_phase3.py                                  # train models -> models/rf_v2.joblib (about 4 min)
python scripts/run_phase4.py                                  # early-warning evaluation -> results/
pytest -q                                                     # run the tests
```

## Run the app
```
streamlit run app.py
```
Optional demo children (D-0001 to D-0005) for presentations:
```
python scripts/seed_demo.py
```
| Code | Demonstrates |
|---|---|
| D-0001 | Deteriorating child: the model alerts early |
| D-0002 | Recovering child: not treated as worsening |
| D-0003 | Stable normal child |
| D-0004 | Persistently low child: flagged by the WHO rule |
| D-0005 | One visit only: insufficient history |

## How the model is evaluated
- Splits follow the dataset's own child-level 70/15/15 split (train/validation/test); no child appears in two splits.
- Hyper-parameters are tuned by grouped cross-validation inside the training split.
- The alert threshold is chosen on the validation split so the model alerts on non-deteriorating visits at the same rate as the
  WHO rule; the test split is used once for reporting.
- Ground-truth columns (label, trajectory type, split, onset) are never model inputs.

## Limitations
- All results come from synthetic data with assumed trajectory shapes; they show technical feasibility, not clinical performance.
- The synthetic population has many borderline and chronically low children, so alert rates are high for both methods.
- The gain over the WHO rule is measured in visits and depends on the visit schedule used in the generator.
- The model flags some deteriorating children before their true onset (pre-onset alarms).
- SHAP values explain the model's score; they are not causal explanations.
- Validated against WHO macro output, not yet against real clinical records.

## Disclaimer
This is a decision-support prototype for educational use. It is not a medical device and must not be used for clinical decisions.
