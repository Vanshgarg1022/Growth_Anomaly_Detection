"""Phase 4: early-warning evaluation on the TEST split, at the threshold stored in the model artifact
(chosen on validation). Reports detection, delay from true onset, pre-onset alarms, false alarms, explanations."""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import pandas as pd

from growth.early_warning import alert_rates, bootstrap_lead_time_ci, evaluate_delay, summarise_delay
from growth.explain import explain_single_visit, get_permutation_importances, readable_name
from growth.models import split_by_dataset_split

MODEL_FILE = os.path.join(PROJECT_ROOT, "models", "rf_v2.joblib")


def main():
    feat_path = os.path.join(PROJECT_ROOT, "data", "asof_features.csv")
    if not os.path.exists(feat_path) or not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("Run scripts/run_phase2.py and scripts/run_phase3.py first.")
    art = joblib.load(MODEL_FILE)
    model, fcols, thr = art["model"], art["features"], art["threshold"]

    te = split_by_dataset_split(pd.read_csv(feat_path))["test"].reset_index(drop=True)
    te["prob"] = model.predict_proba(te[fcols])[:, 1]
    print(f"Test children {te.child_id.nunique()}, visits {len(te)}; threshold {thr:.4f} (chosen on validation)")

    rates = alert_rates(te, thr)
    print("\nAlert / false-alarm rates (test):")
    for k, v in rates.items():
        print(f"  {k:55s} {v:.3f}")

    delay = evaluate_delay(te, thr)
    summ = summarise_delay(delay)
    print("\n--- DETECTION DELAY FROM TRUE ONSET (visits; undetected children counted as censored) ---")
    print(summ.to_string())
    med, (lo, hi) = bootstrap_lead_time_ci(delay["gain_visits"])
    print(f"\nMedian gain over conventional rule: {med:.1f} visits (95% bootstrap CI {lo:.1f} to {hi:.1f}), "
          f"n={len(delay)} children. Positive = model flags earlier. Lower bound where the rule never flags.")
    print(f"Children flagged only by the model: {int((delay.model_detected & ~delay.base_detected).sum())}; "
          f"only by the rule: {int((~delay.model_detected & delay.base_detected).sum())}")

    imp = get_permutation_importances(model, te[fcols], te["label"], fcols)
    print("\nTop 8 features by permutation importance (drop in average precision):")
    for c, v in imp.head(8).items():
        print(f"  {readable_name(c):55s} {v:.4f}")

    alert = te[(te.prob >= thr) & (te.label == 1)].iloc[[0]]
    print(f"\nSample alert ({alert.child_id.iloc[0]}, visit {int(alert.visit_index.iloc[0])}, prob {alert.prob.iloc[0]:.3f}):")
    print("  " + explain_single_visit(model, alert[fcols]))

    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
    delay.to_csv(os.path.join(PROJECT_ROOT, "results", "phase4_delay_per_child.csv"), index=False)
    summ.to_csv(os.path.join(PROJECT_ROOT, "results", "phase4_delay_summary.csv"))
    imp.to_csv(os.path.join(PROJECT_ROOT, "results", "phase4_permutation_importance.csv"), header=["importance"])
    pd.Series(rates).to_csv(os.path.join(PROJECT_ROOT, "results", "phase4_alert_rates.csv"), header=["value"])


if __name__ == "__main__":
    main()
