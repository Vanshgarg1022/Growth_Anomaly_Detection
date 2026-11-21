"""Phase 3: train on TRAIN, tune on CV inside train, pick the alert threshold on VALIDATION, report TEST once."""
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from growth.early_warning import baseline_flagged, matched_threshold
from growth.models import (
    get_feature_columns,
    split_by_dataset_split,
    train_isolation_forest,
    train_logistic_regression,
    train_random_forest,
)

VERSION = "rf_v2"


def score_row(name, y, prob=None, pred=None, note=""):
    row = {"Model": name, "ROC-AUC": np.nan, "Avg-Precision": np.nan}
    if prob is not None:
        row["ROC-AUC"] = round(roc_auc_score(y, prob), 4)
        row["Avg-Precision"] = round(average_precision_score(y, prob), 4)
    if pred is not None:
        row.update({
            "Precision": round(precision_score(y, pred, zero_division=0), 4),
            "Recall": round(recall_score(y, pred, zero_division=0), 4),
            "F1": round(f1_score(y, pred, zero_division=0), 4),
        })
    row["Note"] = note
    return row


def main():
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "asof_features.csv"))
    parts = split_by_dataset_split(df)
    fcols = get_feature_columns(df)
    tr, va, te = parts["train"], parts["validation"], parts["test"]
    print(f"children  train {tr.child_id.nunique()}  val {va.child_id.nunique()}  test {te.child_id.nunique()}"
          f" | positive rate train {tr.label.mean():.3f}  val {va.label.mean():.3f}  test {te.label.mean():.3f}")

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    print("Training Logistic Regression ..."); lr = train_logistic_regression(tr[fcols], tr["label"], tr["child_id"], cv)
    print("Training Random Forest ...");       rf = train_random_forest(tr[fcols], tr["label"], tr["child_id"], cv)
    print("Training Isolation Forest (unsupervised reference) ..."); iso = train_isolation_forest(tr[fcols])

    # threshold on VALIDATION: model alert rate on label-0 visits = conventional rule's rate
    va = va.assign(prob=rf.predict_proba(va[fcols])[:, 1])
    thr = matched_threshold(va)
    watch_thr = 0.6 * thr            # project-defined "Watch" band, not a clinical threshold
    print(f"Threshold chosen on validation: {thr:.4f} (watch band from {watch_thr:.4f})")

    # TEST, touched once
    y = te["label"]
    p_lr = lr.predict_proba(te[fcols])[:, 1]
    p_rf = rf.predict_proba(te[fcols])[:, 1]
    p_iso = -iso.score_samples(te[fcols])
    base = baseline_flagged(te).astype(int)
    rows = [
        score_row("Logistic Regression", y, p_lr, (p_lr >= 0.5).astype(int), "threshold 0.5"),
        score_row("Random Forest", y, p_rf, (p_rf >= 0.5).astype(int), "threshold 0.5"),
        score_row("Random Forest @ matched threshold", y, p_rf, (p_rf >= thr).astype(int),
                  f"threshold {thr:.3f}: same alert rate as WHO rule on non-deteriorating visits (chosen on validation)"),
        score_row("Isolation Forest", y, p_iso, None, "unsupervised, no labels used; ranking metrics only"),
        score_row("Conventional WHO rule (R1-R4)", y, None, base.to_numpy(), "binary rule: no AUC"),
    ]
    summary = pd.DataFrame(rows)
    print("\n--- TEST SET RESULTS ---")
    print(summary.to_string(index=False))

    te_out = te.assign(prob=p_rf)
    per_arch = (te_out[te_out.label == 1].assign(model=lambda d: d.prob >= thr, base=lambda d: baseline_flagged(d))
                .groupby("archetype")[["model", "base"]].mean().round(3).rename(columns={"model": "recall_model", "base": "recall_baseline"}))
    print("\nVisit-level recall on positive visits by trajectory type:\n", per_arch.to_string())

    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
    artifact = {
        "model": rf.best_estimator_,
        "features": fcols,
        "version": VERSION,
        "threshold": thr,
        "watch_threshold": watch_thr,
        "threshold_rule": "model alert rate on label-0 validation visits == WHO-rule alert rate",
        "trained_on": f"synthetic dataset, {df.child_id.nunique()} children (train split only: {tr.child_id.nunique()})",
        "best_params": rf.best_params_,
        "test_metrics": summary.to_dict("records"),
    }
    joblib.dump(artifact, os.path.join(PROJECT_ROOT, "models", f"{VERSION}.joblib"))
    summary.to_csv(os.path.join(PROJECT_ROOT, "results", "phase3_test_metrics.csv"), index=False)
    per_arch.to_csv(os.path.join(PROJECT_ROOT, "results", "phase3_recall_by_type.csv"))
    print(f"\nSaved models/{VERSION}.joblib and results/phase3_*.csv")


if __name__ == "__main__":
    main()
