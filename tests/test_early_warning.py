import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from growth.early_warning import (
    bootstrap_lead_time_ci,
    evaluate_delay,
    matched_threshold,
)

FLAG, OK = "Potential Growth Anomaly", "Normal"


def toy(child, onset, probs, base_flags, arch="gradual_weight_faltering"):
    n = len(probs)
    return pd.DataFrame({
        "child_id": child, "archetype": arch, "visit_index": range(2, 2 + n), "onset_visit_index": onset,
        "label": [int(i + 2 >= onset) for i in range(n)], "prob": probs,
        "baseline_status": [FLAG if b else OK for b in base_flags],
    })


def test_bootstrap_ci_validity():
    med, (lo, hi) = bootstrap_lead_time_ci(pd.Series([1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 1.0, np.nan]), n_boot=200)
    assert lo <= med <= hi


def test_delay_counts_misses_as_censored_and_reports_pre_onset_alarms():
    # child A: onset 4; model flags at visit 5 (delay 1), baseline never flags (censored at last+1)
    a = toy("A", 4, [0.0, 0.0, 0.0, 0.9, 0.9, 0.9], [0, 0, 0, 0, 0, 0])
    # child B: onset 3; model flags at 2 (before onset = alarm) then at 3; baseline flags at 6
    b = toy("B", 3, [0.9, 0.9, 0.9, 0.9, 0.9], [0, 0, 0, 0, 1])
    d = evaluate_delay(pd.concat([a, b]), thr=0.5).set_index("child_id")
    assert d.loc["A", "model_delay"] == 1 and bool(d.loc["A", "model_detected"])
    assert not d.loc["A", "base_detected"] and d.loc["A", "base_delay"] == (7 + 1) - 4   # last visit 7 -> censored 8
    assert d.loc["A", "gain_visits"] == d.loc["A", "base_delay"] - 1
    assert d.loc["B", "model_pre_onset_alarm"] and d.loc["B", "model_delay"] == 0 and d.loc["B", "base_delay"] == 3


def test_matched_threshold_equalises_alert_rates_on_validation():
    rng = np.random.default_rng(0)
    n = 4000
    val = pd.DataFrame({"label": 0, "prob": rng.random(n), "baseline_status": np.where(rng.random(n) < 0.2, FLAG, OK)})
    thr = matched_threshold(val)
    assert abs((val["prob"] >= thr).mean() - 0.2) < 0.01
