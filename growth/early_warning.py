import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from growth.features import DETERIORATING_ARCHETYPES

BASELINE_FLAG = "Potential Growth Anomaly"


def baseline_flagged(df: pd.DataFrame) -> pd.Series:
    return df["baseline_status"] == BASELINE_FLAG


def matched_threshold(val_df: pd.DataFrame, prob_col: str = "prob") -> float:
    """Probability threshold at which the MODEL raises alerts on label-0 visits at the same rate as the
    conventional WHO rule does. Call it on VALIDATION data, never on the test set.

    Label-0 visits = visits that are not (yet) deteriorating: normal, chronic, recovering,
    transient dips, outliers and the pre-onset visits of deteriorating children."""
    neg = val_df[val_df["label"] == 0]
    far = baseline_flagged(neg).mean()
    return float(np.quantile(neg[prob_col], 1.0 - far))


def alert_rates(df: pd.DataFrame, thr: float, prob_col: str = "prob") -> dict:
    """Per-visit alert rate on label-0 visits, and share of non-deteriorating CHILDREN ever alerted."""
    neg = df[df["label"] == 0]
    out = {
        "baseline_alert_rate_label0_visits": float(baseline_flagged(neg).mean()),
        "model_alert_rate_label0_visits": float((neg[prob_col] >= thr).mean()),
    }
    nd = df[~df["archetype"].isin(DETERIORATING_ARCHETYPES)]
    g = nd.assign(b=baseline_flagged(nd), m=nd[prob_col] >= thr).groupby("child_id")[["b", "m"]].max()
    out["baseline_children_ever_alerted_nondeteriorating"] = float(g["b"].mean())
    out["model_children_ever_alerted_nondeteriorating"] = float(g["m"].mean())
    return out


def evaluate_delay(test_df: pd.DataFrame, thr: float, prob_col: str = "prob") -> pd.DataFrame:
    """Detection delay per deteriorating child, measured from the TRUE onset visit.

    delay = (first flagged visit at or after onset) - onset.  A method that never flags the child is
    CENSORED at (last visit + 1) - onset, so missing a child counts against the method instead of
    dropping the child from the statistics. gain = baseline_delay - model_delay (positive = model earlier);
    for baseline-misses the gain is therefore a LOWER bound.
    Flags before onset are reported separately as pre-onset alarms."""
    rows = []
    det = test_df[test_df["archetype"].isin(DETERIORATING_ARCHETYPES)]
    for child, g in det.groupby("child_id"):
        g = g.sort_values("visit_index")
        onset = int(g["onset_visit_index"].iloc[0])
        last = int(g["visit_index"].max())
        post = g[g["visit_index"] >= onset]
        pre = g[g["visit_index"] < onset]
        rec = {"child_id": child, "archetype": g["archetype"].iloc[0], "onset": onset}
        for name, flag_post, flag_pre in (
            ("base", baseline_flagged(post), baseline_flagged(pre)),
            ("model", post[prob_col] >= thr, pre[prob_col] >= thr),
        ):
            hits = post.loc[flag_post, "visit_index"]
            rec[f"{name}_detected"] = bool(len(hits))
            rec[f"{name}_delay"] = float((hits.min() if len(hits) else last + 1) - onset)
            rec[f"{name}_pre_onset_alarm"] = bool(flag_pre.any())
        rec["gain_visits"] = rec["base_delay"] - rec["model_delay"]
        rows.append(rec)
    return pd.DataFrame(rows)


def summarise_delay(delay_df: pd.DataFrame) -> pd.DataFrame:
    def one(g):
        return pd.Series({
            "children": len(g),
            "model_detected_%": round(100 * g["model_detected"].mean(), 1),
            "baseline_detected_%": round(100 * g["base_detected"].mean(), 1),
            "median_delay_model": g["model_delay"].median(),
            "median_delay_baseline": g["base_delay"].median(),
            "median_gain_visits": g["gain_visits"].median(),
            "model_only": int((g["model_detected"] & ~g["base_detected"]).sum()),
            "baseline_only": int((~g["model_detected"] & g["base_detected"]).sum()),
            "model_pre_onset_alarm_%": round(100 * g["model_pre_onset_alarm"].mean(), 1),
            "baseline_pre_onset_alarm_%": round(100 * g["base_pre_onset_alarm"].mean(), 1),
        })
    by = delay_df.groupby("archetype").apply(one, include_groups=False)
    by.loc["ALL deteriorating"] = one(delay_df)
    return by


def bootstrap_lead_time_ci(series: pd.Series, n_boot: int = 1000, seed: int = 42):
    """Median and 95% bootstrap CI (resampling children)."""
    rng = np.random.default_rng(seed)
    vals = series.dropna().to_numpy()
    if len(vals) == 0:
        return np.nan, (np.nan, np.nan)
    boots = [np.median(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(np.median(vals)), (float(lo), float(hi))
