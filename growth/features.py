import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import trajectory as T

INDICATORS = ("waz", "haz", "bmiz", "hcz")

# SINGLE source of truth for "which trajectory types are deteriorating".
# Must match growth/synth.py (tests/test_labels.py checks that it does).
DETERIORATING_ARCHETYPES = frozenset(
    {
        "gradual_weight_faltering",
        "rapid_weight_loss",
        "progressive_stunting",
        "rising_bmi",
        "multi_indicator_failure",
        "hc_deceleration",
        "hc_acceleration",
    }
)


def compute_asof_features(df_child_visits: pd.DataFrame) -> pd.DataFrame:
    """One row per visit k >= 2, using ONLY visits 0..k (no look-ahead)."""
    rows = []
    # k >= 2 ensures at least 3 visits for slope and variability calculation
    for k in range(2, len(df_child_visits)):
        h = df_child_visits.iloc[: k + 1]

        row = {
            "child_id": (
                h["patient_code"].iloc[0]
                if "patient_code" in h.columns
                else h["child"].iloc[0]
            ),
            "visit_index": k,
            "age_months": h["age_months"].iloc[-1],
            "visit_id": (
                h["visit_id"].iloc[-1] if "visit_id" in h.columns else None
            ),
        }

        for ind in INDICATORS:
            feats = T.indicator_features(
                h["age_months"].to_numpy(), h[ind].to_numpy()
            )
            for name, val in feats.items():
                if name != "n_visits":
                    row[f"{ind}_{name}"] = val

        rows.append(row)

    return pd.DataFrame(rows)


def derive_label(archetype: pd.Series, onset: pd.Series, visit_index: pd.Series) -> pd.Series:
    """1 if the child's type is deteriorating AND this visit is at/after the onset visit."""
    onset = pd.to_numeric(onset, errors="coerce")        # '' (in-memory generator output) or NaN (CSV) -> NaN
    is_det = archetype.isin(DETERIORATING_ARCHETYPES)
    after = onset.notna() & (visit_index >= onset)
    return (is_det & after).astype(int)


def build_feature_matrix(
    df_all_visits: pd.DataFrame, df_truth: pd.DataFrame = None
) -> pd.DataFrame:
    all_features = []
    id_col = "patient_code" if "patient_code" in df_all_visits.columns else "child"

    for _, g in df_all_visits.groupby(id_col):
        g_sorted = g.sort_values("age_months").reset_index(drop=True)
        feats_k = compute_asof_features(g_sorted)
        if not feats_k.empty:
            all_features.append(feats_k)

    if not all_features:
        return pd.DataFrame()

    feat_df = pd.concat(all_features, ignore_index=True)

    if df_truth is not None and "patient_code" in df_truth.columns:
        # Ground truth is attached for LABELS / EVALUATION ONLY. These columns must never be features
        # (growth.models.get_feature_columns only selects <indicator>_* columns).
        truth_cols = [
            c
            for c in ["patient_code", "archetype", "onset_visit_index", "split"]
            if c in df_truth.columns
        ]
        feat_df = feat_df.merge(
            df_truth[truth_cols], left_on="child_id", right_on="patient_code", how="left"
        )

        label = derive_label(
            feat_df["archetype"], feat_df["onset_visit_index"], feat_df["visit_index"]
        )

        # If the visits file carries the generator's own per-visit label, use it and make sure
        # it agrees with the derived one (catches a stale archetype list immediately).
        if "label_emerging" in df_all_visits.columns:
            truth_lbl = df_all_visits[[id_col, "visit_index", "label_emerging"]].rename(
                columns={id_col: "child_id", "visit_index": "_vi"}
            )
            tmp = feat_df[["child_id", "visit_index"]].merge(
                truth_lbl, left_on=["child_id", "visit_index"], right_on=["child_id", "_vi"], how="left"
            )
            if tmp["label_emerging"].notna().all():
                official = tmp["label_emerging"].astype(int).to_numpy()
                if (official != label.to_numpy()).any():
                    raise ValueError(
                        "Derived labels disagree with label_emerging for "
                        f"{int((official != label.to_numpy()).sum())} visits - check DETERIORATING_ARCHETYPES."
                    )
                label = pd.Series(official, index=feat_df.index)

        feat_df["label"] = label.astype(int)

    return feat_df
