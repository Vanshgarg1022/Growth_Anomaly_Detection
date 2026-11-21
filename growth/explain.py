import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import trajectory as T

_IND_NAMES = {"waz": "weight", "haz": "length/height", "bmiz": "BMI", "hcz": "head circumference"}
_EXPLAINERS: dict = {}


def readable_name(col: str) -> str:
    """'waz_slope_per_month' -> 'weight: Slope (z-score per month)'."""
    ind, _, feat = col.partition("_")
    return f"{_IND_NAMES.get(ind, ind)}: {T.FEATURE_LABELS.get(feat, feat)}"


def get_permutation_importances(model, X, y, feature_names, scoring="average_precision"):
    """Permutation importance on held-out data (drop in average precision when a feature is shuffled)."""
    result = permutation_importance(
        model, X, y, n_repeats=10, random_state=42, n_jobs=-1, scoring=scoring
    )
    return pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)


def _positive_class_shap(explainer, X: pd.DataFrame) -> np.ndarray:
    """SHAP values for class 1, shape (n_rows, n_features), across shap versions."""
    sv = explainer.shap_values(X)
    if isinstance(sv, list):                 # old shap: [class0, class1]
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    if sv.ndim == 3:                         # new shap: (n, features, classes)
        return sv[:, :, 1]
    return sv


def shap_reasons(model, X_row: pd.DataFrame, top_n: int = 3):
    """Top-N SHAP contributions for ONE visit as a list of dicts (feature, value, shap).

    Positive shap = pushes the concern score UP. Returns None if shap is not installed.
    """
    try:
        import shap
    except ImportError:
        return None
    key = id(model)
    if key not in _EXPLAINERS:
        _EXPLAINERS[key] = shap.TreeExplainer(model)
    contrib = _positive_class_shap(_EXPLAINERS[key], X_row)[0]
    order = np.argsort(np.abs(contrib))[::-1][:top_n]
    cols = list(X_row.columns)
    return [
        {"feature": cols[i], "label": readable_name(cols[i]),
         "value": float(X_row.iloc[0, i]), "shap": float(contrib[i])}
        for i in order
    ]


def explain_single_visit(model, X_row: pd.DataFrame, top_n: int = 3) -> str:
    """Plain-text explanation for the app / PDF. Uses SHAP; falls back to a clearly labelled global-importance list."""
    reasons = shap_reasons(model, X_row, top_n)
    if reasons is not None:
        return "; ".join(
            f"{r['label']} = {r['value']:+.2f} ({'raises' if r['shap'] > 0 else 'lowers'} concern, SHAP {r['shap']:+.3f})"
            for r in reasons
        )
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return "No explanation available."
    cols = list(X_row.columns)
    top = np.argsort(imp)[::-1][:top_n]
    return "Most influential features overall (shap not installed, not specific to this visit): " + "; ".join(
        f"{readable_name(cols[i])} = {X_row.iloc[0, i]:+.2f}" for i in top
    )
