import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

INDICATORS = ("waz", "haz", "bmiz", "hcz")


def get_feature_columns(df: pd.DataFrame) -> list:
    """Trajectory feature columns only (<indicator>_<feature>). Ground-truth / bookkeeping columns
    (label, archetype, split, onset_visit_index, visit_index, child_id ...) never match."""
    return [c for c in df.columns if any(c.startswith(f"{ind}_") for ind in INDICATORS)]


def split_by_dataset_split(df: pd.DataFrame):
    """Use the dataset's own child-level, archetype-stratified split (70/15/15).

    Returns dict with 'train', 'validation', 'test' DataFrames. Raises if any child appears in two parts.
    """
    if "split" not in df.columns:
        raise ValueError("Feature matrix has no 'split' column. Re-run scripts/run_phase2.py.")
    parts = {name: df[df["split"] == name].copy() for name in ("train", "validation", "test")}
    for name, p in parts.items():
        if p.empty:
            raise ValueError(f"Split '{name}' is empty.")
    kids = {n: set(p["child_id"]) for n, p in parts.items()}
    if (kids["train"] & kids["validation"]) or (kids["train"] & kids["test"]) or (kids["validation"] & kids["test"]):
        raise ValueError("Data leakage: a child appears in more than one split.")
    return parts


def train_logistic_regression(X_train, y_train, groups_train, cv):
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
    )
    grid = GridSearchCV(
        pipe, {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]},
        cv=cv, scoring="average_precision", n_jobs=-1, refit=True,
    )
    grid.fit(X_train, y_train, groups=groups_train)
    return grid


def train_random_forest(X_train, y_train, groups_train, cv):
    rf = RandomForestClassifier(class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [200, 300],
        "min_samples_leaf": [3, 5, 10],
        "max_depth": [8, 12, None],
    }
    grid = GridSearchCV(rf, param_grid, cv=cv, scoring="average_precision", n_jobs=-1, refit=True)
    grid.fit(X_train, y_train, groups=groups_train)
    return grid


def train_isolation_forest(X_train):
    """Unsupervised reference only (no labels used). Higher -score_samples = more anomalous."""
    iso = IsolationForest(n_estimators=300, random_state=42, contamination="auto", n_jobs=-1)
    iso.fit(X_train)
    return iso
