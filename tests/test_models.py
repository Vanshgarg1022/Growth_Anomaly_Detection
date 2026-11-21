import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import joblib
import pandas as pd
import pytest

from growth import synth
from growth.features import build_feature_matrix
from growth.models import split_by_dataset_split


def test_splits_use_dataset_split_and_are_child_disjoint():
    visits, children = synth.generate_dataset(150, seed=3)
    parts = split_by_dataset_split(build_feature_matrix(visits, children))
    ids = {k: set(v.child_id) for k, v in parts.items()}
    assert ids["train"].isdisjoint(ids["validation"]) and ids["train"].isdisjoint(ids["test"]) and ids["validation"].isdisjoint(ids["test"])
    assert all(len(v) for v in parts.values())


def test_leaky_split_is_detected():
    df = pd.DataFrame({"child_id": ["a", "a", "b", "c"], "split": ["train", "test", "validation", "test"], "label": [0, 1, 0, 1]})
    with pytest.raises(ValueError, match="leakage"):
        split_by_dataset_split(df)


def test_model_artifact_integrity():
    path = os.path.join(PROJECT_ROOT, "models", "rf_v2.joblib")
    if not os.path.exists(path):
        pytest.skip("models/rf_v2.joblib not trained yet (run scripts/run_phase3.py)")
    art = joblib.load(path)
    for key in ("model", "features", "threshold", "watch_threshold", "version"):
        assert key in art
    assert len(art["features"]) == 52
    assert 0 < art["threshold"] < 1 and art["watch_threshold"] < art["threshold"]
