"""Guards against the label bug found in review: the deteriorating-type list must match the generator."""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import pytest

from growth import synth
from growth.early_warning import DETERIORATING_ARCHETYPES as EW_SET
from growth.features import DETERIORATING_ARCHETYPES, build_feature_matrix


def test_deteriorating_set_matches_generator():
    expected = {a.name for a in synth.ARCHETYPES if a.deteriorating}
    assert set(DETERIORATING_ARCHETYPES) == expected
    assert set(EW_SET) == expected          # one definition used everywhere


def test_feature_labels_equal_generator_labels():
    visits, children = synth.generate_dataset(120, seed=5)
    feats = build_feature_matrix(visits, children)           # raises if derived != label_emerging
    truth = visits.rename(columns={"patient_code": "child_id"})[["child_id", "visit_index", "label_emerging"]]
    m = feats.merge(truth, on=["child_id", "visit_index"])
    assert len(m) == len(feats)
    assert (m["label"] == m["label_emerging"]).all()
    assert m["label"].sum() > 0.15 * len(m)                  # ~24%: not 4.7% as with the stale list


def test_derived_labels_without_label_column():
    visits, children = synth.generate_dataset(60, seed=6)
    a = build_feature_matrix(visits.drop(columns=["label_emerging"]), children)
    b = build_feature_matrix(visits, children)
    assert (a["label"].to_numpy() == b["label"].to_numpy()).all()


def test_ground_truth_columns_are_not_model_features():
    from growth.models import get_feature_columns
    visits, children = synth.generate_dataset(40, seed=7)
    cols = get_feature_columns(build_feature_matrix(visits, children))
    bad = {"label", "archetype", "split", "onset_visit_index", "visit_index", "child_id", "age_months"}
    assert not bad & set(cols)
    assert len(cols) == 52
