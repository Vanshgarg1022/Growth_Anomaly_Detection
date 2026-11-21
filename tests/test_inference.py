"""Risk status logic: the app must never show Normal for a child the WHO rule flags."""
import os
import sys
import types

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import growth.inference as inf
from growth.inference import analyze_patient_trajectory, combine_status


def test_combine_status_takes_the_more_severe():
    assert combine_status("Potential Growth Anomaly", "Normal") == ("Potential Growth Anomaly", "WHO cut-off rule")
    assert combine_status("Normal", "Potential Growth Anomaly") == ("Potential Growth Anomaly", "trajectory model")
    assert combine_status("Watch", "Potential Growth Anomaly")[0] == "Potential Growth Anomaly"
    assert combine_status("Potential Growth Anomaly", "Potential Growth Anomaly")[1] == "both"
    assert combine_status("Normal", "Normal") == ("Normal", "neither")
    assert combine_status("Potential Growth Anomaly", None) == ("Potential Growth Anomaly", "WHO cut-off rule only")


def make_session(zs, dob="2025-01-01", gap_days=90):
    """zs: list of (waz, haz, bmiz, hcz) per visit."""
    eng = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    with eng.begin() as c:
        c.execute(text("CREATE TABLE patients(patient_id INTEGER PRIMARY KEY, patient_code TEXT, date_of_birth DATE, sex TEXT)"))
        c.execute(text("CREATE TABLE visits(visit_id INTEGER PRIMARY KEY, patient_id INT, visit_date DATE, weight_kg REAL, "
                       "height_cm REAL, head_circumference_cm REAL, waz REAL, haz REAL, bmiz REAL, hcz REAL)"))
        c.execute(text("INSERT INTO patients VALUES (1,'T-1',:d,'Girl')"), {"d": dob})
        d0 = pd.Timestamp(dob)
        for i, z in enumerate(zs):
            c.execute(text("INSERT INTO visits(patient_id,visit_date,weight_kg,height_cm,head_circumference_cm,waz,haz,bmiz,hcz) "
                           "VALUES (1,:d,8,70,44,:a,:b,:m,:h)"),
                      {"d": (d0 + pd.Timedelta(days=gap_days * (i + 1))).date(), "a": z[0], "b": z[1], "m": z[2], "h": z[3]})
    return sessionmaker(bind=eng)()


LOW = [(-2.5, -2.4, -1.0, -1.0)] * 6        # chronically low: rule R1/R2 fire


def test_flagged_child_is_never_normal_even_if_model_is_missing(monkeypatch):
    monkeypatch.setattr(inf, "MODEL_FILE", os.path.join(PROJECT_ROOT, "models", "does_not_exist.joblib"))
    with make_session(LOW) as s:
        r = analyze_patient_trajectory("T-1", s)
    assert r["risk_status"] == "Potential Growth Anomaly"
    assert "model" in r["model_error"].lower() and r["rf_prob"] is None


def test_single_visit_with_low_zscore_is_not_hidden_as_insufficient_history():
    with make_session([(-3.0, -1.0, -2.5, 0.0)]) as s:
        r = analyze_patient_trajectory("T-1", s)
    assert r["risk_status"] == "Potential Growth Anomaly" and r["n_visits"] == 1


def test_single_normal_visit_is_insufficient_history():
    with make_session([(0.1, 0.0, 0.1, 0.0)]) as s:
        assert analyze_patient_trajectory("T-1", s)["risk_status"] == "Insufficient history"


def test_with_model_flagged_child_stays_flagged_and_score_present():
    if not os.path.exists(inf.MODEL_FILE):
        pytest.skip("model not trained")
    with make_session(LOW) as s:
        r = analyze_patient_trajectory("T-1", s)
    assert r["model_available"] and 0 <= r["rf_prob"] <= 1
    assert r["risk_status"] == "Potential Growth Anomaly"          # model low-score cannot downgrade the rule
    assert "SHAP" in r["explanation"] or "raises" in r["explanation"] or "lowers" in r["explanation"]
