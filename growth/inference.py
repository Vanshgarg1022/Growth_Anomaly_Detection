import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import pandas as pd
from sqlalchemy import text

from growth.baseline import evaluate_baseline_rules
from growth.explain import explain_single_visit
from growth.features import compute_asof_features

MODEL_FILE = os.path.join(PROJECT_ROOT, "models", "rf_v2.joblib")
MIN_VISITS = 3
SEVERITY = {"Normal": 0, "Watch": 1, "Potential Growth Anomaly": 2}
_MODEL_CACHE: dict = {}


def load_cached_model():
    """Return the model artifact dict, or None if the file is missing (callers must handle None loudly)."""
    if not os.path.exists(MODEL_FILE):
        return None
    mtime = os.path.getmtime(MODEL_FILE)
    if _MODEL_CACHE.get("mtime") != mtime:
        _MODEL_CACHE["artifact"] = joblib.load(MODEL_FILE)
        _MODEL_CACHE["mtime"] = mtime
    return _MODEL_CACHE["artifact"]


def combine_status(baseline_status: str, model_status: str | None):
    """Final status = the MORE SEVERE of the conventional rule and the trajectory model.
    The model can only add alerts; it can never downgrade a conventional-rule alert.
    Returns (status, source)."""
    if model_status is None:
        return baseline_status, "WHO cut-off rule only"
    b, m = SEVERITY.get(baseline_status, 0), SEVERITY.get(model_status, 0)
    status = baseline_status if b >= m else model_status
    if status == "Normal":
        return status, "neither"
    if b == m:
        return status, "both"
    return status, "WHO cut-off rule" if b > m else "trajectory model"


def analyze_patient_trajectory(patient_code: str, session) -> dict:
    """Fetch visits, compute as-of features, run baseline + model, return a dict for the app / PDF."""
    sql = text("""
        SELECT v.visit_id, v.visit_date, v.weight_kg, v.height_cm, v.head_circumference_cm,
               v.waz, v.haz, v.bmiz, v.hcz, p.patient_id, p.date_of_birth, p.sex
        FROM visits v
        JOIN patients p ON p.patient_id = v.patient_id
        WHERE p.patient_code = :c
        ORDER BY v.visit_date ASC
    """)
    rows = session.execute(sql, {"c": patient_code}).mappings().all()
    if not rows:
        return {"status": "No visits found", "n_visits": 0}

    df = pd.DataFrame(rows)
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"])
    # NOT rounded: training used unrounded age_months, so inference must too (display rounding is the UI's job)
    df["age_months"] = (df["visit_date"] - df["date_of_birth"]).dt.days / 30.4375
    for col in ("waz", "haz", "bmiz", "hcz", "weight_kg", "height_cm", "head_circumference_cm"):
        df[col] = df[col].astype(float)

    n_visits = len(df)
    latest = df.iloc[-1]
    latest_z = {k: float(latest[k]) for k in ("waz", "haz", "bmiz", "hcz")}

    # Conventional rule ALWAYS runs, even with 1 visit: a single very low z-score must never be hidden.
    base_status, fired = evaluate_baseline_rules(latest_z)
    result = {
        "patient_id": int(latest["patient_id"]),
        "visit_id": int(latest["visit_id"]),
        "patient_code": patient_code,
        "n_visits": n_visits,
        "df_visits": df,
        "latest_z": latest_z,
        "baseline_status": base_status,
        "fired_rules": fired,
        "rf_prob": None,
        "model_status": None,
        "model_available": False,
    }

    if n_visits < MIN_VISITS:
        result["risk_status"] = base_status if base_status != "Normal" else "Insufficient history"
        result["alert_source"] = "WHO cut-off rule only" if base_status != "Normal" else "n/a"
        result["explanation"] = (
            f"Only {n_visits} visit(s): at least {MIN_VISITS} are needed for trajectory features and the "
            "model score. The conventional WHO rule result above still applies."
        )
        return result

    artifact = load_cached_model()
    if artifact is None:
        result["risk_status"], result["alert_source"] = combine_status(base_status, None)
        result["model_error"] = (
            f"Trajectory model file not found ({os.path.relpath(MODEL_FILE, PROJECT_ROOT)}). "
            "Run scripts/run_phase3.py. Showing the conventional WHO rule result only."
        )
        result["explanation"] = result["model_error"]
        return result

    model, fcols = artifact["model"], artifact["features"]
    thr, watch = float(artifact["threshold"]), float(artifact["watch_threshold"])

    df["patient_code"] = patient_code
    feats = compute_asof_features(df)
    latest_feat = feats.iloc[[-1]][fcols]
    prob = float(model.predict_proba(latest_feat)[:, 1][0])
    model_status = "Potential Growth Anomaly" if prob >= thr else ("Watch" if prob >= watch else "Normal")

    result.update({
        "rf_prob": prob,
        "model_status": model_status,
        "model_available": True,
        "model_version": artifact.get("version"),
        "model_threshold": thr,
        "model_watch_threshold": watch,
        "explanation": explain_single_visit(model, latest_feat),
    })
    result["risk_status"], result["alert_source"] = combine_status(base_status, model_status)
    return result


def save_analysis_result(result: dict, session):
    """Persist the outcome into analysis_results (see sql/analysis_results.sql). Needs the model score."""
    if result.get("n_visits", 0) < MIN_VISITS or result.get("rf_prob") is None:
        return
    sql = text("""
        INSERT INTO analysis_results
            (patient_id, visit_id, method, model_version, risk_status, score, n_valid_visits, explanation)
        VALUES (:pid, :vid, 'random_forest', :ver, :status, :score, :n_vis, :exp)
        ON DUPLICATE KEY UPDATE
            risk_status = VALUES(risk_status),
            score = VALUES(score),
            n_valid_visits = VALUES(n_valid_visits),
            explanation = VALUES(explanation)
    """)
    session.execute(sql, {
        "pid": result["patient_id"],
        "vid": result["visit_id"],
        "ver": result.get("model_version", "rf_v2"),
        "status": result["risk_status"],
        "score": result["rf_prob"],
        "n_vis": result["n_visits"],
        "exp": json.dumps({
            "reason": result.get("explanation", ""),
            "baseline_status": result.get("baseline_status"),
            "fired_rules": result.get("fired_rules", []),
            "model_status": result.get("model_status"),
            "alert_source": result.get("alert_source"),
        }),
    })
    session.commit()
