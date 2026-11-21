"""Seed demo children (codes D-0001..D-0005) into MySQL from TEST-split synthetic children, so the live demo
shows data the model never saw in training.  Run next to db.py with Streamlit secrets configured:
    python scripts/seed_demo.py
Re-runnable: existing D- rows are replaced."""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from sqlalchemy import text

import who_zscore as who

DEMOS = [
    ("D-0001", "gradual_weight_faltering", "Deteriorating: model should alert before the WHO cut-off rule does"),
    ("D-0002", "recovery_catchup", "Recovering child: must not be treated as worsening"),
    ("D-0003", "normal_typical", "Stable normal child"),
    ("D-0004", "persistent_underweight", "Persistently low: the conventional WHO rule must flag it"),
]


def pick_child(children, visits, arch, prefer_gain=False):
    """First TEST-split child of this type with >= 8 visits (best early-warning gain if results exist)."""
    c = children[(children.archetype == arch) & (children.split == "test") & (children.n_visits >= 8)]
    if c.empty:
        c = children[(children.archetype == arch) & (children.n_visits >= 8)]
    gain_path = os.path.join(PROJECT_ROOT, "results", "phase4_delay_per_child.csv")
    if prefer_gain and os.path.exists(gain_path):
        g = pd.read_csv(gain_path)
        g = g[g.child_id.isin(c.patient_code)].sort_values("gain_visits", ascending=False)
        if not g.empty:
            return g.child_id.iloc[0]
    return c.patient_code.iloc[0]


def insert_visit(session, pid, vdate, w, h, hc, mtype, a):
    r = a["results"]
    session.execute(text("""
        INSERT INTO visits (patient_id, visit_date, weight_kg, height_cm, head_circumference_cm,
                            waz, haz, bmiz, hcz, measurement_type, adjusted_length_height_cm, bmi)
        VALUES (:pid, :vd, :w, :h, :hc, :waz, :haz, :bmiz, :hcz, :mt, :adj, :bmi)"""),
        {"pid": pid, "vd": vdate, "w": w, "h": h, "hc": hc, "waz": r["wfa"]["z"], "haz": r["lhfa"]["z"],
         "bmiz": r["bfa"]["z"], "hcz": r["hcfa"]["z"], "mt": mtype,
         "adj": a["adjusted_length_height"], "bmi": a["bmi"]})


def seed_demo_patients():
    from db import SessionLocal  # imported here so the helpers above can be unit-tested without secrets

    visits = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "synthetic_visits.csv"))
    children = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "synthetic_children.csv"))
    today = pd.Timestamp.now().normalize()

    with SessionLocal.begin() as session:
        if _has_table(session):
            session.execute(text("DELETE FROM analysis_results WHERE patient_id IN "
                                 "(SELECT patient_id FROM patients WHERE patient_code LIKE 'D-%')"))
        session.execute(text("DELETE FROM visits WHERE patient_id IN (SELECT patient_id FROM patients WHERE patient_code LIKE 'D-%')"))
        session.execute(text("DELETE FROM patients WHERE patient_code LIKE 'D-%'"))

        for demo_code, arch, desc in DEMOS:
            src = pick_child(children, visits, arch, prefer_gain=(demo_code == "D-0001"))
            ch = children[children.patient_code == src].iloc[0]
            cv = visits[visits.patient_code == src].sort_values("age_days")
            sex = ch["sex"]                                       # real sex from the data (NOT guessed from the code)
            dob = (today - pd.Timedelta(days=int(cv.age_days.max()))).date()
            pid = session.execute(text("INSERT INTO patients (patient_code, date_of_birth, sex) VALUES (:c, :d, :s)"),
                                  {"c": demo_code, "d": dob, "s": sex}).lastrowid
            for v in cv.itertuples():
                a = who.analyse_child(sex, int(v.age_days), float(v.weight_kg), float(v.height_cm),
                                      v.measurement_type, float(v.head_circumference_cm))
                insert_visit(session, pid, dob + pd.Timedelta(days=int(v.age_days)),
                             float(v.weight_kg), float(v.height_cm), float(v.head_circumference_cm), v.measurement_type, a)
            print(f"{demo_code}: {desc}  (from {src}, {len(cv)} visits, {sex})")

        # one-visit child (insufficient-history demo)
        dob5 = (today - pd.Timedelta(days=60)).date()
        pid5 = session.execute(text("INSERT INTO patients (patient_code, date_of_birth, sex) VALUES ('D-0005', :d, 'Boy')"),
                               {"d": dob5}).lastrowid
        a = who.analyse_child("Boy", 60, 4.9, 58.0, who.RECUMBENT, 39.0)
        insert_visit(session, pid5, today.date(), 4.9, 58.0, 39.0, who.RECUMBENT, a)
        print("D-0005: one visit only (insufficient-history demo)")


def _has_table(session) -> bool:
    try:
        session.execute(text("SELECT 1 FROM analysis_results LIMIT 1"))
        return True
    except Exception:
        return False


if __name__ == "__main__":
    seed_demo_patients()
