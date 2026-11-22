import os
import sys

# Ensure root folder is accessible for module imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import calendar
from datetime import date
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sqlalchemy import text

import db
from growth.inference import analyze_patient_trajectory, save_analysis_result
from growth.validation import validate_who_measurement_limits
from growth import ui
import who_zscore as who

st.set_page_config(page_title="Paediatric Growth Anomaly System", page_icon="👶", layout="wide")

ui.inject_css()

MAX_AGE_YEARS = 5  # Project scope: 0 to 5 years


def calculate_age(dob: date, visit: date):
    """Return (years, months, days, total_days) between dob and visit."""
    years = visit.year - dob.year
    months = visit.month - dob.month
    days = visit.day - dob.day

    if days < 0:
        months -= 1
        prev_month = visit.month - 1 or 12
        prev_year = visit.year if visit.month != 1 else visit.year - 1
        days += calendar.monthrange(prev_year, prev_month)[1]
    if months < 0:
        years -= 1
        months += 12

    return years, months, days, (visit - dob).days


def peek_next_patient_code(session) -> str:
    """Reads last_number from patient_code_sequence without incrementing it."""
    try:
        sql = text("SELECT last_number FROM patient_code_sequence WHERE sequence_id = 1")
        last_num = session.execute(sql).scalar()
        if last_num is None:
            sql_max = text("SELECT MAX(patient_id) FROM patients WHERE patient_code LIKE 'A-%'")
            last_num = session.execute(sql_max).scalar() or 0
        return f"A-{(last_num + 1):04d}"
    except Exception:
        return "Auto-Assigned on Submit"


def get_and_increment_patient_code(session) -> str:
    """Increments last_number in patient_code_sequence atomically upon form submission."""
    try:
        sql_select = text("SELECT last_number FROM patient_code_sequence WHERE sequence_id = 1 FOR UPDATE")
        last_num = session.execute(sql_select).scalar()

        if last_num is not None:
            next_num = last_num + 1
            session.execute(
                text("UPDATE patient_code_sequence SET last_number = :n WHERE sequence_id = 1"),
                {"n": next_num}
            )
        else:
            # Seed sequence if row is missing
            next_num = 1
            session.execute(
                text("INSERT INTO patient_code_sequence (sequence_id, last_number) VALUES (1, 1) "
                     "ON DUPLICATE KEY UPDATE last_number = VALUES(last_number)")
            )
        return f"A-{next_num:04d}"
    except Exception as e:
        # Fallback query based on existing A- codes
        sql_fallback = text("""
            SELECT COALESCE(MAX(CAST(SUBSTRING(patient_code, 3) AS UNSIGNED)), 0) + 1 
            FROM patients WHERE patient_code LIKE 'A-%'
        """)
        next_num = session.execute(sql_fallback).scalar() or 1
        return f"A-{next_num:04d}"


def fetch_patient_demographics(patient_code: str, session):
    """Fetch patient sex and DOB for existing patients."""
    sql = text("SELECT patient_id, date_of_birth, sex FROM patients WHERE patient_code = :c")
    row = session.execute(sql, {"c": patient_code.strip()}).mappings().first()
    return row


def validate_inputs(patient_code, dob, visit_date, weight, height, head_circ, measurement_type, is_new_patient=False, session=None):
    """Return a list of validation errors, including single-visit limits and previous-visit drop checks."""
    errors = []
    today = date.today()

    if not is_new_patient:
        if not patient_code or not patient_code.strip():
            errors.append("Patient Code is required.")

    if visit_date < dob:
        errors.append("Visit Date cannot be earlier than Date of Birth.")
    if visit_date > today:
        errors.append(f"Visit Date cannot be in the future (today is {today:%d %b %Y}).")
    if dob > today:
        errors.append("Date of Birth cannot be in the future.")

    if weight is None or weight <= 0:
        errors.append("Weight is required and must be > 0 kg.")
    if height is None or height <= 0:
        errors.append("Length / Height is required and must be > 0 cm.")
    if head_circ is None or head_circ <= 0:
        errors.append("Head Circumference is required and must be > 0 cm.")
    if measurement_type is None:
        errors.append("Select the Measurement Type.")

    if visit_date >= dob:
        years, months, days, _ = calculate_age(dob, visit_date)
        if years > MAX_AGE_YEARS or (years == MAX_AGE_YEARS and (months > 0 or days > 0)):
            errors.append(
                f"Calculated age ({years} yrs, {months} mos, {days} days) exceeds 0–{MAX_AGE_YEARS} year project scope."
            )

    # Checks against the child's existing visits (existing patients only)
    if not is_new_patient and session and patient_code and height and head_circ and measurement_type \
            and visit_date >= dob:
        try:
            code = patient_code.strip()
            dup = session.execute(
                text("""SELECT 1 FROM visits v JOIN patients p ON p.patient_id = v.patient_id
                        WHERE p.patient_code = :c AND v.visit_date = :vd LIMIT 1"""),
                {"c": code, "vd": visit_date},
            ).first()
            if dup:
                errors.append(f"A visit for {code} on {visit_date:%d %b %Y} already exists. "
                              "Choose a different date (one record per child per day).")

            sql = text("""
                SELECT v.adjusted_length_height_cm, v.head_circumference_cm, v.visit_date
                FROM visits v
                JOIN patients p ON p.patient_id = v.patient_id
                WHERE p.patient_code = :c AND v.visit_date < :vd
                ORDER BY v.visit_date DESC LIMIT 1
            """)
            prev = session.execute(sql, {"c": code, "vd": visit_date}).mappings().first()
            if prev:
                # compare WHO-adjusted heights so a recumbent->standing switch is not mistaken for a drop
                adj_h, _ = who.adjust_length_height(height, measurement_type, (visit_date - dob).days)
                prev_h = float(prev["adjusted_length_height_cm"]) if prev["adjusted_length_height_cm"] is not None else None
                prev_hc = float(prev["head_circumference_cm"]) if prev["head_circumference_cm"] is not None else None
                if prev_h is not None and prev_h - adj_h > 1.0:
                    errors.append(
                        f"Implausible height drop: entered (WHO-adjusted) {adj_h:.1f} cm is > 1.0 cm below "
                        f"the previous visit ({prev_h:.1f} cm on {prev['visit_date']}). Please verify."
                    )
                if prev_hc is not None and prev_hc - head_circ > 0.5:
                    errors.append(
                        f"Implausible head circumference drop: entered {head_circ:.1f} cm is > 0.5 cm below "
                        f"the previous visit ({prev_hc:.1f} cm on {prev['visit_date']}). Please verify."
                    )
        except Exception as exc:  # do not hide the failure: tell the user the history check did not run
            errors.append(f"Could not compare with previous visits ({exc}). Check the database connection.")

    return errors


# Session state initialization
if "selected_code" not in st.session_state:
    st.session_state.selected_code = "S-0001"

ui.hero()

tab_entry, tab_hist, tab_risk, tab_report = st.tabs(
    ["Entry", "History & Trajectory", "Risk Summary", "Report Export"]
)

# -----------------------------------------------------------------------------
# TAB 1: ENTRY FORM
# -----------------------------------------------------------------------------
with tab_entry:
    ui.section("Visit entry", "Record a visit. Measurements are checked against WHO limits before anything is saved.")
    
    patient_type = st.radio(
        "Select Entry Workflow:",
        ["Existing Patient", "New Patient"],
        horizontal=True
    )

    with st.form("growth_entry_form", clear_on_submit=False):
        st.subheader("Administrative & Demographics")
        
        if patient_type == "New Patient":
            try:
                with db.SessionLocal() as session:
                    preview_code = peek_next_patient_code(session)
            except Exception:
                preview_code = "Next Available Code"

            patient_code_input = st.text_input("Patient Code (Auto-Assigned on Submit)", value=preview_code, disabled=True)
            gender = st.radio("Gender", ["Boy", "Girl"], horizontal=True)
            dob = st.date_input("Date of Birth (DOB)", max_value=date.today())
            
            patient_exists = False
            confirm_new_code = False
        else:
            col_ex1, col_ex2 = st.columns([2, 1])
            with col_ex1:
                patient_code_input = st.text_input(
                    "Enter Existing Patient Code",
                    value=st.session_state.selected_code,
                    placeholder="e.g. S-0001 or D-0001"
                )
            
            demo_dob = date.today()
            demo_gender = "Boy"
            patient_exists = False
            confirm_new_code = False

            if patient_code_input.strip():
                try:
                    with db.SessionLocal() as session:
                        demo = fetch_patient_demographics(patient_code_input.strip(), session)
                        if demo:
                            demo_dob = pd.to_datetime(demo["date_of_birth"]).date()
                            demo_gender = "Boy" if demo["sex"].lower() in ("boy", "male") else "Girl"
                            patient_exists = True
                except Exception:
                    pass

            if patient_exists:
                st.success(f"Fetched existing demographics for **{patient_code_input.strip()}**")
                g_col1, g_col2 = st.columns(2)
                with g_col1:
                    gender = st.radio("Gender (Read-Only)", ["Boy", "Girl"], index=0 if demo_gender == "Boy" else 1, disabled=True)
                with g_col2:
                    dob = st.date_input("Date of Birth (Read-Only)", value=demo_dob, disabled=True)
            else:
                st.warning(f"Patient code '{patient_code_input.strip()}' not found in database. "
                           "If this is a typo, correct the code. If it is a new child, tick the box below and enter demographics.")
                confirm_new_code = st.checkbox("I confirm this is a NEW patient with exactly this code")
                g_col1, g_col2 = st.columns(2)
                with g_col1:
                    gender = st.radio("Gender", ["Boy", "Girl"], horizontal=True)
                with g_col2:
                    dob = st.date_input("Date of Birth (DOB)", max_value=date.today())

        st.subheader("Visit & Measurements")
        visit_date = st.date_input("Visit Date", max_value=date.today())

        col_meas1, col_meas2 = st.columns(2)
        with col_meas1:
            weight = st.number_input(
                "Weight (kg)",
                min_value=0.0,
                max_value=50.0,
                value=None,
                step=0.1,
                format="%.2f",
                placeholder="Enter weight",
            )
            measurement_type = st.selectbox(
                "Measurement Type",
                [who.RECUMBENT, who.STANDING],
                index=None,
                placeholder="Select type",
            )
        with col_meas2:
            height = st.number_input(
                "Length / Height (cm)",
                min_value=0.0,
                max_value=150.0,
                value=None,
                step=0.5,
                format="%.1f",
                placeholder="Enter height",
            )
            head_circ = st.number_input(
                "Head Circumference (cm)",
                min_value=0.0,
                max_value=60.0,
                value=None,
                step=0.1,
                format="%.1f",
                placeholder="Enter head circ",
            )

        submit_button = st.form_submit_button("Analyse & Save Visit")

    if submit_button:
        is_new = (patient_type == "New Patient")
        
        try:
            with db.SessionLocal() as session:
                errors = validate_inputs(
                    patient_code_input, dob, visit_date, weight, height, head_circ, measurement_type,
                    is_new_patient=is_new, session=session
                )
        except Exception:
            errors = validate_inputs(
                patient_code_input, dob, visit_date, weight, height, head_circ, measurement_type,
                is_new_patient=is_new
            )

        if not is_new and not patient_exists and not confirm_new_code and patient_code_input.strip():
            errors.append(f"Patient code '{patient_code_input.strip()}' does not exist. Correct it, or tick the "
                          "confirmation box to register it as a new patient.")

        # Age/sex-specific WHO plausibility limits on the raw measurements (needs valid dates and values)
        if not errors:
            try:
                years, months, days, delta_days = calculate_age(dob, visit_date)
                errors.extend(validate_who_measurement_limits(
                    gender, delta_days, weight, height, measurement_type, head_circ))
            except (FileNotFoundError, ValueError) as exc:
                errors.append(f"WHO reference validation could not be completed: {exc}")

        if errors:
            st.error("⚠️ Nothing was saved. Please correct the following:")
            for msg in errors:
                st.markdown(f"- {msg}")
        else:
            years, months, days, delta_days = calculate_age(dob, visit_date)
            try:
                analysis = who.analyse_child(
                    sex=gender,
                    age_days=delta_days,
                    weight_kg=weight,
                    length_height_cm=height,
                    measurement_type=measurement_type,
                    head_circ_cm=head_circ,
                )
                
                with db.SessionLocal.begin() as session:
                    # Atomically generate and increment code ONLY upon successful submit if New Patient
                    if is_new:
                        final_patient_code = get_and_increment_patient_code(session)
                    else:
                        final_patient_code = patient_code_input.strip()

                    # Save patient row
                    sql_p = text("""
                        INSERT INTO patients (patient_code, date_of_birth, sex)
                        VALUES (:c, :d, :s)
                        ON DUPLICATE KEY UPDATE date_of_birth=VALUES(date_of_birth)
                    """)
                    session.execute(sql_p, {"c": final_patient_code, "d": dob, "s": gender})
                    
                    # Fetch patient_id
                    pid = session.execute(
                        text("SELECT patient_id FROM patients WHERE patient_code = :c"),
                        {"c": final_patient_code}
                    ).scalar()

                    # Save visit row
                    sql_v = text("""
                        INSERT INTO visits 
                        (patient_id, visit_date, weight_kg, height_cm, head_circumference_cm, 
                         waz, haz, bmiz, hcz, measurement_type, adjusted_length_height_cm, bmi)
                        VALUES (:pid, :vd, :w, :h, :hc, :waz, :haz, :bmiz, :hcz, :mt, :adj, :bmi)
                    """)
                    res_dict = analysis["results"]
                    session.execute(sql_v, {
                        "pid": pid,
                        "vd": visit_date,
                        "w": weight,
                        "h": height,
                        "hc": head_circ,
                        "waz": res_dict["wfa"]["z"],
                        "haz": res_dict["lhfa"]["z"],
                        "bmiz": res_dict["bfa"]["z"],
                        "hcz": res_dict.get("hcfa", {}).get("z", 0.0),
                        "mt": measurement_type,
                        "adj": analysis["adjusted_length_height"],
                        "bmi": analysis["bmi"]
                    })

                st.session_state.selected_code = final_patient_code
                st.success(f"✅ Visit successfully saved and analyzed for patient **{final_patient_code}**!")
            except Exception as exc:
                st.error(f"WHO Calculation / Database Save Failed: {exc}")

# -----------------------------------------------------------------------------
# TAB 2: HISTORY & TRAJECTORY
# -----------------------------------------------------------------------------
with tab_hist:
    ui.section("History and trajectory", "How each z-score has moved across visits. Dashed lines mark -2 and +2, dotted lines -3 and +3.")
    code = st.session_state.get("selected_code")

    col_sel1, col_sel2 = st.columns([3, 1])
    with col_sel1:
        search_code = st.text_input("Active Patient Code:", value=code)
    with col_sel2:
        if st.button("Load Profile"):
            st.session_state.selected_code = search_code.strip()
            st.rerun()

    code = st.session_state.get("selected_code")
    if code:
        try:
            with db.SessionLocal() as session:
                res = analyze_patient_trajectory(code, session)
        except Exception:
            res = {"n_visits": 0, "status": "Database connection offline"}

        if res.get("n_visits", 0) > 0:
            df_v = res["df_visits"]
            st.subheader(f"Visit Records ({res['n_visits']} Visits)")
            st.dataframe(ui.visit_table(df_v), use_container_width=True, hide_index=True)
            st.caption("Amber cells are beyond ±2 SD and red cells beyond ±3 SD of the WHO standard.")

            # 2x2 Plotly Trajectory Charts
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Weight-for-Age (WAZ)",
                    "Length/Height-for-Age (HAZ)",
                    "BMI-for-Age (BMIZ)",
                    "Head Circumference-for-Age (HCZ)",
                ),
            )

            indicators = [
                ("waz", 1, 1, ui.SERIES_COLOURS["waz"]),
                ("haz", 1, 2, ui.SERIES_COLOURS["haz"]),
                ("bmiz", 2, 1, ui.SERIES_COLOURS["bmiz"]),
                ("hcz", 2, 2, ui.SERIES_COLOURS["hcz"]),
            ]

            for ind, r, c, color in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=df_v["age_months"],
                        y=df_v[ind],
                        mode="lines+markers",
                        name=ind.upper(),
                        line=dict(color=color, width=2),
                    ),
                    row=r,
                    col=c,
                )
                fig.add_hline(y=0, line_dash="solid", line_color="#9BB3B6", opacity=0.6, row=r, col=c)
                fig.add_hline(y=2.0, line_dash="dash", line_color="#D9A441", row=r, col=c)
                fig.add_hline(y=-2.0, line_dash="dash", line_color="#D9A441", row=r, col=c)
                fig.add_hline(y=3.0, line_dash="dot", line_color="#C2334B", row=r, col=c)
                fig.add_hline(y=-3.0, line_dash="dot", line_color="#C2334B", row=r, col=c)

            fig.update_layout(title_text=f"z-score trajectories for {code}")
            ui.style_figure(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No visit records found in database for patient code '{code}'.")

# -----------------------------------------------------------------------------
# TAB 3: RISK SUMMARY & DIAGNOSIS SUPPORT
# -----------------------------------------------------------------------------
with tab_risk:
    ui.section("Risk summary", "Where this child stands today, and what drove the result.")
    code = st.session_state.get("selected_code")

    if code:
        try:
            with db.SessionLocal() as session:
                res = analyze_patient_trajectory(code, session)
                try:
                    save_analysis_result(res, session)
                except Exception as exc:  # the analysis is still valid even if it could not be stored
                    session.rollback()
                    st.warning(f"Analysis shown but NOT stored in analysis_results: {exc}")
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            res = {"n_visits": 0, "risk_status": "Insufficient history"}

        n_vis = res.get("n_visits", 0)

        if n_vis == 0:
            st.info(f"No visits found for '{code}'.")
        else:
            if res.get("model_error"):
                st.error(f"🛑 {res['model_error']}")

            ui.status_card(res.get("risk_status", "Insufficient history"), res.get("alert_source"),
                           res.get("rf_prob"), res.get("model_threshold"), res.get("model_watch_threshold"))

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("WHO cut-off rule", res.get("baseline_status", "-"))
                st.caption("Rules fired: " + (", ".join(res.get("fired_rules", [])) or "none"))
            with col_m2:
                rf_score = res.get("rf_prob")
                st.metric("Trajectory model score", f"{rf_score:.3f}" if rf_score is not None else "N/A")
                if res.get("model_status"):
                    st.caption(f"Model status: {res['model_status']} (alert threshold {res['model_threshold']:.3f})")
            with col_m3:
                st.metric("Visits used", n_vis)

            if n_vis < 3:
                st.info(res.get("explanation", ""))
            else:
                st.subheader("Top contributing trajectory factors (SHAP)")
                st.info(res.get("explanation", "No explanation available."))

            st.caption(
                "ℹ️ **Clinical safety disclaimer:** decision-support prototype based on WHO standards (0–5 years), "
                "trained on synthetic data. The overall status is the more severe of the conventional rule and the "
                "model. Not a clinical diagnosis; clinician review required."
            )

# -----------------------------------------------------------------------------
# TAB 4: REPORT EXPORT
# -----------------------------------------------------------------------------
with tab_report:
    ui.section("Report export", "A PDF with the visit table, trajectory charts and risk summary for the clinician.")
    code = st.session_state.get("selected_code")
    st.write(f"Generate PDF report summary for patient: **{code}**")

    if code:
        try:
            with db.SessionLocal() as session:
                res = analyze_patient_trajectory(code, session)
        except Exception:
            res = {"n_visits": 0}

        if res.get("n_visits", 0) > 0:
            from growth.report import generate_patient_pdf_report

            pdf_bytes = generate_patient_pdf_report(res)
            st.download_button(
                label=f"📄 Download PDF Report for {code}",
                data=pdf_bytes,
                file_name=f"{code}_growth_report.pdf",
                mime="application/pdf",
            )
        else:
            st.warning(f"No visit records available to build PDF report for '{code}'.")