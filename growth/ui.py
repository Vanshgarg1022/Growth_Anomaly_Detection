"""Look and feel for the Streamlit app: one stylesheet, a header, a status card and chart styling.

Design notes
  * Calm clinical palette: ink-teal text and accents; red / amber / green are reserved for clinical status only.
  * The header draws WHO-style centile curves, because the whole app is about a child's line against those curves.
  * The risk status is the one big element on the Risk tab; everything else stays quiet.
"""
from __future__ import annotations

import html

import streamlit as st

INK = "#15343C"
TEAL = "#1B7F86"
MIST = "#E9F1F1"
PAPER = "#F7FAFA"

STATUS_STYLE = {
    # status text         (accent,     tint,      short line shown under the status)
    "Normal":                    ("#2A8F6A", "#E3F3EC", "No concerning pattern in the latest measurements."),
    "Watch":                     ("#B26A00", "#FBF0D9", "Trending towards a concern. Plan an earlier follow-up."),
    "Potential Growth Anomaly":  ("#C2334B", "#FBE5E9", "Needs clinical review."),
    "Insufficient history":      ("#566A72", "#ECF1F2", "At least 3 visits are needed for the trajectory model."),
}

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=Figtree:wght@400;500;600&display=swap');

html, body, [class*="st-"], .stApp {{ font-family: 'Figtree', system-ui, sans-serif; }}
.stApp {{ background: {PAPER}; }}
.block-container {{ padding-top: 1.4rem; max-width: 1180px; }}
header[data-testid="stHeader"] {{ background: transparent; }}

h1, h2, h3 {{ font-family: 'Bricolage Grotesque', 'Figtree', sans-serif; color: {INK}; letter-spacing: -0.01em; }}
h2 {{ font-size: 1.55rem; font-weight: 700; margin-top: .4rem; }}
h3 {{ font-size: 1.2rem !important; font-weight: 600; margin-top: 1.1rem; }}
div[data-testid="stForm"] h3 {{ margin-top: .2rem; }}
p, label, li {{ color: {INK}; }}

/* header band */
.gx-hero {{ position: relative; overflow: hidden; background: {INK}; border-radius: 18px;
            padding: 28px 34px 26px; margin-bottom: 18px; }}
.gx-hero svg {{ position: absolute; right: 0; top: 0; height: 100%; width: 50%; opacity: .95;
                 -webkit-mask-image: linear-gradient(to right, transparent, #000 28%); mask-image: linear-gradient(to right, transparent, #000 28%); }}
.gx-hero-text {{ position: relative; max-width: 52%; }}
.gx-hero h1 {{ color: #F2FAFA; font-size: 2.05rem; line-height: 1.1; margin: 0 0 8px; padding: 0; font-weight: 700; }}
.gx-hero p {{ color: #B9D4D6; margin: 0; font-size: 1.02rem; line-height: 1.45; }}
.gx-hero .gx-tag {{ display: inline-block; margin-top: 14px; padding: 4px 11px; border-radius: 999px;
            background: rgba(255,255,255,.10); color: #D5E9EA; font-size: .82rem; }}
@media (max-width: 800px) {{ .gx-hero svg {{ opacity: .25; width: 100%; }} .gx-hero-text {{ max-width: 100%; }} .gx-hero h1 {{ font-size: 1.6rem; }} }}

/* tabs (role-based selectors work across Streamlit versions) */
[role="tablist"] {{ gap: 6px; border-bottom: 1px solid #CFE0E1; }}
[role="tab"] {{ height: 46px; padding: 0 16px; border-radius: 10px 10px 0 0; color: #4E6A72; font-weight: 500; }}
[role="tab"][aria-selected="true"] {{ color: {INK}; font-weight: 600; background: {MIST}; box-shadow: inset 0 -3px 0 {TEAL}; }}
[role="tab"]:focus-visible {{ outline: 3px solid #8CCFD3; outline-offset: -3px; }}
[data-baseweb="tab-highlight"] {{ background-color: {TEAL}; }}

/* section intro */
.gx-section {{ margin: 6px 0 14px; }}
.gx-section h2 {{ margin: 0 0 2px; }}
.gx-section p {{ margin: 0; color: #4E6A72; font-size: .98rem; }}

/* forms and inputs */
div[data-testid="stForm"] {{ background: #FFFFFF; border: 1px solid #D5E5E6; border-radius: 16px; padding: 22px 24px 8px; }}
.stTextInput input, .stNumberInput input, .stDateInput input, div[data-baseweb="select"] > div {{ border-radius: 10px !important; }}
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{
    border-radius: 10px; font-weight: 600; padding: .55rem 1.3rem; border: 1px solid {TEAL}; }}
.stFormSubmitButton > button, .stDownloadButton > button {{ background: {TEAL}; color: #fff; }}
.stFormSubmitButton > button:hover, .stDownloadButton > button:hover {{ background: #166A70; border-color: #166A70; color: #fff; }}
.stButton > button:focus-visible, .stFormSubmitButton > button:focus-visible {{ outline: 3px solid #8CCFD3; outline-offset: 2px; }}

/* metrics */
div[data-testid="stMetric"] {{ border-top: 2px solid #CFE0E1; padding-top: 10px; }}
div[data-testid="stMetricLabel"] p {{ color: #4E6A72; font-weight: 500; }}
div[data-testid="stMetricValue"] {{ font-family: 'Bricolage Grotesque', sans-serif; color: {INK}; }}

/* status card */
.gx-status {{ display: grid; grid-template-columns: 10px 1fr; border-radius: 16px; overflow: hidden; margin: 8px 0 18px;
              border: 1px solid rgba(0,0,0,.06); }}
.gx-status-body {{ padding: 20px 26px 22px; }}
.gx-status-label {{ font-size: .95rem; font-weight: 500; opacity: .8; margin: 0; }}
.gx-status-word {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 2.3rem; font-weight: 700; line-height: 1.1; margin: 2px 0 4px; }}
.gx-status-line {{ margin: 0; font-size: 1.02rem; }}
.gx-status-src {{ margin: 6px 0 0; font-size: .92rem; opacity: .85; }}

/* model score track */
.gx-track {{ position: relative; height: 10px; border-radius: 999px; margin: 26px 0 6px; background: rgba(21,52,60,.12); }}
.gx-zone {{ position: absolute; top: 0; bottom: 0; border-radius: 999px; }}
.gx-dot {{ position: absolute; top: 50%; width: 18px; height: 18px; border-radius: 50%; background: {INK};
           border: 3px solid #fff; transform: translate(-50%, -50%); box-shadow: 0 0 0 1px rgba(0,0,0,.25); }}
.gx-tick {{ position: absolute; top: -17px; transform: translateX(-50%); font-size: .74rem; color: #4E6A72; white-space: nowrap; }}
.gx-track-caption {{ font-size: .86rem; color: #4E6A72; margin: 0; }}

/* dataframes and alerts */
div[data-testid="stDataFrame"] {{ border: 1px solid #D5E5E6; border-radius: 12px; overflow: hidden; }}
div[data-testid="stAlert"] {{ border-radius: 12px; }}
"""


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def hero() -> None:
    """Title band with WHO-style centile curves and one child's trajectory."""
    curves = ""
    for i, (y0, y1, op) in enumerate([(24, 8, .30), (50, 38, .38), (78, 76, .55), (106, 114, .38), (132, 150, .30)]):
        curves += (f'<path d="M0 {y0 + 40} C 180 {y0 + 28}, 330 {y1 + 34}, 520 {y1 + 20}" fill="none" '
                   f'stroke="#7FC6C9" stroke-width="{2.4 if i == 2 else 1.4}" opacity="{op}"/>')
    pts = [(40, 118), (120, 108), (200, 100), (290, 88), (360, 84), (420, 96), (480, 114)]
    path = " ".join(("M" if k == 0 else "L") + f"{x} {y}" for k, (x, y) in enumerate(pts))
    dots = "".join(f'<circle cx="{x}" cy="{y}" r="4.2" fill="{"#FF8FA0" if k >= 5 else "#F2FAFA"}"/>' for k, (x, y) in enumerate(pts))
    svg = (f'<svg viewBox="0 0 520 190" preserveAspectRatio="xMaxYMid meet" aria-hidden="true">{curves}'
           f'<path d="{path}" fill="none" stroke="#F2FAFA" stroke-width="2.2"/>{dots}</svg>')
    st.markdown(
        f'<div class="gx-hero">{svg}<div class="gx-hero-text">'
        "<h1>Paediatric Growth Anomaly Early Warning</h1>"
        "<p>Follows each child's growth against WHO standards and flags a falling or rising trajectory "
        "before it crosses the usual cut-off.</p>"
        '<span class="gx-tag">Birth to 5 years, WHO Child Growth Standards</span>'
        "</div></div>",
        unsafe_allow_html=True,
    )


def section(title: str, hint: str = "") -> None:
    hint_html = f"<p>{html.escape(hint)}</p>" if hint else ""
    st.markdown(f'<div class="gx-section"><h2>{html.escape(title)}</h2>{hint_html}</div>', unsafe_allow_html=True)


def status_card(status: str, source: str | None = None, score: float | None = None,
                alert_threshold: float | None = None, watch_threshold: float | None = None) -> None:
    """Big status panel. With a model score it also draws where the score sits against the watch/alert thresholds."""
    accent, tint, line = STATUS_STYLE.get(status, STATUS_STYLE["Insufficient history"])
    src = ""
    if source and source not in ("n/a", "neither"):
        src = f'<div class="gx-status-src">Raised by: {html.escape(source)}</div>'

    track = ""
    if score is not None and alert_threshold:
        top = max(0.6, alert_threshold * 2.0, score * 1.15)
        pct = lambda v: f"{min(max(v / top, 0), 1) * 100:.1f}%"
        w = watch_threshold or alert_threshold * 0.6
        track = (
            '<div class="gx-track" role="img" aria-label="Model score against watch and alert thresholds">'
            f'<div class="gx-zone" style="left:0;width:{pct(w)};background:#BFE3D2"></div>'
            f'<div class="gx-zone" style="left:{pct(w)};width:{pct(alert_threshold - w)};background:#F2D79B"></div>'
            f'<div class="gx-zone" style="left:{pct(alert_threshold)};right:0;background:#EDB3BD"></div>'
            f'<span class="gx-tick" style="left:{pct(w)}">watch {w:.2f}</span>'
            f'<span class="gx-tick" style="left:{pct(alert_threshold)}">alert {alert_threshold:.2f}</span>'
            f'<div class="gx-dot" style="left:{pct(score)}"></div></div>'
            f'<div class="gx-track-caption">Trajectory model score {score:.3f}. The overall status is the more severe of this '
            "model and the WHO cut-off rule.</div>"
        )
    st.markdown(
        f'<div class="gx-status" style="background:{tint};color:{INK}">'
        f'<div style="background:{accent}"></div><div class="gx-status-body">'
        '<div class="gx-status-label">Overall status</div>'
        f'<div class="gx-status-word" style="color:{accent}">{html.escape(status)}</div>'
        f'<div class="gx-status-line">{html.escape(line)}</div>{src}{track}</div></div>',
        unsafe_allow_html=True,
    )


SERIES_COLOURS = {"waz": "#1B7F86", "haz": "#3F6FB5", "bmiz": "#8A5BA8", "hcz": "#C2754A"}


def style_figure(fig, height: int = 640) -> None:
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
        font=dict(family="Figtree, system-ui, sans-serif", color=INK, size=13),
        margin=dict(l=40, r=20, t=70, b=40), showlegend=False,
        title=dict(font=dict(family="Bricolage Grotesque, Figtree, sans-serif", size=19)),
    )
    fig.update_xaxes(showgrid=False, linecolor="#CFE0E1", title_text="Age (months)", title_font=dict(size=11))
    fig.update_yaxes(gridcolor="#E5EEEE", zerolinecolor="#CFE0E1", title_text="z-score", title_font=dict(size=11))
    for ann in fig.layout.annotations:           # subplot titles
        ann.font = dict(size=14, color=INK)


_Z_COLS = {"waz": "Weight z", "haz": "Length/height z", "bmiz": "BMI z", "hcz": "Head z"}


def _z_colour(v):
    """Cell tint for a z-score: amber beyond +/-2, red beyond +/-3 (the WHO cut-offs)."""
    try:
        a = abs(float(v))
    except (TypeError, ValueError):
        return ""
    if a > 3:
        return "background-color:#F6CBD3;color:#7D1428;font-weight:600"
    if a > 2:
        return "background-color:#FAE6BC;color:#6B4300"
    return ""


def visit_table(df):
    """Readable, colour-coded version of the visit records for st.dataframe(..., hide_index=True)."""
    t = df.assign(visit_date=df["visit_date"].dt.strftime("%d %b %Y"), age_months=df["age_months"].round(1))[
        ["visit_date", "age_months", "weight_kg", "height_cm", "head_circumference_cm", *_Z_COLS]
    ].rename(columns={"visit_date": "Visit date", "age_months": "Age (months)", "weight_kg": "Weight (kg)",
                      "height_cm": "Length/height (cm)", "head_circumference_cm": "Head circ. (cm)", **_Z_COLS})
    styler = t.style.format({"Age (months)": "{:.1f}", "Weight (kg)": "{:.2f}", "Length/height (cm)": "{:.1f}",
                             "Head circ. (cm)": "{:.1f}", **{v: "{:+.2f}" for v in _Z_COLS.values()}})
    apply = getattr(styler, "map", None) or styler.applymap
    return apply(_z_colour, subset=list(_Z_COLS.values()))
