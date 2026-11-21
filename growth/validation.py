"""Age/sex-specific WHO plausibility limits for RAW measurements (kg / cm), derived from the WHO engine's own
flag limits so the app can reject typos (pounds, mm, swapped fields ...) BEFORE anything is saved."""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import who_zscore as who


def who_measurement_limits(indicator, sex, age_days):
    """(low, high) measurement bounds implied by the engine's biological flag limits."""
    key, _ = who.select_table(indicator, sex, age_days)
    L, M, S = who.lms_at(key, who.age_in_months(age_days))
    z_min, z_max = who.FLAG_LIMITS[indicator]

    if indicator in who.EXTEND_BEYOND_3SD:
        # match zscore_from_lms()'s linear extension beyond +/-3 SD
        sd3n, sd2n = who._cutoff(L, M, S, -3), who._cutoff(L, M, S, -2)
        sd2, sd3 = who._cutoff(L, M, S, 2), who._cutoff(L, M, S, 3)
        low = sd3n + (z_min + 3) * (sd2n - sd3n)
        high = sd3 + (z_max - 3) * (sd3 - sd2)
    else:
        low = who._cutoff(L, M, S, z_min)
        high = who._cutoff(L, M, S, z_max)
    return low, high


def validate_who_measurement_limits(sex, age_days, weight, height, measurement_type, head_circ):
    """Return a list of error strings; empty list = all four measurements are plausible."""
    errors = []
    adjusted_height, _ = who.adjust_length_height(height, measurement_type, age_days)
    bmi = weight / (adjusted_height / 100) ** 2
    checks = [
        ("Weight", "wfa", weight, "kg"),
        ("Length/height (WHO-adjusted)", "lhfa", adjusted_height, "cm"),
        ("BMI", "bfa", bmi, "kg/m²"),
        ("Head circumference", "hcfa", head_circ, "cm"),
    ]
    for label, indicator, value, unit in checks:
        low, high = who_measurement_limits(indicator, sex, age_days)
        if value < low or value > high:
            errors.append(
                f"{label} is outside the WHO plausible recording limits for this child's age and sex "
                f"({low:.2f}–{high:.2f} {unit}). Entered: {value:.2f} {unit}. "
                "Please verify the dates, units, measurement technique and value."
            )
    return errors
