import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pytest

import who_zscore as who
from growth.validation import validate_who_measurement_limits, who_measurement_limits


def test_typical_six_month_boy_is_accepted():
    assert validate_who_measurement_limits("Boy", 183, 7.9, 67.6, who.RECUMBENT, 43.3) == []


@pytest.mark.parametrize("w,h,hc,what", [
    (17.4, 67.6, 43.3, "Weight"),      # 7.9 kg entered in pounds
    (7.9, 26.6, 43.3, "Length"),       # height in inches
    (7.9, 67.6, 17.0, "Head"),         # head circumference in inches
])
def test_typical_unit_mistakes_are_rejected(w, h, hc, what):
    errs = validate_who_measurement_limits("Boy", 183, w, h, who.RECUMBENT, hc)
    assert any(what in e for e in errs)


def test_limits_are_ordered_and_age_dependent():
    lo6, hi6 = who_measurement_limits("wfa", "Girl", 183)
    lo48, hi48 = who_measurement_limits("wfa", "Girl", 1461)
    assert lo6 < hi6 and lo48 < hi48 and hi48 > hi6
