import os
import sys
import types

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(PROJECT_ROOT, "app.py")


@pytest.fixture(autouse=True)
def stub_db(monkeypatch):
    """Replace db.py (needs MySQL secrets) with an empty in-memory SQLite session factory."""
    eng = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    stub = types.ModuleType("db")
    stub.SessionLocal = sessionmaker(bind=eng, expire_on_commit=False)
    monkeypatch.setitem(sys.modules, "db", stub)


def page_text(at: AppTest) -> str:
    parts = []
    for coll in (at.markdown, at.caption, at.info, at.warning, at.error):
        parts += [el.value for el in coll]
    return "\n".join(parts)


def test_app_loads_without_exceptions():
    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()
    assert not at.exception


def test_validation_errors_stop_the_calculation():
    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()
    at.text_input[0].set_value("")          # queued; applied together with the submit click (widget is inside a form)
    at.button[0].click().run()
    txt = page_text(at)
    assert "Patient Code is required." in txt
    assert "Weight is required and must be > 0 kg." in txt
    assert "Nothing was saved" in txt
