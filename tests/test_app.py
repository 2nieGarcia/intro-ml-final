from streamlit.testing.v1 import AppTest

from conftest import PROJECT_ROOT

APP = str(PROJECT_ROOT / "app" / "streamlit_app.py")


def fill(at, **values):
    for key, value in values.items():
        widget = at.number_input(key=key) if key in {"age", "weight", "height", "sbp", "dbp", "waist"} else at.selectbox(key=key)
        widget.set_value(value)


def submit(at):
    at.button[0].click()
    return at.run(timeout=60)


def valid_patient(at, **changes):
    values = dict(age=62, sex="Male", weight=80.0, height=170.0, sbp=150, dbp=90, told_htn="Yes",
                  bp_meds="Yes", diabetes="Yes", any_cvd="No", high_chol="Yes", fam_diabetes="Don't know",
                  smoking="Former")
    values.update(changes)
    fill(at, **values)


def test_app_loads_without_errors():
    at = AppTest.from_file(APP).run(timeout=60)
    assert not at.exception
    assert len(at.error) == 0


def test_valid_patient_gets_a_recommendation():
    at = AppTest.from_file(APP).run(timeout=60)
    valid_patient(at)
    at = submit(at)
    assert not at.exception
    assert len(at.error) == 0
    assert len(at.warning) + len(at.success) == 1


def test_out_of_range_age_shows_error_not_crash():
    at = AppTest.from_file(APP).run(timeout=60)
    valid_patient(at, age=12)
    at = submit(at)
    assert not at.exception
    assert "Age" in at.error[0].value


def test_empty_form_lists_required_fields():
    at = AppTest.from_file(APP).run(timeout=60)
    at = submit(at)
    assert not at.exception
    message = at.error[0].value
    for field in ["Age", "Sex", "Weight", "Height", "Systolic"]:
        assert field in message


def test_age_above_80_gives_note():
    at = AppTest.from_file(APP).run(timeout=60)
    valid_patient(at, age=92)
    at = submit(at)
    assert any("80" in info.value for info in at.info)
