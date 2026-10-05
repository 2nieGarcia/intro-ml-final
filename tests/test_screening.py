import numpy as np
import pandas as pd
import pytest

import nhanes
import screening

VALID = {
    "age": 55, "sex": "female", "weight_kg": 70, "height_cm": 160, "waist_cm": 90,
    "sbp": 140, "dbp": 85, "told_htn": "yes", "bp_meds": "yes", "high_chol": "no",
    "any_cvd": "no", "fam_diabetes": "dont_know", "diabetes": "borderline", "smoking": "never",
}


def patient(**changes):
    record = dict(VALID)
    record.update(changes)
    return record


def test_valid_patient_gives_model_features():
    features, errors, _ = screening.validate_patient(VALID)
    assert errors == []
    assert list(features.columns) == nhanes.FEATURES
    assert features.loc[0, "bmi"] == pytest.approx(70 / 1.6 ** 2)
    assert features.loc[0, "female"] == 1
    assert np.isnan(features.loc[0, "fam_diabetes"])


@pytest.mark.parametrize("age, ok", [(19, False), (20, True), (80, True), (95, True), (121, False)])
def test_age_boundaries(age, ok):
    features, errors, notes = screening.validate_patient(patient(age=age))
    assert (errors == []) == ok
    if age == 95:
        assert features.loc[0, "age"] == 80
        assert any("80" in note for note in notes)


@pytest.mark.parametrize("field, value", [
    ("sbp", 59), ("sbp", 261), ("dbp", 29), ("waist_cm", 300), ("age", "fifty"), ("sex", "other"),
    ("diabetes", "maybe"), ("sbp", None), ("weight_kg", ""),
])
def test_invalid_or_missing_inputs_give_errors(field, value):
    features, errors, _ = screening.validate_patient(patient(**{field: value}))
    assert features is None
    assert len(errors) >= 1


def test_diastolic_must_be_below_systolic():
    _, errors, _ = screening.validate_patient(patient(sbp=120, dbp=120))
    assert any("Diastolic" in e for e in errors)


def test_impossible_bmi_is_rejected():
    _, errors, _ = screening.validate_patient(patient(weight_kg=300, height_cm=100))
    assert any("BMI" in e for e in errors)


def test_optional_waist_and_blank_answers():
    features, errors, notes = screening.validate_patient(patient(waist_cm="", smoking="", told_htn=None))
    assert errors == []
    assert np.isnan(features.loc[0, "waist"])
    assert pd.isna(features.loc[0, "smoking"])
    assert any("Waist" in n for n in notes)


def test_csv_reports_bad_rows_and_keeps_good_ones():
    table = pd.DataFrame([VALID, patient(age=10), VALID])
    features, problems = screening.validate_csv(table)
    assert list(features.index) == [1, 3]
    assert problems[0][0] == 2


def test_csv_missing_columns():
    with pytest.raises(ValueError):
        screening.validate_csv(pd.DataFrame([{"age": 50}]))
