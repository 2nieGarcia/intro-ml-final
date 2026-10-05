"""Turns what a health worker enters into the model's 13 features, with input checks.

The Streamlit form and the CSV upload both go through validate_patient, so they follow
exactly the same rules. A record is a dict with these keys (the CSV column names):

    age, sex, weight_kg, height_cm, waist_cm, sbp, dbp, told_htn, bp_meds,
    high_chol, any_cvd, fam_diabetes, diabetes, smoking

Text answers use: sex = female / male, yes-no questions = yes / no / dont_know,
diabetes = no / borderline / yes / dont_know, smoking = never / former / current / dont_know.
waist_cm is optional. A blank answer to a yes-no question counts as dont_know.
"""

import numpy as np
import pandas as pd

import nhanes

INPUT_COLUMNS = [
    "age", "sex", "weight_kg", "height_cm", "waist_cm", "sbp", "dbp", "told_htn", "bp_meds",
    "high_chol", "any_cvd", "fam_diabetes", "diabetes", "smoking",
]

# What the form accepts, with the label used in error messages
NUMBER_INPUTS = {
    "age": ("Age", "years", 20, 120),
    "weight_kg": ("Weight", "kg", 20, 300),
    "height_cm": ("Height", "cm", 100, 230),
    "waist_cm": ("Waist circumference", "cm", 40, 200),
    "sbp": ("Systolic blood pressure", "mmHg", 60, 260),
    "dbp": ("Diastolic blood pressure", "mmHg", 30, 150),
}
OPTIONAL_NUMBERS = ["waist_cm"]

# The oldest age in the training data (NHANES records everyone 80 or older as 80)
MODEL_MAX_AGE = 80

YES_NO_INPUTS = {
    "told_htn": "Told they have high blood pressure",
    "bp_meds": "Taking blood pressure medicine",
    "high_chol": "Told they have high cholesterol",
    "any_cvd": "Heart disease or stroke",
    "fam_diabetes": "Family history of diabetes",
}
YES_NO = {"yes": 1, "no": 0, "dont_know": np.nan}
DIABETES = {"no": "no", "borderline": "borderline", "yes": "yes", "dont_know": np.nan}
SMOKING = {"never": "never", "former": "former", "current": "current", "dont_know": np.nan}
SEX = {"female": 1, "male": 0}


def is_blank(value):
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == ""


def read_number(record, key, errors):
    """Returns the number, or None if blank. Adds a readable error for bad input."""
    label, unit, low, high = NUMBER_INPUTS[key]
    value = record.get(key)
    if is_blank(value):
        if key not in OPTIONAL_NUMBERS:
            errors.append(f"{label} is required.")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be a number, got '{value}'.")
        return None
    if not low <= number <= high:
        errors.append(f"{label} must be between {low} and {high} {unit}, got {number:g}.")
        return None
    return number


def read_choice(record, key, options, label, errors, blank_means=None):
    value = record.get(key)
    if is_blank(value):
        if blank_means is None:
            errors.append(f"{label} is required.")
            return None
        value = blank_means
    value = str(value).strip().lower().replace("'", "").replace(" ", "_")
    if value not in options:
        errors.append(f"{label} must be one of: {', '.join(options)}. Got '{record.get(key)}'.")
        return None
    return options[value]


def validate_patient(record):
    """Check one patient's inputs.

    Returns (features, errors, notes). features is a one-row DataFrame with the model's
    13 columns, or None if there are errors. notes are things the user should know but
    that do not stop the prediction.
    """
    errors, notes = [], []
    numbers = {key: read_number(record, key, errors) for key in NUMBER_INPUTS}

    if numbers["age"] is not None and numbers["age"] > MODEL_MAX_AGE:
        notes.append(f"Age {numbers['age']:g} is treated as {MODEL_MAX_AGE}, the oldest age in the training data.")
        numbers["age"] = MODEL_MAX_AGE

    bmi = None
    if numbers["weight_kg"] is not None and numbers["height_cm"] is not None:
        bmi = numbers["weight_kg"] / (numbers["height_cm"] / 100) ** 2
        low, high = nhanes.PLAUSIBLE_RANGES["bmi"]
        if not low <= bmi <= high:
            errors.append(f"Weight and height give a BMI of {bmi:.1f}, which is not possible. Check both values.")

    if numbers["sbp"] is not None and numbers["dbp"] is not None and numbers["dbp"] >= numbers["sbp"]:
        errors.append("Diastolic blood pressure must be lower than systolic blood pressure.")

    if numbers["waist_cm"] is None and not errors:
        notes.append("Waist was not entered, so the typical value from the training data is used.")

    female = read_choice(record, "sex", SEX, "Sex", errors)
    answers = {key: read_choice(record, key, YES_NO, label, errors, blank_means="dont_know")
               for key, label in YES_NO_INPUTS.items()}
    diabetes = read_choice(record, "diabetes", DIABETES, "Diabetes", errors, blank_means="dont_know")
    smoking = read_choice(record, "smoking", SMOKING, "Smoking", errors, blank_means="dont_know")

    if errors:
        return None, errors, notes

    features = {
        "age": numbers["age"],
        "bmi": bmi,
        "waist": np.nan if numbers["waist_cm"] is None else numbers["waist_cm"],
        "sbp": numbers["sbp"],
        "dbp": numbers["dbp"],
        "female": female,
        **answers,
        "diabetes": diabetes,
        "smoking": smoking,
    }
    frame = pd.DataFrame([features])[nhanes.FEATURES]
    frame[nhanes.CATEGORICAL_FEATURES] = frame[nhanes.CATEGORICAL_FEATURES].astype(object)
    return frame, errors, notes


def validate_csv(table):
    """Validate every row of an uploaded table.

    Returns (features for the valid rows, list of (row number, errors) for invalid rows).
    Row numbers start at 1, the way a spreadsheet user would count them.
    """
    missing_columns = [c for c in INPUT_COLUMNS if c not in table.columns]
    if missing_columns:
        raise ValueError(f"The file is missing these columns: {', '.join(missing_columns)}")
    valid, problems = [], []
    for row_number, record in enumerate(table.to_dict("records"), start=1):
        features, errors, _ = validate_patient(record)
        if errors:
            problems.append((row_number, errors))
        else:
            features.index = [row_number]
            valid.append(features)
    features = pd.concat(valid) if valid else pd.DataFrame(columns=nhanes.FEATURES)
    return features, problems


def observed_rate(score, table):
    """Share of training patients with a similar score who had CKD (from model_card.json)."""
    for group in table:
        if score <= group["high"]:
            return group["observed_rate"]
    return table[-1]["observed_rate"]


def predict(model, card, features):
    """Score, referral decision and observed rate for each row of features."""
    scores = model.predict_proba(features)[:, 1]
    refer = model.predict(features) == 1
    return pd.DataFrame({
        "score": scores,
        "refer": refer,
        "observed_rate": [observed_rate(s, card["risk_table"]) for s in scores],
    }, index=features.index)
