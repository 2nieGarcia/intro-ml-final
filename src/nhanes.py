"""NHANES 2017-March 2020: load the raw files, build the cohort, the CKD label and the Tier 3 features.

Every rule in this file comes from documentation/preregistration.md.
The order of use is:
    raw = load_raw(data_dir)
    raw = replace_refused_and_dont_know(raw)
    cohort, flow = build_cohort(raw)
    cohort = add_label(cohort)
    features = build_features(cohort)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# Which columns to keep from each file. P_DEMO comes first so every other file is joined onto it.
RAW_FILES = {
    "P_DEMO": ["RIDAGEYR", "RIAGENDR", "RIDRETH3", "RIDSTATR", "RIDEXPRG"],
    "P_BMX": ["BMXBMI", "BMXWAIST"],
    "P_BPXO": ["BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXODI1", "BPXODI2", "BPXODI3"],
    "P_BPQ": ["BPQ020", "BPQ040A", "BPQ050A", "BPQ080"],
    "P_DIQ": ["DIQ010"],
    "P_SMQ": ["SMQ020", "SMQ040"],
    "P_MCQ": ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F", "MCQ300C"],
    "P_KIQ_U": ["KIQ022", "KIQ025"],
    "P_BIOPRO": ["LBXSCR"],
    "P_ALB_CR": ["URDACT"],
}

# Questionnaire columns where 7 means "refused" and 9 means "don't know"
QUESTION_COLUMNS = [
    "BPQ020", "BPQ040A", "BPQ050A", "BPQ080", "DIQ010", "SMQ020", "SMQ040",
    "MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F", "MCQ300C", "KIQ022", "KIQ025",
]
REFUSED_OR_DONT_KNOW = [7, 9]

# Congestive heart failure, coronary heart disease, angina, heart attack, stroke
CVD_COLUMNS = ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]

NUMERIC_FEATURES = ["age", "bmi", "waist", "sbp", "dbp"]
BINARY_FEATURES = ["female", "told_htn", "bp_meds", "high_chol", "any_cvd", "fam_diabetes"]
CATEGORICAL_FEATURES = ["diabetes", "smoking"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
TARGET = "ckd"

# Data dictionary for the features: (NHANES source, unit or coding, meaning)
FEATURE_INFO = {
    "age": ("RIDAGEYR", "years", "Age at screening. 80 means 80 or older."),
    "bmi": ("BMXBMI", "kg/m2", "Body mass index"),
    "waist": ("BMXWAIST", "cm", "Waist circumference"),
    "sbp": ("BPXOSY1-3", "mmHg", "Systolic blood pressure, mean of up to 3 readings"),
    "dbp": ("BPXODI1-3", "mmHg", "Diastolic blood pressure, mean of up to 3 readings"),
    "female": ("RIAGENDR", "1 = female, 0 = male", "Sex"),
    "told_htn": ("BPQ020", "1 = yes, 0 = no", "Ever told by a doctor they have high blood pressure"),
    "bp_meds": ("BPQ040A, BPQ050A", "1 = yes, 0 = no", "Currently taking prescribed blood pressure medicine"),
    "high_chol": ("BPQ080", "1 = yes, 0 = no", "Ever told by a doctor they have high cholesterol"),
    "any_cvd": ("MCQ160B-F", "1 = yes, 0 = no", "Ever told they had heart failure, coronary heart disease, angina, heart attack or stroke"),
    "fam_diabetes": ("MCQ300C", "1 = yes, 0 = no", "A close relative has diabetes"),
    "diabetes": ("DIQ010", "no / borderline / yes", "Ever told by a doctor they have diabetes"),
    "smoking": ("SMQ020, SMQ040", "never / former / current", "Never = under 100 cigarettes in their life"),
}

# Values outside these ranges are physically impossible or almost certainly entry errors
PLAUSIBLE_RANGES = {
    "age": (20, 80),
    "bmi": (10, 100),
    "waist": (40, 200),
    "sbp": (60, 260),
    "dbp": (30, 150),
}


def load_raw(data_dir):
    """Read the 10 NHANES files and join them onto the demographics file by participant ID."""
    data_dir = Path(data_dir)
    merged = None
    for name, columns in RAW_FILES.items():
        table = pd.read_sas(data_dir / f"{name}.xpt", format="xport")
        table = table[["SEQN"] + columns]
        if merged is None:
            merged = table
        else:
            merged = merged.merge(table, on="SEQN", how="left")
    merged["SEQN"] = merged["SEQN"].astype(int)
    return merged


def replace_refused_and_dont_know(raw):
    """Codes 7 and 9 are not real answers, so they become missing."""
    raw = raw.copy()
    raw[QUESTION_COLUMNS] = raw[QUESTION_COLUMNS].replace(REFUSED_OR_DONT_KNOW, np.nan)
    return raw


def build_cohort(raw):
    """Apply the inclusion rules in order.

    Returns the cohort and a table showing how many people were left after each rule.
    """
    df = raw
    flow = [("All participants", len(df))]

    df = df[df["RIDAGEYR"] >= 20]
    flow.append(("Age 20 or older", len(df)))

    df = df[df["RIDSTATR"] == 2]
    flow.append(("Examined at the mobile exam center", len(df)))

    # RIDEXPRG is only recorded for women 20-44, so it is missing for everyone else and they are kept
    df = df[df["RIDEXPRG"] != 1]
    flow.append(("Not pregnant", len(df)))

    df = df.dropna(subset=["LBXSCR", "URDACT"])
    flow.append(("Has serum creatinine and urine ACR", len(df)))

    df = df[(df["KIQ022"] != 1) & (df["KIQ025"] != 1)]
    flow.append(("Not told of weak or failing kidneys, no dialysis", len(df)))

    flow = pd.DataFrame(flow, columns=["step", "remaining"])
    flow["removed"] = (-flow["remaining"].diff()).fillna(0).astype(int)
    return df.copy(), flow


def egfr_ckd_epi_2021(scr, age, female):
    """eGFR in mL/min/1.73m2 from the CKD-EPI 2021 equation without race (Inker et al., 2021).

    scr is serum creatinine in mg/dL, age is in years, female is True or False for each person.
    """
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    ratio = scr / kappa
    egfr = 142 * np.minimum(ratio, 1) ** alpha * np.maximum(ratio, 1) ** -1.200 * 0.9938 ** age
    return egfr * np.where(female, 1.012, 1.0)


def add_label(cohort):
    """CKD = 1 if eGFR < 60 or urine albumin-to-creatinine ratio >= 30 mg/g."""
    cohort = cohort.copy()
    female = cohort["RIAGENDR"] == 2
    cohort["egfr"] = egfr_ckd_epi_2021(cohort["LBXSCR"], cohort["RIDAGEYR"], female)
    cohort["low_egfr"] = (cohort["egfr"] < 60).astype(int)
    cohort["albuminuria"] = (cohort["URDACT"] >= 30).astype(int)
    cohort[TARGET] = ((cohort["low_egfr"] == 1) | (cohort["albuminuria"] == 1)).astype(int)
    return cohort


def yes_no(column):
    """NHANES 1 = yes, 2 = no. Becomes 1 / 0, anything else missing."""
    return column.map({1: 1, 2: 0})


def any_yes(columns):
    """1 if any column is yes, 0 if all are no, missing if unsure."""
    result = pd.Series(np.nan, index=columns.index)
    result.loc[(columns == 2).all(axis=1)] = 0
    result.loc[(columns == 1).any(axis=1)] = 1
    return result


def smoking_status(smoked_100, smokes_now):
    """never / former / current from SMQ020 and SMQ040.

    SMQ040 is only asked if SMQ020 = 1, so a never smoker has a blank SMQ040 by design.
    """
    status = pd.Series(np.nan, index=smoked_100.index, dtype=object)
    status.loc[smoked_100 == 2] = "never"
    status.loc[smokes_now == 3] = "former"
    status.loc[smokes_now.isin([1, 2])] = "current"
    return status


def bp_medication(told_htn, told_take_meds, taking_meds):
    """1 / 0 for taking blood pressure medicine.

    BPQ050A is only asked if BPQ040A = 1, and BPQ040A only if BPQ020 = 1.
    A blank caused by that skip pattern means "not taking it", so it becomes 0 instead of missing.
    """
    result = yes_no(taking_meds)
    result.loc[(told_htn == 2) | (told_take_meds == 2)] = 0
    return result


def build_features(cohort):
    """Turn the raw NHANES columns into the Tier 3 features (names match FEATURES)."""
    features = pd.DataFrame(index=cohort.index)
    features["age"] = cohort["RIDAGEYR"]
    features["bmi"] = cohort["BMXBMI"]
    features["waist"] = cohort["BMXWAIST"]
    # average of whichever of the 3 readings were taken, missing if none were
    features["sbp"] = cohort[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1)
    features["dbp"] = cohort[["BPXODI1", "BPXODI2", "BPXODI3"]].mean(axis=1)
    features["female"] = (cohort["RIAGENDR"] == 2).astype(int)
    features["told_htn"] = yes_no(cohort["BPQ020"])
    features["bp_meds"] = bp_medication(cohort["BPQ020"], cohort["BPQ040A"], cohort["BPQ050A"])
    features["high_chol"] = yes_no(cohort["BPQ080"])
    features["any_cvd"] = any_yes(cohort[CVD_COLUMNS])
    features["fam_diabetes"] = yes_no(cohort["MCQ300C"])
    features["diabetes"] = cohort["DIQ010"].map({1: "yes", 2: "no", 3: "borderline"}).astype(object)
    features["smoking"] = smoking_status(cohort["SMQ020"], cohort["SMQ040"])
    return features


def add_split(df, test_size=0.2):
    """Mark each row as train or test: 80/20, stratified on the label, random_state 42."""
    train_index, test_index = train_test_split(
        df.index, test_size=test_size, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    df = df.copy()
    df["split"] = "train"
    df.loc[test_index, "split"] = "test"
    return df
