"""UCI Chronic Kidney Disease dataset: cleaning and the feature tiers.

The tiers are copied from Section 3 of documentation/preregistration.md.
"""

import numpy as np
import pandas as pd

TARGET = "ckd"

# Section 3 of the pre-registration
TIERS = {
    "tier1": ["sc", "bu", "al", "sg", "rbc"],
    "tier2": ["hemo", "pcv", "rbcc", "ane", "pot", "sod", "pe", "appet"],
    "other_lab": ["su", "bgr", "wbcc", "pc", "pcc", "ba"],
    "tier3": ["age", "bp", "htn", "dm", "cad"],
}

ALL_FEATURES = TIERS["tier1"] + TIERS["tier2"] + TIERS["other_lab"] + TIERS["tier3"]

# The four ablation steps, from all features down to what is known before any lab test
FEATURE_SETS = {
    "All features": ALL_FEATURES,
    "Without Tier 1": TIERS["tier2"] + TIERS["other_lab"] + TIERS["tier3"],
    "Without Tier 1 and 2": TIERS["other_lab"] + TIERS["tier3"],
    "Tier 3 only": TIERS["tier3"],
}

# Text columns. Everything else is numeric.
CATEGORICAL = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]

# Values outside these ranges cannot be real lab results for a living patient
IMPOSSIBLE = {
    "sod": (100, 200),   # sodium, mEq/L
    "pot": (1.5, 15),    # potassium, mEq/L
}


def clean(raw):
    """Fix the known data problems. Returns a new DataFrame with a 0/1 `ckd` column."""
    df = raw.copy()
    for column in CATEGORICAL + ["class"]:
        # some entries have stray tabs, like "ckd\t" and "\tno"
        df[column] = df[column].str.strip().astype(object)
    for column, (low, high) in IMPOSSIBLE.items():
        df.loc[(df[column] < low) | (df[column] > high), column] = np.nan
    df[TARGET] = (df["class"] == "ckd").astype(int)
    return df.drop(columns="class")


def column_types(features):
    """Split a feature list into numeric and categorical, for the preprocessor."""
    numeric = [f for f in features if f not in CATEGORICAL]
    categorical = [f for f in features if f in CATEGORICAL]
    return numeric, categorical
