import pandas as pd

import uci
from conftest import PROJECT_ROOT


def load_clean():
    return uci.clean(pd.read_csv(PROJECT_ROOT / "data" / "original" / "uci" / "chronic_kidney_disease.csv"))


def test_tiers_cover_every_feature_exactly_once():
    assert len(uci.ALL_FEATURES) == 24
    assert len(set(uci.ALL_FEATURES)) == 24


def test_label_is_binary_with_known_counts():
    df = load_clean()
    assert set(df["ckd"].unique()) == {0, 1}
    assert df["ckd"].sum() == 250


def test_no_stray_whitespace_left():
    df = load_clean()
    for column in uci.CATEGORICAL:
        values = df[column].dropna()
        assert (values == values.str.strip()).all()
    assert set(df["dm"].dropna().unique()) == {"yes", "no"}


def test_impossible_values_removed():
    df = load_clean()
    assert df["sod"].min() >= 100
    assert df["pot"].max() <= 15
