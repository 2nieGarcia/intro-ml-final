import numpy as np
import pandas as pd
import pytest

import nhanes
from conftest import PROJECT_ROOT


def test_egfr_matches_hand_worked_example():
    # 60 year old man, creatinine 1.4 mg/dL, worked out by hand as 57.5
    egfr = nhanes.egfr_ckd_epi_2021(pd.Series([1.4]), pd.Series([60]), pd.Series([False]))
    assert egfr[0] == pytest.approx(57.5, abs=0.1)


def test_egfr_female_branch():
    # woman at creatinine = kappa (0.7): both ratio terms are 1, so eGFR = 142 * 0.9938^age * 1.012
    egfr = nhanes.egfr_ckd_epi_2021(pd.Series([0.7]), pd.Series([60]), pd.Series([True]))
    assert egfr[0] == pytest.approx(142 * 0.9938 ** 60 * 1.012)


def test_smoking_skip_pattern_blank_means_never():
    smoked_100 = pd.Series([2, 1, 1, 1, np.nan])
    smokes_now = pd.Series([np.nan, 1, 2, 3, np.nan])
    status = nhanes.smoking_status(smoked_100, smokes_now)
    assert status.tolist()[:4] == ["never", "current", "current", "former"]
    assert pd.isna(status.iloc[4])


def test_bp_medication_skip_pattern_blank_means_not_taking():
    told_htn = pd.Series([2, 1, 1, 1])
    told_take = pd.Series([np.nan, 2, 1, 1])
    taking = pd.Series([np.nan, np.nan, 1, 2])
    assert nhanes.bp_medication(told_htn, told_take, taking).tolist() == [0, 0, 1, 0]


def test_any_yes():
    columns = pd.DataFrame({"a": [1, 2, 2, np.nan], "b": [2, 2, np.nan, 1]})
    result = nhanes.any_yes(columns)
    assert result.iloc[0] == 1 and result.iloc[1] == 0 and pd.isna(result.iloc[2]) and result.iloc[3] == 1


def test_cohort_flow_matches_eda():
    raw = nhanes.replace_refused_and_dont_know(nhanes.load_raw(PROJECT_ROOT / "data" / "original" / "nhanes"))
    cohort, flow = nhanes.build_cohort(raw)
    assert flow["remaining"].tolist() == [15560, 9232, 8544, 8457, 7667, 7363]
    labeled = nhanes.add_label(cohort)
    assert labeled["ckd"].sum() == 1217
