import numpy as np
import pandas as pd
import pytest

import modeling


def test_scores_match_worked_example():
    # 100 patients, 13 with CKD. The model refers 30 and catches 10 of the 13.
    y_true = np.array([1] * 13 + [0] * 87)
    y_pred = np.array([1] * 10 + [0] * 3 + [1] * 20 + [0] * 67)
    scores = modeling.score_predictions(y_true, y_pred)
    assert scores["recall"] == pytest.approx(10 / 13)
    assert scores["precision"] == pytest.approx(10 / 30)
    assert scores["specificity"] == pytest.approx(67 / 87)
    assert scores["f2"] == pytest.approx(0.610, abs=0.001)
    assert np.isnan(scores["roc_auc"])


def test_refer_everyone_f2_floor():
    y_true = np.array([1] * 13 + [0] * 87)
    scores = modeling.score_predictions(y_true, np.ones(100, dtype=int))
    p = 0.13
    assert scores["f2"] == pytest.approx(5 * p / (4 * p + 1))


def folds(values):
    return pd.DataFrame({"f2": values})


def test_tie_break_picks_simpler_model_when_within_one_sd():
    scores = {"Logistic Regression": folds([0.60, 0.61, 0.62, 0.60, 0.61]),
              "Gradient Boosting": folds([0.60, 0.64, 0.62, 0.66, 0.60])}
    name, _ = modeling.select_model(scores)
    assert name == "Logistic Regression"


def test_clear_winner_is_kept():
    scores = {"Logistic Regression": folds([0.50, 0.51, 0.50, 0.51, 0.50]),
              "Gradient Boosting": folds([0.60, 0.61, 0.60, 0.61, 0.60])}
    name, _ = modeling.select_model(scores)
    assert name == "Gradient Boosting"
