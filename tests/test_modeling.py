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


def test_cases_caught_counts_ties_in_proportion():
    # two people share the score 0.8 and one of them has CKD
    score = np.array([0.9, 0.8, 0.8, 0.1])
    y = np.array([1, 0, 1, 0])
    assert modeling.cases_caught(score, y, 1) == pytest.approx(1)
    assert modeling.cases_caught(score, y, 2) == pytest.approx(1.5)  # half of the tied pair
    assert modeling.cases_caught(score, y, 3) == pytest.approx(2)
    assert modeling.cases_caught(score, y, 4) == pytest.approx(2)


def test_cases_caught_for_a_yes_no_rule():
    # a yes/no rule cannot rank: picking 2 of its 4 positives catches half of their 2 cases
    rule = np.array([1, 1, 1, 1, 0, 0])
    y = np.array([1, 0, 1, 0, 1, 0])
    assert modeling.cases_caught(rule, y, 2) == pytest.approx(1)
