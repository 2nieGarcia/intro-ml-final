"""Preprocessing, tuning and evaluation shared by the UCI and NHANES notebooks.

All three models get the same preprocessing, the same folds and the same primary metric,
as written in documentation/preregistration.md.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
BETA = 2

# F2 for the CKD class (label 1). Primary metric for tuning, thresholds and model selection.
F2 = make_scorer(fbeta_score, beta=BETA)

# The same 5 folds are used everywhere
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Simplest first. Used by the tie-break rule.
SIMPLICITY_ORDER = ["Logistic Regression", "Random Forest", "Gradient Boosting"]

METRICS = ["f2", "recall", "precision", "specificity", "f1", "accuracy", "roc_auc", "pr_auc", "brier"]


def make_preprocessor(numeric, binary, categorical):
    """Same steps for every model. Fit only on whatever data the pipeline is fit on."""
    transformers = []
    if numeric:
        numeric_steps = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_steps, numeric))
    if binary:
        transformers.append(("binary", SimpleImputer(strategy="most_frequent"), binary))
    if categorical:
        categorical_steps = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_steps, categorical))
    return ColumnTransformer(transformers)


def make_models():
    """The three algorithms and the hyperparameter values tried for each."""
    return {
        "Logistic Regression": (
            LogisticRegression(class_weight="balanced", max_iter=1000),
            {"C": [0.001, 0.01, 0.1, 1, 10]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE),
            {"max_depth": [5, 10, None], "min_samples_leaf": [5, 20]},
        ),
        "Gradient Boosting": (
            HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {"learning_rate": [0.05, 0.1], "max_depth": [3, None], "max_iter": [100, 300]},
        ),
    }


def build_pipeline(preprocessor, model):
    """Preprocessing plus model, wrapped so the decision threshold is tuned for F2.

    TunedThresholdClassifierCV runs its own cross-validation on the data it is given,
    so the threshold is never chosen on the fold it is scored on.
    """
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    return TunedThresholdClassifierCV(pipeline, scoring=F2, cv=CV)


def tune(preprocessor, model, grid, X, y):
    """Grid search over the model's hyperparameters, scored by F2 on the shared folds."""
    param_grid = {f"estimator__model__{name}": values for name, values in grid.items()}
    search = GridSearchCV(build_pipeline(preprocessor, model), param_grid, scoring=F2, cv=CV, n_jobs=-1)
    search.fit(X, y)
    return search


def score_predictions(y_true, y_pred, proba=None):
    """All reported metrics for one set of predictions. CKD (1) is the positive class."""
    scores = {
        "f2": fbeta_score(y_true, y_pred, beta=BETA, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "brier": np.nan,
    }
    # rule-based baselines have no probabilities, so these stay empty for them
    if proba is not None:
        scores["roc_auc"] = roc_auc_score(y_true, proba)
        scores["pr_auc"] = average_precision_score(y_true, proba)
        scores["brier"] = brier_score_loss(y_true, proba)
    return scores


def cross_validate_model(estimator, X, y):
    """Fit on each training fold, score on its validation fold.

    Returns one row of scores per fold, plus the out-of-fold probabilities and predictions,
    so every training row gets a prediction from a model that never saw it.
    """
    rows = []
    oof_proba = pd.Series(np.nan, index=X.index)
    oof_pred = pd.Series(0, index=X.index)
    for fold, (train_idx, val_idx) in enumerate(CV.split(X, y), start=1):
        model = clone(estimator).fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        pred = model.predict(X.iloc[val_idx])
        scores = score_predictions(y.iloc[val_idx], pred, proba)
        scores["fold"] = fold
        scores["threshold"] = model.best_threshold_
        rows.append(scores)
        oof_proba.iloc[val_idx] = proba
        oof_pred.iloc[val_idx] = pred
    return pd.DataFrame(rows), oof_proba, oof_pred


def cross_validate_rule(rule_predictions, y):
    """Score a fixed rule (no fitting) on the same validation folds as the models."""
    rows = []
    for fold, (_, val_idx) in enumerate(CV.split(rule_predictions.to_frame(), y), start=1):
        scores = score_predictions(y.iloc[val_idx], rule_predictions.iloc[val_idx])
        scores["fold"] = fold
        rows.append(scores)
    return pd.DataFrame(rows)


def summarize(fold_scores):
    """Mean and standard deviation across folds for each model. fold_scores: name -> DataFrame."""
    rows = {}
    for name, scores in fold_scores.items():
        row = {}
        for metric in METRICS:
            row[f"{metric} mean"] = scores[metric].mean()
            row[f"{metric} sd"] = scores[metric].std()
        rows[name] = row
    return pd.DataFrame(rows).T


def select_model(fold_scores):
    """Pre-registered rule: highest mean F2, but if the top two differ by less than one
    standard deviation of the better model's fold F2, take the simpler of the two.
    """
    means = {name: scores["f2"].mean() for name, scores in fold_scores.items()}
    ranked = sorted(means, key=means.get, reverse=True)
    best, second = ranked[0], ranked[1]
    gap = means[best] - means[second]
    sd = fold_scores[best]["f2"].std()
    if gap < sd and SIMPLICITY_ORDER.index(second) < SIMPLICITY_ORDER.index(best):
        reason = (f"{best} had the highest mean F2 ({means[best]:.3f}), but {second} ({means[second]:.3f}) "
                  f"is within one SD ({sd:.3f}), so the simpler model, {second}, is selected.")
        return second, reason
    if gap < sd:
        reason = (f"{best} had the highest mean F2 ({means[best]:.3f}) and is also simpler than "
                  f"{second} ({means[second]:.3f}), so it is selected.")
    else:
        reason = (f"{best} had the highest mean F2 ({means[best]:.3f}), more than one SD ({sd:.3f}) "
                  f"above {second} ({means[second]:.3f}), so it is selected.")
    return best, reason


def risk_table(proba, y, n_bins=10):
    """Observed CKD rate for groups of people with similar scores (equal-sized groups).

    class_weight="balanced" pushes the raw scores up, so a score of 0.6 does not mean a 60%
    chance. This table turns a score into what actually happened to similar people.
    """
    bins = pd.qcut(proba, q=n_bins, duplicates="drop")
    table = pd.DataFrame({"proba": proba, "ckd": y}).groupby(bins, observed=True).agg(
        low=("proba", "min"), high=("proba", "max"), people=("ckd", "size"), observed_rate=("ckd", "mean")
    )
    return table.reset_index(drop=True)
