"""
tabular/reliability.py - how much the test scores can be trusted.

A test score is measured on one sample of examples. A different sample
would give a slightly different number, so a single score hides how sure
we are. This module answers three questions:

    Is the model better than guessing?
        A baseline that ignores the inputs (always the most common class, or
        always the average value) sets the floor every model must beat.
    How precise is each score?
        Bootstrap: draw the test set again with replacement 1000 times,
        re-score each draw, and keep the middle 95% of the scores. That
        range is the 95% confidence interval.
    Is the best model really better than the others?
        Paired tests compare two models on the same test examples.
        McNemar's test (classification) looks only at the examples where
        exactly one of the two models is right. The Wilcoxon signed-rank
        test (regression) compares the sizes of their errors row by row.
        A p-value below 0.05 means a gap that big would rarely appear by
        chance if the models were equally good.

It also draws learning curves: the score as the training set grows, which
shows whether more data would help and whether a model overfits.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import learning_curve

from pxlab.common import RANDOM_STATE, save_figure
from pxlab.tabular.evaluate import (
    classification_scores_from_predictions,
    positive_probabilities,
    regression_scores_from_predictions,
)
from pxlab.tabular.train import cross_validation_folds, scorer_for

BASELINE_NAME = "Baseline (ignores the inputs)"
BOOTSTRAP_ROUNDS = 1000
CONFIDENCE_LEVEL = 0.95
SIGNIFICANCE_LEVEL = 0.05
TRAINING_SHARES = np.linspace(0.15, 1.0, 6)


def headline_scores(spec):
    """
    The scores the reliability tables report.

    Input:
        spec: the project SPEC.
    Returns:
        A list of score names (keys of the evaluate.py score dictionaries).
    """
    if spec["task"] == "regression":
        return ["MAE", "R²"]
    if spec.get("positive_class") is not None:
        return ["Accuracy", "F1", "ROC-AUC"]
    return ["Accuracy", "F1"]


def fit_baseline(spec, train_features, train_target):
    """
    A model that ignores the inputs: it always predicts the most common
    class (classification) or the average value (regression).

    Inputs:
        spec: the project SPEC.
        train_features, train_target: the training data.
    Returns:
        The fitted baseline model.
    """
    if spec["task"] == "classification":
        baseline = DummyClassifier(strategy="most_frequent")
    else:
        baseline = DummyRegressor(strategy="mean")
    return baseline.fit(train_features, train_target)


def test_predictions(models, test_features, spec):
    """
    Every model's predictions on the test set, made once so the bootstrap
    can re-score them without predicting again.

    Inputs:
        models: model name -> fitted pipeline (the baseline included).
        test_features: the test inputs.
        spec: the project SPEC.
    Returns:
        model name -> (predictions, positive-class probabilities or None).
    """
    predictions = {}
    for model_name, pipeline in models.items():
        probabilities = None
        if spec.get("positive_class") is not None:
            probabilities = positive_probabilities(pipeline, test_features, spec["positive_class"])
        predictions[model_name] = (np.asarray(pipeline.predict(test_features)), probabilities)
    return predictions


def scores_for(spec, target, predictions, probabilities):
    """
    The score dictionary for already-made predictions.

    Inputs:
        spec: the project SPEC.
        target: the real values.
        predictions: the predictions.
        probabilities: positive-class probabilities or None.
    Returns:
        A dictionary of scores.
    """
    if spec["task"] == "regression":
        return regression_scores_from_predictions(target, predictions)
    return classification_scores_from_predictions(target, predictions, spec.get("positive_class"), probabilities)


def bootstrap_intervals(predictions, test_target, spec, rounds=BOOTSTRAP_ROUNDS):
    """
    A 95% confidence interval for every headline score of every model.

    The same resampled rows are used for every model, so the intervals are
    comparable with each other.

    Inputs:
        predictions: from test_predictions().
        test_target: the real test values.
        spec: the project SPEC.
        rounds: how many bootstrap samples.
    Returns:
        A long DataFrame: Model, Score, Test value, Low, High.
    """
    target = np.asarray(test_target)
    generator = np.random.default_rng(RANDOM_STATE)
    samples = [generator.integers(0, len(target), len(target)) for _ in range(rounds)]
    tail = (1 - CONFIDENCE_LEVEL) / 2 * 100
    rows = []
    for model_name, (predicted, probabilities) in predictions.items():
        point = scores_for(spec, target, predicted, probabilities)
        draws = {name: [] for name in headline_scores(spec) if name in point}
        for rows_drawn in samples:
            sample_probabilities = None if probabilities is None else probabilities[rows_drawn]
            scores = scores_for(spec, target[rows_drawn], predicted[rows_drawn], sample_probabilities)
            for name in draws:
                draws[name].append(scores[name])
        for name, values in draws.items():
            low, high = np.percentile(values, [tail, 100 - tail])
            rows.append({"Model": model_name, "Score": name, "Test value": point[name], "Low": low, "High": high})
    return pd.DataFrame(rows)


def plot_intervals(intervals, baseline, filename):
    """
    Each model's headline scores as a dot with its 95% confidence interval.
    Overlapping bars mean the test set can't tell the models apart.

    Inputs:
        intervals: from bootstrap_intervals().
        baseline: the baseline model's name (drawn in grey).
        filename: the image file name.
    Returns:
        The saved path.
    """
    score_names = list(dict.fromkeys(intervals["Score"]))
    figure, axes = plt.subplots(
        1, len(score_names), figsize=(4.6 * len(score_names), 0.5 * intervals["Model"].nunique() + 2)
    )
    axes = np.atleast_1d(axes)
    for axis, score_name in zip(axes, score_names):
        part = intervals[intervals["Score"] == score_name].iloc[::-1]
        positions = np.arange(len(part))
        for position, row in zip(positions, part.itertuples(index=False)):
            colour = "grey" if row.Model == baseline else "#4C72B0"
            axis.errorbar(
                row[2],  # the test value
                position,
                xerr=[[row[2] - row.Low], [row.High - row[2]]],
                fmt="o",
                color=colour,
                capsize=4,
            )
        axis.set_yticks(positions, part["Model"] if axis is axes[0] else [""] * len(part))
        axis.set_xlabel(score_name)
        axis.set_title(score_name)
    figure.suptitle(f"Test scores with {CONFIDENCE_LEVEL:.0%} bootstrap confidence intervals")
    figure.tight_layout()
    return save_figure(figure, filename)


def paired_test(spec, target, predictions_a, predictions_b):
    """
    Compare two models on the same test examples.

    Classification, McNemar's test: count the examples only model A gets
    right and those only model B gets right. If the models were equally
    good, each such example would be a coin flip between A and B.
    Regression, Wilcoxon signed-rank test: for each example, which model's
    error is smaller, weighted by how much smaller.

    Inputs:
        spec: the project SPEC.
        target: the real test values.
        predictions_a, predictions_b: the two models' predictions.
    Returns:
        (examples where A is better, examples where B is better, p-value).
    """
    target = np.asarray(target)
    if spec["task"] == "classification":
        a_right = predictions_a == target
        b_right = predictions_b == target
        a_wins = int(np.sum(a_right & ~b_right))
        b_wins = int(np.sum(b_right & ~a_right))
        if a_wins + b_wins == 0:
            return a_wins, b_wins, 1.0  # they never disagree
        return a_wins, b_wins, binomtest(min(a_wins, b_wins), a_wins + b_wins, 0.5).pvalue
    a_errors = np.abs(target - predictions_a)
    b_errors = np.abs(target - predictions_b)
    a_wins = int(np.sum(a_errors < b_errors))
    b_wins = int(np.sum(b_errors < a_errors))
    if np.all(a_errors == b_errors):
        return a_wins, b_wins, 1.0  # identical errors
    return a_wins, b_wins, wilcoxon(a_errors, b_errors).pvalue


def significance_table(predictions, test_target, spec, best):
    """
    The best model against every other model and the baseline.

    Inputs:
        predictions: from test_predictions().
        test_target: the real test values.
        spec: the project SPEC.
        best: the best model's name.
    Returns:
        A DataFrame: one row per comparison.
    """
    test_name = "McNemar" if spec["task"] == "classification" else "Wilcoxon signed-rank"
    rows = []
    for other in predictions:
        if other == best:
            continue
        best_wins, other_wins, p_value = paired_test(spec, test_target, predictions[best][0], predictions[other][0])
        verdict = "Real difference" if p_value < SIGNIFICANCE_LEVEL else "Could be chance"
        rows.append(
            {
                "Best model": best,
                "Compared with": other,
                "Best is better on": best_wins,
                "Other is better on": other_wins,
                "Test": test_name,
                "p-value": round(float(p_value), 4),
                "Verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def learning_curves(results, train_features, train_target, spec):
    """
    Cross-validated score of every model trained on growing shares of the
    training set, with the best settings found by the grid search.

    Inputs:
        results: from train.train_models().
        train_features, train_target: the training data.
        spec: the project SPEC.
    Returns:
        model name -> (training sizes, training scores, validation scores),
        scores as arrays of shape (sizes, folds). Errors that
        scikit-learn negates (neg_ scorings) are made positive again.
    """
    sign = -1 if spec["scoring"].startswith("neg_") else 1
    curves = {}
    for model_name, result in results.items():
        sizes, train_scores, validation_scores = learning_curve(
            clone(result["pipeline"]),
            train_features,
            train_target,
            train_sizes=TRAINING_SHARES,
            cv=cross_validation_folds(spec),
            scoring=scorer_for(spec),
            # Time-ordered data must keep its order; otherwise shuffle so the
            # small training sets still contain every class.
            shuffle=not spec.get("time_order", False),
            random_state=RANDOM_STATE,
            error_score=np.nan,
            n_jobs=-1,
        )
        curves[model_name] = (sizes, sign * train_scores, sign * validation_scores)
    return curves


def learning_curve_table(curves):
    """
    The learning curves as a table: mean scores per training size.

    Input:
        curves: from learning_curves().
    Returns:
        A DataFrame: Model, Training rows, Training score, Validation score.
    """
    rows = []
    for model_name, (sizes, train_scores, validation_scores) in curves.items():
        for size, train_row, validation_row in zip(sizes, train_scores, validation_scores):
            rows.append(
                {
                    "Model": model_name,
                    "Training rows": int(size),
                    "Training score": np.nanmean(train_row),
                    "Validation score": np.nanmean(validation_row),
                }
            )
    return pd.DataFrame(rows)


def plot_learning_curves(curves, score_name, filename):
    """
    One panel per model: training and validation score against training
    size, with a band of one standard deviation across the folds.

    A big gap between the lines means overfitting (the model memorises the
    training rows). A validation line still rising at the right edge means
    more data would likely help.

    Inputs:
        curves: from learning_curves().
        score_name: the y-axis label.
        filename: the image file name.
    Returns:
        The saved path.
    """
    names = list(curves)
    columns = min(3, len(names))
    rows = int(np.ceil(len(names) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.8 * columns, 3.9 * rows), squeeze=False, sharey=True)
    for axis, model_name in zip(axes.flat, names):
        sizes, train_scores, validation_scores = curves[model_name]
        for scores, colour, label in [
            (train_scores, "#C44E52", "Training"),
            (validation_scores, "#4C72B0", "Validation"),
        ]:
            mean = np.nanmean(scores, axis=1)
            spread = np.nanstd(scores, axis=1)
            axis.plot(sizes, mean, marker="o", color=colour, label=label)
            axis.fill_between(sizes, mean - spread, mean + spread, color=colour, alpha=0.15)
        axis.set_title(model_name)
        axis.set_xlabel("Training rows")
        axis.set_ylabel(score_name)
        axis.legend(loc="lower right")
    for axis in list(axes.flat)[len(names) :]:
        axis.set_visible(False)
    figure.suptitle("Learning curves (5-fold cross-validation)")
    figure.tight_layout()
    return save_figure(figure, filename)


def learning_curve_findings(curves, spec, score_name):
    """
    One plain sentence per model on what its learning curve shows.

    Inputs:
        curves: from learning_curves().
        spec: the project SPEC (errors like RMSE are better when lower).
        score_name: the score's name, used in the sentences.
    Returns:
        A list of strings.
    """
    better = -1 if spec["scoring"].startswith("neg_") else 1
    findings = []
    for model_name, (_sizes, train_scores, validation_scores) in curves.items():
        train_mean = np.nanmean(train_scores, axis=1)
        validation_mean = np.nanmean(validation_scores, axis=1)
        gap = better * (train_mean[-1] - validation_mean[-1])
        rise = better * (validation_mean[-1] - validation_mean[-2])
        scale = max(abs(validation_mean[-1]), 1e-9)
        if gap > 0.1 * scale:
            fit = f"overfits: training {score_name} {train_mean[-1]:.3f} against validation {validation_mean[-1]:.3f}"
        else:
            fit = f"generalises well: training and validation {score_name} are close ({gap:+.3f})"
        if rise > 0.01 * scale:
            data = "validation is still rising, so more data would likely help"
        else:
            data = "validation has levelled off, so more data alone would help little"
        findings.append(f"{model_name} {fit}; {data}.")
    return findings
