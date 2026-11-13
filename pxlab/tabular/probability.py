"""
tabular/probability.py - using the predicted probabilities well (two-class
problems only).

A classifier gives a probability, and predict() turns it into a yes/no by
comparing it with 0.5. Two questions follow:

    Is 0.5 the right cut-off?
        Not always. If missing a sick patient is five times worse than a
        false alarm, the model should say "sick" at a lower probability.
        With honest probabilities the cheapest cut-off is
            cost of a false alarm / (cost of a false alarm + cost of a miss),
        e.g. 1 / (1 + 5) = 0.17. Without costs we pick the cut-off with the
        best F1 instead. The cut-off is chosen on the training data
        (cross-validated predictions), never on the test set, and the test
        set then shows whether it holds up.
    Are the probabilities honest ("calibrated")?
        Of all the cases a model gives 80%, about 80% should really be
        positive. A reliability diagram plots predicted against observed
        frequency; the diagonal is perfect. The Brier score is the mean
        squared gap between probability and outcome (0 is perfect).
        Naive Bayes is famous for being overconfident; Platt scaling or
        isotonic regression can repair it.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import cross_val_predict

from pxlab.common import save_figure
from pxlab.tabular.evaluate import positive_probabilities
from pxlab.tabular.train import cross_validation_folds

THRESHOLDS = np.round(np.arange(0.02, 0.99, 0.01), 2)
NUMBER_OF_BINS = 10


def out_of_fold_probabilities(pipeline, train_features, train_target, spec):
    """
    Positive-class probabilities for every training row, each made by a
    model that did not see that row (5-fold cross-validation). Choosing the
    cut-off on these keeps the test set untouched.

    Inputs:
        pipeline: a fitted pipeline (cloned, its settings are reused).
        train_features, train_target: the training data.
        spec: the project SPEC.
    Returns:
        A numpy array of probabilities.
    """
    probabilities = cross_val_predict(
        clone(pipeline), train_features, train_target, cv=cross_validation_folds(spec), method="predict_proba"
    )
    column = list(np.unique(train_target)).index(spec["positive_class"])
    return probabilities[:, column]


def outcome_counts(is_positive, probabilities, threshold):
    """
    True and false positives and negatives at one cut-off.

    Inputs:
        is_positive: boolean array, True for real positives.
        probabilities: predicted positive-class probabilities.
        threshold: the cut-off.
    Returns:
        A dictionary with TP, FP, FN, TN.
    """
    said_positive = probabilities >= threshold
    return {
        "TP": int(np.sum(said_positive & is_positive)),
        "FP": int(np.sum(said_positive & ~is_positive)),
        "FN": int(np.sum(~said_positive & is_positive)),
        "TN": int(np.sum(~said_positive & ~is_positive)),
    }


def scores_at(is_positive, probabilities, threshold, spec):
    """
    Precision, recall, F1, accuracy and (when the SPEC has error costs) the
    average cost per case at one cut-off.

    Inputs:
        is_positive: boolean array, True for real positives.
        probabilities: predicted positive-class probabilities.
        threshold: the cut-off.
        spec: the project SPEC.
    Returns:
        A dictionary of scores and counts.
    """
    counts = outcome_counts(is_positive, probabilities, threshold)
    precision = counts["TP"] / max(counts["TP"] + counts["FP"], 1)
    recall = counts["TP"] / max(counts["TP"] + counts["FN"], 1)
    scores = {
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": 2 * precision * recall / max(precision + recall, 1e-12),
        "Accuracy": (counts["TP"] + counts["TN"]) / len(is_positive),
    }
    costs = spec.get("error_costs")
    if costs:
        total = counts["FP"] * costs["false_positive"] + counts["FN"] * costs["false_negative"]
        scores["Cost per 100 cases"] = 100 * total / len(is_positive)
    scores.update(counts)
    return scores


def threshold_curve(is_positive, probabilities, spec):
    """
    Scores at every cut-off from 0.02 to 0.98.

    Inputs:
        is_positive: boolean array, True for real positives.
        probabilities: predicted positive-class probabilities.
        spec: the project SPEC.
    Returns:
        A DataFrame with one row per cut-off.
    """
    return pd.DataFrame([scores_at(is_positive, probabilities, t, spec) for t in THRESHOLDS])


def choose_threshold(curve, spec):
    """
    The cut-off to use: the cheapest when the SPEC has error costs,
    otherwise the one with the best F1.

    Inputs:
        curve: from threshold_curve() on the training predictions.
        spec: the project SPEC.
    Returns:
        The chosen cut-off as a float.
    """
    if spec.get("error_costs"):
        return float(curve.loc[curve["Cost per 100 cases"].idxmin(), "Threshold"])
    return float(curve.loc[curve["F1"].idxmax(), "Threshold"])


def tune_thresholds(results, train_features, train_target, test_features, test_target, spec):
    """
    Choose a cut-off for every model on the training data, then compare the
    default 0.5 with the chosen cut-off on the test set.

    Inputs:
        results: from train.train_models().
        train_features, train_target, test_features, test_target: the data.
        spec: the project SPEC.
    Returns:
        (test table, chosen cut-off per model, training curve per model).
    """
    positive = spec["positive_class"]
    train_is_positive = np.asarray(train_target) == positive
    test_is_positive = np.asarray(test_target) == positive
    rows, chosen, curves = [], {}, {}
    for model_name, result in results.items():
        pipeline = result["pipeline"]
        if not hasattr(pipeline, "predict_proba"):
            continue  # no probabilities, nothing to move
        curves[model_name] = threshold_curve(
            train_is_positive, out_of_fold_probabilities(pipeline, train_features, train_target, spec), spec
        )
        chosen[model_name] = choose_threshold(curves[model_name], spec)
        test_probabilities = positive_probabilities(pipeline, test_features, positive)
        for label, threshold in [("Default", 0.5), ("Tuned", chosen[model_name])]:
            row = {"Model": model_name, "Cut-off": label}
            row.update(scores_at(test_is_positive, test_probabilities, threshold, spec))
            rows.append(row)
    return pd.DataFrame(rows).round(3), chosen, curves


def plot_threshold_curve(curve, chosen, spec, model_name, filename):
    """
    How precision, recall and F1 (and the cost, when the SPEC has one)
    change with the cut-off, on the cross-validated training predictions.

    Inputs:
        curve: from threshold_curve().
        chosen: the chosen cut-off.
        spec: the project SPEC.
        model_name: shown in the title.
        filename: the image file name.
    Returns:
        The saved path.
    """
    has_cost = bool(spec.get("error_costs"))
    figure, axes = plt.subplots(1, 2 if has_cost else 1, figsize=(12 if has_cost else 7.5, 4.4), squeeze=False)
    left = axes[0, 0]
    for column in ["Precision", "Recall", "F1"]:
        left.plot(curve["Threshold"], curve[column], label=column)
    left.set_xlabel("Cut-off (say positive when the probability is at least this)")
    left.set_ylabel("Score")
    left.set_title("Scores against the cut-off")
    if has_cost:
        right = axes[0, 1]
        right.plot(curve["Threshold"], curve["Cost per 100 cases"], color="#C44E52")
        costs = spec["error_costs"]
        right.set_title(f"Cost (false alarm = {costs['false_positive']}, miss = {costs['false_negative']})")
        right.set_xlabel("Cut-off")
        right.set_ylabel("Cost per 100 cases")
    for axis in axes.flat:
        axis.axvline(0.5, color="grey", linestyle=":", label="Default 0.5")
        axis.axvline(chosen, color="black", linestyle="--", label=f"Chosen {chosen:.2f}")
        axis.legend()
    figure.suptitle(f"{model_name}: choosing the cut-off (cross-validated training predictions)")
    figure.tight_layout()
    return save_figure(figure, filename)


def calibrated_copy(pipeline, train_features, train_target, spec):
    """
    A recalibrated version of a model: CalibratedClassifierCV refits it on
    folds of the training data and learns a correction from its
    probabilities to observed frequencies. Isotonic regression needs a lot
    of data, so small datasets use Platt (sigmoid) scaling.

    Inputs:
        pipeline: a fitted pipeline (cloned, its settings are reused).
        train_features, train_target: the training data.
        spec: the project SPEC.
    Returns:
        (fitted calibrated model, method name).
    """
    method = "isotonic" if len(train_target) >= 1000 else "sigmoid"
    calibrated = CalibratedClassifierCV(clone(pipeline), method=method, cv=cross_validation_folds(spec))
    return calibrated.fit(train_features, train_target), method


def expected_calibration_error(is_positive, probabilities, bins=NUMBER_OF_BINS):
    """
    The average gap between predicted and observed frequency, weighted by
    how many cases fall in each probability bin.

    Inputs:
        is_positive: boolean array, True for real positives.
        probabilities: predicted positive-class probabilities.
        bins: how many equal-width bins.
    Returns:
        The error as a float (0 is perfect).
    """
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for b in range(bins):
        inside = which == b
        if inside.any():
            error += inside.mean() * abs(probabilities[inside].mean() - is_positive[inside].mean())
    return float(error)


def calibration_analysis(results, best, train_features, train_target, test_features, test_target, spec):
    """
    Brier score and calibration error of every model on the test set, plus
    a recalibrated copy of the best model.

    Inputs:
        results: from train.train_models().
        best: the best model's name.
        train_features, train_target, test_features, test_target: the data.
        spec: the project SPEC.
    Returns:
        (table, probabilities per model name for the diagram).
    """
    positive = spec["positive_class"]
    is_positive = np.asarray(test_target) == positive
    probabilities = {}
    for model_name, result in results.items():
        model_probabilities = positive_probabilities(result["pipeline"], test_features, positive)
        if model_probabilities is not None:
            probabilities[model_name] = model_probabilities
    calibrated, method = calibrated_copy(results[best]["pipeline"], train_features, train_target, spec)
    probabilities[f"{best} recalibrated ({method})"] = positive_probabilities(calibrated, test_features, positive)
    rows = []
    for model_name, model_probabilities in probabilities.items():
        rows.append(
            {
                "Model": model_name,
                "Brier score": brier_score_loss(is_positive, model_probabilities),
                "Calibration error": expected_calibration_error(is_positive, model_probabilities),
                "Mean predicted": model_probabilities.mean(),
                "Real positive share": is_positive.mean(),
            }
        )
    return pd.DataFrame(rows).round(4), probabilities


def plot_reliability(probabilities, test_target, spec, filename):
    """
    Reliability diagram (top) and how the predicted probabilities spread
    (bottom). Points below the diagonal mean the model is overconfident
    about the positive class; above it, underconfident.

    Inputs:
        probabilities: from calibration_analysis().
        test_target: the real test classes.
        spec: the project SPEC.
        filename: the image file name.
    Returns:
        The saved path.
    """
    is_positive = np.asarray(test_target) == spec["positive_class"]
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 8), height_ratios=[2, 1], sharex=True)
    top.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
    for model_name, model_probabilities in probabilities.items():
        observed, predicted = calibration_curve(
            is_positive, model_probabilities, n_bins=NUMBER_OF_BINS, strategy="quantile"
        )
        top.plot(predicted, observed, marker="o", label=model_name)
        bottom.hist(model_probabilities, bins=20, range=(0, 1), histtype="step", linewidth=1.6, label=model_name)
    top.set_ylabel(f"Real share of {spec['positive_class']}")
    top.set_title("Reliability diagram on the test set")
    top.legend(fontsize=8)
    bottom.set_xlabel(f"Predicted probability of {spec['positive_class']}")
    bottom.set_ylabel("Test cases")
    figure.tight_layout()
    return save_figure(figure, filename)
