"""
tabular/evaluate.py - test-set scores, comparison tables and plots.

Classification scores:
    accuracy: share of all predictions that are right.
    precision: of the examples predicted as a class, the share that really
        are that class ("when it says spam, how often is it spam?").
    recall: of the examples that really are a class, the share the model
        finds ("how much of the spam does it catch?").
    F1: the balance (harmonic mean) of precision and recall.
    ROC-AUC (two classes only): how well the model ranks positives above
        negatives, from 0.5 (random) to 1 (perfect).
With more than two classes, precision, recall and F1 are averaged over the
classes with equal weight ("macro" average).

Regression scores:
    MAE: the average size of the error, in the target's own units.
    RMSE: like MAE but punishes big errors more.
    R²: the share of the target's variation the model explains (1 is
        perfect, 0 is no better than always predicting the average).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from pxlab.common import save_figure, save_table
from pxlab.tabular.data import label_for


def classification_scores_from_predictions(target, predictions, positive_class=None, probabilities=None):
    """
    Classification scores from predictions that were already made. Used
    directly by the bootstrap (which re-scores the same predictions many
    times) and by classification_scores().

    Inputs:
        target: the real classes.
        predictions: the predicted classes.
        positive_class: for two classes, the class that counts as
            "positive" (e.g. "spam"). None for more than two classes.
        probabilities: for two classes, the predicted chance of the
            positive class, or None if the model has no probabilities.
    Returns:
        A dictionary of scores.
    """
    scores = {"Accuracy": accuracy_score(target, predictions)}
    if positive_class is not None:
        # Balanced accuracy averages the recall of both classes, so a model
        # that always answers the bigger class scores 0.5, not its share.
        scores["Balanced accuracy"] = balanced_accuracy_score(target, predictions)
        scores["Precision"] = precision_score(target, predictions, pos_label=positive_class, zero_division=0)
        scores["Recall"] = recall_score(target, predictions, pos_label=positive_class, zero_division=0)
        scores["F1"] = f1_score(target, predictions, pos_label=positive_class, zero_division=0)
        if probabilities is not None:
            scores["ROC-AUC"] = roc_auc_score(np.asarray(target) == positive_class, probabilities)
    else:
        scores["Precision"] = precision_score(target, predictions, average="macro", zero_division=0)
        scores["Recall"] = recall_score(target, predictions, average="macro", zero_division=0)
        scores["F1"] = f1_score(target, predictions, average="macro", zero_division=0)
    return scores


def positive_probabilities(pipeline, features, positive_class):
    """
    The model's predicted chance of the positive class for every row.

    Inputs:
        pipeline: a fitted classifier pipeline.
        features: the inputs.
        positive_class: the class that counts as positive.
    Returns:
        A numpy array, or None when the model can't give probabilities.
    """
    if not hasattr(pipeline, "predict_proba"):
        return None
    column = list(pipeline.classes_).index(positive_class)
    return pipeline.predict_proba(features)[:, column]


def classification_scores(pipeline, features, target, positive_class=None):
    """
    Score a classifier on labelled data.

    Inputs:
        pipeline: a fitted classifier pipeline.
        features, target: the data to score on.
        positive_class: for two classes, the class that counts as
            "positive" (e.g. "spam"). None for more than two classes.
    Returns:
        A dictionary of scores.
    """
    probabilities = None
    if positive_class is not None:
        probabilities = positive_probabilities(pipeline, features, positive_class)
    return classification_scores_from_predictions(target, pipeline.predict(features), positive_class, probabilities)


def regression_scores_from_predictions(target, predictions):
    """
    Regression scores from predictions that were already made.

    Inputs:
        target: the real values.
        predictions: the predicted values.
    Returns:
        A dictionary of scores.
    """
    return {
        "MAE": mean_absolute_error(target, predictions),
        "RMSE": root_mean_squared_error(target, predictions),
        "R²": r2_score(target, predictions),
    }


def regression_scores(pipeline, features, target):
    """
    Score a regression model on labelled data.

    Inputs:
        pipeline: a fitted regression pipeline.
        features, target: the data to score on.
    Returns:
        A dictionary of scores.
    """
    return regression_scores_from_predictions(target, pipeline.predict(features))


def test_scores(pipeline, features, target, spec):
    """
    The right set of scores for the SPEC's task.

    Inputs:
        pipeline: a fitted pipeline.
        features, target: the data to score on.
        spec: the project SPEC.
    Returns:
        A dictionary of scores.
    """
    if spec["task"] == "classification":
        return classification_scores(pipeline, features, target, spec.get("positive_class"))
    return regression_scores(pipeline, features, target)


def comparison_table(results, test_features, test_target, spec, filename="model_comparison.csv"):
    """
    One row per model: its role, cross-validated score and test scores.
    Saved as outputs/model_comparison.csv.

    Inputs:
        results: from train.train_models().
        test_features, test_target: the test set.
        spec: the project SPEC.
        filename: the CSV file inside outputs/.
    Returns:
        The table as a DataFrame.
    """
    rows = []
    cv_label = "CV " + spec["scoring"].replace("neg_", "").replace("_", " ")
    for model_name, result in results.items():
        cv_score = result["cv_score"]
        if spec["scoring"].startswith("neg_"):
            cv_score = -cv_score  # scikit-learn negates errors so "higher is better"
        row = {"Model": model_name, "Role": result["role"], cv_label: cv_score}
        row.update(test_scores(result["pipeline"], test_features, test_target, spec))
        rows.append(row)
    table = pd.DataFrame(rows).round(3)
    save_table(table, filename)
    return table


def plot_confusion_matrices(results, test_features, test_target, filename="confusion_matrices.png"):
    """
    One confusion matrix per model: rows are the real classes, columns the
    predicted ones, so the diagonal counts the right answers.

    Inputs:
        results: from train.train_models().
        test_features, test_target: the test set.
        filename: the image file name.
    Returns:
        The saved path.
    """
    names = list(results)
    columns = min(3, len(names))
    rows = int(np.ceil(len(names) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 4.2 * rows), squeeze=False)
    for axis, model_name in zip(axes.flat, names):
        ConfusionMatrixDisplay.from_estimator(
            results[model_name]["pipeline"], test_features, test_target, ax=axis, colorbar=False, cmap="Blues"
        )
        axis.set_title(model_name)
        axis.grid(False)
    for axis in list(axes.flat)[len(names) :]:
        axis.set_visible(False)
    figure.suptitle("Confusion matrices on the test set")
    figure.tight_layout()
    return save_figure(figure, filename)


def plot_roc_curves(results, test_features, test_target, positive_class, filename="roc_curves.png"):
    """
    ROC curves of every model (two-class problems only).

    Inputs:
        results: from train.train_models().
        test_features, test_target: the test set.
        positive_class: the class that counts as positive.
        filename: the image file name.
    Returns:
        The saved path.
    """
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    for model_name, result in results.items():
        RocCurveDisplay.from_estimator(
            result["pipeline"], test_features, test_target, pos_label=positive_class, ax=axis, name=model_name
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random guess")
    axis.set_title("ROC curves on the test set")
    axis.legend(loc="lower right")
    return save_figure(figure, filename)


def plot_predicted_vs_actual(results, test_features, test_target, spec, filename="predicted_vs_actual.png"):
    """
    Scatter of predicted against real values for every regression model.
    Points on the diagonal are perfect predictions.

    Inputs:
        results: from train.train_models().
        test_features, test_target: the test set.
        spec: the project SPEC.
        filename: the image file name.
    Returns:
        The saved path.
    """
    names = list(results)
    figure, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.6), squeeze=False)
    lowest, highest = float(np.min(test_target)), float(np.max(test_target))
    for axis, model_name in zip(axes.flat, names):
        predictions = results[model_name]["pipeline"].predict(test_features)
        axis.scatter(test_target, predictions, s=10, alpha=0.4)
        axis.plot([lowest, highest], [lowest, highest], color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("Real " + label_for(spec, spec["target"]))
        axis.set_ylabel("Predicted")
        axis.set_title(model_name)
    figure.suptitle("Predicted against real values on the test set")
    figure.tight_layout()
    return save_figure(figure, filename)


def plot_residuals(pipeline, test_features, test_target, spec, model_name, filename="residuals.png"):
    """
    Residuals (real minus predicted) against the prediction, and their
    histogram. A good model has residuals centred on 0 with no pattern.

    Inputs:
        pipeline: a fitted regression pipeline.
        test_features, test_target: the test set.
        spec: the project SPEC.
        model_name: shown in the title.
        filename: the image file name.
    Returns:
        The saved path.
    """
    predictions = pipeline.predict(test_features)
    residuals = np.asarray(test_target) - predictions
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4))
    left.scatter(predictions, residuals, s=10, alpha=0.4)
    left.axhline(0, color="black", linestyle="--", linewidth=1)
    left.set_xlabel("Predicted " + label_for(spec, spec["target"]))
    left.set_ylabel("Real minus predicted")
    left.set_title("Residuals against the prediction")
    right.hist(residuals, bins=40)
    right.set_xlabel("Real minus predicted")
    right.set_title("How big the errors are")
    figure.suptitle(model_name + " on the test set")
    figure.tight_layout()
    return save_figure(figure, filename)


def wrong_predictions(pipeline, test_features, test_target, spec, limit=20):
    """
    The examples the model got wrong (classification) or got most wrong
    (regression), with the real and predicted values next to the inputs.

    Inputs:
        pipeline: a fitted pipeline.
        test_features, test_target: the test set.
        spec: the project SPEC.
        limit: the most rows to return.
    Returns:
        A DataFrame.
    """
    table = test_features.copy()
    table["Real"] = test_target.to_numpy()
    table["Predicted"] = pipeline.predict(test_features)
    if spec["task"] == "classification":
        wrong = table[table["Real"] != table["Predicted"]]
        return wrong.head(limit)
    table["Error"] = table["Real"] - table["Predicted"]
    order = table["Error"].abs().sort_values(ascending=False).index
    return table.loc[order].head(limit)


def largest_errors(pipeline, test_features, test_target, spec, limit=20):
    """
    The test examples a regression model got most wrong, for error analysis.

    Inputs:
        pipeline: a fitted regression pipeline.
        test_features, test_target: the test data.
        spec: the project SPEC.
        limit: how many rows to keep.
    Returns:
        A DataFrame with the inputs, the real and predicted values and the
        error, biggest error first.
    """
    predictions = pipeline.predict(test_features)
    table = test_features.copy()
    table["Real"] = test_target.to_numpy()
    table["Predicted"] = np.round(predictions, 2)
    table["Error"] = np.round(table["Real"] - table["Predicted"], 2)
    table = table.reindex(table["Error"].abs().sort_values(ascending=False).index).head(limit)
    return table.rename(columns=lambda column: label_for(spec, column))
