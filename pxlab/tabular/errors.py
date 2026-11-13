"""
tabular/errors.py - where and why the best model goes wrong.

A single score says how often a model is wrong, not which cases it gets
wrong. Error analysis splits the test set into groups and compares them:

    Two classes: caught positives (true positives), missed positives
        (false negatives), false alarms (false positives) and correct
        negatives (true negatives). If the missed positives look like the
        correct negatives on every input, the inputs simply can't separate
        them, and no model can.
    More classes: which pairs of classes the model mixes up.
    Regression: how big and how biased the error is across the range of the
        target, and per category.
    Confidence: is the model less sure when it is wrong? If its confident
        answers are almost always right, its probability is useful on its
        own (e.g. send only the unsure cases to a human).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from pxlab.common import save_figure
from pxlab.tabular.data import label_for

CONFIDENCE_BANDS = [0.0, 0.6, 0.7, 0.8, 0.9, 1.0001]
SHOWN_INPUTS = 8


def prediction_groups(predictions, test_target, spec):
    """
    Name the outcome of every test prediction.

    Inputs:
        predictions: the predicted classes.
        test_target: the real classes.
        spec: the project SPEC.
    Returns:
        (a pandas Series of group names, the group names in display order).
    """
    real = np.asarray(test_target)
    if spec.get("positive_class") is None:
        names = np.where(predictions == real, "Right", "Wrong")
        return pd.Series(names), ["Right", "Wrong"]
    positive = spec["positive_class"]
    order = [f"Caught {positive} (TP)", f"Missed {positive} (FN)", "False alarm (FP)", "Correct negative (TN)"]
    is_positive = real == positive
    said_positive = predictions == positive
    names = np.select(
        [is_positive & said_positive, is_positive & ~said_positive, ~is_positive & said_positive],
        order[:3],
        default=order[3],
    )
    return pd.Series(names), order


def group_profile(test_features, groups, order, spec):
    """
    The average of each numeric input in each group, for the inputs whose
    averages differ most between the groups (measured in standard
    deviations, so inputs on different scales can be compared).

    Inputs:
        test_features: the test inputs.
        groups: from prediction_groups().
        order: the group names in display order.
        spec: the project SPEC.
    Returns:
        (table of averages with a Cases row, table of differences in
        standard deviations for the heatmap). Empty tables when the SPEC
        has no numeric inputs.
    """
    numeric = test_features[list(spec["numeric"])].reset_index(drop=True).astype(float)
    present = [name for name in order if (groups == name).any()]
    if numeric.empty or not present:
        return pd.DataFrame(), pd.DataFrame()
    means = numeric.groupby(groups.to_numpy()).mean().reindex(present).T
    spread = numeric.std().replace(0, 1)
    shifted = means.sub(numeric.mean(), axis=0).div(spread, axis=0)
    keep = shifted.abs().max(axis=1).sort_values(ascending=False).index[:SHOWN_INPUTS]
    means, shifted = means.loc[keep], shifted.loc[keep]
    means.index = shifted.index = [label_for(spec, column) for column in keep]
    counts = pd.DataFrame([groups.value_counts().reindex(present).to_numpy()], index=["Cases"], columns=present)
    table = pd.concat([counts, means.round(3)])
    table.index.name = "Input"
    return table, shifted


def plot_group_profile(shifted, title, filename):
    """
    Heatmap of how far each group's average sits from the overall average,
    in standard deviations. Red is above average, blue below.

    Inputs:
        shifted: the second table from group_profile().
        title: the chart title.
        filename: the image file name.
    Returns:
        The saved path.
    """
    figure, axis = plt.subplots(figsize=(1.9 * len(shifted.columns) + 4, 0.5 * len(shifted) + 2))
    limit = max(float(np.nanmax(np.abs(shifted.to_numpy()))), 0.1)
    image = axis.imshow(shifted.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(shifted.columns)), shifted.columns, rotation=15, ha="right")
    axis.set_yticks(range(len(shifted)), shifted.index)
    for row in range(len(shifted)):
        for column in range(len(shifted.columns)):
            axis.text(column, row, f"{shifted.iat[row, column]:+.2f}", ha="center", va="center", fontsize=8)
    axis.grid(False)
    figure.colorbar(image, ax=axis, label="Group average minus overall average (standard deviations)")
    axis.set_title(title)
    return save_figure(figure, filename)


def confused_pairs(predictions, test_target, limit=10):
    """
    The pairs of classes the model mixes up most (more than two classes).

    Inputs:
        predictions: the predicted classes.
        test_target: the real classes.
        limit: how many pairs to keep.
    Returns:
        A DataFrame: Real class, Predicted as, Cases, Share of the real class.
    """
    labels = sorted(set(np.asarray(test_target)) | set(predictions))
    matrix = confusion_matrix(test_target, predictions, labels=labels)
    rows = []
    for i, real in enumerate(labels):
        for j, predicted in enumerate(labels):
            if i != j and matrix[i, j] > 0:
                rows.append(
                    {
                        "Real class": real,
                        "Predicted as": predicted,
                        "Cases": int(matrix[i, j]),
                        "Share of the real class": round(matrix[i, j] / matrix[i].sum(), 3),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["Real class", "Predicted as", "Cases", "Share of the real class"])
    return pd.DataFrame(rows).sort_values("Cases", ascending=False).head(limit)


def confidence_table(pipeline, test_features, test_target):
    """
    How often the model is right at each level of confidence (its highest
    class probability).

    Inputs:
        pipeline: a fitted classifier pipeline with predict_proba.
        test_features, test_target: the test set.
    Returns:
        (table per confidence band, confidence per test row, right/wrong per row).
    """
    probabilities = pipeline.predict_proba(test_features)
    confidence = probabilities.max(axis=1)
    right = pipeline.classes_[probabilities.argmax(axis=1)] == np.asarray(test_target)
    bands = pd.cut(confidence, CONFIDENCE_BANDS, right=False)
    rows = []
    for band in bands.categories:
        inside = np.asarray(bands == band)
        if inside.any():
            high = min(band.right, 1.0)
            rows.append(
                {
                    "Confidence": f"{band.left:.0%} to {high:.0%}",
                    "Cases": int(inside.sum()),
                    "Share of the test set": round(inside.mean(), 3),
                    "Accuracy": round(right[inside].mean(), 3),
                }
            )
    return pd.DataFrame(rows), confidence, right


def plot_confidence(confidence, right, model_name, filename):
    """
    Histogram of the model's confidence, split into right and wrong answers.

    Inputs:
        confidence, right: from confidence_table().
        model_name: shown in the title.
        filename: the image file name.
    Returns:
        The saved path.
    """
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    bins = np.linspace(min(confidence.min(), 0.5), 1, 21)
    axis.hist(confidence[right], bins=bins, alpha=0.7, label=f"Right ({right.sum()})", color="#4C72B0")
    axis.hist(confidence[~right], bins=bins, alpha=0.8, label=f"Wrong ({(~right).sum()})", color="#C44E52")
    axis.set_xlabel("Confidence (highest class probability)")
    axis.set_ylabel("Test cases")
    axis.set_title(f"{model_name}: is it less sure when it is wrong?")
    axis.legend()
    return save_figure(figure, filename)


def error_by_range(predictions, test_target, spec, bins=5):
    """
    Regression error across the range of the real value: the test rows are
    split into equal-sized groups from lowest to highest real value.

    Mean error is real minus predicted: positive means the model predicts
    too low in that range, negative too high. Many models pull predictions
    towards the average, so they predict too high at the bottom of the
    range and too low at the top.

    Inputs:
        predictions: the predicted values.
        test_target: the real values.
        spec: the project SPEC.
        bins: how many groups.
    Returns:
        A DataFrame: Range, Cases, MAE, Mean error.
    """
    real = np.asarray(test_target, dtype=float)
    errors = real - predictions
    groups = pd.qcut(real, bins, duplicates="drop")
    rows = []
    for group in groups.categories:
        inside = np.asarray(groups == group)
        rows.append(
            {
                "Range of " + label_for(spec, spec["target"]): f"{group.left:.4g} to {group.right:.4g}",
                "Cases": int(inside.sum()),
                "MAE": round(float(np.abs(errors[inside]).mean()), 4),
                "Mean error": round(float(errors[inside].mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def error_by_category(predictions, test_features, test_target, spec, top=8):
    """
    Regression error per value of every category input.

    Inputs:
        predictions: the predicted values.
        test_features, test_target: the test set.
        spec: the project SPEC.
        top: the most common values to keep per input.
    Returns:
        A DataFrame: Input, Value, Cases, MAE, Mean error.
    """
    errors = np.asarray(test_target, dtype=float) - predictions
    rows = []
    for column in spec["categorical"]:
        values = test_features[column].astype(str).to_numpy()
        for value in pd.Series(values).value_counts().index[:top]:
            inside = values == value
            rows.append(
                {
                    "Input": label_for(spec, column),
                    "Value": value,
                    "Cases": int(inside.sum()),
                    "MAE": round(float(np.abs(errors[inside]).mean()), 4),
                    "Mean error": round(float(errors[inside].mean()), 4),
                }
            )
    return pd.DataFrame(rows)


def plot_error_by_range(table, model_name, filename):
    """
    Bars of the error size (MAE) and bias (mean error) per range.

    Inputs:
        table: from error_by_range().
        model_name: shown in the title.
        filename: the image file name.
    Returns:
        The saved path.
    """
    labels = table.iloc[:, 0]
    positions = np.arange(len(table))
    figure, axis = plt.subplots(figsize=(8.5, 4.4))
    axis.bar(positions - 0.2, table["MAE"], width=0.4, label="MAE (size of the error)", color="#4C72B0")
    axis.bar(
        positions + 0.2, table["Mean error"], width=0.4, label="Mean error (+ = predicts too low)", color="#DD8452"
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(positions, labels, rotation=15, ha="right")
    axis.set_xlabel(table.columns[0])
    axis.set_title(f"{model_name}: error across the range of the real value")
    axis.legend()
    return save_figure(figure, filename)
