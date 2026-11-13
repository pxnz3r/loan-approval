"""
tabular/explain.py - which inputs drive the predictions.

Tools:
    permutation importance: shuffle one input column and measure how much
        the test score drops. A big drop means the model relies on it.
        Works for every model, and on the original columns.
    tree rules: a Decision Tree is a flowchart; we can print it.
    decision path: the exact questions the tree asked for one example.
    coefficients: a linear model's weight per input.
    parameter sweep: how a setting (K in KNN, tree depth, hidden neurons)
        changes the score.
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_text, plot_tree

from pxlab.common import OUTPUTS_FOLDER, RANDOM_STATE, save_figure, save_table
from pxlab.tabular.data import feature_columns, label_for
from pxlab.tabular.train import build_pipeline, cross_validation_folds


def permutation_importance_table(
    results, test_features, test_target, spec, repeats=10, filename="permutation_importance.csv"
):
    """
    Permutation importance of every input for every model, on the test set.

    Inputs:
        results: from train.train_models().
        test_features, test_target: the test set.
        spec: the project SPEC (its "scoring" is the score that drops).
        repeats: how many times each column is shuffled (the drops are averaged).
        filename: the CSV file inside outputs/.
    Returns:
        A DataFrame: one row per input (readable names), one column per
        model, sorted by the best model's importance. Saved as CSV.
    """
    table = {}
    for model_name, result in results.items():
        importance = permutation_importance(
            result["pipeline"],
            test_features,
            test_target,
            scoring=spec["scoring"],
            n_repeats=repeats,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        table[model_name] = importance.importances_mean
    frame = pd.DataFrame(table, index=[label_for(spec, c) for c in feature_columns(spec)])
    frame = frame.sort_values(frame.columns[0], ascending=False)
    save_table(frame.round(4), filename, index=True)
    return frame


def plot_permutation_importance(table, top=12, filename="permutation_importance.png"):
    """
    Heatmap of the permutation importance table.

    Inputs:
        table: from permutation_importance_table().
        top: how many inputs to show.
        filename: the image file name.
    Returns:
        The saved path.
    """
    shown = table.head(top)
    figure, axis = plt.subplots(figsize=(1.6 * len(shown.columns) + 4, 0.45 * len(shown) + 1.8))
    image = axis.imshow(shown.to_numpy(), cmap="Blues", aspect="auto")
    axis.set_xticks(range(len(shown.columns)), shown.columns, rotation=20, ha="right")
    axis.set_yticks(range(len(shown)), shown.index)
    for row in range(len(shown)):
        for column in range(len(shown.columns)):
            axis.text(column, row, f"{shown.iat[row, column]:.3f}", ha="center", va="center", fontsize=8)
    axis.grid(False)
    figure.colorbar(image, ax=axis, label="Drop in score when shuffled")
    axis.set_title("Permutation importance on the test set")
    return save_figure(figure, filename)


def readable_feature_names(pipeline, spec):
    """
    Names of the columns the model sees after preprocessing, made readable:
    "Colour_red" becomes "Colour = red".

    Inputs:
        pipeline: a fitted pipeline.
        spec: the project SPEC.
    Returns:
        A list of strings.
    """
    names = []
    for name in pipeline.named_steps["preprocess"].get_feature_names_out():
        readable = label_for(spec, name)
        for column in spec["categorical"]:
            if name.startswith(column + "_"):
                readable = label_for(spec, column) + " = " + name[len(column) + 1 :]
        names.append(readable)
    return names


def tree_rules(pipeline, spec, max_depth=4, filename="tree_rules.txt"):
    """
    The Decision Tree's rules as indented text, saved to outputs/tree_rules.txt.

    Inputs:
        pipeline: a fitted Decision Tree pipeline.
        spec: the project SPEC.
        max_depth: how many levels to print.
        filename: the text file inside outputs/.
    Returns:
        The text.
    """
    text = export_text(
        pipeline.named_steps["model"],
        feature_names=readable_feature_names(pipeline, spec),
        max_depth=max_depth,
        decimals=2,
    )
    # One-hot splits print as "Savings = none <= 0.50"; say them in words.
    text = re.sub(r"(\w[^|\n]*?) = ([^\n]*?) <= 0\.50", r"\1 is not \2", text)
    text = re.sub(r"(\w[^|\n]*?) = ([^\n]*?) >  0\.50", r"\1 is \2", text)
    path = os.path.join(OUTPUTS_FOLDER, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)
    return text


def plot_tree_top(pipeline, spec, depth=3, filename="decision_tree.png"):
    """
    Draw the top levels of a fitted Decision Tree.

    Inputs:
        pipeline: a fitted Decision Tree pipeline.
        spec: the project SPEC.
        depth: how many levels to draw.
        filename: the image file name.
    Returns:
        The saved path.
    """
    model = pipeline.named_steps["model"]
    figure, axis = plt.subplots(figsize=(4 * 2**depth / 2 + 4, 2.4 * depth + 2))
    class_names = [str(c) for c in model.classes_] if hasattr(model, "classes_") else None
    plot_tree(
        model,
        max_depth=depth,
        feature_names=readable_feature_names(pipeline, spec),
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        precision=2,
        fontsize=9,
        ax=axis,
    )
    axis.set_title(f"Top {depth} levels of the Decision Tree")
    return save_figure(figure, filename)


def decision_path(pipeline, example, spec):
    """
    The questions a Decision Tree asked for one example, from the root to
    the leaf that made the prediction.

    Inputs:
        pipeline: a fitted Decision Tree pipeline.
        example: a one-row DataFrame with the raw input columns.
        spec: the project SPEC.
    Returns:
        A list of readable rules, e.g. "Petal length is 1.40, which is <= 2.45".
    """
    model = pipeline.named_steps["model"]
    transformed = pipeline.named_steps["preprocess"].transform(example)
    raw_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    rules = []
    for node in model.decision_path(transformed).indices:
        feature_index = model.tree_.feature[node]
        if feature_index < 0:
            continue  # a leaf: no question here
        name = raw_names[feature_index]
        split = model.tree_.threshold[node]
        value = transformed[0, feature_index]
        goes_left = value <= split
        category_column = None
        for column in spec["categorical"]:
            if name.startswith(column + "_"):
                category_column = column
        if category_column is not None:
            category = name[len(category_column) + 1 :]
            verb = "is not" if goes_left else "is"
            rules.append(f"{label_for(spec, category_column)} {verb} {category}")
        else:
            sign = "<=" if goes_left else ">"
            rules.append(f"{label_for(spec, name)} is {value:.2f}, which is {sign} {split:.2f}")
    return rules


def coefficients_table(pipeline, spec, filename="coefficients.csv"):
    """
    The weight a linear model gives each input, saved as CSV.

    With standardised inputs, a weight is the change in the prediction when
    that input rises by one standard deviation and the others stay fixed,
    so weights can be compared with each other.

    Inputs:
        pipeline: a fitted linear-model pipeline.
        spec: the project SPEC.
        filename: the CSV file inside outputs/.
    Returns:
        A DataFrame sorted by the size of the weight.
    """
    model = pipeline.named_steps["model"]
    weights = np.ravel(model.coef_)
    table = pd.DataFrame({"Input": readable_feature_names(pipeline, spec), "Weight": weights})
    table = table.reindex(table["Weight"].abs().sort_values(ascending=False).index)
    save_table(table.round(4), filename)
    return table


def plot_coefficients(table, top=15, filename="coefficients.png"):
    """
    Horizontal bars of the largest linear-model weights.

    Inputs:
        table: from coefficients_table().
        top: how many to show.
        filename: the image file name.
    Returns:
        The saved path.
    """
    shown = table.head(top).iloc[::-1]
    figure, axis = plt.subplots(figsize=(8, 0.4 * len(shown) + 1.5))
    colours = ["#C44E52" if w < 0 else "#4C72B0" for w in shown["Weight"]]
    axis.barh(shown["Input"], shown["Weight"], color=colours)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Weight (change per standard deviation of the input)")
    axis.set_title("Linear model weights")
    return save_figure(figure, filename)


def parameter_sweep(spec, model_spec, parameter, values, train_features, train_target, test_features, test_target):
    """
    Train one model for each value of a setting and record the
    cross-validated and test scores, to see how the setting matters.

    Inputs:
        spec: the project SPEC.
        model_spec: one entry of SPEC["models"] (its other best settings
            can be put in model_spec["estimator"] beforehand).
        parameter: the setting's name, e.g. "n_neighbors".
        values: the values to try.
        train_features, train_target, test_features, test_target: the data.
    Returns:
        A DataFrame: value, CV score, test score.
    """
    scorer_sign = -1 if spec["scoring"].startswith("neg_") else 1
    rows = []
    for value in values:
        pipeline = build_pipeline(spec, model_spec)
        pipeline.set_params(**{"model__" + parameter: value})
        cv = cross_val_score(
            clone(pipeline), train_features, train_target, scoring=spec["scoring"], cv=cross_validation_folds(spec)
        )
        pipeline.fit(train_features, train_target)
        test = get_scorer(spec["scoring"])(pipeline, test_features, test_target)
        rows.append({parameter: value, "CV score": scorer_sign * cv.mean(), "Test score": scorer_sign * test})
    return pd.DataFrame(rows)


def plot_parameter_sweep(table, parameter, score_name, title, filename):
    """
    Line chart of a parameter sweep.

    Inputs:
        table: from parameter_sweep().
        parameter: the column with the setting's values.
        score_name: label for the y axis (e.g. "Accuracy").
        title: the chart title.
        filename: the image file name.
    Returns:
        The saved path.
    """
    figure, axis = plt.subplots(figsize=(7.5, 4.4))
    x_values = [str(v) for v in table[parameter]]
    axis.plot(x_values, table["CV score"], marker="o", label="Cross-validation (training data)")
    axis.plot(x_values, table["Test score"], marker="s", label="Test set")
    axis.set_xlabel(parameter)
    axis.set_ylabel(score_name)
    axis.set_title(title)
    axis.legend()
    return save_figure(figure, filename)
