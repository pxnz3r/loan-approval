"""
tabular/data.py - checking, describing and splitting a table of examples.

Every project describes its dataset in a SPEC dictionary (see spec.py in
the project). The keys used here:
    "task": "classification" or "regression"
    "target": the column to predict
    "numeric": list of number columns
    "categorical": list of category columns
    "labels": readable name for each column (optional entries)
    "test_size": share of rows kept back for the final test (e.g. 0.2)
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from pxlab.common import RANDOM_STATE, save_figure, save_table


def feature_columns(spec):
    """
    All input columns, numeric first.

    Input:
        spec: the project SPEC.
    Returns:
        A list of column names.
    """
    return list(spec["numeric"]) + list(spec["categorical"])


def label_for(spec, column):
    """
    The readable name of a column, or the column name if it has none.

    Inputs:
        spec: the project SPEC.
        column: a column name.
    Returns:
        A string.
    """
    return spec.get("labels", {}).get(column, column)


def check_columns(data, spec):
    """
    Stop early with a clear message if the data lacks a column the SPEC names.

    Inputs:
        data: the loaded DataFrame.
        spec: the project SPEC.
    Returns:
        Nothing. Raises ValueError when a column is missing.
    """
    missing = [column for column in feature_columns(spec) + [spec["target"]] if column not in data.columns]
    if missing:
        raise ValueError("The dataset is missing these columns: " + ", ".join(missing))


def quality_report(data, spec, filename="data_quality.csv"):
    """
    Count rows, missing values and duplicates, print them and save a table.

    Inputs:
        data: the loaded DataFrame.
        spec: the project SPEC.
        filename: the CSV file inside outputs/.
    Returns:
        The report as a DataFrame (one row per column).
    """
    columns = feature_columns(spec) + [spec["target"]]
    report = pd.DataFrame(
        {
            "Column": columns,
            "Type": [
                "numeric" if c in spec["numeric"] else "target" if c == spec["target"] else "category" for c in columns
            ],
            "Missing": [int(data[c].isna().sum()) for c in columns],
            "Distinct values": [int(data[c].nunique()) for c in columns],
        }
    )
    print(f"  Rows: {len(data)}, columns used: {len(columns)}, duplicate rows: {int(data.duplicated().sum())}")
    if report["Missing"].sum() > 0:
        print("  Missing values:", int(report["Missing"].sum()))
    save_table(report, filename)
    return report


def split_data(data, spec):
    """
    Split into a training set and a test set.

    The test set is kept apart until the very end, so the final scores show
    how the model does on examples it has never seen. For classification
    the split is stratified: both parts keep the same share of each class.

    Inputs:
        data: the loaded DataFrame.
        spec: the project SPEC.
    Returns:
        (train_features, test_features, train_target, test_target)
    """
    features = data[feature_columns(spec)]
    target = data[spec["target"]]
    if spec.get("time_order"):
        # Time series: the test set is the most recent rows, so the model is
        # judged on days that come after everything it learned from.
        train_features, test_features, train_target, test_target = train_test_split(
            features, target, test_size=spec.get("test_size", 0.2), shuffle=False
        )
    else:
        stratify = target if spec["task"] == "classification" else None
        train_features, test_features, train_target, test_target = train_test_split(
            features,
            target,
            test_size=spec.get("test_size", 0.2),
            random_state=RANDOM_STATE,
            stratify=stratify,
        )
    print(f"  Training rows: {len(train_features)}, test rows: {len(test_features)}")
    return train_features, test_features, train_target, test_target


def input_summary(train_features, spec):
    """
    Typical values, ranges and category options of every input, taken from
    the training data. The app uses them as starting values and limits.

    Inputs:
        train_features: the training inputs.
        spec: the project SPEC.
    Returns:
        A dictionary with "default_values", "feature_ranges", "category_options".
    """
    defaults = {}
    ranges = {}
    options = {}
    for column in spec["numeric"]:
        defaults[column] = float(train_features[column].median())
        ranges[column] = [float(train_features[column].min()), float(train_features[column].max())]
    for column in spec["categorical"]:
        # Numbers sort as numbers (month 2 before 10), text alphabetically.
        values = sorted(train_features[column].dropna().unique().tolist())
        options[column] = [value.item() if hasattr(value, "item") else value for value in values]
        mode = train_features[column].mode().iloc[0]
        defaults[column] = mode.item() if hasattr(mode, "item") else mode
    return {"default_values": defaults, "feature_ranges": ranges, "category_options": options}


def plot_target(data, spec, filename="target.png"):
    """
    Bar chart of the classes (classification) or histogram of the target
    (regression).

    Inputs:
        data: the loaded DataFrame.
        spec: the project SPEC.
        filename: the image file name.
    Returns:
        The saved path.
    """
    figure, axis = plt.subplots(figsize=(7, 4))
    target = spec["target"]
    if spec["task"] == "classification":
        counts = data[target].value_counts().sort_index()
        sns.barplot(x=counts.index.astype(str), y=counts.values, ax=axis, color="#4C72B0")
        for position, count in enumerate(counts.values):
            axis.text(position, count, f"{count} ({count / len(data):.0%})", ha="center", va="bottom")
        axis.set_ylabel("Rows")
        axis.set_title("How many examples of each class")
    else:
        sns.histplot(data[target], bins=40, ax=axis, color="#4C72B0")
        axis.set_title("Distribution of " + label_for(spec, target))
    axis.set_xlabel(label_for(spec, target))
    return save_figure(figure, filename)


def plot_correlations(data, spec, filename="correlations.png"):
    """
    Heatmap of the correlations between numeric columns (and the target,
    when the target is a number).

    Inputs:
        data: the loaded DataFrame.
        spec: the project SPEC.
        filename: the image file name.
    Returns:
        The saved path, or None when there are fewer than two numeric columns.
    """
    columns = list(spec["numeric"])
    if spec["task"] == "regression":
        columns.append(spec["target"])
    if len(columns) < 2:
        return None
    table = data[columns].rename(columns={c: label_for(spec, c) for c in columns})
    size = max(5, 0.6 * len(columns) + 2)
    figure, axis = plt.subplots(figsize=(size, size * 0.8))
    sns.heatmap(table.corr(), cmap="vlag", vmin=-1, vmax=1, annot=len(columns) <= 12, fmt=".2f", ax=axis)
    axis.set_title("Correlation between numeric columns")
    return save_figure(figure, filename)


def plot_pairs(data, spec, filename="pairs.png"):
    """
    Scatter plot of every pair of numeric inputs, coloured by class. Shows
    which inputs separate the classes (useful for small datasets).

    Inputs:
        data: the loaded DataFrame.
        spec: the project SPEC (classification).
        filename: the image file name.
    Returns:
        The saved path.
    """
    columns = list(spec["numeric"]) + [spec["target"]]
    table = data[columns].rename(columns={c: label_for(spec, c) for c in columns})
    grid = sns.pairplot(table, hue=label_for(spec, spec["target"]), corner=True, plot_kws={"s": 14, "alpha": 0.7})
    grid.figure.suptitle("Every pair of inputs, coloured by class", y=1.02)
    return save_figure(grid.figure, filename)
