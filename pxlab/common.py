"""
common.py - small helpers every project uses: folders, saving figures and
tables, downloading a dataset once, and the random seed.
"""

import os
import urllib.request

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # draw to files, no window (works on servers too)

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

# One seed everywhere, so every run gives the same split and the same models.
RANDOM_STATE = 42

DATA_FOLDER = "data"
OUTPUTS_FOLDER = "outputs"
FIGURES_FOLDER = os.path.join(OUTPUTS_FOLDER, "figures")
MODELS_FOLDER = "models"


def set_plot_style():
    """
    Use one clean plot style for every figure.

    Returns:
        Nothing.
    """
    sns.set_theme(style="whitegrid", context="notebook")


def save_figure(figure, filename):
    """
    Save a matplotlib figure into outputs/figures/ and close it.

    Inputs:
        figure: the matplotlib figure.
        filename: e.g. "confusion_matrices.png" or "iris/confusion_matrices.png"
            (a subfolder keeps several datasets' figures apart).
    Returns:
        The path of the saved image.
    """
    path = os.path.join(FIGURES_FOLDER, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return path


def save_table(table, filename, index=False):
    """
    Save a pandas table as CSV into outputs/.

    Inputs:
        table: the DataFrame.
        filename: e.g. "model_comparison.csv" or "iris/model_comparison.csv".
        index: whether to write the index column.
    Returns:
        The path of the saved file.
    """
    path = os.path.join(OUTPUTS_FOLDER, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table.to_csv(path, index=index)
    return path


def download_once(url, path, timeout_seconds=60):
    """
    Download a file unless it is already on disk.

    Inputs:
        url: where to download from.
        path: where to save it.
        timeout_seconds: give up after this long without data.
    Returns:
        The path.
    """
    if os.path.isfile(path):
        return path
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    print("  Downloading", url)
    request = urllib.request.Request(url, headers={"User-Agent": "pxlab-dataset-download"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        data = response.read()
    with open(path, "wb") as file:
        file.write(data)
    return path


def markdown_table(table):
    """
    A pandas table as a Markdown table (for docs/results.md).

    Input:
        table: a DataFrame.
    Returns:
        Markdown text.
    """
    header = "| " + " | ".join(str(column) for column in table.columns) + " |"
    divider = "| " + " | ".join("---" for _ in table.columns) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def write_results_doc(title, intro, sections, path="docs/results.md"):
    """
    Write docs/results.md from the saved tables and figures. It takes the
    same sections list as the app's Results tab, so the two always match.

    Inputs:
        title: the page title.
        intro: a paragraph under the title.
        sections: a list of (heading, items). Each item is ("table", title,
            csv file in outputs/) or ("figure", image file in
            outputs/figures/) or ("text", markdown).
        path: where to write.
    Returns:
        The path written.
    """
    lines = [f"# {title}", "", intro, ""]
    for heading, items in sections:
        lines += [f"## {heading}", ""]
        for item in items:
            if item[0] == "table":
                csv_path = os.path.join(OUTPUTS_FOLDER, item[2])
                if os.path.isfile(csv_path):
                    table = pd.read_csv(csv_path)
                    table = table.rename(columns=lambda c: "" if str(c).startswith("Unnamed") else c)
                    lines += [f"### {item[1]}", "", markdown_table(table), ""]
            elif item[0] == "figure":
                if not os.path.isfile(os.path.join(FIGURES_FOLDER, item[1])):
                    continue
                alt_text = os.path.splitext(os.path.basename(item[1]))[0].replace("_", " ")
                lines += [f"![{alt_text}](../{FIGURES_FOLDER.replace(os.sep, '/')}/{item[1]})", ""]
            else:
                lines += [item[1], ""]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(lines))
    return path
