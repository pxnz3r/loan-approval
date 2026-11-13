"""
app.py - Streamlit building blocks shared by the project demos: the page
setup, an overview page read from Markdown, input forms built from the
SPEC, saved tables and figures, and a probability bar.
"""

import os

import pandas as pd
import streamlit as st

from pxlab.common import FIGURES_FOLDER, OUTPUTS_FOLDER


def setup_page(title, browser_title=None):
    """
    Wide layout, the browser tab title and the page title.

    Inputs:
        title: the title shown on the page.
        browser_title: the browser tab title (defaults to the title).
    Returns:
        Nothing.
    """
    st.set_page_config(page_title=browser_title or title, layout="wide")
    st.title(title, anchor=False)


def show_markdown_file(path):
    """
    Show a Markdown file (the overview page lives in one, so its words can
    be edited without touching code).

    Input:
        path: the .md file.
    Returns:
        Nothing.
    """
    with open(path, encoding="utf-8") as file:
        st.markdown(file.read())


def show_table(title, file_name):
    """
    Show a CSV from outputs/ under a subheader, if the file exists.

    Inputs:
        title: the subheader.
        file_name: the CSV path inside outputs/.
    Returns:
        Nothing.
    """
    path = os.path.join(OUTPUTS_FOLDER, file_name)
    if os.path.isfile(path):
        st.subheader(title, anchor=False)
        st.dataframe(pd.read_csv(path), hide_index=True, width="stretch")


def show_figure(file_name):
    """
    Show an image from outputs/figures/, if it exists. Every figure carries
    its own title, so no caption is added.

    Input:
        file_name: the image path inside outputs/figures/.
    Returns:
        Nothing.
    """
    path = os.path.join(FIGURES_FOLDER, file_name)
    if os.path.isfile(path):
        st.image(path, width="stretch")


def show_results_tabs(sections):
    """
    One tab per section of saved results.

    Input:
        sections: a list of (tab name, items). Each item is ("table",
            title, csv file) or ("figure", image file) or ("text", markdown).
    Returns:
        Nothing.
    """
    tabs = st.tabs([name for name, items in sections])
    for tab, (_name, items) in zip(tabs, sections):
        with tab:
            for item in items:
                if item[0] == "table":
                    show_table(item[1], item[2])
                elif item[0] == "figure":
                    show_figure(item[1])
                else:
                    st.markdown(item[1])


def help_text(spec, column):
    """
    The help text for an input: the SPEC's explanation and the dataset
    column name.

    Inputs:
        spec: the project SPEC.
        column: the column name.
    Returns:
        Markdown text.
    """
    note = spec.get("help", {}).get(column, "")
    if note:
        return note + "  \n" + f"Column: `{column}`"
    return f"Column: `{column}`"


def number_box(spec, model_info, column, value, key):
    """
    One number input for a numeric column, limited to a sensible range.

    Inputs:
        spec: the project SPEC.
        model_info: holds the training ranges.
        column: the column name.
        value: the starting value.
        key: a unique widget key.
    Returns:
        The entered number.
    """
    lowest, highest = model_info["feature_ranges"][column]
    whole_numbers = column in spec.get("whole_numbers", [])
    span = highest - lowest
    if whole_numbers:
        return st.number_input(
            spec.get("labels", {}).get(column, column),
            min_value=int(min(0, lowest)),
            max_value=int(highest + span),
            value=int(round(value)),
            step=1,
            help=help_text(spec, column),
            key=key,
        )
    if span >= 100:
        step = 10.0
    elif span >= 10:
        step = 1.0
    elif span >= 1:
        step = 0.1
    else:
        step = 0.01
    return st.number_input(
        spec.get("labels", {}).get(column, column),
        min_value=float(min(0.0, lowest)),
        max_value=float(highest + span),
        value=float(value),
        step=step,
        format="%.2f",
        help=help_text(spec, column),
        key=key,
    )


def choice_box(spec, model_info, column, value, key):
    """
    One drop-down for a category column.

    Inputs:
        spec: the project SPEC.
        model_info: holds the category options.
        column: the column name.
        value: the starting value.
        key: a unique widget key.
    Returns:
        The picked category.
    """
    options = model_info["category_options"][column]
    index = options.index(value) if value in options else 0
    return st.selectbox(
        spec.get("labels", {}).get(column, column), options, index=index, help=help_text(spec, column), key=key
    )


def input_form(spec, model_info, start_values, key_prefix, columns=2):
    """
    Inputs for every feature, in a grid, starting from start_values.

    Inputs:
        spec: the project SPEC.
        model_info: holds ranges and category options.
        start_values: dictionary column -> starting value.
        key_prefix: makes widget keys unique per starting example, so
            picking a new example resets the inputs.
        columns: how many columns of inputs.
    Returns:
        A one-row DataFrame with the entered values, in SPEC column order.
    """
    names = list(spec["numeric"]) + list(spec["categorical"])
    values = {}
    grid = st.columns(columns)
    for position, column in enumerate(names):
        with grid[position % columns]:
            key = key_prefix + column
            if column in spec["numeric"]:
                values[column] = number_box(spec, model_info, column, start_values[column], key)
            else:
                values[column] = choice_box(spec, model_info, column, start_values[column], key)
    return pd.DataFrame([values], columns=names)


def probability_bar(probability, threshold=None):
    """
    A bar from 0 to 100% filled to the probability, with an optional line
    at a threshold. Plain HTML in the text colour, so it suits light and
    dark themes.

    Inputs:
        probability: a number from 0 to 1.
        threshold: where to draw the line, or None.
    Returns:
        Nothing (draws on the page).
    """
    filled = min(max(probability, 0.0), 1.0) * 100
    line = ""
    if threshold is not None:
        position = min(max(threshold, 0.0), 1.0) * 100
        line = (
            f'<div style="position: absolute; left: calc({position:.1f}% - 1px); top: -6px; bottom: -6px;'
            ' width: 2px; border-radius: 1px; background: currentColor;"></div>'
        )
    st.markdown(
        f"""
<div style="position: relative; height: 12px; margin: 14px 0 6px; border-radius: 6px;
            background: color-mix(in srgb, currentColor 12%, transparent);">
  <div style="width: {filled:.1f}%; height: 100%; border-radius: 6px;
              background: color-mix(in srgb, currentColor 55%, transparent);"></div>
  {line}
</div>
<div style="display: flex; justify-content: space-between; font-size: 0.8rem; opacity: 0.6;">
  <span>0%</span><span>100%</span>
</div>
""",
        unsafe_allow_html=True,
    )


def class_probabilities(pipeline, example):
    """
    The predicted probability of every class for one example, highest first.

    Inputs:
        pipeline: a fitted classifier pipeline.
        example: a one-row DataFrame.
    Returns:
        A pandas Series: class -> probability.
    """
    probabilities = pipeline.predict_proba(example)[0]
    return pd.Series(probabilities, index=[str(c) for c in pipeline.classes_]).sort_values(ascending=False)


def signed_bars(values, format_value=lambda value: f"{value:+.2f}"):
    """
    One row per item with a bar going right for positive values and left
    for negative ones, from a centre line. Used to show how much each input
    pushes a prediction up or down. Plain HTML in the text colour, so it
    suits light and dark themes.

    Inputs:
        values: a pandas Series, label -> signed number.
        format_value: a function that turns a number into display text.
    Returns:
        Nothing (draws on the page).
    """
    largest = max(float(values.abs().max()), 1e-12)
    rows = []
    for label, value in values.items():
        width = abs(float(value)) / largest * 50
        side = f"left: 50%; width: {width:.1f}%;" if value >= 0 else f"left: {50 - width:.1f}%; width: {width:.1f}%;"
        rows.append(
            f"""
<div style="display: grid; grid-template-columns: minmax(0, 2fr) minmax(0, 3fr) 6.5rem; gap: 0.75rem;
            align-items: center; margin: 4px 0;">
  <span style="overflow-wrap: anywhere;">{label}</span>
  <div style="position: relative; height: 10px;">
    <div style="position: absolute; left: calc(50% - 0.5px); top: -4px; bottom: -4px; width: 1px;
                background: color-mix(in srgb, currentColor 30%, transparent);"></div>
    <div style="position: absolute; top: 0; bottom: 0; {side} border-radius: 3px;
                background: color-mix(in srgb, currentColor {55 if value >= 0 else 30}%, transparent);"></div>
  </div>
  <span style="text-align: right; font-variant-numeric: tabular-nums;">{format_value(value)}</span>
</div>"""
        )
    st.markdown("".join(rows), unsafe_allow_html=True)
