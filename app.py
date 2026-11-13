"""
app.py - the loan approval demo. Run it with:

    streamlit run app.py

Tabs:
    Overview   what the project is and how a Decision Tree decides,
    Decide     fill in an application (or start from a real test applicant)
               and see the decision and the questions the tree asked,
    Rules      every rule the tree learned, and how depth changes it,
    Results    the saved tables and figures from main.py.
It needs the saved models, so run `python main.py` once first.
"""

import os

import pandas as pd
import streamlit as st

from loan_approval.report import FOLDER, results_sections
from loan_approval.spec import LOAN_SPEC
from pxlab.app import (
    cutoff_choice,
    input_form,
    probability_bar,
    setup_page,
    show_figure,
    show_markdown_file,
    show_results_tabs,
)
from pxlab.common import MODELS_FOLDER, OUTPUTS_FOLDER
from pxlab.tabular.explain import decision_path
from pxlab.tabular.train import load_models, models_are_saved

MODEL_FOLDER = os.path.join(MODELS_FOLDER, FOLDER)
TYPICAL = "Typical applicant"
TEST_EXAMPLES = 15


@st.cache_resource
def load_everything():
    """
    Load the saved models and the test set (cached, so this happens once).

    Returns:
        (model_info, models, test table)
    """
    model_info, models = load_models(MODEL_FOLDER)
    test = pd.read_csv(os.path.join(MODEL_FOLDER, "test.csv"))
    return model_info, models, test


def show_decide_page(model_info, models, test):
    """
    The Decide tab: the application on the left, the decision and the
    tree's questions on the right.

    Inputs:
        model_info, models: from load_models().
        test: the saved test table.
    Returns:
        Nothing.
    """
    spec = LOAN_SPEC
    inputs = list(spec["numeric"]) + list(spec["categorical"])
    left, right = st.columns([3, 2], gap="large")
    with left:
        picker, model_box = st.columns(2)
        test_options = [
            f"Test applicant {number + 1} (bank: {test['decision'].iloc[number]})"
            for number in range(min(TEST_EXAMPLES, len(test)))
        ]
        start = picker.selectbox(
            "Start from",
            [TYPICAL, *test_options],
            help="The typical applicant has the training medians and most common answers. "
            "Test applicants are real rows the models never saw.",
        )
        model_name = model_box.selectbox("Model", list(models), index=list(models).index("Decision Tree"))
        if start == TYPICAL:
            start_values = model_info["default_values"]
        else:
            start_values = test.iloc[test_options.index(start)][inputs].to_dict()
        application = input_form(spec, model_info, start_values, key_prefix=f"{start}-")

    pipeline = models[model_name]
    approve_column = list(pipeline.classes_).index("approved")
    chance = float(pipeline.predict_proba(application)[0, approve_column])
    with right:
        cutoff = cutoff_choice(model_info, model_name, key=f"cutoff-{model_name}")
        st.subheader("Approved" if chance >= cutoff else "Rejected", anchor=False)
        st.markdown(f"Approval score **{chance:.0%}**")
        probability_bar(chance, threshold=cutoff)
        st.caption(
            f"Approved when the score is at least {cutoff:.0%} (the line). A bad loan costs five times a lost "
            "good customer, so the tuned cut-off asks for a higher score than 50%."
        )
        if model_name == "Decision Tree":
            st.caption("The score is the share of approved training applicants in the leaf this application lands in.")
            st.subheader("The questions the tree asked", anchor=False)
            steps = decision_path(pipeline, application, spec)
            st.markdown("\n".join(f"{number}. {step}" for number, step in enumerate(steps, start=1)))
        else:
            st.caption("Pick Decision Tree to see the questions that led to the decision.")


def show_rules_page(model_info):
    """
    The Rules tab: the full rule list, the top of the tree and the depth
    sweep.

    Input:
        model_info: from load_models().
    Returns:
        Nothing.
    """
    best = model_info["best_params"]["Decision Tree"]
    st.markdown(
        f"Cross-validation chose a tree of depth {best['max_depth']} with at least {best['min_samples_leaf']} "
        f"training applicants per leaf, giving {model_info['tree_leaves']} leaves. Read each rule from the top: "
        "every line is a question, and indented lines apply only when it is true."
    )
    show_figure(f"{FOLDER}/decision_tree.png")
    with open(os.path.join(OUTPUTS_FOLDER, FOLDER, "tree_rules.txt"), encoding="utf-8") as file:
        st.code(file.read(), language=None)
    st.subheader("How deep should the tree be?", anchor=False)
    st.markdown(
        "A shallow tree asks too few questions and misses patterns (underfitting). A deep tree keeps splitting "
        "until it memorises the training applicants (overfitting). Cross-validation picks the depth in between."
    )
    show_figure(f"{FOLDER}/depth_sweep.png")


def main():
    """
    Build the page.

    Returns:
        Nothing.
    """
    setup_page("Loan approval")
    if not models_are_saved(MODEL_FOLDER):
        st.error("No saved models yet. Run `python main.py` first.")
        st.stop()
    model_info, models, test = load_everything()

    overview, decide, rules, results = st.tabs(["Overview", "Decide", "Rules", "Results"])
    with overview:
        show_markdown_file("app_overview.md")
    with decide:
        show_decide_page(model_info, models, test)
    with rules:
        show_rules_page(model_info)
    with results:
        show_results_tabs(results_sections())


main()
