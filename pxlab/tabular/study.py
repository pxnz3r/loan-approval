"""
tabular/study.py - the in-depth analysis every tabular project runs after
training: reliability of the scores, learning curves, cut-off tuning and
calibration (two classes), and error analysis. One call writes every table
and figure; study_sections() lists them for the app and docs/results.md.

    findings = run_study(spec, results, best, train_features, train_target,
                         test_features, test_target, folder="loans")
"""

import numpy as np
import pandas as pd

from pxlab.common import save_table
from pxlab.tabular import errors, probability, reliability
from pxlab.tabular.data import label_for
from pxlab.tabular.evaluate import positive_probabilities

# Which column of the score tables matches each tuning score.
MAIN_SCORE = {"roc_auc": "ROC-AUC", "f1": "F1", "f1_macro": "F1", "accuracy": "Accuracy", "r2": "R²"}


def output_name(folder, name):
    """
    A file name inside the project's output subfolder.

    Inputs:
        folder: the subfolder ("" for none).
        name: the file name.
    Returns:
        The path relative to outputs/ or outputs/figures/.
    """
    return f"{folder}/{name}" if folder else name


def uses_probability_analysis(spec):
    """
    Cut-off tuning and calibration need two classes and folds that can be
    shuffled (cross_val_predict can't use time-ordered folds).

    Input:
        spec: the project SPEC.
    Returns:
        True when the probability analysis runs.
    """
    return spec.get("positive_class") is not None and not spec.get("time_order", False)


def main_score_name(spec):
    """
    The score the findings quote: the one the models were tuned on.

    Input:
        spec: the project SPEC.
    Returns:
        A score name.
    """
    return MAIN_SCORE.get(spec["scoring"], reliability.headline_scores(spec)[0])


def reliability_part(spec, results, best, train_features, train_target, test_features, test_target, folder):
    """
    Baseline, bootstrap intervals, paired tests and learning curves.

    Returns:
        A list of finding sentences.
    """
    models = {name: result["pipeline"] for name, result in results.items()}
    # Use the project's own baseline model when it has one; otherwise add one.
    own_baselines = [name for name, result in results.items() if result["role"] == "Baseline"]
    baseline = own_baselines[0] if own_baselines else reliability.BASELINE_NAME
    if not own_baselines:
        models[baseline] = reliability.fit_baseline(spec, train_features, train_target)
    predictions = reliability.test_predictions(models, test_features, spec)

    intervals = reliability.bootstrap_intervals(predictions, test_target, spec)
    save_table(intervals.round(4), output_name(folder, "confidence_intervals.csv"))
    reliability.plot_intervals(intervals, baseline, output_name(folder, "confidence_intervals.png"))

    tests = reliability.significance_table(predictions, test_target, spec, best)
    save_table(tests, output_name(folder, "significance.csv"))

    learners = {name: result for name, result in results.items() if result["role"] != "Baseline"}
    curves = reliability.learning_curves(learners, train_features, train_target, spec)
    save_table(reliability.learning_curve_table(curves).round(4), output_name(folder, "learning_curves.csv"))
    score_label = spec["scoring"].replace("neg_", "").replace("_", " ")
    reliability.plot_learning_curves(curves, score_label, output_name(folder, "learning_curves.png"))

    score = main_score_name(spec)
    row = intervals[(intervals["Model"] == best) & (intervals["Score"] == score)].iloc[0]
    floor = intervals[(intervals["Model"] == baseline) & (intervals["Score"] == score)].iloc[0]
    named = "" if baseline == reliability.BASELINE_NAME else f" ({baseline})"
    findings = [
        f"{best} scores {score} {row['Test value']:.3f} on the test set (95% interval {row['Low']:.3f} to "
        f"{row['High']:.3f}); the baseline that ignores the inputs{named} scores {floor['Test value']:.3f}."
    ]
    compared_on = "right or wrong answers" if spec["task"] == "classification" else "error sizes"
    for test in tests.itertuples(index=False):
        findings.append(
            f"{best} against {test[1]} on {compared_on} ({test[4]}): {test[6].lower()} (p = {test[5]:.3f})."
        )
    findings += reliability.learning_curve_findings(curves, spec, score_label)
    return findings


def probability_part(spec, results, best, train_features, train_target, test_features, test_target, folder):
    """
    Cut-off tuning and calibration (two classes only).

    Returns:
        (finding sentences, chosen cut-off per model name).
    """
    table, chosen, curves = probability.tune_thresholds(
        results, train_features, train_target, test_features, test_target, spec
    )
    save_table(table, output_name(folder, "threshold_tuning.csv"))
    probability.plot_threshold_curve(curves[best], chosen[best], spec, best, output_name(folder, "threshold_curve.png"))

    calibration, probabilities = probability.calibration_analysis(
        results, best, train_features, train_target, test_features, test_target, spec
    )
    save_table(calibration, output_name(folder, "calibration.csv"))
    probability.plot_reliability(probabilities, test_target, spec, output_name(folder, "reliability.png"))

    rule = "the lowest cost" if spec.get("error_costs") else "the best F1"
    default = table[(table["Model"] == best) & (table["Cut-off"] == "Default")].iloc[0]
    tuned = table[(table["Model"] == best) & (table["Cut-off"] == "Tuned")].iloc[0]
    change = (
        f"recall {default['Recall']:.2f} to {tuned['Recall']:.2f}, precision {default['Precision']:.2f} to "
        f"{tuned['Precision']:.2f}"
    )
    if spec.get("error_costs"):
        change += f", cost per 100 cases {default['Cost per 100 cases']:.1f} to {tuned['Cost per 100 cases']:.1f}"
    findings = [f"Cut-off for {best}: {chosen[best]:.2f} instead of 0.5, chosen for {rule}. On the test set: {change}."]
    best_calibrated = calibration.loc[calibration["Brier score"].idxmin()]
    own = calibration[calibration["Model"] == best].iloc[0]
    findings.append(
        f"Most honest probabilities: {best_calibrated['Model']} (Brier {best_calibrated['Brier score']:.3f}); "
        f"{best} has Brier {own['Brier score']:.3f} and calibration error {own['Calibration error']:.3f}."
    )
    return findings, chosen


def classification_errors_part(spec, pipeline, best, test_features, test_target, threshold, folder):
    """
    Error groups, confused pairs and confidence for the best classifier.

    Returns:
        A list of finding sentences.
    """
    if threshold is None:
        predictions = np.asarray(pipeline.predict(test_features))
    else:
        positive = spec["positive_class"]
        negative = [c for c in pipeline.classes_ if c != positive][0]
        chance = positive_probabilities(pipeline, test_features, positive)
        predictions = np.where(chance >= threshold, positive, negative)
    groups, order = errors.prediction_groups(predictions, test_target, spec)
    findings = []

    profile, shifted = errors.group_profile(test_features, groups, order, spec)
    if not profile.empty:
        save_table(profile, output_name(folder, "error_groups.csv"), index=True)
        errors.plot_group_profile(
            shifted, f"{best}: how each outcome group differs", output_name(folder, "error_groups.png")
        )
        if len(order) == 4 and {order[0], order[1]} <= set(shifted.columns):
            gap = (shifted[order[0]] - shifted[order[1]]).abs().sort_values(ascending=False)
            name = gap.index[0]
            findings.append(
                f"The input that most separates caught from missed {spec['positive_class']} is {name}: "
                f"average {profile.loc[name, order[0]]:,.4g} when caught, {profile.loc[name, order[1]]:,.4g} when missed."
            )

    if spec.get("positive_class") is None:
        pairs = errors.confused_pairs(predictions, test_target)
        save_table(pairs, output_name(folder, "confused_pairs.csv"))
        if not pairs.empty:
            top = pairs.iloc[0]
            findings.append(
                f"The most common mix-up: {top['Real class']} predicted as {top['Predicted as']} "
                f"({top['Cases']} cases, {top['Share of the real class']:.0%} of {top['Real class']})."
            )

    if hasattr(pipeline, "predict_proba"):
        bands, confidence, right = errors.confidence_table(pipeline, test_features, test_target)
        save_table(bands, output_name(folder, "confidence.csv"))
        errors.plot_confidence(confidence, right, best, output_name(folder, "confidence.png"))
        sure = bands[bands["Confidence"].str.startswith("90%")]
        if not sure.empty:
            findings.append(
                f"When {best} is at least 90% sure it is right {sure.iloc[0]['Accuracy']:.0%} of the time "
                f"({sure.iloc[0]['Share of the test set']:.0%} of test cases)."
            )
    return findings


def regression_errors_part(spec, pipeline, best, test_features, test_target, folder):
    """
    Error by range of the target and by category for the best regressor.

    Returns:
        A list of finding sentences.
    """
    predictions = pipeline.predict(test_features)
    by_range = errors.error_by_range(predictions, test_target, spec)
    save_table(by_range, output_name(folder, "error_by_range.csv"))
    errors.plot_error_by_range(by_range, best, output_name(folder, "error_by_range.png"))
    if spec["categorical"]:
        save_table(
            errors.error_by_category(predictions, test_features, test_target, spec),
            output_name(folder, "error_by_category.csv"),
        )
    worst = by_range.loc[by_range["Mean error"].abs().idxmax()]
    direction = "too low" if worst["Mean error"] > 0 else "too high"
    return [
        f"{best} is most biased for {label_for(spec, spec['target'])} {worst.iloc[0]}: it predicts {direction} by "
        f"{errors.readable(abs(worst['Mean error']))} on average there (MAE {errors.readable(worst['MAE'])})."
    ]


def run_study(spec, results, best, train_features, train_target, test_features, test_target, folder=""):
    """
    Run every analysis that fits the SPEC and save its tables and figures,
    plus outputs/<folder>/key_findings.csv with one sentence per finding.

    Inputs:
        spec: the project SPEC.
        results: from train.train_models().
        best: the best model's name.
        train_features, train_target, test_features, test_target: the data.
        folder: the output subfolder ("" for none).
    Returns:
        A dictionary for model_info.json: the findings, and for two-class
        problems the tuned cut-off of every model ("thresholds").
    """
    print("In-depth analysis:")
    findings = reliability_part(spec, results, best, train_features, train_target, test_features, test_target, folder)
    summary = {}
    threshold = None
    if uses_probability_analysis(spec) and hasattr(results[best]["pipeline"], "predict_proba"):
        more, chosen = probability_part(
            spec, results, best, train_features, train_target, test_features, test_target, folder
        )
        findings += more
        threshold = chosen[best]
        summary["thresholds"] = chosen
        summary["threshold_rule"] = "lowest cost" if spec.get("error_costs") else "best F1"
    pipeline = results[best]["pipeline"]
    if spec["task"] == "classification":
        findings += classification_errors_part(spec, pipeline, best, test_features, test_target, threshold, folder)
    else:
        findings += regression_errors_part(spec, pipeline, best, test_features, test_target, folder)
    save_table(pd.DataFrame({"Finding": findings}), output_name(folder, "key_findings.csv"))
    for finding in findings:
        print("  " + finding)
    summary["findings"] = findings
    return summary


def study_sections(spec, folder=""):
    """
    The study's tables and figures as results sections, in the format of
    write_results_doc() and the app's show_results_tabs().

    Inputs:
        spec: the project SPEC.
        folder: the output subfolder ("" for none).
    Returns:
        A list of (section name, items).
    """

    def name(file_name):
        return output_name(folder, file_name)

    sections = [
        ("Key findings", [("table", "What the in-depth analysis found", name("key_findings.csv"))]),
        (
            "Reliability",
            [
                ("text", "Bootstrap 95% intervals: how much each test score could move with a different test sample."),
                ("figure", name("confidence_intervals.png")),
                ("table", "Confidence intervals", name("confidence_intervals.csv")),
                ("table", "Is the best model really better? (paired tests)", name("significance.csv")),
                ("figure", name("learning_curves.png")),
                ("table", "Learning curves", name("learning_curves.csv")),
            ],
        ),
    ]
    if uses_probability_analysis(spec):
        costs = spec.get("error_costs")
        note = "The cut-off is chosen on cross-validated training predictions, then checked on the test set."
        if costs:
            note += (
                f" Costs: a false alarm costs {costs['false_positive']}, a miss costs {costs['false_negative']}. "
                + costs.get("reason", "")
            )
        sections.append(
            (
                "Cut-off and calibration",
                [
                    ("text", note),
                    ("figure", name("threshold_curve.png")),
                    ("table", "Default against tuned cut-off on the test set", name("threshold_tuning.csv")),
                    ("figure", name("reliability.png")),
                    ("table", "How honest the probabilities are", name("calibration.csv")),
                ],
            )
        )
    if spec["task"] == "classification":
        items = [
            ("figure", name("error_groups.png")),
            ("table", "Average inputs per outcome group", name("error_groups.csv")),
        ]
        if spec.get("positive_class") is None:
            items.append(("table", "Classes the model mixes up", name("confused_pairs.csv")))
        items += [("figure", name("confidence.png")), ("table", "Accuracy by confidence", name("confidence.csv"))]
    else:
        items = [
            ("figure", name("error_by_range.png")),
            ("table", "Error across the range of the real value", name("error_by_range.csv")),
        ]
        if spec["categorical"]:
            items.append(("table", "Error per category", name("error_by_category.csv")))
    sections.append(("Error analysis", items))
    return sections
