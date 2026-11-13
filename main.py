"""
main.py - run the whole loan approval study with one command:

    python main.py

It:
    1. downloads (once) and checks the German Credit data,
    2. keeps 25% of the applicants back as a test set,
    3. tunes a Decision Tree (depth, leaf size, class weights) with 5-fold
       cross-validation, plus a Random Forest benchmark and an "always
       approve" baseline,
    4. scores them on the test set and draws confusion matrices and ROC
       curves,
    5. writes out the rules the tree learned and draws its top levels,
    6. sweeps the tree depth to show over- and underfitting,
    7. measures which inputs matter and lists the mistakes,
    8. saves the models for the app and writes docs/results.md.
Figures and tables go to outputs/loans/, models to models/loans/.
"""

import os

from sklearn.tree import DecisionTreeClassifier

from loan_approval.data import load_loans
from loan_approval.report import FOLDER, results_sections
from loan_approval.spec import LOAN_SPEC
from pxlab.common import MODELS_FOLDER, RANDOM_STATE, save_table, set_plot_style, write_results_doc
from pxlab.tabular.data import (
    check_columns,
    input_summary,
    plot_correlations,
    plot_target,
    quality_report,
    split_data,
)
from pxlab.tabular.evaluate import (
    comparison_table,
    plot_confusion_matrices,
    plot_roc_curves,
    test_scores,
    wrong_predictions,
)
from pxlab.tabular.explain import (
    parameter_sweep,
    permutation_importance_table,
    plot_parameter_sweep,
    plot_permutation_importance,
    plot_tree_top,
    tree_rules,
)
from pxlab.tabular.train import best_model_name, save_models, train_models

SWEEP_DEPTHS = list(range(1, 16))


def main():
    """
    The full study.

    Returns:
        Nothing.
    """
    spec = LOAN_SPEC
    set_plot_style()
    data = load_loans()
    check_columns(data, spec)
    quality_report(data, spec, f"{FOLDER}/data_quality.csv")
    plot_target(data, spec, f"{FOLDER}/target.png")
    plot_correlations(data, spec, f"{FOLDER}/correlations.png")

    train_features, test_features, train_target, test_target = split_data(data, spec)
    model_folder = os.path.join(MODELS_FOLDER, FOLDER)
    os.makedirs(model_folder, exist_ok=True)
    train_features.assign(decision=train_target).to_csv(os.path.join(model_folder, "train.csv"), index=False)
    test_features.assign(decision=test_target).to_csv(os.path.join(model_folder, "test.csv"), index=False)

    print("Tuning with 5-fold cross-validation:")
    results = train_models(spec, train_features, train_target)
    table = comparison_table(results, test_features, test_target, spec, f"{FOLDER}/model_comparison.csv")
    print(table.to_string(index=False))
    plot_confusion_matrices(results, test_features, test_target, f"{FOLDER}/confusion_matrices.png")
    plot_roc_curves(results, test_features, test_target, spec["positive_class"], f"{FOLDER}/roc_curves.png")

    tree = results["Decision Tree"]["pipeline"]
    rules = tree_rules(tree, spec, max_depth=20, filename=f"{FOLDER}/tree_rules.txt")
    print(f"  The tree has {tree.named_steps['model'].get_n_leaves()} leaves; rules in outputs/{FOLDER}/tree_rules.txt")
    plot_tree_top(tree, spec, depth=3, filename=f"{FOLDER}/decision_tree.png")
    save_table(wrong_predictions(tree, test_features, test_target, spec), f"{FOLDER}/wrong_predictions.csv")

    # Depth sweep: keep the tuned leaf size and class weights, change only the depth.
    best = results["Decision Tree"]["best_params"]
    tree_spec = dict(spec["models"]["Decision Tree"])
    tree_spec["estimator"] = DecisionTreeClassifier(
        random_state=RANDOM_STATE, min_samples_leaf=best["min_samples_leaf"], class_weight=best["class_weight"]
    )
    sweep = parameter_sweep(
        spec, tree_spec, "max_depth", SWEEP_DEPTHS, train_features, train_target, test_features, test_target
    )
    save_table(sweep.round(4), f"{FOLDER}/depth_sweep.csv")
    plot_parameter_sweep(
        sweep, "max_depth", "ROC-AUC", "Decision Tree: ROC-AUC against depth", f"{FOLDER}/depth_sweep.png"
    )

    explained = {name: result for name, result in results.items() if result["role"] != "Baseline"}
    importance = permutation_importance_table(
        explained, test_features, test_target, spec, filename=f"{FOLDER}/permutation_importance.csv"
    )
    plot_permutation_importance(importance, filename=f"{FOLDER}/permutation_importance.png")

    model_info = {
        "dataset": "Statlog German Credit",
        "classes": ["approved", "rejected"],
        "best_model": best_model_name(results),
        "test_scores": {
            name: test_scores(r["pipeline"], test_features, test_target, spec) for name, r in results.items()
        },
        "cv_scores": {name: r["cv_score"] for name, r in results.items()},
        "tree_leaves": int(tree.named_steps["model"].get_n_leaves()),
        "tree_depth": int(tree.named_steps["model"].get_depth()),
    }
    model_info.update(input_summary(train_features, spec))
    save_models(results, model_info, folder=model_folder)

    write_results_doc(
        "Results",
        "Produced by `python main.py`. The test set is 25% of the applicants, stratified, and was not used for "
        "tuning or model choice. The full rules of the tree are in `outputs/loans/tree_rules.txt`.\n\n"
        f"![Top of the Decision Tree](../outputs/figures/{FOLDER}/decision_tree.png)",
        results_sections(),
    )
    print(f"  Rules ({len(rules.splitlines())} lines) and figures saved. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
