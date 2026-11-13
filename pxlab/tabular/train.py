"""
tabular/train.py - preprocessing, model tuning with cross-validation, and
saving or loading the trained models.

SPEC["models"] lists the models to train:
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "grid": {"max_depth": [3, 5, None]},   # values to try
        "scale": False,                         # standardise numbers first?
        "role": "Main",                         # shown in tables
    }
SPEC["scoring"] is the scikit-learn score the tuning maximises, e.g.
"f1_macro" or "neg_root_mean_squared_error".
"""

import json
import os
import re
import time

import joblib
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pxlab.common import MODELS_FOLDER, RANDOM_STATE

MODEL_INFO_FILE = "model_info.json"


def build_preprocessor(spec, scale):
    """
    The preprocessing step: fill gaps, one-hot encode categories, and
    optionally standardise numbers.

    Why standardise? Distance-based models (KNN) and gradient-based ones
    (neural networks, linear models with regularisation) treat a feature
    measured in thousands as far more important than one measured in
    fractions. Scaling every number to mean 0 and spread 1 puts them on
    equal footing. Trees split one feature at a time, so they don't need it.

    One-hot encoding turns a category column like Colour = red/green into
    0/1 columns Colour_red and Colour_green, because models need numbers.

    Inputs:
        spec: the project SPEC.
        scale: True to standardise the numeric columns.
    Returns:
        An unfitted ColumnTransformer.
    """
    numeric_steps = [("fill", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    parts = []
    if spec["numeric"]:
        parts.append(("numeric", Pipeline(numeric_steps), list(spec["numeric"])))
    if spec["categorical"]:
        categorical_steps = Pipeline(
            [
                ("fill", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        parts.append(("categorical", categorical_steps, list(spec["categorical"])))
    return ColumnTransformer(parts, verbose_feature_names_out=False)


def build_pipeline(spec, model_spec):
    """
    Preprocessing + model in one scikit-learn Pipeline.

    Keeping them together means the preprocessing is fitted only on the
    training folds during cross-validation, so no information from the
    validation or test rows leaks into training.

    Inputs:
        spec: the project SPEC.
        model_spec: one entry of SPEC["models"].
    Returns:
        An unfitted Pipeline with steps "preprocess" and "model".
    """
    return Pipeline(
        [
            ("preprocess", build_preprocessor(spec, model_spec.get("scale", False))),
            ("model", clone(model_spec["estimator"])),
        ]
    )


def cross_validation_folds(spec, folds=5):
    """
    The folds used for tuning: stratified for classification, plain shuffled
    folds for regression, time-ordered folds when SPEC["time_order"] is set.

    Inputs:
        spec: the project SPEC.
        folds: how many folds.
    Returns:
        A scikit-learn splitter.
    """
    if spec.get("time_order"):
        # Rows are in time order: each fold trains on the past and validates
        # on the period right after it, never on the future.
        return TimeSeriesSplit(n_splits=folds)
    if spec["task"] == "classification":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    return KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)


def scorer_for(spec):
    """
    The score used for tuning and sweeps. F1, precision and recall need to
    know which class is "positive"; scikit-learn assumes the label 1, so for
    text labels like "spam" the scorer is built with the SPEC's
    positive_class.

    Input:
        spec: the project SPEC.
    Returns:
        A scoring name or a scorer that scikit-learn accepts.
    """
    metrics = {"f1": f1_score, "precision": precision_score, "recall": recall_score}
    if spec.get("positive_class") is not None and spec["scoring"] in metrics:
        return make_scorer(metrics[spec["scoring"]], pos_label=spec["positive_class"])
    return spec["scoring"]


def train_models(spec, train_features, train_target):
    """
    Tune and train every model in SPEC["models"] with grid search and
    5-fold cross-validation.

    Grid search tries every combination of the values in "grid". For each
    combination, cross-validation trains on 4/5 of the training data and
    scores on the remaining 1/5, five times, and averages the scores. The
    best combination is then refitted on the whole training set.

    Inputs:
        spec: the project SPEC.
        train_features, train_target: the training data.
    Returns:
        A dictionary: model name -> {"pipeline", "best_params", "cv_score",
        "role", "seconds"}.
    """
    results = {}
    for model_name, model_spec in spec["models"].items():
        started = time.time()
        grid = {"model__" + name: values for name, values in model_spec.get("grid", {}).items()}
        search = GridSearchCV(
            build_pipeline(spec, model_spec),
            grid,
            scoring=scorer_for(spec),
            cv=cross_validation_folds(spec),
            n_jobs=-1,
        )
        search.fit(train_features, train_target)
        best_params = {name.replace("model__", ""): value for name, value in search.best_params_.items()}
        results[model_name] = {
            "pipeline": search.best_estimator_,
            "best_params": best_params,
            "cv_score": float(search.best_score_),
            "role": model_spec.get("role", ""),
            "seconds": round(time.time() - started, 1),
        }
        print(f"  {model_name}: CV {spec['scoring']} = {search.best_score_:.3f} with {best_params}")
    return results


def best_model_name(results):
    """
    The model with the highest cross-validated score.

    Inputs:
        results: from train_models().
    Returns:
        The model name.
    """
    return max(results, key=lambda name: results[name]["cv_score"])


def file_name_for(model_name):
    """
    A safe file name for a model, e.g. "Decision Tree" -> "decision_tree.joblib".

    Input:
        model_name: the model's name.
    Returns:
        The file name.
    """
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_") + ".joblib"


def plain_value(value):
    """
    Turn a parameter value into something JSON can store.

    Input:
        value: e.g. 5, None, "gini" or an object.
    Returns:
        The value, or its text form when JSON can't store it.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def save_models(results, model_info, folder=MODELS_FOLDER):
    """
    Save every trained pipeline with joblib and the model information as JSON.

    Inputs:
        results: from train_models().
        model_info: a dictionary with anything the app needs (scores,
            defaults, ranges...). The file names are added to it.
        folder: where to save.
    Returns:
        The model_info dictionary that was written.
    """
    os.makedirs(folder, exist_ok=True)
    model_info["model_files"] = {}
    model_info["best_params"] = {}
    for model_name, result in results.items():
        file_name = file_name_for(model_name)
        # compress=3 keeps big models (random forests) small enough for GitHub.
        joblib.dump(result["pipeline"], os.path.join(folder, file_name), compress=3)
        model_info["model_files"][model_name] = file_name
        model_info["best_params"][model_name] = {k: plain_value(v) for k, v in result["best_params"].items()}
    with open(os.path.join(folder, MODEL_INFO_FILE), "w") as file:
        json.dump(model_info, file, indent=2)
    print("  Saved", len(results), "models to", folder + "/")
    return model_info


def models_are_saved(folder=MODELS_FOLDER):
    """
    Check that main.py has been run and every saved model exists.

    Input:
        folder: the models folder.
    Returns:
        True when model_info.json and every model file are present.
    """
    info_path = os.path.join(folder, MODEL_INFO_FILE)
    if not os.path.isfile(info_path):
        return False
    with open(info_path) as file:
        files = json.load(file)["model_files"].values()
    return all(os.path.isfile(os.path.join(folder, name)) for name in files)


def load_models(folder=MODELS_FOLDER):
    """
    Load model_info.json and every saved pipeline.

    Input:
        folder: the models folder.
    Returns:
        (model_info, models) where models maps name -> fitted pipeline.
    """
    with open(os.path.join(folder, MODEL_INFO_FILE)) as file:
        model_info = json.load(file)
    models = {}
    for model_name, file_name in model_info["model_files"].items():
        models[model_name] = joblib.load(os.path.join(folder, file_name))
    return model_info, models
