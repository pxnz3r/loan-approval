"""
spec.py - the loan dataset and the models trained on it.

The pxlab engine reads this dictionary: which column to predict, which
columns are inputs, readable names, and the models with the settings that
grid search tries.

Left out on purpose: sex and marital status, age and foreign-worker status.
They are in the original data, but lenders may not decide on them, and a
rule like "if single male, approve" is exactly what a model shouldn't learn.
"""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from pxlab.common import RANDOM_STATE

# Tree depths tried by grid search and drawn on the depth sweep.
DEPTHS = [2, 3, 4, 5, 6, 8]

LOAN_SPEC = {
    "name": "Loan applications",
    "folder": "loans",
    "task": "classification",
    "target": "decision",
    "positive_class": "approved",
    "numeric": ["credit_amount", "duration", "installment_commitment", "existing_credits"],
    "categorical": [
        "checking_status",
        "savings_status",
        "credit_history",
        "employment",
        "job",
        "purpose",
        "housing",
        "property_magnitude",
        "other_parties",
        "other_payment_plans",
    ],
    "whole_numbers": ["credit_amount", "duration", "installment_commitment", "existing_credits"],
    "labels": {
        "credit_amount": "Loan amount (DM)",
        "duration": "Loan length (months)",
        "installment_commitment": "Repayment as share of income (1 to 4)",
        "existing_credits": "Loans already at this bank",
        "checking_status": "Current account balance",
        "savings_status": "Savings",
        "credit_history": "Credit history",
        "employment": "Years in current job",
        "job": "Job type",
        "purpose": "Purpose of the loan",
        "housing": "Housing",
        "property_magnitude": "Most valuable property",
        "other_parties": "Co-applicant or guarantor",
        "other_payment_plans": "Other repayment plans",
        "decision": "Decision",
    },
    "help": {
        "credit_amount": "In Deutsche Mark, the currency of the original 1990s data.",
        "installment_commitment": "Monthly repayment as a share of disposable income: 1 = under 20%, 4 = over 35%. "
        "The dataset has no income column; this and the balances stand in for it.",
        "checking_status": "Balance of the applicant's current account at this bank.",
        "credit_history": "How past loans were repaid.",
    },
    "test_size": 0.25,
    "scoring": "roc_auc",
    "models": {
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "grid": {"max_depth": DEPTHS, "min_samples_leaf": [5, 10, 20, 40], "class_weight": [None, "balanced"]},
            "scale": False,
            "role": "Main",
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1),
            "grid": {"min_samples_leaf": [1, 5, 10], "class_weight": [None, "balanced"]},
            "scale": False,
            "role": "Benchmark beyond the course",
        },
        "Always approve": {
            "estimator": DummyClassifier(strategy="most_frequent"),
            "grid": {},
            "scale": False,
            "role": "Baseline",
        },
    },
}
