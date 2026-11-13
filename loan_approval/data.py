"""
data.py - the Statlog German Credit data: 1,000 loan applicants from a
German bank in the 1990s, each judged a good or bad credit risk
(Hofmann, 1994; UCI Machine Learning Repository, CC BY 4.0). Downloaded
once from OpenML (dataset 31) into data/.

A good risk is treated as "approved" and a bad risk as "rejected": the
bank's judgement is the decision the model learns to copy.
"""

import os

from sklearn.datasets import fetch_openml

from pxlab.common import DATA_FOLDER

# The original codes made readable for the app's drop-downs.
READABLE_VALUES = {
    "checking_status": {
        "<0": "Overdrawn",
        "0<=X<200": "0 to 200 DM",
        ">=200": "200 DM or more",
        "no checking": "No current account",
    },
    "savings_status": {
        "<100": "Under 100 DM",
        "100<=X<500": "100 to 500 DM",
        "500<=X<1000": "500 to 1,000 DM",
        ">=1000": "1,000 DM or more",
        "no known savings": "None known",
    },
    "credit_history": {
        "no credits/all paid": "No loans taken, or all repaid",
        "all paid": "All loans at this bank repaid",
        "existing paid": "Current loans repaid on time so far",
        "delayed previously": "Paid late in the past",
        "critical/other existing credit": "Critical account or loans at other banks",
    },
    "employment": {
        "unemployed": "Unemployed",
        "<1": "Under 1 year",
        "1<=X<4": "1 to 4 years",
        "4<=X<7": "4 to 7 years",
        ">=7": "7 years or more",
    },
    "job": {
        "unemp/unskilled non res": "Unskilled, non-resident",
        "unskilled resident": "Unskilled",
        "skilled": "Skilled",
        "high qualif/self emp/mgmt": "Highly qualified, self-employed or manager",
    },
    "housing": {"for free": "Lives for free", "own": "Owns", "rent": "Rents"},
    "property_magnitude": {
        "real estate": "Real estate",
        "life insurance": "Savings or life insurance",
        "car": "Car or other",
        "no known property": "None known",
    },
    "other_parties": {"none": "None", "co applicant": "Co-applicant", "guarantor": "Guarantor"},
    "other_payment_plans": {"none": "None", "bank": "At another bank", "stores": "At stores"},
}


def load_loans():
    """
    The applicants with readable values and the decision column.

    Returns:
        A DataFrame with the SPEC's input columns and "decision".
    """
    frame = fetch_openml(data_id=31, as_frame=True, data_home=os.path.join(DATA_FOLDER, "openml"), parser="auto").frame
    for column, mapping in READABLE_VALUES.items():
        frame[column] = frame[column].astype(str).map(mapping)
    frame["purpose"] = frame["purpose"].astype(str).str.capitalize()
    frame["decision"] = frame["class"].astype(str).map({"good": "approved", "bad": "rejected"})
    columns = [
        "credit_amount",
        "duration",
        "installment_commitment",
        "existing_credits",
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
        "decision",
    ]
    return frame[columns]
