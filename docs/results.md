# Results

Produced by `python main.py`. The test set is 25% of the applicants, stratified, and was not used for tuning or model choice. The full rules of the tree are in `outputs/loans/tree_rules.txt`.

![Top of the Decision Tree](../outputs/figures/loans/decision_tree.png)

## Models

### Test-set scores

| Model | Role | CV roc auc | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decision Tree | Main | 0.772 | 0.7 | 0.607 | 0.758 | 0.84 | 0.797 | 0.682 |
| Random Forest | Benchmark beyond the course | 0.789 | 0.744 | 0.749 | 0.878 | 0.737 | 0.801 | 0.785 |
| Always approve | Baseline | 0.5 | 0.7 | 0.5 | 0.7 | 1.0 | 0.824 | 0.5 |

![confusion matrices](../outputs/figures/loans/confusion_matrices.png)

![roc curves](../outputs/figures/loans/roc_curves.png)

## Tree depth

![depth sweep](../outputs/figures/loans/depth_sweep.png)

### ROC-AUC against tree depth

| max_depth | CV score | Test score |
| --- | --- | --- |
| 1 | 0.6708 | 0.6752 |
| 2 | 0.6927 | 0.7048 |
| 3 | 0.741 | 0.6918 |
| 4 | 0.7589 | 0.7029 |
| 5 | 0.7642 | 0.6806 |
| 6 | 0.7715 | 0.6822 |
| 7 | 0.7699 | 0.6712 |
| 8 | 0.7702 | 0.6722 |
| 9 | 0.7702 | 0.6722 |
| 10 | 0.7702 | 0.6722 |
| 11 | 0.7702 | 0.6722 |
| 12 | 0.7702 | 0.6722 |
| 13 | 0.7702 | 0.6722 |
| 14 | 0.7702 | 0.6722 |
| 15 | 0.7702 | 0.6722 |

## Which inputs matter

![permutation importance](../outputs/figures/loans/permutation_importance.png)

### Permutation importance (drop in ROC-AUC when the input is shuffled)

|  | Decision Tree | Random Forest |
| --- | --- | --- |
| Current account balance | 0.1216 | 0.138 |
| Loan length (months) | 0.0311 | 0.0145 |
| Loan amount (DM) | 0.0101 | 0.0083 |
| Credit history | 0.0076 | 0.0112 |
| Other repayment plans | 0.0068 | 0.0036 |
| Loans already at this bank | 0.0 | 0.0006 |
| Co-applicant or guarantor | 0.0 | 0.0071 |
| Housing | 0.0 | -0.0017 |
| Purpose of the loan | 0.0 | 0.0063 |
| Job type | 0.0 | 0.0039 |
| Most valuable property | -0.0021 | -0.0017 |
| Savings | -0.0042 | 0.0135 |
| Repayment as share of income (1 to 4) | -0.0065 | 0.0006 |
| Years in current job | -0.0156 | -0.0009 |

## Mistakes

### Test applicants the Decision Tree got wrong

| credit_amount | duration | installment_commitment | existing_credits | checking_status | savings_status | credit_history | employment | job | purpose | housing | property_magnitude | other_parties | other_payment_plans | Real | Predicted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5152 | 24 | 4 | 1 | 200 DM or more | Under 100 DM | Current loans repaid on time so far | 4 to 7 years | Skilled | Radio/tv | Owns | Car or other | nan | At another bank | approved | rejected |
| 2570 | 27 | 3 | 1 | No current account | Under 100 DM | Current loans repaid on time so far | 1 to 4 years | Skilled | New car | Rents | Real estate | nan | nan | rejected | approved |
| 3857 | 30 | 4 | 1 | Overdrawn | Under 100 DM | Current loans repaid on time so far | 1 to 4 years | Highly qualified, self-employed or manager | Used car | Owns | Savings or life insurance | nan | nan | approved | rejected |
| 1282 | 12 | 2 | 1 | Overdrawn | Under 100 DM | Current loans repaid on time so far | 1 to 4 years | Skilled | Furniture/equipment | Rents | Car or other | nan | nan | rejected | approved |
| 9398 | 36 | 1 | 1 | 0 to 200 DM | Under 100 DM | Current loans repaid on time so far | Under 1 year | Highly qualified, self-employed or manager | Used car | Rents | Car or other | nan | nan | rejected | approved |
| 1199 | 24 | 4 | 2 | Overdrawn | Under 100 DM | Critical account or loans at other banks | 7 years or more | Unskilled | New car | Owns | Car or other | nan | nan | rejected | approved |
| 2292 | 12 | 4 | 2 | No current account | Under 100 DM | Critical account or loans at other banks | Unemployed | Highly qualified, self-employed or manager | Business | Owns | Car or other | nan | At stores | rejected | approved |
| 7511 | 18 | 1 | 1 | Overdrawn | None known | Current loans repaid on time so far | 7 years or more | Skilled | Used car | Lives for free | Savings or life insurance | nan | nan | rejected | approved |
| 1275 | 24 | 2 | 2 | 200 DM or more | 1,000 DM or more | Critical account or loans at other banks | 1 to 4 years | Skilled | Business | Owns | Real estate | nan | nan | approved | rejected |
| 2384 | 24 | 4 | 1 | Overdrawn | Under 100 DM | Current loans repaid on time so far | 7 years or more | Unskilled | Radio/tv | Rents | Real estate | nan | At another bank | approved | rejected |
| 1484 | 12 | 2 | 1 | 0 to 200 DM | None known | Current loans repaid on time so far | 1 to 4 years | Skilled | Radio/tv | Owns | Real estate | nan | nan | rejected | approved |
| 5951 | 48 | 2 | 1 | 0 to 200 DM | Under 100 DM | Current loans repaid on time so far | 1 to 4 years | Skilled | Radio/tv | Owns | Real estate | nan | nan | rejected | approved |
| 1209 | 6 | 4 | 1 | 0 to 200 DM | Under 100 DM | Paid late in the past | Unemployed | Highly qualified, self-employed or manager | New car | Owns | Savings or life insurance | nan | nan | rejected | approved |
| 2149 | 12 | 4 | 1 | Overdrawn | Under 100 DM | All loans at this bank repaid | 1 to 4 years | Skilled | Radio/tv | Lives for free | None known | nan | nan | rejected | approved |
| 6350 | 30 | 4 | 1 | Overdrawn | None known | Current loans repaid on time so far | 7 years or more | Skilled | Furniture/equipment | Owns | Savings or life insurance | nan | nan | rejected | approved |
| 1603 | 24 | 4 | 1 | Overdrawn | Under 100 DM | Current loans repaid on time so far | 7 years or more | Skilled | Radio/tv | Owns | Car or other | nan | nan | approved | rejected |
| 2820 | 36 | 4 | 2 | 0 to 200 DM | Under 100 DM | Critical account or loans at other banks | Under 1 year | Skilled | New car | Owns | Car or other | nan | nan | rejected | approved |
| 626 | 12 | 4 | 1 | Overdrawn | Under 100 DM | All loans at this bank repaid | 1 to 4 years | Unskilled | Radio/tv | Owns | Real estate | nan | At another bank | rejected | approved |
| 12389 | 36 | 1 | 1 | 0 to 200 DM | None known | Current loans repaid on time so far | 1 to 4 years | Skilled | New car | Lives for free | None known | nan | nan | rejected | approved |
| 3499 | 12 | 3 | 2 | Overdrawn | Under 100 DM | Critical account or loans at other banks | 1 to 4 years | Skilled | New car | Owns | Real estate | Co-applicant | nan | rejected | approved |

## Data

### Data quality

| Column | Type | Missing | Distinct values |
| --- | --- | --- | --- |
| credit_amount | numeric | 0 | 921 |
| duration | numeric | 0 | 33 |
| installment_commitment | numeric | 0 | 4 |
| existing_credits | numeric | 0 | 4 |
| checking_status | category | 0 | 4 |
| savings_status | category | 0 | 5 |
| credit_history | category | 0 | 5 |
| employment | category | 0 | 5 |
| job | category | 0 | 4 |
| purpose | category | 0 | 10 |
| housing | category | 0 | 3 |
| property_magnitude | category | 0 | 4 |
| other_parties | category | 0 | 3 |
| other_payment_plans | category | 0 | 3 |
| decision | target | 0 | 2 |

![target](../outputs/figures/loans/target.png)

![correlations](../outputs/figures/loans/correlations.png)
