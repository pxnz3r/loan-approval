# Methodology

## Question

Can a Decision Tree learn to approve or reject loan applications the way a bank did, and what rules does it learn?

## Data

Statlog German Credit: 1,000 applicants, 700 judged good credit risks and 300 bad. A good risk is treated as approved and a bad risk as rejected.

| Group | Inputs used |
| --- | --- |
| Loan | amount (DM), length (months), purpose |
| Credit history | how past loans were repaid, loans already at this bank, other repayment plans |
| Money (income stand-ins) | repayment as a share of disposable income (1 to 4), current account balance, savings |
| Work and home | years in current job, job type, housing, most valuable property, co-applicant or guarantor |

The original category codes (for example `A11`, `<0`) are turned into readable values in `loan_approval/data.py`.

**Left out:** sex and marital status, age, foreign-worker status (protected characteristics in lending), plus telephone, years at current address and number of dependents (little to do with the decision and they lengthen the form).

## Steps

1. **Checks**: missing values, duplicates, the approved/rejected split and correlations.
2. **Split**: 75% training, 25% test, stratified so both keep the 70/30 split.
3. **Models**, each in a scikit-learn Pipeline (median fill, one-hot encoding for categories):
   - Decision Tree, tuned over depth (2 to 8), minimum applicants per leaf (5 to 40) and class weights (none or balanced).
   - Random Forest, 300 trees, a benchmark beyond the course.
   - Always approve, the baseline.
4. **Tuning**: 5-fold stratified cross-validation scored by ROC-AUC, which judges how well the model ranks applicants regardless of the 50% cut-off and isn't fooled by the 70/30 imbalance.
5. **Test scores**: accuracy, balanced accuracy (the average of the recall for each class), precision, recall and F1 for "approved", and ROC-AUC. Confusion matrices and ROC curves.
6. **Rules**: the full tree as text (`outputs/loans/tree_rules.txt`), its top three levels as a picture, and in the app the path one application takes.
7. **Depth sweep**: depth 1 to 15 with the other tuned settings fixed, cross-validation and test ROC-AUC.
8. **Importance**: permutation importance on the test set, 10 shuffles per input.
9. **In-depth analysis** (`pxlab.tabular.study`): 1,000-round bootstrap 95% intervals for the headline test scores, next to a baseline that ignores the inputs; McNemar's test of the best model against each other model; learning curves; a cut-off chosen on cross-validated training predictions for the lowest cost (a wrongly approved bad borrower costs 5, a wrongly rejected good one costs 1, the dataset's own cost matrix) and then checked on the test set; a reliability diagram, Brier score and a recalibrated copy of the best model; and an error analysis that compares the inputs of caught, missed and falsely flagged cases and shows how accuracy depends on the model's confidence. Each finding is one sentence in `outputs/loans/key_findings.csv`.

## Limits

- 1,000 applicants is small. Test ROC-AUC (0.68) is well below cross-validation (0.77): with 250 test applicants a few odd cases move it a lot.
- The labels are one bank's judgement from the 1990s, so the model copies that judgement, including any of its biases, not the truth about who repays.
- The original data comes with a cost matrix (approving a bad risk costs five times more than rejecting a good one). This project uses the plain 50% cut-off; raising the cut-off would reject more applicants and lose fewer loans.
- Leaving protected inputs out doesn't make the model fair by itself: other inputs, like job type or housing, can stand in for them.
