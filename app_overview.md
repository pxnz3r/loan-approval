### What this is

A model that decides whether a loan application is **approved or rejected**, built with a **Decision Tree** so every decision can be read as a short list of questions.

It learns from 1,000 real applicants to a German bank in the 1990s (the Statlog German Credit data). The bank judged each one a good or bad credit risk; here a good risk counts as approved and a bad risk as rejected. 70% were approved.

### How a Decision Tree decides

It asks one yes-or-no question at a time, such as *"Is the current account overdrawn?"* or *"Is the loan longer than 11.5 months?"*, and follows the answer down to the next question until it reaches a leaf. The leaf gives the decision: whatever most training applicants who reached it got.

Training picks, at every step, the question that best separates approved from rejected applicants. Left alone, the tree keeps asking until it has memorised the training data, so its depth and the minimum number of applicants per leaf are tuned with cross-validation.

### What to try

- **Decide**: fill in an application, or start from a real test applicant, and see the decision and the exact questions that led to it.
- **Rules**: every rule the tree learned, the top of the tree as a picture, and what happens as the tree gets deeper.
- **Results**: scores on held-back applicants, the mistakes, which inputs matter and the data checks.

### What was found

- **The current account decides most.** Applicants with no current account at the bank, or a positive balance, are approved far more often than those who are overdrawn. Shuffling this input costs the tree more than every other input together.
- **Then loan length and amount.** Longer and larger loans to people with little savings are rejected.
- **A readable tree costs accuracy.** On 250 test applicants the tree scores ROC-AUC 0.68 and balanced accuracy 0.61. A Random Forest (300 trees voting, not readable) reaches 0.79 and 0.75. Always approving would be right 70% of the time but catches no bad risks.

### Left out on purpose

Sex and marital status, age and foreign-worker status are in the original data but are not used. Lenders may not decide on them, and a rule like "if single male, approve" is exactly what a model shouldn't learn. The data has no income column: the share of income going to repayments, the account and savings balances, and the job stand in for it.

This is a teaching project on old data, not a lending tool.

Code and every figure: [github.com/pxnz3r/loan-approval](https://github.com/pxnz3r/loan-approval).
