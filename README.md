# Loan Approval Prediction

A machine learning project that predicts whether a bank will **approve or reject** a loan application from the applicant's details, and compares three classification algorithms to find the best one.

This is a **classification** problem: the goal is to predict a category (Approved / Rejected), not a number.

## Objective

When a loan application arrives, the bank must decide: approve (`Y`) or reject (`N`)? We use the applicant's details to predict that decision.

## Dataset

The "Loan Approval Prediction Dataset" (4,269 loan applications, no missing values). Save it as `dataset/loan_data.csv`. Amounts are in rupees and `loan_term` is in years.

| Column | Meaning |
|---|---|
| `loan_id` | Application ID (dropped, it carries no information) |
| `no_of_dependents` | Number of people depending on the applicant |
| `education` | Graduate or Not Graduate |
| `self_employed` | Yes or No |
| `income_annum` | Annual income |
| `loan_amount` | Amount requested |
| `loan_term` | Loan duration in years |
| `cibil_score` | Credit score (300 to 900) |
| `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, `bank_asset_value` | Value of the applicant's assets |
| `loan_status` | **Target**: Approved or Rejected |

The raw file has extra spaces in its column names and text values (for example `" education"` and `" Graduate"`), which the script removes. It also drops 28 rows that have a negative asset value, which is impossible.

## Project Phases

1. Dataset collection
2. Data exploration
3. Data cleaning
4. Data visualization
5. Data preprocessing (feature building, train/test split, pipelines)
6. Feature selection
7. Model training
8. Model evaluation
9. Algorithm comparison
10. Final report

See [WORKFLOW.md](WORKFLOW.md) for a diagram of how the data flows through these phases.

## Algorithms Compared

| Algorithm | Purpose |
|---|---|
| Logistic Regression | Simple, fast model that estimates the probability of approval |
| Decision Tree Classifier | Makes decisions using yes/no rules |
| Random Forest Classifier | Combines many decision trees for steadier predictions |
| Baseline (always approve) | A "lazy" model every real model must beat |

## Evaluation Metrics

- **Accuracy**: share of predictions that were correct.
- **Precision**: of the loans the model approved, how many deserved approval.
- **Recall**: of the loans that deserved approval, how many the model found.
- **F1 score**: a balance of precision and recall.
- **Macro F1**: F1 averaged over both Approved and Rejected, so a model cannot win just by approving almost everyone. This is the score used to pick the best model.
- **Reject recall**: of the loans that should be rejected, how many the model caught.
- **ROC AUC**: how well the model separates approved from rejected (0.5 = guessing, 1.0 = perfect).
- **Confusion matrix**: a table of which predictions were right and wrong.

### Why not accuracy alone?

About 62% of applications in this dataset are approved. A model that says "approve everyone" would already be right about 62% of the time, but it would never catch a bad loan. That is why the project includes the baseline and the extra metrics above.

## Project Structure

```
Loan_Approval_Prediction/
│
├── dataset/
│   └── loan_data.csv
│
├── notebooks/
│
├── results/
│   ├── visualizations/
│   ├── model_comparison.csv
│   ├── final_report.txt
│   └── best_model.joblib
│
├── src/
│
├── main.py
├── requirements.txt
├── README.md
└── WORKFLOW.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Loan_Approval_Prediction
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
```

### 3. Install the libraries

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Save the loan dataset as `dataset/loan_data.csv`. If the file is missing, `main.py` tells you what to do and stops.

### 5. Run the project

```bash
python main.py
```

## Outputs

After running, you will find:

- `results/visualizations/`: charts (approval counts, CIBIL score by decision and by score band, approval rate by other details, income and loan amount charts, correlation heatmap, feature importance, model comparison, confusion matrix, ROC curves)
- `results/model_comparison.csv`: all scores for each model
- `results/final_report.txt`: short summary with the best model
- `results/best_model.joblib`: the trained best model, ready to reuse

### Using the saved model

```python
import joblib
import pandas as pd
from main import engineer_features  # or copy the function into your own script

model = joblib.load("results/best_model.joblib")
applicant = pd.DataFrame([{
    "no_of_dependents": 2, "education": "Graduate", "self_employed": "No",
    "income_annum": 6000000, "loan_amount": 15000000, "loan_term": 12,
    "cibil_score": 750, "residential_assets_value": 5000000,
    "commercial_assets_value": 3000000, "luxury_assets_value": 10000000,
    "bank_asset_value": 4000000,
}])
print(model.predict_proba(engineer_features(applicant))[0, 1])
```

Note: importing from `main` runs the whole script. For reuse in other code, move `engineer_features` into `src/`.

## Results

Fill this table in with the numbers from your `results/model_comparison.csv` after running the project.

| Model | Accuracy | Precision | Recall | F1 | Macro F1 | Reject recall | ROC AUC |
|---|---|---|---|---|---|---|---|
| Baseline (always approve) | | | | | | | |
| Logistic Regression | | | | | | | |
| Decision Tree | | | | | | | |
| Random Forest | | | | | | | |

**Best model:** _to be filled in_

## Key Findings

_Add your own conclusions here after viewing the charts and results, for example which features affect approval most and whether the best model clearly beats the baseline._

## Notes on Method

- **No data leakage:** any missing values are filled inside the model pipeline using training data only, so the test set stays unseen. (This dataset has none, but the pipeline is built to handle them.)
- **Feature building:** `total_assets` adds the four asset values together, and `loan_to_income` shows how many years of income the loan equals.
- **Imbalance:** models use `class_weight="balanced"` so the smaller group (rejected loans) gets extra attention.
- **One feature dominates:** in this dataset the CIBIL score drives most decisions. Applicants with a score of about 550 or more are almost always approved, and those below are almost always rejected. This is why tree models score very high. Discuss it in your Key Findings, and treat the very high scores with care, since real bank decisions are rarely this clean.

## Tech Stack

- Python
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn, joblib
- Jupyter Notebook

## Future Improvements

- Tune hyperparameters with grid search
- Try Gradient Boosting or XGBoost
- Re-run without `cibil_score` to see how well the other details predict approval on their own
- Add a threshold choice (approve only when probability is above a chosen cut-off)
- Build a small web app where a user enters applicant details and gets a prediction
