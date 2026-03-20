# Home Credit Default Risk — Predicting Loan Repayment

## Context

This project is based on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) Kaggle competition. Home Credit is an international lender that serves people with little or no credit history. The goal is to predict whether a borrower will have difficulty repaying a loan, using historical application data.

This is a **binary classification** problem: each loan is labeled 0 (repaid on time) or 1 (payment difficulties). The evaluation metric is **AUC-ROC** (Area Under the ROC Curve).

## Why this matters

Credit scoring is at the core of banking and financial risk management. Being able to identify risky borrowers before granting a loan directly impacts a bank's profitability and stability. This project applies machine learning techniques to a real-world financial dataset of 307,000+ loan applications.

## Approach

The notebook follows a standard end-to-end ML pipeline:

1. **Exploratory Data Analysis** — target distribution, missing values, feature types
2. **Preprocessing** — one-hot encoding, removal of columns with >50% missing values, median imputation
3. **Correlation analysis** — identifying the strongest predictors of default
4. **Modeling** — three models of increasing complexity:
   - Logistic Regression (baseline)
   - Random Forest
   - XGBoost
5. **Model comparison** — ROC curves and AUC scores side by side
6. **Feature importance** — comparison across all three models
7. **Kaggle submission** — generating the final prediction file

## Tools

- Python (pandas, numpy, matplotlib, scikit-learn, xgboost)
- Jupyter Notebook

## Dataset

Available on Kaggle: [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data). Only the main tables (`application_train.csv` and `application_test.csv`) are used in this project.
