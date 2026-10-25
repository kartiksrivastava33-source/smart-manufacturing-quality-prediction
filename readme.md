# Smart Manufacturing Machine Failure Prediction

## Overview
An end-to-end machine learning project that predicts potential machine failures using operating conditions such as temperature, rotational speed, torque, and tool wear. The project uses XGBoost and feature engineering to identify patterns associated with machine failure.

## Problem Statement
Unexpected machine failures can cause downtime and maintenance costs. This project explores how machine sensor data can be used to estimate failure risk and support predictive maintenance decisions.

## Dataset
- **Dataset:** AI4I 2020 Predictive Maintenance Dataset
- **Target:** `Machine failure`
- **Task:** Binary classification
- **Challenge:** Class imbalance, with failures representing approximately 3.39% of the dataset.

## Features Used
- Machine type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- **Engineered feature:** Temperature difference = Process temperature − Air temperature

Identifier columns and failure-mode indicators were excluded from model inputs to reduce leakage risk.

## Machine Learning Workflow
1. Data cleaning and exploratory data analysis
2. Train-test split with stratification
3. Numerical imputation and scaling
4. Categorical imputation and one-hot encoding
5. Scikit-learn preprocessing and model pipelines
6. Comparison of Logistic Regression, Decision Tree, Random Forest, and XGBoost
7. Hyperparameter tuning with GridSearchCV and cross-validation
8. Feature engineering and cross-validation comparison
9. Class-imbalance experiments using class weighting and SMOTE
10. Model interpretation using SHAP and permutation importance
11. Evaluation using precision, recall, F1-score, Average Precision, and a confusion matrix
12. Model persistence using joblib

## Final Model
- **Algorithm:** XGBoost
- **Number of estimators:** 100
- **Maximum tree depth:** 5
- **Learning rate:** 0.05
- **Decision threshold used for final evaluation:** 0.4

The fitted preprocessing and classification pipeline is saved as `models/machine_failure_model.joblib`.

## Final Test Results
The recorded results on the final test split were:

| Metric | Result |
|---|---:|
| Average Precision | 0.8411 |
| Accuracy | 98.6% |
| Failure precision | 90.0% |
| Failure recall | 66.2% |
| Failure F1-score | 76.3% |

### Confusion Matrix

| | Predicted No Failure | Predicted Failure |
|---|---:|---:|
| Actual No Failure | 1927 | 5 |
| Actual Failure | 23 | 45 |

The model detected 45 of 68 failures and missed 23. The results illustrate a trade-off between detecting failures and avoiding false alarms.

**Evaluation caveat:** Earlier experiments influenced model selection using the same overall dataset. These metrics are useful for this project, but should not be presented as an entirely independent estimate of real-world performance. External validation would provide stronger evidence of generalization.

## Model Explainability
SHAP was used to investigate global feature importance and individual predictions. Permutation importance was also used to measure how shuffling individual features affected Average Precision.

These methods describe model behavior and predictive relationships; they do not establish causation.

## Technologies
- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- SHAP
- imbalanced-learn
- joblib
- Jupyter Notebook

## Current Status
The core ML workflow, model evaluation, explainability experiments, and model saving/loading have been implemented. API development, automated tests, containerization, CI, and monitoring are planned as future steps.

## Limitations
- Performance has been evaluated on a single dataset split.
- Failure events are rare, so recall and precision should be considered alongside accuracy.
- The selected