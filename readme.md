# Smart Manufacturing Quality Prediction

## Overview
An end-to-end machine learning project that predicts potential machine failures from operating conditions such as temperature, rotational speed, torque, and tool wear. The project uses XGBoost, feature engineering, model explainability, and interactive prediction interfaces to support predictive maintenance.

## Problem Statement
Unexpected machine failures can cause production downtime and maintenance costs. This project explores how machine sensor data can estimate failure risk and help prioritize maintenance decisions.

## Dataset
- **Dataset:** AI4I 2020 Predictive Maintenance Dataset
- **Target:** `Machine failure`
- **Task:** Binary classification
- **Class imbalance:** Approximately 3.39% of records represent machine failures.

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
2. Stratified train-test splitting
3. Missing-value imputation and numerical scaling
4. Categorical imputation and one-hot encoding
5. Scikit-learn preprocessing and model pipelines
6. Comparison of Logistic Regression, Decision Tree, Random Forest, and XGBoost
7. Hyperparameter tuning using GridSearchCV and cross-validation
8. Feature engineering and cross-validation comparison
9. Class-imbalance experiments using class weighting and SMOTE
10. Model interpretation using SHAP and permutation importance
11. Threshold exploration and evaluation using precision, recall, F1-score, Average Precision, and a confusion matrix
12. Model persistence using joblib

## Final Model
- **Algorithm:** XGBoost
- **Number of estimators:** 100
- **Maximum tree depth:** 5
- **Learning rate:** 0.05
- **Evaluation threshold:** 0.4

The fitted preprocessing and classification pipeline is saved as `models/machine_failure_model.joblib`.

## Evaluation Results

The following results were recorded on the final test split using a classification threshold of 0.4.

| Metric | Result |
|---|---:|
| Average Precision | 0.8411 |
| Accuracy | 98.65% |
| Failure precision | 85.96% |
| Failure recall | 72.06% |
| Failure F1-score | 78.40% |

### Confusion Matrix

| | Predicted No Failure | Predicted Failure |
|---|---:|---:|
| Actual No Failure | 1924 | 8 |
| Actual Failure | 19 | 49 |

The model correctly detected 49 of 68 failures and missed 19. It also generated 8 false alarms.

**Evaluation caveat:** Earlier experiments used the same overall dataset to inform model and threshold selection. Therefore, these results should be treated as experimental rather than a fully independent estimate of real-world performance. Selecting the threshold using training-only validation or out-of-fold predictions, followed by evaluation on an untouched test set, would provide a more rigorous estimate.

## Model Explainability
SHAP was used to examine global feature importance and individual predictions. Permutation importance was also used to assess how shuffling individual features affected Average Precision.

These methods describe predictive relationships and model behavior; they do not establish causation.

## API and Interactive Dashboard
- **FastAPI:** Provides a `/predict` endpoint that accepts machine operating conditions and returns a failure probability and classification.
- **Streamlit:** Provides an interactive dashboard for entering machine readings and viewing predicted failure risk.
- **Pytest:** Includes automated API tests for successful requests and invalid inputs.
- **Decision threshold:** The API and dashboard use a threshold of 0.4.

## Technologies
- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- SHAP
- imbalanced-learn
- joblib
- FastAPI and Uvicorn
- Streamlit
- Pytest
- Jupyter Notebook

## Project Structure

```text
Smart Manufacturing Quality Prediction/
├── models/
│   └── machine_failure_model.joblib
├── notebook/
│   └── main.ipynb
├── src/
│   └── api.py
├── tests/
│   └── test_api.py
├── app.py
├── dataset.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/kartiksrivastava33-source/smart-manufacturing-quality-prediction.git
cd smart-manufacturing-quality-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit dashboard

```bash
streamlit run app.py
```

### 5. Run the FastAPI service

```bash
python -m uvicorn src.api:app --reload
```

Open `http://127.0.0.1:8000/docs` to test the API through Swagger UI.

### 6. Run automated tests

```bash
python -m pytest tests/ -v
```

## Current Status
The project includes the ML training and evaluation workflow, feature engineering, explainability experiments, a saved model, a FastAPI prediction endpoint, a Streamlit dashboard, and automated API tests.

Containerization, continuous integration, production deployment, and monitoring have not been implemented.

## Limitations
- Evaluation is based on a single held-out split, and earlier experiments influenced model selection.
- Failure events are rare, so accuracy alone is not a sufficient performance measure.
- The threshold of 0.4 is provisional and should be validated without using the final test set for threshold selection.
- The dataset is not a substitute for validation on real operating machines.
- Predicted failure probabilities should not be interpreted as calibrated real-world risk without a dedicated calibration assessment.

## Future Improvements
- Validate the threshold using training-only validation or out-of-fold predictions.
- Evaluate on independent data from real manufacturing equipment.
- Add model and data drift monitoring.
- Containerize the API and dashboard.
- Deploy the application for remote access.
