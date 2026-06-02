# Credit Card Fraud Detection API

## Overview

This project implements an end-to-end machine learning pipeline for detecting fraudulent credit card transactions. The workflow includes data exploration, preprocessing, model training, model persistence, and deployment through a FastAPI REST API.

The objective is to identify potentially fraudulent transactions while minimizing false positives.

---

## Project Structure

```text
credit-card-fraud-mlops/
│
├── README.md
├── requirements.txt
├── main.py
│
├── data/
│   └── raw/
│       └── creditcard.csv
│
├── models/
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   └── threshold.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_preprocessing.ipynb
│
└── src/
    ├── __init__.py
    ├── train.py
    └── predict.py
```

---

## Dataset

The project uses the Credit Card Fraud Detection dataset.

Features:

* `Time`
* `Amount`
* `V1` to `V28` (PCA-transformed features)

Target:

* `Class = 0` → Legitimate transaction
* `Class = 1` → Fraudulent transaction

The dataset is highly imbalanced, with fraudulent transactions representing a very small percentage of all transactions.

---

## Data Preprocessing

The preprocessing pipeline includes:

* Train/Test split using stratification
* Standardization of the `Amount` feature using `StandardScaler`
* Handling class imbalance using SMOTE
* Feature preparation for model training

---

## Models Evaluated

### Logistic Regression

* Baseline model
* Trained on SMOTE-balanced data

### Random Forest

* Ensemble tree-based model
* Evaluated as an alternative to Logistic Regression

### XGBoost

* Gradient boosting model
* Best performing model
* Uses `scale_pos_weight` to address class imbalance
* Custom classification threshold applied

---

## Model Artifacts

The following artifacts are generated after training:

| File                | Description                              |
| ------------------- | ---------------------------------------- |
| `xgboost_model.pkl` | Trained XGBoost model                    |
| `scaler.pkl`        | StandardScaler used for Amount           |
| `threshold.pkl`     | Decision threshold used during inference |

---

## API Deployment

The model is served through a FastAPI application.

### Start the API

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Prediction Endpoint

### Request

**POST** `/predict`

Example:

```json
{
  "Time": 0,
  "V1": -1.359807,
  "V2": -0.072781,
  "V3": 2.536347,
  "V4": 1.378155,
  "V5": -0.338321,
  "V6": 0.462388,
  "V7": 0.239599,
  "V8": 0.098698,
  "V9": 0.363787,
  "V10": 0.090794,
  "V11": -0.5516,
  "V12": -0.617801,
  "V13": -0.99139,
  "V14": -0.311169,
  "V15": 1.468177,
  "V16": -0.470401,
  "V17": 0.207971,
  "V18": 0.025791,
  "V19": 0.403993,
  "V20": 0.251412,
  "V21": -0.018307,
  "V22": 0.277838,
  "V23": -0.110474,
  "V24": 0.066928,
  "V25": 0.128539,
  "V26": -0.189115,
  "V27": 0.133558,
  "V28": -0.021053,
  "Amount": 149.62
}
```

### Response

```json
{
  "fraud_probability": 0.00019,
  "prediction": 0
}
```

Where:

* `prediction = 0` → Legitimate transaction
* `prediction = 1` → Fraudulent transaction

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd credit-card-fraud-mlops
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-Learn (SMOTE)
* XGBoost
* FastAPI
* Uvicorn
* Joblib
* Jupyter Notebook

---

## Future Improvements

* Docker containerization
* Automated testing
* CI/CD pipeline
* MLflow experiment tracking
* Model monitoring
* Cloud deployment

---

## Author

Brahim AZREG

Machine Learning & MLOps Portfolio Project
