# 💳 Credit Card Fraud Detection - End-to-End MLOps Project

## Overview

This project implements an end-to-end Machine Learning and MLOps pipeline for detecting fraudulent credit card transactions.

The solution includes:

* XGBoost fraud detection model
* FastAPI REST API
* Docker containerization
* Deployment on Render
* Prometheus monitoring
* GitHub Actions CI
* Automated testing with Pytest

The goal is not only to build a machine learning model, but also to deploy, monitor, and maintain it following MLOps best practices.

---

## 🚀 Live Application

| Service      | URL                                                     |
| ------------ | ------------------------------------------------------- |
| API          | https://credit-card-fraud-api-tfga.onrender.com/        |
| Swagger Docs | https://credit-card-fraud-api-tfga.onrender.com/docs    |
| Health Check | https://credit-card-fraud-api-tfga.onrender.com/health  |
| Metrics      | https://credit-card-fraud-api-tfga.onrender.com/metrics |

---

## 🏗️ Architecture

```text
Client
   │
   ▼
FastAPI API
   │
   ▼
XGBoost Model
   │
   ▼
Prediction Response

Monitoring
   │
   └── Prometheus Metrics

CI Pipeline
   │
   └── GitHub Actions
```

---

## 🛠️ Tech Stack

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost

### API Development

* FastAPI
* Pydantic
* Uvicorn

### MLOps

* Docker
* Render
* Prometheus
* GitHub Actions
* Pytest

---

## 📂 Project Structure

```text
credit-card-fraud-mlops/
│
├── .github/workflows/ci.yml
├── data/
├── models/
├── notebooks/
├── src/
│   ├── train.py
│   └── predict.py
├── tests/
├── main.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ✨ Features

### Machine Learning

* Fraud detection using XGBoost
* Feature preprocessing and scaling
* Probability-based predictions

### API

* RESTful prediction endpoint
* Input validation using Pydantic
* Interactive Swagger documentation

### Monitoring

* Request metrics
* Response latency
* Memory usage
* CPU usage
* Custom ML metrics:

  * predictions_total
  * fraud_predictions_total

### Testing & CI

* Unit testing with Pytest
* Coverage reporting
* Automated CI pipeline using GitHub Actions

---

## 📊 Example Prediction

### Request

```json
{
  "Time": 0,
  "V1": 1,
  "V2": 1,
  "V3": 1,
  "V4": 1,
  "V5": 1,
  "V6": 1,
  "V7": 1,
  "V8": 1,
  "V9": 1,
  "V10": 1,
  "V11": 1,
  "V12": 1,
  "V13": 1,
  "V14": 1,
  "V15": 1,
  "V16": 1,
  "V17": 1,
  "V18": 1,
  "V19": 1,
  "V20": 1,
  "V21": 1,
  "V22": 1,
  "V23": 1,
  "V24": 1,
  "V25": 1,
  "V26": 1,
  "V27": 1,
  "V28": 1,
  "Amount": 100
}
```

### Response

```json
{
  "fraud_probability": 0.000182,
  "prediction": 0
}
```

---

## 🧪 Running Locally

### Clone the repository

```bash
git clone <your-repository-url>
cd credit-card-fraud-mlops
```

### Create virtual environment

```bash
python -m venv venv
```

### Activate environment

Windows:

```bash
venv\Scripts\activate
```

Linux / Mac:

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the API

```bash
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## 🔍 Monitoring

Prometheus metrics are available at:

```text
/metrics
```

Example custom metrics:

```text
predictions_total 6
fraud_predictions_total 0
```

These metrics provide visibility into model usage and fraud detection activity in production.

---

## ✅ Continuous Integration

GitHub Actions automatically runs:

* Dependency installation
* Unit tests
* Validation on every push to main

Workflow:

```text
.github/workflows/ci.yml
```

---

## 🎯 MLOps Capabilities

* Model Serving with FastAPI
* Containerization with Docker
* Cloud Deployment on Render
* Monitoring with Prometheus
* Automated Testing with Pytest
* CI Pipeline with GitHub Actions

---

## 👨‍💻 Author

**Brahim Azreg**

Machine Learning • Data Science • MLOps
