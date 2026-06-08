import joblib
import pandas as pd
from prometheus_client import Counter

model = joblib.load("models/xgboost_model.pkl")
scaler = joblib.load("models/scaler.pkl")
threshold = joblib.load("models/threshold.pkl")

prediction_counter = Counter(
    "predictions_total",
    "Total predictions made"
)

fraud_counter = Counter(
    "fraud_predictions_total",
    "Fraud predictions made"
)

def predict_transaction(data):
    df = pd.DataFrame([data])

    df["Amount"] = scaler.transform(df[["Amount"]])

    proba = model.predict_proba(df)[:, 1][0]
    prediction_counter.inc()
    pred = int(proba > threshold)
    if pred == 1:
        fraud_counter.inc()

    return {
        "fraud_probability": float(proba),
        "prediction": pred
    }