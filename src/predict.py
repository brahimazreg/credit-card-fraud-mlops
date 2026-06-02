import joblib
import pandas as pd

model = joblib.load("models/xgboost_model.pkl")
scaler = joblib.load("models/scaler.pkl")
threshold = joblib.load("models/threshold.pkl")

def predict_transaction(data):
    df = pd.DataFrame([data])

    df["Amount"] = scaler.transform(df[["Amount"]])

    proba = model.predict_proba(df)[:, 1][0]
    pred = int(proba > threshold)

    return {
        "fraud_probability": float(proba),
        "prediction": pred
    }