from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()
model  = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

class InputData(BaseModel):
    score: float
    controversiality: float
    user_comment_karma: float
    user_link_karma: float
    user_total_karma: float
    account_age_days: float
    comment_length: float

@app.get("/")
def root():
    return {"status": "Model is running!"}

@app.post("/predict")
def predict(data: InputData):
    features = np.array([[
        data.score, data.controversiality,
        data.user_comment_karma, data.user_link_karma,
        data.user_total_karma, data.account_age_days,
        data.comment_length
    ]])

    features_scaled = scaler.transform(features)
    proba      = model.predict_proba(features_scaled)[0][1]
    prediction = 1 if proba >= 0.30 else 0
    label      = "Akun Buzzer" if prediction == 1 else "Pengguna Asli"

    return {
        "prediction": prediction,
        "label": label,
        "confidence": round(float(proba), 4)
    }
