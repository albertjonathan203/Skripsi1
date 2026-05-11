from fastapi import FastAPI
import joblib
import numpy as np
import re
from scipy.sparse import hstack, csr_matrix
from pydantic import BaseModel

app = FastAPI()
model  = joblib.load("model.pkl")
tfidf  = joblib.load("tfidf.pkl")
scaler = joblib.load("scaler.pkl")

class InputData(BaseModel):
    text: str
    score: float
    controversiality: float
    user_comment_karma: float
    user_link_karma: float
    user_total_karma: float
    account_age_days: float
    comment_length: float

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.get("/")
def root():
    return {"status": "Model Hybrid is running!"}

@app.post("/predict")
def predict(data: InputData):
    # Fitur teks
    clean = clean_text(data.text)
    X_tfidf = tfidf.transform([clean])

    # Fitur numerik
    numerik = np.array([[
        data.score, data.controversiality,
        data.user_comment_karma, data.user_link_karma,
        data.user_total_karma, data.account_age_days,
        data.comment_length
    ]])
    X_num_scaled = scaler.transform(numerik)

    # Gabungkan
    X_combined = hstack([X_tfidf, csr_matrix(X_num_scaled)])

    proba = model.predict_proba(X_combined)[0][1]
    prediction = 1 if proba >= 0.30 else 0
    label = "Akun Buzzer" if prediction == 1 else "Pengguna Asli"

    return {
        "prediction": prediction,
        "label": label,
        "confidence": round(float(proba), 4)
    }
