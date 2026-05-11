from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("model.pkl")

class InputData(BaseModel):
    features: list

@app.get("/")
def root():
    return {"status": "Model is running!"}

@app.post("/predict")
def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)
    
    label = "Akun Buzzer" if prediction[0] == 1 else "Pengguna Asli"
    
    return {
        "prediction": int(prediction[0]),
        "label": label
    }
