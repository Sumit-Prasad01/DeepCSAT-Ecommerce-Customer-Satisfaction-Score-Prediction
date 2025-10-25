from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from config.paths_config import *

# ----------------------------
# Initialize FastAPI app
# ----------------------------
app = FastAPI(title="CSAT Prediction API", version="1.0")

# ----------------------------
# Load model and preprocessing
# ----------------------------
MODEL_PATH = MODEL_OUTPUT_PATH
SCALER_PATH = SCALER_PATH
ENCODER_PATH = ENCODER_OUTPUT_PATH

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        label_encoders = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Error loading model or preprocessors: {e}")

# ----------------------------
# Request and Response Schemas
# ----------------------------
class CSATInput(BaseModel):
    channel_name: str
    category: str
    sub_category: str
    agent_name: str
    supervisor: str
    manager: str
    tenure_bucket: str
    agent_shift: str
    item_price: float = 0.0
    connected_handling_time: float = 0.0


class CSATResponse(BaseModel):
    predicted_csat_score: int
    confidence: float

# ----------------------------
# Helper: Preprocess Input
# ----------------------------
def preprocess_input(data: dict) -> np.ndarray:
    df = pd.DataFrame([data])

    # Map tenure bucket
    tenure_map = {
        "On Job Training": 0,
        "0-30": 1,
        "31-60": 2,
        "61-90": 3,
        ">90": 4
    }
    df["tenure_bucket"] = df["tenure_bucket"].map(tenure_map).fillna(0)

    # Label encode categorical columns
    for col, encoder in label_encoders.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
            df[col] = encoder.transform(df[col])

    # Scale numeric data
    X = scaler.transform(df)
    return X

# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def home():
    return {"message": "Welcome to the CSAT Prediction API"}

@app.post("/predict", response_model=CSATResponse)
def predict(input_data: CSATInput):
    try:
        X = preprocess_input(input_data.dict())
        preds = model.predict(X, verbose=0)
        predicted_class = int(np.argmax(preds[0])) + 1
        confidence = float(np.max(preds[0]))
        return {"predicted_csat_score": predicted_class, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
