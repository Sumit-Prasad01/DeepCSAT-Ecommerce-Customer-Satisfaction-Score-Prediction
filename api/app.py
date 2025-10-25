from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from config.paths_config import *

app = FastAPI(title="CSAT Prediction API", version="1.0")

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

CATEGORICAL_FEATURES = [
    "channel_name",
    "category",
    "Sub-category",
    "Customer Remarks",
    "Customer_City",
    "Product_category",
    "Agent_name",
    "Supervisor",
    "Manager",
    "Agent Shift"
]

ALL_FEATURES_ORDER = [
    "channel_name",
    "category",
    "Sub-category",
    "Customer Remarks",
    "Customer_City",
    "Product_category",
    "Item_price",
    "connected_handling_time",
    "Agent_name",
    "Supervisor",
    "Manager",
    "Tenure Bucket",
    "Agent Shift"
]

COLUMN_RENAME_MAP = {
    "sub_category": "Sub-category",
    "customer_remarks": "Customer Remarks",
    "customer_city": "Customer_City",
    "product_category": "Product_category",
    "item_price": "Item_price",
    "agent_name": "Agent_name",
    "supervisor": "Supervisor",
    "manager": "Manager",
    "tenure_bucket": "Tenure Bucket",
    "agent_shift": "Agent Shift"
}

class CSATInput(BaseModel):
    channel_name: str
    category: str
    sub_category: str
    customer_remarks: str
    customer_city: str
    product_category: str
    item_price: float
    connected_handling_time: float
    agent_name: str
    supervisor: str
    manager: str
    tenure_bucket: str
    agent_shift: str

class CSATResponse(BaseModel):
    predicted_csat_score: int
    confidence: float

def preprocess_input(data: dict) -> np.ndarray:
    df = pd.DataFrame([data])
    df.rename(columns=COLUMN_RENAME_MAP, inplace=True)

    tenure_map = {
        "On Job Training": 0,
        "0-30": 1,
        "31-60": 2,
        "61-90": 3,
        ">90": 4
    }
    df["Tenure Bucket"] = df["Tenure Bucket"].map(tenure_map).fillna(0)

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            if col in label_encoders:
                encoder = label_encoders[col]
                df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
                df[col] = encoder.transform(df[col])
            else:
                df[col] = 0

    try:
        df = df[ALL_FEATURES_ORDER]
    except KeyError as e:
        raise ValueError(f"Missing expected feature: {e}")

    X = scaler.transform(df)
    return X

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
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)