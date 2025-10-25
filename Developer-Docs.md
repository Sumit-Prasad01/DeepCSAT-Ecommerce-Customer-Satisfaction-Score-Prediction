# CSAT Prediction System - Developer Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Data Ingestion](#data-ingestion)
4. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
5. [Model Training](#model-training)
6. [API Documentation](#api-documentation)
7. [Frontend Application](#frontend-application)
8. [Setup Instructions](#setup-instructions)
9. [API Usage Examples](#api-usage-examples)

## Overview

The CSAT Prediction System is an end-to-end machine learning solution designed to predict customer satisfaction scores (1-5) for e-commerce transactions. The system consists of:

- Data preprocessing pipeline
- Deep learning model training
- REST API for predictions
- Interactive web interface

## System Architecture

```
┌─────────────────┐
│   Raw CSV Data  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Ingestion  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI Server │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Streamlit UI    │
└─────────────────┘
```

## Data Ingestion

### Input Data Format

The system expects CSV files with the following schema:

```csv
Unique id,channel_name,category,Sub-category,Customer Remarks,Order_id,
order_date_time,Issue_reported at,issue_responded,Survey_response_Date,
Customer_City,Product_category,Item_price,connected_handling_time,
Agent_name,Supervisor,Manager,Tenure Bucket,Agent Shift,CSAT Score
```

### Data Fields

| Field Name | Type | Description |
|------------|------|-------------|
| `Unique id` | String | Unique transaction identifier (dropped during preprocessing) |
| `channel_name` | Categorical | Communication channel (Outcall, Inbound, Chat, Email) |
| `category` | Categorical | Main issue category |
| `Sub-category` | Categorical | Detailed issue subcategory |
| `Customer Remarks` | Categorical | Customer feedback text |
| `Order_id` | String | Order identifier (dropped during preprocessing) |
| `order_date_time` | DateTime | Order timestamp (dropped during preprocessing) |
| `Issue_reported at` | DateTime | Issue report timestamp (dropped during preprocessing) |
| `issue_responded` | DateTime | Response timestamp (dropped during preprocessing) |
| `Survey_response_Date` | DateTime | Survey completion date (dropped during preprocessing) |
| `Customer_City` | Categorical | Customer location |
| `Product_category` | Categorical | Product type |
| `Item_price` | Numerical | Transaction amount |
| `connected_handling_time` | Numerical | Call/chat duration in minutes |
| `Agent_name` | Categorical | Support agent identifier |
| `Supervisor` | Categorical | Supervising agent identifier |
| `Manager` | Categorical | Manager identifier |
| `Tenure Bucket` | Ordinal | Agent experience level |
| `Agent Shift` | Categorical | Work shift (Morning, Evening, Night) |
| `CSAT Score` | Target | Customer satisfaction score (1-5) |

### Data Loading

```python
import pandas as pd

train_df = pd.read_csv('data/raw/train.csv')
test_df = pd.read_csv('data/raw/test.csv')
```

### Expected Data Statistics

- **Training Set**: ~70-80% of total data
- **Test Set**: ~20-30% of total data
- **Target Distribution**: CSAT scores from 1 (lowest) to 5 (highest)
- **Missing Values**: Handled automatically by preprocessing pipeline

## Data Preprocessing Pipeline

### DataProcessor Class

Located in: `src/data_preprocessing.py`

```python
from src.data_preprocessing import DataProcessor

processor = DataProcessor(
    train_path='data/raw/train.csv',
    test_path='data/raw/test.csv',
    processed_dir='data/processed'
)
processor.process(
    train_output_path='data/processed/train.csv',
    test_output_path='data/processed/test.csv',
    encoder_path='artifacts/models/encoders.pkl'
)
```

### Preprocessing Steps

#### 1. Data Cleaning
- Drop identifier columns: `Unique id`, `Order_id`
- Drop datetime columns: `order_date_time`, `Issue_reported at`, `issue_responded`, `Survey_response_Date`
- Remove duplicate rows
- Remove unnamed index columns

#### 2. Missing Value Handling

**Categorical Features:**
- Fill with mode (most frequent value)
- Special case: `Customer Remarks` filled with "No remarks"

**Numerical Features:**
- Fill with median value
- Applied to: `Item_price`, `connected_handling_time`

#### 3. Feature Engineering

**Tenure Bucket Mapping:**
```python
tenure_map = {
    "On Job Training": 0,
    "0-30": 1,
    "31-60": 2,
    "61-90": 3,
    ">90": 4
}
```

#### 4. Encoding

**Label Encoding:**
All categorical features are label encoded and encoders are saved to `artifacts/models/encoders.pkl`

#### 5. Skewness Correction

Numerical features with skewness > 5 are log-transformed:
```python
df[column] = np.log1p(df[column])
```

#### 6. Data Balancing

**SMOTE (Synthetic Minority Over-sampling Technique):**
- Applied to handle class imbalance in CSAT scores
- Random state: 42 for reproducibility

### Output Files

```
data/processed/
├── train.csv          # Processed training data
└── test.csv           # Processed test data

artifacts/models/
└── encoders.pkl       # Label encoders dictionary
```

## Model Training

### Architecture

Deep learning model built with TensorFlow/Keras:

```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(n_features,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(5, activation='softmax')  # 5 classes for CSAT 1-5
])
```

### Training Configuration

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Artifacts Generated

```
artifacts/models/
├── model.h5           # Trained model
├── scaler.pkl         # StandardScaler for features
└── encoders.pkl       # Label encoders
```

## API Documentation

### FastAPI Server

Located in: `app/api.py`

### Endpoints

#### 1. Health Check
```http
GET /
```

**Response:**
```json
{
    "message": "Welcome to the CSAT Prediction API"
}
```

#### 2. Predict CSAT Score
```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
    "channel_name": "Outcall",
    "category": "Product Queries",
    "sub_category": "Life Insurance",
    "customer_remarks": "Great service",
    "customer_city": "Mumbai",
    "product_category": "Insurance",
    "item_price": 15000.50,
    "connected_handling_time": 120.5,
    "agent_name": "Richard Buchanan",
    "supervisor": "Mason Gupta",
    "manager": "Jennifer Nguyen",
    "tenure_bucket": "On Job Training",
    "agent_shift": "Morning"
}
```

**Response:**
```json
{
    "predicted_csat_score": 5,
    "confidence": 0.92
}
```

**Error Response:**
```json
{
    "detail": "Prediction failed: [error message]"
}
```

### Request Schema

| Field | Type | Required | Valid Values |
|-------|------|----------|--------------|
| `channel_name` | string | Yes | Outcall, Inbound, Chat, Email |
| `category` | string | Yes | Any category string |
| `sub_category` | string | Yes | Any subcategory string |
| `customer_remarks` | string | Yes | Any text or "No remarks" |
| `customer_city` | string | Yes | Any city name |
| `product_category` | string | Yes | Electronics, Insurance, etc. |
| `item_price` | float | Yes | >= 0.0 |
| `connected_handling_time` | float | Yes | >= 0.0 |
| `agent_name` | string | Yes | Any agent name |
| `supervisor` | string | Yes | Any supervisor name |
| `manager` | string | Yes | Any manager name |
| `tenure_bucket` | string | Yes | On Job Training, 0-30, 31-60, 61-90, >90 |
| `agent_shift` | string | Yes | Morning, Evening, Night |

### Internal Processing Flow

1. **Column Renaming**: Snake case to training format
2. **Tenure Mapping**: Convert bucket to ordinal values
3. **Label Encoding**: Transform categorical features
4. **Feature Ordering**: Arrange columns to match training
5. **Scaling**: Apply StandardScaler transformation
6. **Prediction**: Run through neural network
7. **Post-processing**: Extract class and confidence

## Frontend Application

### Streamlit Interface

Located in: `app/streamlit_app.py`

### Features

- Interactive dropdown selections for categorical features
- Numerical input fields with validation
- Real-time prediction via FastAPI
- Visual feedback for high/low satisfaction predictions
- Error handling and connection status

### Running the Application

```bash
streamlit run app/streamlit_app.py
```

Access at: `http://localhost:8501`

## Setup Instructions

### Prerequisites

```bash
Python 3.8+
pip
virtualenv (recommended)
```

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

```
pandas
numpy
scikit-learn
imbalanced-learn
tensorflow
fastapi
uvicorn
streamlit
requests
pydantic
```

### Directory Structure

```
project/
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/
│       ├── train.csv
│       └── test.csv
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── logger.py
│   └── custom_exception.py
├── config/
│   └── paths_config.py
├── artifacts/
│   └── models/
│       ├── model.h5
│       ├── scaler.pkl
│       └── encoders.pkl
├── app/
│   ├── api.py
│   └── streamlit_app.py
├── pipeline/
│   └── training_pipeline.py
└── requirements.txt
```

### Running the System

#### 1. Data Preprocessing
```bash
python src/data_preprocessing.py
```

#### 2. Model Training
```bash
python pipeline/training_pipeline.py
```

#### 3. Start API Server
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

#### 4. Start Streamlit UI
```bash
streamlit run app/streamlit_app.py
```

## API Usage Examples

### Python

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "channel_name": "Outcall",
    "category": "Product Queries",
    "sub_category": "Life Insurance",
    "customer_remarks": "No remarks",
    "customer_city": "Mumbai",
    "product_category": "Electronics",
    "item_price": 15000.50,
    "connected_handling_time": 120.5,
    "agent_name": "Richard Buchanan",
    "supervisor": "Mason Gupta",
    "manager": "Jennifer Nguyen",
    "tenure_bucket": "On Job Training",
    "agent_shift": "Morning"
}

response = requests.post(url, json=data)
print(response.json())
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "channel_name": "Outcall",
    "category": "Product Queries",
    "sub_category": "Life Insurance",
    "customer_remarks": "No remarks",
    "customer_city": "Mumbai",
    "product_category": "Electronics",
    "item_price": 15000.50,
    "connected_handling_time": 120.5,
    "agent_name": "Richard Buchanan",
    "supervisor": "Mason Gupta",
    "manager": "Jennifer Nguyen",
    "tenure_bucket": "On Job Training",
    "agent_shift": "Morning"
  }'
```

### JavaScript (Fetch API)

```javascript
const url = 'http://localhost:8000/predict';
const data = {
    channel_name: "Outcall",
    category: "Product Queries",
    sub_category: "Life Insurance",
    customer_remarks: "No remarks",
    customer_city: "Mumbai",
    product_category: "Electronics",
    item_price: 15000.50,
    connected_handling_time: 120.5,
    agent_name: "Richard Buchanan",
    supervisor: "Mason Gupta",
    manager: "Jennifer Nguyen",
    tenure_bucket: "On Job Training",
    agent_shift: "Morning"
};

fetch(url, {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
})
.then(response => response.json())
.then(result => console.log(result))
.catch(error => console.error('Error:', error));
```

## Troubleshooting

### Common Issues

#### 1. Feature Name Mismatch
**Error:** "Feature names should match those that were passed during fit"

**Solution:** Ensure column names match exactly (case-sensitive). Check the `COLUMN_RENAME_MAP` in `api.py`.

#### 2. Missing Model Files
**Error:** "Error loading model or preprocessors"

**Solution:** Run the training pipeline to generate required artifacts:
```bash
python pipeline/training_pipeline.py
```

#### 3. Connection Refused
**Error:** "Failed to connect to FastAPI"

**Solution:** Ensure the FastAPI server is running:
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

#### 4. Unknown Category Values
**Error:** Label encoder encounters unknown values

**Solution:** The API automatically maps unknown values to the first class in the encoder. Ensure your training data covers all expected categories.

## Performance Considerations

### Scalability
- API can handle concurrent requests
- Consider load balancing for production
- Cache model artifacts in memory

### Optimization
- Batch predictions for multiple records
- Use async endpoints for I/O operations
- Implement request queuing for high traffic

## Security Recommendations

1. Add authentication/authorization to API endpoints
2. Implement rate limiting
3. Validate and sanitize all inputs
4. Use HTTPS in production
5. Implement logging and monitoring
6. Add input validation middleware

