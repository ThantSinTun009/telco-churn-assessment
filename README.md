# Telco Customer Churn Prediction

This repository contains the solution for the Telco Customer Churn assessment.

It includes a data analysis, preprocessing notebook, a training pipeline, and a prediction API.

---

### Submission Detail:

#### 1. Data Analysis Notebook (`analysis.ipynb`)
- A Jupyter notebook showing inspection of the dataset.
- Must identify which factors correlate most with customer churn.  
  *Example:* People with Fiber Optic are more likely to churn than DSL users.
- Visualization: Simple charts (Bar charts or Histograms) showing the data distribution.

#### 2. The Training Pipeline (`train.py`)
- A script to clean the data (convert "Yes/No" text to 1/0).
- Train a classifier model.
- Tested Models: XGBoost, LightGBM, or Random Forest (industry standards for tabular data).
- Crucial: Handle **class imbalance** with SMOTE.  
  *Note:* SInce only reporting Accuracy is insufficient, Used **F1-Score** or **ROC-AUC** as the main metric.

#### 3. The Prediction API (`app.py`)
- A Flask API endpoint that accepts customer details as input.
- Returns churn prediction.

---

## Repository Structure

```bash
telco_churn_submission/

├── analysis.ipynb # Data exploration and visualization

├── notebook.ipynb # Data cleaning, training pipeline, and handling class imbalance

├── train.py # Best Model Training

├── app.py # Flask API for predictions

├── requirements.txt # Python dependencies

├── README.md 

└── final_tuned_lgbm_pipeline.pkl
```

---

## Usage Instructions

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model: 

```bash
python train.py
```

Run the API:
```bash
python app.py
```

The server will start at:
```
http://127.0.0.1:5000
```

#### Create a New Request:

- Open Postman and create a new HTTP request.

Set the request method to:
```
POST
```
Endpoint URL:
```
http://127.0.0.1:5000/predict_api
```
3. Add Request Body

- Go to Body → raw → JSON and paste the following example input:
```
{
  "SeniorCitizen": 0,
  "gender": 1,
  "Partner": 1,
  "Dependents": 0,
  "InternetService": "Fiber optic",
  "Contract": "Month-to-month",
  "tenure_group": "0-1yr",
  "PaymentMethod": "Electronic check",
  "tenure": 12,
  "PhoneService": 1,
  "MultipleLines": 0,
  "OnlineSecurity": 0,
  "OnlineBackup": 1,
  "DeviceProtection": 1,
  "TechSupport": 0,
  "StreamingTV": 1,
  "StreamingMovies": 1,
  "PaperlessBilling": 1,
  "MonthlyCharges": 70,
  "TotalCharges": 840,
  "charge_per_tenure": 70,
  "avg_monthly_spend": 70,
  "num_services": 4
}
```

Send the Request

```
Click Send in Postman.
```
