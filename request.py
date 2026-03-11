import requests
import json

url = "http://127.0.0.1:5000/predict_api"

data = {
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

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response:", response.json())