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

---
