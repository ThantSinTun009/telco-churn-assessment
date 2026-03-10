# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# modelling libraries
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, RocCurveDisplay, average_precision_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import  XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

df = pd.read_csv('Telco-Customer-Churn.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


# customerID variables don't carry information so we can drop these
df.drop(['customerID'], inplace=True, axis=1)


# Change the label format
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})


# Data Cleaning
# Remove duplicated data
df.drop_duplicates(inplace=True)

df.dropna(inplace=True)

# Feature Extraction
df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"]
)

# Encode ordinally
df['tenure_group'] = df['tenure_group'].map({
    '0-1yr':0,
    '1-2yr':1,
    '2-4yr':2,
    '4-6yr':3
})

df['tenure_group'] = df['tenure_group'].astype(int)

# Create montly charge ratio
df['charge_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)

# total charge per month within customer duration
df['avg_monthly_spend'] = df['TotalCharges'] / (df['tenure'] + 1)

# Number of services
services = [
    'PhoneService',
    'MultipleLines',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies'
]

df['num_services'] = (df[services] == 1).sum(axis=1)

# No internet service & No phone service --> No
service_cols = ['MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']

for col in service_cols:
    df[col] = df[col].replace({
        'No internet service':'No',
        'No phone service':'No'
    })

binary_cols = [
'Partner','Dependents','PhoneService','PaperlessBilling',
'MultipleLines','OnlineSecurity','OnlineBackup',
'DeviceProtection','TechSupport','StreamingTV','StreamingMovies'
]

for col in binary_cols:
    df[col] = df[col].map({'Yes':1,'No':0})

df['gender'] = df['gender'].map({'Male':1,'Female':0})

df['Contract'] = df['Contract'].map({
    'Month-to-month':0,
    'One year':1,
    'Two year':2
})

final_df = df.copy()



df = pd.get_dummies(df, columns=['PaymentMethod'], drop_first=True)
df = pd.get_dummies(df, columns=['InternetService'], drop_first=True)




X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


# Model Training
lgbm_params = {
    "n_estimators": [200, 400, 600],
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [31, 63, 127],
    "max_depth": [-1, 10, 20],
    "min_child_samples": [20, 40, 60],
    "subsample": [0.7, 0.9, 1.0]
}

lgbm_search = RandomizedSearchCV(
    LGBMClassifier(class_weight="balanced", random_state=42),
    lgbm_params,
    n_iter=30,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=0
)

lgbm_search.fit(X_resampled, y_resampled)

print("Best LGBM Params:", lgbm_search.best_params_)

best_lgbm = lgbm_search.best_estimator_


# Model Pipeline

# Column Transformer
numeric_features = ['tenure_group', 'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
                    'charge_per_tenure', 'avg_monthly_spend', 'num_services']

binary_cats = ['gender', 'Partner', 'PaperlessBilling',
'Dependents', 'PhoneService','PaperlessBilling', 'MultipleLines',
'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

mul_cats = ['InternetService', 'PaymentMethod']

ord_cats = ['Contract']
# Contract order
contract_order = ['Month-to-month', 'One year', 'Two year']

preprocessor = ColumnTransformer([
    ("num", 'passthrough', numeric_features),          # just keep numeric features
    ("binary", 'passthrough', binary_cats),           # already 0/1
    ("multiple", OneHotEncoder(handle_unknown='ignore'), mul_cats),
    ("ordinal", 'passthrough', ord_cats) # already order
])


# Pipeline
final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", LGBMClassifier(
        n_estimators=400,
        learning_rate=0.01,
        num_leaves=63,
        max_depth=10,
        min_child_samples=20,
        subsample=0.7,
        class_weight="balanced",
        random_state=42
    ))
])


# New final train test split
X = final_df.drop('Churn', axis=1)
y = final_df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

final_pipeline.fit(X_train, y_train)

y_prob = final_pipeline.predict_proba(X_test)[:,1]
y_pred = (y_prob > 0.35).astype(int)   # tuned threshold

print("-- Confusion Matrix --")
print(confusion_matrix(y_test, y_pred))
print("\n-- Classification Report --")
print(classification_report(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("PR-AUC:", average_precision_score(y_test, y_prob))


import joblib
joblib.dump(final_pipeline, "final_tuned_lgbm_pipeline.pkl")



