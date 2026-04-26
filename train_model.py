
# TRAINING FILE: train_model.py


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
data = pd.read_csv("Synthetic_Financial_datasets_log.csv")

# Feature Engineering
data['diffOrig'] = data['oldbalanceOrg'] - data['newbalanceOrig']
data['diffDest'] = data['newbalanceDest'] - data['oldbalanceDest']

# Drop unwanted columns
data.drop(['nameOrig', 'nameDest', 'isFlaggedFraud', 'step'], axis=1, inplace=True)

# Drop missing target
data.dropna(subset=['isFraud'], inplace=True)

# Input & Output
X = data.drop('isFraud', axis=1)
y = data['isFraud']

# One-hot encoding
X = pd.get_dummies(X, columns=['type'])

# Save column names (VERY IMPORTANT)
joblib.dump(X.columns.tolist(), "columns.pkl")

# Fill missing
X = X.fillna(0)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_scaled, y_train)

# Evaluation
preds = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# Save everything
joblib.dump(model, "simple_fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model, scaler, and columns saved successfully!")