import streamlit as st
import pandas as pd
import joblib

# Load model files
model = joblib.load("simple_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("💳 Financial Fraud Detection")

st.write("Enter transaction details:")

# Inputs
amount = st.number_input("Amount", min_value=0.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0)

type_trans = st.selectbox("Transaction Type", 
                         ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT'])

# Feature engineering
diffOrig = oldbalanceOrg - newbalanceOrig
diffDest = newbalanceDest - oldbalanceDest

# Create input
input_data = {
    'amount': amount,
    'oldbalanceOrg': oldbalanceOrg,
    'newbalanceOrig': newbalanceOrig,
    'oldbalanceDest': oldbalanceDest,
    'newbalanceDest': newbalanceDest,
    'diffOrig': diffOrig,
    'diffDest': diffDest
}

input_df = pd.DataFrame([input_data])

# Add missing columns
for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Set transaction type
type_col = f"type_{type_trans}"
if type_col in input_df.columns:
    input_df[type_col] = 1

# Arrange columns
input_df = input_df[columns]

# Scale
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict"):
    result = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if result == 1:
        st.error("🚨 Fraud Transaction")
    else:
        st.success("✅ Legit Transaction")

    st.write(f"Fraud Probability: {prob:.2f}")