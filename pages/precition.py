import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(__file__)

with open(os.path.join(BASE_DIR, "fraud_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE_DIR, "encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)

st.title("💳 Fraud Detection Prediction")
st.write("---")


col1, col2, col3 = st.columns(3)

with col1:
    type_val = st.selectbox(
        "Transaction Type",
        ["","CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    )

with col2:
    amount = st.number_input("Amount", min_value=0.0)


with col1:
    oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0)

with col2:
    newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0)

with col3:
    oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0)

newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0)


nameOrig = st.text_input("Sender Account (nameOrig)", value="")
nameDest = st.text_input("Receiver Account (nameDest)", value="")


input_df = pd.DataFrame([{
    "type": type_val,
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "newbalanceOrig": newbalanceOrig,
    "oldbalanceDest": oldbalanceDest,
    "newbalanceDest": newbalanceDest,

    "nameOrig": nameOrig,
    "nameDest": nameDest
}])


num_cols = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",

]

cat_cols = ["type", "nameOrig", "nameDest"]

num_inputs = input_df[num_cols]
cat_inputs = input_df[cat_cols]


X_cat = encoder.transform(cat_inputs)
X_num = scaler.transform(num_inputs)

X = np.hstack([X_num, X_cat])

prediction = model.predict(X)[0]

st.write("---")
if st.button("Predict"):
    if prediction == 1:
        st.error("⚠️ Transaction is LIKELY FRAUDULENT")
    else:
        st.success("✅ Transaction looks SAFE")