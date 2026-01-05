import streamlit as st

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("AIML Dataset.csv")

st.markdown("<h1 style='text-align:center;'>Fraud Detect APP System</h1>",unsafe_allow_html=True)
st.markdown("---")

st.subheader("Summary")

st.write("Fraudulent transactions cause huge financial losses every year.This project builds a machine-learning model that analyzes transaction patterns and predicts whether a transaction is legitimate or fraudulent.The system learns from historical data containing both normal and fraudulent behavior.During prediction, the model evaluates key features such as transaction amount, frequency, account activity, and risk signals.")

st.write("he goal of this project is to:")
st.write("✔️ Detect suspicious transactions early")
st.write("✔️ Reduce financial risk")
st.write("✔️ Support decision-making for banks and payment systems")
st.write("Users can input transaction details in the dashboard, and the model instantly provides a Fraud / Not Fraud prediction along with helpful insights")


st. subheader("Key Features of This App")

st.write("🚀 App Features")
st.write("User-friendly dashboard")
st.write("Real-time prediction")
st.write("Handles both numeric and categorical values")
st.write("Pre-trained model using industry-standard ML algorithms")

st.subheader("Download Dataset")
st.write("Download Report")
st.download_button("Download CSV", df.to_csv().encode(), "AIML Dataset.csv")
st.markdown("<h1 style='text-align:center;'>Basic Visualization </h1>",unsafe_allow_html=True)
st.markdown("---")

st.subheader("1️⃣ Fraud vs Non-Fraud")

fig, ax = plt.subplots()
sns.countplot(data=df, x="isFraud", ax=ax)
ax.set_xticklabels(["Not Fraud", "Fraud"])
ax.set_xlabel("")
ax.set_ylabel("Count")
st.pyplot(fig)

st.markdown("> Fraud cases are usually much smaller — which makes detection hard.")

st.subheader("2️⃣ Transaction Type vs Fraud")
fig, ax = plt.subplots()
sns.countplot(data=df, x="type", hue="isFraud", ax=ax)
ax.set_xlabel("Transaction Type")
ax.set_ylabel("Count")
ax.legend(["Not Fraud", "Fraud"])
plt.xticks(rotation=30)
st.pyplot(fig)

st.markdown("> Transfers & Cash-outs usually carry most fraud risk.")


st.subheader("3️⃣ Transaction Amount Distribution")

fig, ax = plt.subplots()
sns.kdeplot(data=df, x="amount", hue="isFraud", ax=ax, fill=True, common_norm=False)
ax.set_xlabel("Amount")
st.pyplot(fig)

st.markdown("> Fraud transactions often involve higher amounts.")


st.subheader("4️⃣ Feature Correlation (Important Numeric Columns)")

numeric_cols = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
]

corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(5,3))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.markdown("> Helps identify features that strongly relate to fraud.")

st.subheader("5️⃣ Suspicious Balance Changes")

df["balance_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]

fig, ax = plt.subplots()
sns.boxplot(data=df, x="isFraud", y="balance_diff", ax=ax)
ax.set_xticklabels(["Not Fraud", "Fraud"])
ax.set_xlabel("")
ax.set_ylabel("Balance Difference")
st.pyplot(fig)

st.markdown("> Large sudden balance drops are strong fraud signals.")