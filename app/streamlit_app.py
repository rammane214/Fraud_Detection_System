 
# ============================================================
# FRAUD DETECTION DASHBOARD (CLEAN VERSION)
# File: app/streamlit_app.py
# Run: streamlit run app/streamlit_app.py
# ============================================================

import streamlit as st
import numpy as np
import joblib
import pandas as pd
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# -----------------------------
# LOAD MODEL & SCALER
# -----------------------------
@st.cache_resource
def load_assets():
    # Get the directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to project root, then to models
    model_path = os.path.join(base_dir, "..", "models", "xgboost.pkl")
    scaler_path = os.path.join(base_dir, "..", "models", "scaler.pkl")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_assets()

# -----------------------------
# HEADER
# -----------------------------
st.title("💳 Fraud Detection AI Dashboard")
st.markdown("### Detect fraudulent transactions using Machine Learning")

st.divider()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict"])

# ============================================================
# HOME PAGE
# ============================================================
if page == "Home":

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", "284,807")
    col2.metric("Fraud Cases", "492")
    col3.metric("Fraud Rate", "0.17%")

    st.info("Use the sidebar to predict whether a transaction is fraud or not.")

    st.markdown("#### Model Info")
    st.write("• Algorithm: XGBoost")
    st.write("• Features: Scaled PCA components (V1–V28)")
    st.write("• Problem Type: Binary Classification")

# ============================================================
# PREDICTION PAGE
# ============================================================
elif page == "Predict":

    st.subheader("Enter Transaction Details")

    with st.form("fraud_form"):

        col1, col2 = st.columns(2)

        with col1:
            amount = st.number_input("Amount", min_value=0.0, value=100.0)

        with col2:
            time = st.number_input("Time", min_value=0.0, value=50000.0)
            hour = st.number_input("Hour", min_value=0.0, max_value=23.0, value=12.0, step=1.0)

        st.markdown("### V1 – V28 Features")

        v_features = []

        cols = st.columns(4)
        for i in range(1, 29):
            v = cols[(i-1) % 4].number_input(
                f"V{i}",
                value=0.0,
                format="%.4f",
                key=f"v{i}"
            )
            v_features.append(v)

        submit = st.form_submit_button("🔍 Predict Fraud")

    if submit:

        try:
            # -----------------------------
            # Build input (FIXED: no missing feature)
            # -----------------------------
            input_data = np.array([[amount, time] + v_features + [hour]])

            # Scale only if scaler supports it
            try:
                input_data = scaler.transform(input_data)
            except:
                pass

            # Prediction
            pred = model.predict(input_data)[0]

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_data)[0][1]
            else:
                prob = 0.0

            # -----------------------------
            # RESULT UI
            # -----------------------------
            st.divider()

            col1, col2, col3 = st.columns(3)

            col1.metric("Fraud Probability", f"{prob*100:.2f}%")

            risk = "LOW"
            if prob > 0.7:
                risk = "HIGH"
            elif prob > 0.3:
                risk = "MEDIUM"

            col2.metric("Risk Level", risk)

            col3.metric("Decision", "BLOCK" if pred == 1 else "ALLOW")

            # ALERT
            if pred == 1:
                st.error("🚨 Fraudulent Transaction Detected!")
            else:
                st.success("✅ Legitimate Transaction")

            # PROGRESS BAR
            st.progress(float(prob))

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
 
