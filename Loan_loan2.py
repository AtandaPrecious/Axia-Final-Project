# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 16:42:57 2025

@author: USER
"""

# Import libraries
import pandas as pd
import streamlit as st
import joblib
import time

# ------------------- CONFIG -------------------
st.set_page_config(
    page_title="Loan Defaulter Predictor",
    page_icon="💰",
    layout="centered"
)

# ------------------- LOAD MODEL -------------------
model = joblib.load("Loan_defaulter_predict.pkl")

# ------------------- CUSTOM CSS -------------------
st.markdown(
    """
    <style>
    /* Animated gradient background */
    .stApp {
        background: linear-gradient(270deg, #1a2a6c, #b21f1f, #fdbb2d);
        background-size: 600% 600%;
        animation: GradientShift 20s ease infinite;
        color: white;
    }
    @keyframes GradientShift {
        0% {background-position:0% 50%}
        50% {background-position:100% 50%}
        100% {background-position:0% 50%}
    }

    /* Glassmorphism effect for cards */
    .glass-card {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 25px;
        border-radius: 18px;
        margin-bottom: 20px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
    }

    /* Glowing prediction */
    .prediction-default {
        color: #ff4b5c;
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        text-shadow: 0px 0px 20px #ff4b5c;
    }
    .prediction-safe {
        color: #00e676;
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        text-shadow: 0px 0px 20px #00e676;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------- HEADER -------------------
st.markdown('<div class="glass-card" style="text-align:center;font-size:30px;">💰 Loan Defaulter Predictor</div>', unsafe_allow_html=True)

# ------------------- INPUT SECTION -------------------
with st.container():
    st.markdown('<div class="glass-card">📥 Loan Information</div>', unsafe_allow_html=True)
    loanamount = st.number_input("Loan Amount", min_value=1000, max_value=1_000_000, step=1000)
    totaldue = st.number_input("Total Due", min_value=1000, max_value=2_000_000, step=1000)

with st.container():
    st.markdown('<div class="glass-card">👤 Customer Profile</div>', unsafe_allow_html=True)
    referredby = st.selectbox("Referred By", ["Not Referred", "Referred"])
    bank_account_type = st.selectbox("Bank Account Type", ["Savings", "Current", "Other"])
    employment_status_clients = st.selectbox(
        "Employment Status",
        ["Permanent", "Not Provided", "Unemployed", "Self-Employed", "Student", "Retired", "Contract"]
    )
    customer_age = st.number_input("Customer Age", min_value=18, max_value=60, step=1)
    Loan_Term_Category = st.selectbox("Loan Term Category", ["Short Term", "Long Term"])

# ------------------- FEATURE ENGINEERING -------------------
if 18 <= customer_age <= 24:
    age_category = "Youth"
elif 25 <= customer_age <= 39:
    age_category = "Adult"
else:
    age_category = "Middle Age"

interest_rate = ((totaldue - loanamount) / loanamount) * 100 if loanamount > 0 else 0

# ------------------- DERIVED INFO -------------------
st.markdown('<div class="glass-card">🔍 Derived Features</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
col1.metric("Age Category", age_category)
col2.metric("Interest Rate (%)", f"{round(interest_rate, 2)}")

# ------------------- PREDICTION -------------------
input_data = pd.DataFrame({
    "loanamount": [loanamount],
    "totaldue": [totaldue],
    "referredby": [referredby],
    "bank_account_type": [bank_account_type],
    "employment_status_clients": [employment_status_clients],
    "customer_age": [customer_age],
    "Loan_Term_Category": [Loan_Term_Category],
    "age_category": [age_category],
    "interest_rate": [interest_rate]
})

st.markdown("---")
if st.button("⚡ Predict Now"):
    with st.spinner("Analyzing loan risk..."):
        time.sleep(1.5)  # little suspense effect
    prediction = model.predict(input_data)[0]

    if prediction == 0:
        st.markdown('<p class="prediction-default">⚠️ HIGH RISK: Customer likely to DEFAULT</p>', unsafe_allow_html=True)
        st.progress(90)  # red risk bar
    else:
        st.markdown('<p class="prediction-safe">✅ SAFE: Customer NOT likely to default</p>', unsafe_allow_html=True)
        st.progress(20)  # green risk bar

# ------------------- FOOTER -------------------
st.markdown("---")
st.markdown("<p style='text-align:center;color:white;'>🚀 Designed with a neon glow | Powered by Streamlit</p>", unsafe_allow_html=True)
