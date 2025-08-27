# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 16:42:57 2025

@author: USER

Streamlined badass version:
- Only the Predictor page shows its hero image.
- Removed LOAN_ICON, PROFILE_ICON, DERIVED_ICON, and other hero pics.
- Sidebar now has "Explore pages here..." guidance.
- About page expanded with more info about loan default prediction.
- Batch Predictor page explains required input columns.
"""

import pandas as pd
import streamlit as st
import joblib
import time
import logging
import base64
from sklearn.base import ClassifierMixin
import plotly.graph_objects as go

# ------------------- CONFIG -------------------
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("Loan_defaulter_predict.pkl")
        logger.info("Model loaded successfully.")
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'Loan_defaulter_predict.pkl' is in the correct directory.")
        logger.error("Model file not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        logger.error(f"Error loading model: {str(e)}")
        st.stop()

model = load_model()
supports_proba = hasattr(model, 'predict_proba') and isinstance(model, ClassifierMixin)

# ------------------- CUSTOM CSS -------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
        font-family: 'Montserrat', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 35px;
        margin-bottom: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0px 10px 40px rgba(0, 0, 0, 0.5);
    }

    .prediction-default {
        color: #ff3b30;
        font-size: 36px;
        font-weight: 800;
        text-align: center;
        text-shadow: 0px 0px 30px #ff3b30, 0px 0px 50px #ff3b30;
        animation: epicGlow 1.5s ease-in-out infinite;
    }
    .prediction-safe {
        color: #34c759;
        font-size: 36px;
        font-weight: 800;
        text-align: center;
        text-shadow: 0px 0px 30px #34c759, 0px 0px 50px #34c759;
        animation: epicGlow 1.5s ease-in-out infinite;
    }
    @keyframes epicGlow {
        0% { text-shadow: 0px 0px 20px; }
        50% { text-shadow: 0px 0px 40px; }
        100% { text-shadow: 0px 0px 20px; }
    }

    hr {
        border: none;
        height: 3px;
        background: linear-gradient(90deg, #ff9500, #ff2d55, #34c759);
        border-radius: 2px;
        margin: 30px 0;
    }

    .stButton>button {
        background: linear-gradient(45deg, #ff9500, #ff2d55);
        color: white;
        border-radius: 15px;
        padding: 15px 30px;
        font-weight: 700;
        font-size: 18px;
        transition: transform 0.4s ease, box-shadow 0.4s ease;
    }
    .stButton>button:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0px 6px 25px rgba(255, 149, 0, 0.5);
    }

    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.6);
        border-right: 1px solid rgba(255, 255, 255, 0.15);
    }

    .hero-image {
        width: 100%;
        height: auto;
        border-radius: 25px;
        box-shadow: 0px 10px 40px rgba(0, 0, 0, 0.5);
        display: block;
        margin: 0 auto 20px auto;
    }

    .footer-link {
        color: #ff9500;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    .footer-link:hover {
        color: #ff2d55;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------- IMAGE -------------------
HEADER_IMAGE = "https://st.depositphotos.com/3246463/5167/i/600/depositphotos_51674189-stock-photo-loan-approval-concept.jpg"

# ------------------- SIDEBAR -------------------
with st.sidebar:
    st.header("Navigation", divider="orange")
    st.write("✨ Explore pages here...")
    page = st.radio("Choose a page:", ["Predictor", "Batch Predictor", "About"])

# ------------------- FUNCTIONS -------------------
def categorize_age(age):
    if 18 <= age <= 24:
        return "Youth"
    elif 25 <= age <= 39:
        return "Adult"
    else:
        return "Middle Age"

def calculate_interest_rate(loan_amount, total_due):
    if loan_amount > 0:
        return ((total_due - loan_amount) / loan_amount) * 100
    return 0

def validate_inputs(loanamount, totaldue):
    if loanamount <= 0:
        st.warning("Loan Amount must be greater than 0.")
        return False
    if totaldue < loanamount:
        st.warning("Total Due should be at least equal to Loan Amount.")
        return False
    return True

def get_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="prediction_results.csv" style="color: #ff9500; font-weight: bold;">Download Results CSV</a>'

def process_batch(df):
    required_cols = ["loanamount", "totaldue", "referredby", "bank_account_type", "employment_status_clients", "customer_age", "Loan_Term_Category"]
    if not all(col in df.columns for col in required_cols):
        st.error("⚠️ CSV must include columns: loanamount, totaldue, referredby, bank_account_type, employment_status_clients, customer_age, Loan_Term_Category")
        return None
    
    df['age_category'] = df['customer_age'].apply(categorize_age)
    df['interest_rate'] = df.apply(lambda row: calculate_interest_rate(row['loanamount'], row['totaldue']), axis=1)
    
    predictions = model.predict(df)
    df['prediction'] = ["High Risk (Default)" if p == 0 else "Low Risk (No Default)" for p in predictions]
    
    if supports_proba:
        probas = model.predict_proba(df)
        df['default_probability'] = [f"{round(proba[0] * 100 if p == 0 else proba[1] * 100, 2)}%" for proba, p in zip(probas, predictions)]
    
    return df

# ------------------- MAIN CONTENT -------------------
if page == "Predictor":
    st.markdown(f'<img src="{HEADER_IMAGE}" alt="Hero" class="hero-image" />', unsafe_allow_html=True)

    st.markdown('<div class="glass-card" style="text-align:center;font-size:40px;font-weight:900;">💰 Loan Default Risk Predictor</div>', unsafe_allow_html=True)

    with st.form(key="input_form"):
        st.markdown('<div class="glass-card">📥 Loan Information</div>', unsafe_allow_html=True)
        loanamount = st.number_input("Loan Amount", min_value=1000, max_value=1_000_000, step=1000)
        totaldue = st.number_input("Total Due", min_value=1000, max_value=2_000_000, step=1000)

        st.markdown('<div class="glass-card">👤 Customer Profile</div>', unsafe_allow_html=True)
        referredby = st.selectbox("Referred By", ["Not Referred", "Referred"])
        bank_account_type = st.selectbox("Bank Account Type", ["Savings", "Current", "Other"])
        employment_status_clients = st.selectbox("Employment Status",
            ["Permanent", "Not Provided", "Unemployed", "Self-Employed", "Student", "Retired", "Contract"])
        Loan_Term_Category = st.selectbox("Loan Term Category", ["Short Term", "Long Term"])
        customer_age = st.number_input("Customer Age", min_value=18, max_value=100, step=1)

        submit_button = st.form_submit_button(label="⚡ Unleash Prediction")

    if submit_button:
        if not validate_inputs(loanamount, totaldue):
            st.stop()

        age_category = categorize_age(customer_age)
        interest_rate = calculate_interest_rate(loanamount, totaldue)

        st.markdown('<div class="glass-card">🔍 Derived Insights</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.metric("Age Category", age_category)
        col2.metric("Interest Rate (%)", f"{round(interest_rate, 2)}")

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

        with st.spinner("Decoding Risk Matrix..."):
            time.sleep(1.5)
            try:
                prediction = model.predict(input_data)[0]
                if supports_proba:
                    proba = model.predict_proba(input_data)[0]
                    default_prob = proba[0] if prediction == 0 else proba[1]
                else:
                    default_prob = None
            except Exception as e:
                st.error(f"Prediction anomaly: {str(e)}")
                st.stop()

        st.markdown("---")
        if prediction == 0:
            st.markdown('<p class="prediction-default">⚠️ ALERT: High Risk Default Imminent</p>', unsafe_allow_html=True)
            risk_level = 0.9
            bar_color = "#ff3b30"
        else:
            st.markdown('<p class="prediction-safe">✅ CLEAR: Low Risk, Proceed with Confidence</p>', unsafe_allow_html=True)
            risk_level = 0.2
            bar_color = "#34c759"

        st.progress(risk_level)

        if supports_proba:
            fig_prob = go.Figure(go.Indicator(
                mode="gauge+number",
                value=default_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Default Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': bar_color},
                    'steps': [
                        {'range': [0, 30], 'color': "#34c759"},
                        {'range': [30, 70], 'color': "#ffcc00"},
                        {'range': [70, 100], 'color': "#ff3b30"}
                    ]
                }
            ))
            st.plotly_chart(fig_prob, use_container_width=True)

        results_df = input_data.copy()
        results_df["prediction"] = ["High Risk (Default)" if prediction == 0 else "Low Risk (No Default)"]
        if supports_proba:
            results_df["default_probability"] = [f"{round(default_prob * 100, 2)}%"]

        st.markdown(get_download_link(results_df), unsafe_allow_html=True)

elif page == "Batch Predictor":
    st.markdown('<div class="glass-card" style="text-align:center;font-size:40px;font-weight:900;">📊 Batch Risk Analyzer</div>', unsafe_allow_html=True)
    st.markdown("""<div class="glass-card"><p style="font-size:18px;">
    Upload a CSV file containing customer loan details.  
    <br>⚠️ Required columns: <b>loanamount, totaldue, referredby, bank_account_type, employment_status_clients, customer_age, Loan_Term_Category</b>
    </p></div>""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        processed_df = process_batch(df)
        if processed_df is not None:
            st.dataframe(processed_df.style.highlight_max(axis=0, subset=['interest_rate'], color='#ffcc00'))
            st.markdown(get_download_link(processed_df), unsafe_allow_html=True)

elif page == "About":
    st.markdown('<div class="glass-card" style="text-align:center;font-size:40px;font-weight:900;">About This Beast</div>', unsafe_allow_html=True)
    st.markdown("""<div class="glass-card"><p style="font-size:18px;">
    This AI-powered tool leverages advanced machine learning algorithms to assess the probability of a customer defaulting on their loan.  
    <br><br>
    🌍 Why it matters: Loan defaults cost financial institutions billions annually. Predicting risk early helps reduce losses, safeguard assets, and ensure fairer lending.  
    <br><br>
    🔑 What it does:  
    - Evaluates loan details (loan amount, repayment, term).  
    - Considers customer demographics (age, employment, referral source).  
    - Generates a **risk prediction** (High Risk / Low Risk).  
    - Provides **probability scores** to aid decision-making.  
    <br><br>
    This is not just a predictor — it’s a financial risk shield.
    </p></div>""", unsafe_allow_html=True)

# ------------------- FOOTER -------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;color:#ffffff;font-size:18px;'>
        <p>🔥 Engineered for Excellence | Powered by Streamlit & scikit-learn</p>
        <p>Contact: 
            <a href="mailto:atandaprecious41@gmail.com" class="footer-link">atandaprecious41@gmail.com</a> | 
            <a href="https://www.linkedin.com/in/atanda-precious-5a6695237" class="footer-link" target="_blank">LinkedIn</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
