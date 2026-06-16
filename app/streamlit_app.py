import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
import shap
import matplotlib.pyplot as plt

# Add parent directory to path to allow importing from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Disclaimer
st.set_page_config(page_title="CommissionLens", layout="wide")

st.title("CommissionLens: Mutual Fund Commission Justification Predictor")
st.markdown("""
**Disclaimer:** This project is for educational and research purposes only. 
It does not provide financial advice or mutual fund recommendations.
""")

st.header("1. Project Explanation")
st.write("""
This tool predicts whether a mutual fund's regular plan commission (the expense ratio gap between regular and direct plans) 
is justified by its future alpha. It uses historical NAVs, benchmark returns, expense ratios, macroeconomic variables, and machine learning.
""")

# Load Models
@st.cache_resource
def load_models():
    reg_path = "models/best_regression_model.pkl"
    clf_path = "models/best_classification_model.pkl"
    
    reg_model = joblib.load(reg_path) if os.path.exists(reg_path) else None
    clf_model = joblib.load(clf_path) if os.path.exists(clf_path) else None
    
    return reg_model, clf_model

reg_model, clf_model = load_models()

# Upload dataset option
st.sidebar.header("2. Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Processed Feature Dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    # Try to load default
    default_path = "data/processed/model_ready_data.csv"
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
    else:
        st.warning("No data found. Please run main.py to generate sample data or upload a file.")
        st.stop()

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

# Select fund name
st.sidebar.header("3. Fund Selection")
funds = sorted(df['fund_name'].unique())
selected_fund = st.sidebar.selectbox("Select a Fund", funds)

fund_data = df[df['fund_name'] == selected_fund].sort_values('date')

if len(fund_data) == 0:
    st.error("No data for selected fund.")
    st.stop()

# Get latest features
latest_record = fund_data.iloc[-1:]

from src.modeling import get_features
features = get_features(df)

# Show latest features
st.header("4. Latest Fund Features")
st.write(f"Latest Date: {latest_record['date'].dt.date.iloc[0]}")
display_cols = ['nav', 'aum_cr', 'expense_gap', 'rolling_alpha_4q', 'information_ratio_8q', 'beta_8q']
# Only show columns that exist
display_cols = [c for c in display_cols if c in latest_record.columns]
st.dataframe(latest_record[display_cols].T.rename(columns={latest_record.index[0]: 'Value'}))

# Predictions
st.header("5 & 6 & 7. Predictions")
if reg_model and clf_model:
    X_latest = latest_record[features]
    
    predicted_alpha = reg_model.predict(X_latest)[0]
    is_justified = clf_model.predict(X_latest)[0]
    prob_justified = clf_model.predict_proba(X_latest)[0][1] if hasattr(clf_model, 'predict_proba') else is_justified
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Next-Quarter Net Alpha", f"{predicted_alpha:.4f}")
    col2.metric("Probability Justified", f"{prob_justified:.2%}")
    col3.metric("Commission Justification Score", "Justified" if is_justified == 1 else "Not Justified")
else:
    st.warning("Models not found. Please train models first.")

# SHAP Explainability
st.header("8. Top SHAP-based Explanation")
shap_img_path = "reports/figures/shap_summary_plot.png"
if os.path.exists(shap_img_path):
    st.image(shap_img_path, caption="SHAP Summary Plot")
else:
    st.write("SHAP summary plot not available. Run main.py to generate.")

# SIP Backtest
st.header("9. SIP Backtest Comparison")
sip_img_path = "reports/figures/sip_growth_comparison.png"
if os.path.exists(sip_img_path):
    st.image(sip_img_path, caption="SIP Growth Comparison")
    
    sip_csv_path = "reports/sip_backtest_results.csv"
    if os.path.exists(sip_csv_path):
        sip_res = pd.read_csv(sip_csv_path)
        st.dataframe(sip_res)
else:
    st.write("SIP Backtest results not available. Run main.py to generate.")

