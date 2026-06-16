# CommissionLens: Commission-Adjusted Alpha Prediction in Indian Mutual Funds

## Problem Statement
Indian mutual funds offer "Regular" and "Direct" plans. Regular plans charge a higher expense ratio to pay commissions to distributors. The key question for investors is: **Does the higher cost of a regular plan provide enough additional returns (alpha) to justify the commission?**

## Objective
CommissionLens is an end-to-end Machine Learning pipeline that predicts whether a mutual fund's regular plan commission is justified by its future alpha. By analyzing historical NAV data, fund characteristics, and macroeconomic variables, it classifies funds into "Justified" or "Not Justified" categories.

## Methodology
1. **Data Collection**: Synthesizes or loads historical mutual fund data (NAV, expense ratios, AUM, turnover) and macroeconomic indicators (repo rate, inflation).
2. **Feature Engineering**: Calculates rolling returns, volatility, Sharpe ratio, gross/net alpha, tracking error, information ratio, and beta.
3. **Target Creation**: Defines the future quarter's commission justification based on whether `next_quarter_gross_alpha > quarterly_expense_gap`.
4. **Modeling**: Trains classification (Logistic Regression, Random Forest, XGBoost) and regression models using a temporal train-test split.
5. **Evaluation**: Assesses models using RMSE, Accuracy, Precision, Recall, F1, AUC-ROC, and Precision at top decile.
6. **Explainability**: Utilizes SHAP to explain feature importance and how variables impact predictions.
7. **Backtesting**: Simulates a SIP strategy comparing a naive approach vs. model-guided fund selection.

## Disclaimer
> [!WARNING]
> This project is for **educational and research purposes only**. It does not provide financial advice, mutual fund recommendations, or investment strategies. Always consult a certified financial advisor before investing.

## Directory Structure
```
commissionlens/
├── app/                  # Streamlit dashboard
├── data/                 # Raw, processed, and synthetic sample data
├── models/               # Saved trained ML models
├── notebooks/            # Jupyter notebooks for exploration
├── reports/              # Output metrics, figures, and SHAP summaries
├── src/                  # Core Python modules
├── config.yaml           # Configuration
├── main.py               # Main execution script
├── requirements.txt      # Python dependencies
└── README.md             # This documentation
```

## How to Run

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Pipeline**
   ```bash
   python main.py
   ```
   This will generate synthetic data, preprocess it, engineer features, train models, evaluate them, and run the SIP backtest. All outputs will be saved in the `reports/` and `models/` directories.

3. **Launch the Dashboard**
   ```bash
   streamlit run app/streamlit_app.py
   ```
   Open the provided URL in your browser to interact with the predictions and visualize the SHAP and backtest results.
