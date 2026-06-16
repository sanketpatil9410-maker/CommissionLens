import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from scipy import optimize

def xnpv(rate, cashflows, dates):
    """Calculate Net Present Value for a schedule of cash flows."""
    if rate <= -1.0:
        return float('inf')
    d0 = dates[0]
    return sum([cf / (1 + rate)**((d - d0).days / 365.0) for cf, d in zip(cashflows, dates)])

def xirr(cashflows, dates, guess=0.1):
    """Calculate the Internal Rate of Return for a schedule of cash flows."""
    try:
        return optimize.newton(lambda r: xnpv(r, cashflows, dates), guess)
    except Exception:
        return np.nan

def simulate_sip(df_test: pd.DataFrame, investment_mask: pd.Series, sip_amount_per_quarter: float = 15000.0) -> dict:
    """
    Simulate a SIP.
    df_test: DataFrame containing the test set with 'fund_id', 'date', 'fund_return'
    investment_mask: Boolean series indicating which funds to invest in for each quarter.
    """
    dates = sorted(df_test['date'].unique())
    
    portfolio_value = 0.0
    cashflows = []
    cf_dates = []
    
    total_invested = 0.0
    
    portfolio_history = []
    
    # We maintain a hypothetical number of "units" or just track value
    # To keep it simple, we track value and apply the average return of selected funds.
    for dt in dates:
        current_quarter_data = df_test[df_test['date'] == dt]
        selected_funds = current_quarter_data[investment_mask[current_quarter_data.index]]
        
        if len(selected_funds) > 0:
            avg_return = selected_funds['fund_return'].mean()
        else:
            avg_return = 0.0
            
        # Grow existing portfolio
        portfolio_value = portfolio_value * (1 + avg_return)
        
        # Add new SIP
        portfolio_value += sip_amount_per_quarter
        total_invested += sip_amount_per_quarter
        
        cashflows.append(-sip_amount_per_quarter)
        cf_dates.append(dt)
        
        portfolio_history.append({
            'date': dt,
            'portfolio_value': portfolio_value,
            'total_invested': total_invested
        })
        
    # Terminal value is a positive cash flow
    if len(cashflows) > 0:
        cashflows.append(portfolio_value)
        # Assume terminal date is 1 day after last investment for XIRR calculation to avoid 0 days
        cf_dates.append(dates[-1] + pd.Timedelta(days=1))
        
    calc_xirr = xirr(cashflows, cf_dates)
    
    return {
        'final_corpus': portfolio_value,
        'total_invested': total_invested,
        'xirr': calc_xirr,
        'history': pd.DataFrame(portfolio_history)
    }

def run_backtest(df: pd.DataFrame, features: list, models_dir: str, reports_dir: str):
    """Run the SIP backtest comparing different strategies."""
    
    from src.modeling import temporal_split
    _, _, test_df = temporal_split(df)
    
    clf_model_path = os.path.join(models_dir, 'best_classification_model.pkl')
    if not os.path.exists(clf_model_path):
        print("Model not found for backtest.")
        return
        
    model = joblib.load(clf_model_path)
    X_test = test_df[features]
    predictions = model.predict(X_test)
    
    test_df = test_df.copy()
    test_df['predicted_justified'] = predictions
    
    # Strategy 1: Naive (Invest in all available funds)
    naive_mask = pd.Series(True, index=test_df.index)
    res_naive = simulate_sip(test_df, naive_mask)
    
    # Strategy 2: Model-guided
    model_mask = test_df['predicted_justified'] == 1
    # If model says no funds are justified in a quarter, it will sit in cash (0% return)
    res_model = simulate_sip(test_df, model_mask)
    
    # Strategy 3: Direct Plan Benchmark
    # We adjust the fund_return in test_df to simulate direct plan return
    direct_df = test_df.copy()
    # fund_return was net of regular_expense_ratio. 
    # To get direct return, we add back the regular_expense_ratio and subtract direct_expense_ratio.
    # Note: Expense ratios in data might be annualized, we need to divide by 4.
    expense_gap_q = direct_df['expense_gap'] / 4
    direct_df['fund_return'] = direct_df['fund_return'] + expense_gap_q
    
    res_direct = simulate_sip(direct_df, naive_mask)
    
    # Summary Table
    summary = []
    for name, res in zip(['Naive Regular', 'Model-Guided', 'Direct Plan Benchmark'], [res_naive, res_model, res_direct]):
        summary.append({
            'Strategy': name,
            'Total Invested': res['total_invested'],
            'Final Corpus': res['final_corpus'],
            'XIRR (%)': res['xirr'] * 100 if pd.notnull(res['xirr']) else np.nan
        })
        
    summary_df = pd.DataFrame(summary)
    os.makedirs(reports_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(reports_dir, 'sip_backtest_results.csv'), index=False)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    if len(res_naive['history']) > 0:
        plt.plot(res_naive['history']['date'], res_naive['history']['portfolio_value'], label='Naive Regular')
    if len(res_model['history']) > 0:
        plt.plot(res_model['history']['date'], res_model['history']['portfolio_value'], label='Model-Guided')
    if len(res_direct['history']) > 0:
        plt.plot(res_direct['history']['date'], res_direct['history']['portfolio_value'], label='Direct Plan Benchmark')
        
    # Plot invested amount line
    if len(res_naive['history']) > 0:
        plt.plot(res_naive['history']['date'], res_naive['history']['total_invested'], label='Total Invested', linestyle='--', color='gray')
        
    plt.title('SIP Growth Comparison')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (₹)')
    plt.legend()
    plt.tight_layout()
    
    fig_dir = os.path.join(reports_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, 'sip_growth_comparison.png'))
    plt.close()
    
    print("SIP backtest complete.")
    
if __name__ == "__main__":
    from src.modeling import get_features
    df = pd.read_csv("data/processed/model_ready_data.csv")
    df['date'] = pd.to_datetime(df['date'])
    features = get_features(df)
    run_backtest(df, features, "models", "reports")
