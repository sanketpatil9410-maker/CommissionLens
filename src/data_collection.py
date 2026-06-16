import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta

def load_csv_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    """
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        raise FileNotFoundError(f"File not found: {filepath}")

def fetch_nav_from_api(fund_code: str) -> pd.DataFrame:
    """
    Optional API function to fetch NAV data from mfapi.in.
    Code must not break if API fails.
    """
    url = f"https://api.mfapi.in/mf/{fund_code}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "data" in data:
                df = pd.DataFrame(data["data"])
                df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
                df['nav'] = pd.to_numeric(df['nav'])
                df['fund_id'] = fund_code
                return df
    except Exception as e:
        print(f"Failed to fetch data for fund {fund_code} from API: {e}")
    return pd.DataFrame()

def generate_sample_data(output_path: str = "data/sample/synthetic_funds_data.csv", n_funds: int = 50, years: int = 5) -> pd.DataFrame:
    """
    Generates realistic sample data for Indian equity mutual funds.
    Creates monthly observations.
    """
    np.random.seed(42)
    
    fund_names = [
        "Axis Bluechip Fund", "HDFC Flexi Cap Fund", "ICICI Prudential Large & Mid Cap Fund", 
        "SBI Contra Fund", "Nippon India Growth Fund", "Mirae Asset Large Cap Fund",
        "Kotak Emerging Equity Fund", "DSP Midcap Fund", "UTI Flexi Cap Fund",
        "Aditya Birla Sun Life Frontline Equity Fund", "Tata Digital India Fund",
        "Franklin India Prima Fund", "Canara Robeco Bluechip Equity Fund",
        "Invesco India Growth Opportunities Fund", "Motilal Oswal Midcap 30 Fund"
    ]
    
    categories = ["Large Cap", "Mid Cap", "Small Cap", "Flexi Cap", "Sectoral"]
    
    # Generate fund metadata
    funds_meta = []
    for i in range(1, n_funds + 1):
        fund_name = fund_names[i % len(fund_names)] + f" (Sample {i})"
        category = categories[i % len(categories)]
        
        # Base characteristics
        reg_exp = np.random.uniform(1.0, 2.5) / 100
        dir_exp = reg_exp - np.random.uniform(0.5, 1.5) / 100
        dir_exp = max(0.1 / 100, dir_exp) # ensure direct is positive
        
        funds_meta.append({
            'fund_id': f"F{i:03d}",
            'fund_name': fund_name,
            'category': category,
            'base_reg_exp': reg_exp,
            'base_dir_exp': dir_exp,
            'fund_manager_tenure_years': max(0.5, np.random.normal(5, 3)),
            'base_aum': np.random.uniform(500, 30000)
        })
        
    funds_df = pd.DataFrame(funds_meta)
    
    # Generate dates (monthly)
    end_date = datetime(2023, 12, 31)
    dates = [end_date - timedelta(days=30*i) for i in range(years * 12)]
    dates.reverse()
    
    # Generate macro data
    n_dates = len(dates)
    macro_df = pd.DataFrame({
        'date': dates,
        'repo_rate': np.linspace(5.15, 6.5, n_dates) + np.random.normal(0, 0.2, n_dates),
        'cpi_inflation': np.linspace(4.0, 6.0, n_dates) + np.random.normal(0, 0.5, n_dates),
        'yield_curve_slope': np.random.normal(1.5, 0.5, n_dates),
        'fii_flow_cr': np.random.normal(0, 5000, n_dates),
        'dii_flow_cr': np.random.normal(2000, 3000, n_dates)
    })
    
    # Generate monthly benchmark returns (approx 12% annualized)
    benchmark_returns = np.random.normal(0.12/12, 0.05, n_dates)
    
    records = []
    
    for _, fund in funds_df.iterrows():
        # Fund-specific alpha component
        fund_alpha_base = np.random.normal(0, 0.02/12)
        fund_beta = np.random.normal(1.0, 0.15)
        
        current_nav = 100.0
        
        for i, dt in enumerate(dates):
            # Calculate returns
            bm_ret = benchmark_returns[i]
            
            # Gross return = beta * bm_ret + alpha + noise
            gross_ret = fund_beta * bm_ret + fund_alpha_base + np.random.normal(0, 0.01)
            
            expense_gap = fund['base_reg_exp'] - fund['base_dir_exp']
            
            # Net return
            net_ret = gross_ret - (fund['base_reg_exp'] / 12)
            
            current_nav = current_nav * (1 + net_ret)
            
            records.append({
                'fund_id': fund['fund_id'],
                'fund_name': fund['fund_name'],
                'category': fund['category'],
                'date': dt,
                'nav': current_nav,
                'fund_return': net_ret,
                'benchmark_return': bm_ret,
                'regular_expense_ratio': fund['base_reg_exp'],
                'direct_expense_ratio': fund['base_dir_exp'],
                'expense_gap': expense_gap,
                'aum_cr': fund['base_aum'] * (1 + np.random.normal(0, 0.05)),
                'portfolio_turnover': np.random.uniform(0.1, 2.0),
                'fund_manager_tenure_years': fund['fund_manager_tenure_years'],
                'repo_rate': macro_df.loc[i, 'repo_rate'],
                'cpi_inflation': macro_df.loc[i, 'cpi_inflation'],
                'yield_curve_slope': macro_df.loc[i, 'yield_curve_slope'],
                'fii_flow_cr': macro_df.loc[i, 'fii_flow_cr'],
                'dii_flow_cr': macro_df.loc[i, 'dii_flow_cr']
            })
            
    final_df = pd.DataFrame(records)
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"Sample data generated and saved to {output_path}")
    
    return final_df

if __name__ == "__main__":
    generate_sample_data()
