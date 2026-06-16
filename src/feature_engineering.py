import pandas as pd
import numpy as np
import os

def calculate_rolling_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate rolling returns for 3, 4, and 8 quarters."""
    df = df.copy()
    
    for q in [3, 4, 8]:
        # (1 + r1)(1 + r2)... - 1
        # To compute rolling compounded return, we can use rolling sum of log(1+r)
        df[f'rolling_{q}q_return'] = df.groupby('fund_id')['fund_return'].apply(
            lambda x: np.exp(np.log1p(x).rolling(window=q).sum()) - 1
        ).reset_index(level=0, drop=True)
    
    return df

def calculate_rolling_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate rolling volatility and Sharpe ratio."""
    df = df.copy()
    
    # Assume risk-free rate is quarterly repo_rate / 400 (if it's annualized %)
    df['rf_q'] = df['repo_rate'] / 400
    df['excess_return'] = df['fund_return'] - df['rf_q']
    
    for q in [4, 8]:
        # Volatility
        df[f'rolling_volatility_{q}q'] = df.groupby('fund_id')['fund_return'].transform(
            lambda x: x.rolling(window=q).std()
        )
        
        # Annualize the volatility (multiply by sqrt(4) = 2)
        df[f'rolling_volatility_{q}q'] *= 2
        
        # Sharpe
        rolling_excess = df.groupby('fund_id')['excess_return'].apply(
            lambda x: np.exp(np.log1p(x).rolling(window=q).sum()) - 1
        ).reset_index(level=0, drop=True)
        
        df[f'rolling_sharpe_{q}q'] = rolling_excess / df[f'rolling_volatility_{q}q']
        
    df = df.drop(columns=['rf_q', 'excess_return'])
    return df

def calculate_alpha_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate gross and net alpha and their rolling versions."""
    df = df.copy()
    
    df['gross_alpha'] = df['fund_return'] - df['benchmark_return']
    # expense_gap is annualized, convert to quarterly
    df['expense_gap_q'] = df['expense_gap'] / 4
    df['net_alpha'] = df['gross_alpha'] - df['expense_gap_q']
    
    for q in [4, 8]:
        df[f'rolling_alpha_{q}q'] = df.groupby('fund_id')['net_alpha'].apply(
            lambda x: np.exp(np.log1p(x).rolling(window=q).sum()) - 1
        ).reset_index(level=0, drop=True)
        
    df['alpha_persistence'] = df['rolling_alpha_4q'] - df['rolling_alpha_8q']
    
    df = df.drop(columns=['expense_gap_q'])
    return df

def calculate_beta_and_ir(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate beta, tracking error, and information ratio over 8 quarters."""
    df = df.copy()
    
    def calc_beta(x):
        return x['fund_return'].rolling(window=8).cov(x['benchmark_return']) / x['benchmark_return'].rolling(window=8).var()
        
    def calc_te(x):
        return (x['fund_return'] - x['benchmark_return']).rolling(window=8).std() * 2 # Annualized
    
    df['beta_8q'] = df.groupby('fund_id').apply(calc_beta).reset_index(level=0, drop=True)
    df['tracking_error_8q'] = df.groupby('fund_id').apply(calc_te).reset_index(level=0, drop=True)
    
    # Information Ratio
    df['information_ratio_8q'] = df['rolling_alpha_8q'] / df['tracking_error_8q']
    
    return df

def calculate_cost_and_fund_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate cost features, fund features, and one-hot encode category."""
    df = df.copy()
    
    df['estimated_hidden_cost'] = df['portfolio_turnover'] * 0.001
    df['log_aum'] = np.log1p(df['aum_cr'])
    
    # One-hot encode category
    cat_dummies = pd.get_dummies(df['category'], prefix='cat', drop_first=False)
    # Convert booleans to int to prevent issues later
    cat_dummies = cat_dummies.astype(int)
    df = pd.concat([df, cat_dummies], axis=1)
    
    return df

def engineer_features(input_path: str, output_path: str):
    """Run all feature engineering steps."""
    df = pd.read_csv(input_path)
    # Reconvert date because read_csv imports it as string
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        
    # Sort just in case
    df = df.sort_values(by=['fund_id', 'date'])
    
    df = calculate_rolling_returns(df)
    df = calculate_rolling_risk(df)
    df = calculate_alpha_features(df)
    df = calculate_beta_and_ir(df)
    df = calculate_cost_and_fund_features(df)
    
    # We drop NA values created by rolling windows
    df = df.dropna().reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Engineered features saved to {output_path}")
    
    return df

if __name__ == "__main__":
    engineer_features("data/processed/fund_quarterly_dataset.csv", "data/processed/fund_features.csv")
