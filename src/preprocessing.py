import pandas as pd
import numpy as np
import os

def convert_dates(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """Convert date column to datetime."""
    df[date_col] = pd.to_datetime(df[date_col])
    return df

def clean_nav_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the main dataset by dropping extreme invalid NAVs if any."""
    df = df[df['nav'] > 0]
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate records."""
    df = df.drop_duplicates(subset=['fund_id', 'date'])
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values by forward-filling macro variables
    and dropping rows where essential data is missing.
    """
    # Sort to ensure forward-fill works correctly
    df = df.sort_values(by=['fund_id', 'date'])
    
    macro_cols = ['repo_rate', 'cpi_inflation', 'yield_curve_slope', 'fii_flow_cr', 'dii_flow_cr']
    df[macro_cols] = df.groupby('fund_id')[macro_cols].ffill()
    
    # Drop where target values might be missing
    df = df.dropna(subset=['fund_return', 'benchmark_return'])
    
    return df

def validate_columns(df: pd.DataFrame, required_columns: list) -> bool:
    """Validate that all required columns are present."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True

def create_quarterly_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert monthly data to quarterly data.
    Takes the last value of the quarter for NAV and AUM,
    and cumulative product for returns.
    """
    df = df.copy()
    df['quarter'] = df['date'].dt.to_period('Q')
    
    # Sort for sequential calculations
    df = df.sort_values(by=['fund_id', 'date'])
    
    # Function to calculate quarterly return from monthly returns
    def calc_quarterly_return(x):
        return (1 + x).prod() - 1
    
    agg_funcs = {
        'fund_name': 'first',
        'category': 'first',
        'date': 'last', # End of quarter date
        'nav': 'last',
        'fund_return': calc_quarterly_return,
        'benchmark_return': calc_quarterly_return,
        'regular_expense_ratio': 'last',
        'direct_expense_ratio': 'last',
        'expense_gap': 'last',
        'aum_cr': 'last',
        'portfolio_turnover': 'mean',
        'fund_manager_tenure_years': 'last',
        'repo_rate': 'last',
        'cpi_inflation': 'last',
        'yield_curve_slope': 'last',
        'fii_flow_cr': 'sum',
        'dii_flow_cr': 'sum'
    }
    
    quarterly_df = df.groupby(['fund_id', 'quarter']).agg(agg_funcs).reset_index()
    quarterly_df = quarterly_df.sort_values(by=['fund_id', 'date']).reset_index(drop=True)
    
    return quarterly_df

def preprocess_pipeline(input_path: str, output_path: str):
    """Run the full preprocessing pipeline."""
    df = pd.read_csv(input_path)
    
    required_cols = [
        'fund_id', 'fund_name', 'date', 'nav', 'fund_return', 'benchmark_return',
        'regular_expense_ratio', 'direct_expense_ratio', 'expense_gap',
        'aum_cr', 'portfolio_turnover', 'fund_manager_tenure_years',
        'repo_rate', 'cpi_inflation', 'yield_curve_slope', 'fii_flow_cr', 'dii_flow_cr',
        'category'
    ]
    validate_columns(df, required_cols)
    
    df = convert_dates(df)
    df = clean_nav_data(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    
    # The requirement asks to create quarterly dataset
    q_df = create_quarterly_dataset(df)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    q_df.to_csv(output_path, index=False)
    print(f"Processed quarterly data saved to {output_path}")
    
    return q_df

if __name__ == "__main__":
    preprocess_pipeline("data/sample/synthetic_funds_data.csv", "data/processed/fund_quarterly_dataset.csv")
