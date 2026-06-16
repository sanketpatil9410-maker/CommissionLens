import pandas as pd
import numpy as np
import os

def create_targets(input_path: str, output_path: str):
    """
    Create targets for modeling.
    Requires temporal shift to avoid look-ahead bias in features.
    """
    df = pd.read_csv(input_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        
    df = df.sort_values(by=['fund_id', 'date'])
    
    # Create shifted targets
    # Shift by -1 brings next quarter's value to current row
    df['next_quarter_net_alpha'] = df.groupby('fund_id')['net_alpha'].shift(-1)
    df['next_quarter_gross_alpha'] = df.groupby('fund_id')['gross_alpha'].shift(-1)
    
    # We drop the last quarter for each fund since we don't have its future value
    df = df.dropna(subset=['next_quarter_net_alpha', 'next_quarter_gross_alpha']).copy()
    
    # Expense gap is annualized, so quarterly expense gap is expense_gap / 4
    expense_gap_q = df['expense_gap'] / 4
    
    # Classification targets
    df['commission_justified'] = (df['next_quarter_gross_alpha'] > expense_gap_q).astype(int)
    
    # Strong commission justified
    df['strong_commission_justified'] = (df['next_quarter_net_alpha'] > 0.005).astype(int)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Targets created and dataset saved to {output_path}")
    
    return df

if __name__ == "__main__":
    create_targets("data/processed/fund_features.csv", "data/processed/model_ready_data.csv")
