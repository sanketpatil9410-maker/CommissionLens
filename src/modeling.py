import pandas as pd
import numpy as np
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

def temporal_split(df: pd.DataFrame):
    """
    Split the dataset into train (70%), validation (15%), and test (15%)
    based on unique dates to avoid look-ahead bias.
    """
    unique_dates = sorted(df['date'].unique())
    n_dates = len(unique_dates)
    
    train_idx = int(n_dates * 0.7)
    val_idx = int(n_dates * 0.85)
    
    train_dates = unique_dates[:train_idx]
    val_dates = unique_dates[train_idx:val_idx]
    test_dates = unique_dates[val_idx:]
    
    train_df = df[df['date'].isin(train_dates)].copy()
    val_df = df[df['date'].isin(val_dates)].copy()
    test_df = df[df['date'].isin(test_dates)].copy()
    
    return train_df, val_df, test_df

def get_features(df: pd.DataFrame):
    """Extract features from the dataframe."""
    base_features = [
        'rolling_3q_return', 'rolling_4q_return', 'rolling_8q_return',
        'rolling_volatility_4q', 'rolling_volatility_8q',
        'rolling_sharpe_4q', 'rolling_sharpe_8q',
        'rolling_alpha_4q', 'rolling_alpha_8q', 'alpha_persistence',
        'beta_8q', 'tracking_error_8q', 'information_ratio_8q',
        'expense_gap', 'regular_expense_ratio', 'direct_expense_ratio',
        'portfolio_turnover', 'estimated_hidden_cost',
        'repo_rate', 'cpi_inflation', 'yield_curve_slope', 'fii_flow_cr', 'dii_flow_cr',
        'log_aum', 'fund_manager_tenure_years'
    ]
    
    # Add one-hot encoded categories
    cat_features = [col for col in df.columns if col.startswith('cat_')]
    return base_features + cat_features

def train_models(input_path: str, output_dir: str):
    """Train regression and classification models and save the best ones."""
    df = pd.read_csv(input_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        
    train_df, val_df, test_df = temporal_split(df)
    
    features = get_features(df)
    
    X_train, y_train_reg, y_train_clf = train_df[features], train_df['next_quarter_net_alpha'], train_df['commission_justified']
    X_val, y_val_reg, y_val_clf = val_df[features], val_df['next_quarter_net_alpha'], val_df['commission_justified']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ---------------------------
    # Regression Models
    # ---------------------------
    reg_models = {
        'Linear Regression': Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
        'Random Forest': Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor(n_estimators=100, random_state=42))]),
        'XGBoost': Pipeline([('scaler', StandardScaler()), ('model', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))])
    }
    
    best_reg_model = None
    best_reg_rmse = float('inf')
    best_reg_name = ""
    
    for name, pipeline in reg_models.items():
        pipeline.fit(X_train, y_train_reg)
        val_preds = pipeline.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val_reg, val_preds))
        print(f"Regression - {name} Validation RMSE: {rmse:.4f}")
        
        if rmse < best_reg_rmse:
            best_reg_rmse = rmse
            best_reg_model = pipeline
            best_reg_name = name
            
    print(f"Best Regression Model: {best_reg_name}")
    joblib.dump(best_reg_model, os.path.join(output_dir, 'best_regression_model.pkl'))
    
    # ---------------------------
    # Classification Models
    # ---------------------------
    clf_models = {
        'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(random_state=42, max_iter=1000))]),
        'Random Forest': Pipeline([('scaler', StandardScaler()), ('model', RandomForestClassifier(n_estimators=100, random_state=42))]),
        'XGBoost': Pipeline([('scaler', StandardScaler()), ('model', XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss'))])
    }
    
    best_clf_model = None
    best_clf_acc = -1
    best_clf_name = ""
    
    for name, pipeline in clf_models.items():
        pipeline.fit(X_train, y_train_clf)
        val_preds = pipeline.predict(X_val)
        acc = accuracy_score(y_val_clf, val_preds)
        print(f"Classification - {name} Validation Accuracy: {acc:.4f}")
        
        if acc > best_clf_acc:
            best_clf_acc = acc
            best_clf_model = pipeline
            best_clf_name = name
            
    print(f"Best Classification Model: {best_clf_name}")
    joblib.dump(best_clf_model, os.path.join(output_dir, 'best_classification_model.pkl'))
    
    return train_df, val_df, test_df, features

if __name__ == "__main__":
    train_models("data/processed/model_ready_data.csv", "models")
