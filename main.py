import os
import pandas as pd
from src.data_collection import generate_sample_data
from src.preprocessing import preprocess_pipeline
from src.feature_engineering import engineer_features
from src.target_creation import create_targets
from src.modeling import train_models
from src.evaluation import evaluate_models
from src.explainability import generate_shap_explanations
from src.sip_backtest import run_backtest
from src.utils import ensure_directories_exist

def main():
    print("Starting CommissionLens Pipeline...")
    
    directories = [
        "data/raw",
        "data/processed",
        "data/sample",
        "models",
        "reports/figures"
    ]
    ensure_directories_exist(directories)
    
    # Step 1: Generate or load data
    print("\n--- Step 1: Data Generation ---")
    sample_data_path = "data/sample/synthetic_funds_data.csv"
    if not os.path.exists(sample_data_path):
        generate_sample_data(output_path=sample_data_path, n_funds=250, years=5)
    else:
        print(f"Sample data already exists at {sample_data_path}")
        
    # Step 2: Preprocess data
    print("\n--- Step 2: Preprocessing ---")
    processed_data_path = "data/processed/fund_quarterly_dataset.csv"
    preprocess_pipeline(sample_data_path, processed_data_path)
    
    # Step 3: Engineer features
    print("\n--- Step 3: Feature Engineering ---")
    features_path = "data/processed/fund_features.csv"
    engineer_features(processed_data_path, features_path)
    
    # Step 4: Create targets
    print("\n--- Step 4: Target Creation ---")
    model_ready_path = "data/processed/model_ready_data.csv"
    create_targets(features_path, model_ready_path)
    
    # Step 5: Train models
    print("\n--- Step 5: Model Training ---")
    df = pd.read_csv(model_ready_path)
    df['date'] = pd.to_datetime(df['date'])
    train_df, val_df, test_df, features = train_models(model_ready_path, "models")
    
    # Step 6: Evaluate models
    print("\n--- Step 6: Evaluation ---")
    evaluate_models(test_df, features, "models", "reports")
    
    # Step 7: Run SHAP explainability
    print("\n--- Step 7: SHAP Explainability ---")
    generate_shap_explanations(df, features, "models", "reports")
    
    # Step 8: Run SIP backtest
    print("\n--- Step 8: SIP Backtest ---")
    run_backtest(df, features, "models", "reports")
    
    print("\nPipeline Complete!")
    print("You can now run Streamlit app using: streamlit run app/streamlit_app.py")

if __name__ == "__main__":
    main()
