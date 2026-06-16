import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt

def generate_shap_explanations(df: pd.DataFrame, features: list, models_dir: str, reports_dir: str):
    """Generate SHAP explanations for the best classification model (assumed tree-based)."""
    clf_model_path = os.path.join(models_dir, 'best_classification_model.pkl')
    
    if not os.path.exists(clf_model_path):
        print("Model not found for explainability.")
        return
        
    pipeline = joblib.load(clf_model_path)
    
    model = pipeline.named_steps.get('model')
    scaler = pipeline.named_steps.get('scaler')
    
    # SHAP requires the model directly. TreeExplainer is best for RF/XGBoost.
    if hasattr(model, 'feature_importances_'):
        # We will explain on a subset to save time
        X_explain = df[features].sample(min(1000, len(df)), random_state=42)
        X_explain_scaled = scaler.transform(X_explain)
        
        # Convert back to DataFrame for SHAP to have feature names
        X_explain_scaled_df = pd.DataFrame(X_explain_scaled, columns=features)
        
        explainer = shap.TreeExplainer(model)
        # shap_values might be a list for classification (classes) or an array
        shap_values = explainer.shap_values(X_explain_scaled_df)
        
        # For some models, shap_values is a list of arrays (one per class). 
        # For others it's a 3D array (n_samples, n_features, n_classes).
        # We want the explanation for the positive class (class 1).
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1]
        elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
            shap_values_to_plot = shap_values[:, :, 1]
        else:
            shap_values_to_plot = shap_values
            
        fig_dir = os.path.join(reports_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        
        # Summary Plot
        plt.figure()
        shap.summary_plot(shap_values_to_plot, X_explain_scaled_df, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'shap_summary_plot.png'), bbox_inches='tight')
        plt.close()
        
        # Bar Plot
        plt.figure()
        shap.summary_plot(shap_values_to_plot, X_explain_scaled_df, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'shap_bar_plot.png'), bbox_inches='tight')
        plt.close()
        
        # Top 3 features dependence plots
        mean_abs_shap = np.abs(shap_values_to_plot).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:3]
        top_features = [features[int(i)] for i in top_indices]
        
        for feature in top_features:
            plt.figure()
            shap.dependence_plot(feature, shap_values_to_plot, X_explain_scaled_df, show=False)
            plt.tight_layout()
            # Clean filename
            safe_fname = feature.replace('/', '_')
            plt.savefig(os.path.join(fig_dir, f'shap_dependence_{safe_fname}.png'), bbox_inches='tight')
            plt.close()
            
        # Generate Text Summary
        summary_path = os.path.join(reports_dir, 'shap_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("SHAP Explainability Summary\n")
            f.write("===========================\n\n")
            f.write("Top 5 Features Influencing Commission Justification:\n")
            top_5_features = [features[int(i)] for i in np.argsort(mean_abs_shap)[::-1][:5]]
            for idx, feature in enumerate(top_5_features, 1):
                f.write(f"{idx}. {feature}\n")
                
            f.write("\nInterpretation:\n")
            f.write("- Features with higher SHAP values push the model to predict 'Commission Justified' (1).\n")
            f.write("- Features with lower SHAP values push the model to predict 'Not Justified' (0).\n")
            f.write("- The summary plot shows how high and low values of these features affect the output.\n")
            f.write("- This is useful for understanding the exact drivers behind mutual fund performance metrics.\n")
            
        print(f"SHAP explanations generated and saved to {reports_dir}")
        
if __name__ == "__main__":
    from src.modeling import get_features, temporal_split
    df = pd.read_csv("data/processed/model_ready_data.csv")
    features = get_features(df)
    generate_shap_explanations(df, features, "models", "reports")
