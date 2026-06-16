import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

def evaluate_models(test_df: pd.DataFrame, features: list, models_dir: str, reports_dir: str):
    """Evaluate best models on the test set and generate reports/figures."""
    X_test = test_df[features]
    y_test_reg = test_df['next_quarter_net_alpha']
    y_test_clf = test_df['commission_justified']
    
    reg_model_path = os.path.join(models_dir, 'best_regression_model.pkl')
    clf_model_path = os.path.join(models_dir, 'best_classification_model.pkl')
    
    os.makedirs(reports_dir, exist_ok=True)
    fig_dir = os.path.join(reports_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    results = []
    
    # ---------------------------
    # Regression Evaluation
    # ---------------------------
    if os.path.exists(reg_model_path):
        reg_model = joblib.load(reg_model_path)
        reg_preds = reg_model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test_reg, reg_preds))
        mae = mean_absolute_error(y_test_reg, reg_preds)
        r2 = r2_score(y_test_reg, reg_preds)
        
        results.append({'Task': 'Regression', 'Metric': 'RMSE', 'Value': rmse})
        results.append({'Task': 'Regression', 'Metric': 'MAE', 'Value': mae})
        results.append({'Task': 'Regression', 'Metric': 'R2', 'Value': r2})
        
        # Plot Actual vs Predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test_reg, reg_preds, alpha=0.5)
        plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--')
        plt.xlabel("Actual Next Quarter Net Alpha")
        plt.ylabel("Predicted Next Quarter Net Alpha")
        plt.title("Actual vs Predicted Net Alpha")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'actual_vs_predicted_alpha.png'))
        plt.close()
        
    # ---------------------------
    # Classification Evaluation
    # ---------------------------
    if os.path.exists(clf_model_path):
        clf_model = joblib.load(clf_model_path)
        clf_preds = clf_model.predict(X_test)
        clf_probs = clf_model.predict_proba(X_test)[:, 1] if hasattr(clf_model, 'predict_proba') else clf_preds
        
        acc = accuracy_score(y_test_clf, clf_preds)
        prec = precision_score(y_test_clf, clf_preds, zero_division=0)
        rec = recall_score(y_test_clf, clf_preds, zero_division=0)
        f1 = f1_score(y_test_clf, clf_preds, zero_division=0)
        auc = roc_auc_score(y_test_clf, clf_probs)
        
        # Precision at top decile
        test_df_eval = test_df.copy()
        test_df_eval['prob'] = clf_probs
        top_decile = test_df_eval.nlargest(int(len(test_df_eval) * 0.1), 'prob')
        if len(top_decile) > 0:
            prec_top_decile = top_decile['commission_justified'].mean()
        else:
            prec_top_decile = 0.0
            
        results.append({'Task': 'Classification', 'Metric': 'Accuracy', 'Value': acc})
        results.append({'Task': 'Classification', 'Metric': 'Precision', 'Value': prec})
        results.append({'Task': 'Classification', 'Metric': 'Recall', 'Value': rec})
        results.append({'Task': 'Classification', 'Metric': 'F1 Score', 'Value': f1})
        results.append({'Task': 'Classification', 'Metric': 'AUC-ROC', 'Value': auc})
        results.append({'Task': 'Classification', 'Metric': 'Precision at Top Decile', 'Value': prec_top_decile})
        
        # Confusion Matrix Heatmap
        cm = confusion_matrix(y_test_clf, clf_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'confusion_matrix.png'))
        plt.close()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test_clf, clf_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'roc_curve.png'))
        plt.close()
        
        # Feature Importance (if applicable)
        model_step = clf_model.named_steps.get('model')
        if hasattr(model_step, 'feature_importances_'):
            importances = model_step.feature_importances_
            indices = np.argsort(importances)[::-1][:15] # Top 15
            plt.figure(figsize=(10, 6))
            plt.title("Top 15 Feature Importances (Classification)")
            plt.bar(range(15), importances[indices], align="center")
            plt.xticks(range(15), [features[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'feature_importance.png'))
            plt.close()

    # Save metrics
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(reports_dir, 'model_results.csv'), index=False)
    print(f"Evaluation complete. Results saved to {reports_dir}")
    
    return results_df

if __name__ == "__main__":
    from src.modeling import get_features, temporal_split
    df = pd.read_csv("data/processed/model_ready_data.csv")
    df['date'] = pd.to_datetime(df['date'])
    _, _, test_df = temporal_split(df)
    features = get_features(df)
    evaluate_models(test_df, features, "models", "reports")
