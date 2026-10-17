"""
Model Evaluation Script for Student Performance Prediction
Evaluates Logistic Regression, Decision Tree, and Random Forest on test set.
Generates Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def evaluate_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    models_dir = os.path.join(base_dir, 'models')
    
    print("Loading test data...")
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).values.ravel()
    
    feature_names = list(X_test.columns)
    
    # Load trained models
    models = {
        'Logistic Regression': joblib.load(os.path.join(models_dir, 'logistic_regression.pkl')),
        'Decision Tree': joblib.load(os.path.join(models_dir, 'decision_tree.pkl')),
        'Random Forest': joblib.load(os.path.join(models_dir, 'random_forest.pkl'))
    }
    
    results = {}
    print("\n" + "=" * 70)
    print("                  MODEL EVALUATION REPORT (TEST SET)                  ")
    print("=" * 70)
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        # Calculate probabilities for ROC-AUC
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_prob))
        else:
            roc_auc = 0.0
            
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        # Feature importance / coefficients
        feature_importance_dict = {}
        if hasattr(model, 'feature_importances_'):
            for feat, imp in zip(feature_names, model.feature_importances_):
                feature_importance_dict[feat] = round(float(imp), 4)
        elif hasattr(model, 'coef_'):
            for feat, coef in zip(feature_names, model.coef_[0]):
                feature_importance_dict[feat] = round(float(coef), 4)
                
        results[name] = {
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1_Score': round(f1, 4),
            'ROC_AUC': round(roc_auc, 4),
            'Confusion_Matrix': cm,
            'Feature_Importance': feature_importance_dict
        }
        
        print(f"\n--- {name} ---")
        print(f"Accuracy:  {acc * 100:.2f}%")
        print(f"Precision: {prec * 100:.2f}%")
        print(f"Recall:    {rec * 100:.2f}%")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print("Confusion Matrix:")
        print(f"  [TN: {cm[0][0]}, FP: {cm[0][1]}]")
        print(f"  [FN: {cm[1][0]}, TP: {cm[1][1]}]")
        
    # Save results to JSON
    metrics_path = os.path.join(models_dir, 'model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "=" * 70)
    print(f"Metrics successfully saved to: {metrics_path}")
    
    # Comparison summary table
    summary_df = pd.DataFrame([
        {
            'Model': m,
            'Accuracy (%)': f"{results[m]['Accuracy'] * 100:.2f}%",
            'Precision (%)': f"{results[m]['Precision'] * 100:.2f}%",
            'Recall (%)': f"{results[m]['Recall'] * 100:.2f}%",
            'F1 Score': results[m]['F1_Score'],
            'ROC-AUC': results[m]['ROC_AUC']
        }
        for m in results
    ])
    print("\n--- Model Comparison Summary ---")
    print(summary_df.to_string(index=False))
    print("=" * 70)
    
    return results

if __name__ == '__main__':
    evaluate_models()
