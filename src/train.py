"""
Model Training Script for Student Performance Prediction
Trains and persists Logistic Regression, Decision Tree, and Random Forest models.
"""

import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading preprocessed training data...")
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).values.ravel()
    
    # 1. Logistic Regression Model (Baseline linear model)
    print("\n[1/3] Training Logistic Regression...")
    lr_model = LogisticRegression(
        C=1.0,
        solver='liblinear',
        random_state=42
    )
    lr_model.fit(X_train, y_train)
    lr_train_acc = lr_model.score(X_train, y_train)
    print(f" -> Logistic Regression Train Accuracy: {lr_train_acc * 100:.2f}%")
    joblib.dump(lr_model, os.path.join(models_dir, 'logistic_regression.pkl'))
    
    # 2. Decision Tree Classifier (Interpretable non-linear tree rules)
    print("\n[2/3] Training Decision Tree Classifier...")
    dt_model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        criterion='gini',
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    dt_train_acc = dt_model.score(X_train, y_train)
    print(f" -> Decision Tree Train Accuracy: {dt_train_acc * 100:.2f}%")
    joblib.dump(dt_model, os.path.join(models_dir, 'decision_tree.pkl'))
    
    # 3. Random Forest Classifier (Bagged ensemble of trees)
    print("\n[3/3] Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=8,
        min_samples_leaf=4,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_train_acc = rf_model.score(X_train, y_train)
    print(f" -> Random Forest Train Accuracy: {rf_train_acc * 100:.2f}%")
    joblib.dump(rf_model, os.path.join(models_dir, 'random_forest.pkl'))
    
    print("\nAll 3 models trained and saved successfully in:", models_dir)
    return {
        'Logistic Regression': lr_model,
        'Decision Tree': dt_model,
        'Random Forest': rf_model
    }

if __name__ == '__main__':
    train_models()
