"""
Preprocessing Pipeline for Student Performance Prediction
Handles data cleaning, categorical encoding, feature scaling, and train/test splitting.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Mapping definitions for ordinal and binary categorical variables
BINARY_MAPPINGS = {
    'Extracurricular_Activities': {'No': 0, 'Yes': 1},
    'Internet_Access': {'No': 0, 'Yes': 1}
}

ORDINAL_MAPPINGS = {
    'Parent_Education_Level': {
        'High School': 0,
        'Bachelor': 1,
        'Master': 2,
        'Doctorate': 3
    },
    'Stress_Level': {
        'Low': 0,
        'Medium': 1,
        'High': 2
    }
}

NUMERICAL_FEATURES = [
    'Study_Hours_Per_Week',
    'Attendance_Rate',
    'Past_Exam_Score',
    'Sleep_Hours',
    'Tutoring_Sessions'
]

FEATURE_COLUMNS = [
    'Study_Hours_Per_Week',
    'Attendance_Rate',
    'Past_Exam_Score',
    'Sleep_Hours',
    'Tutoring_Sessions',
    'Extracurricular_Activities',
    'Parent_Education_Level',
    'Internet_Access',
    'Stress_Level'
]

TARGET_COLUMN = 'Passed'


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode binary and categorical features into numeric format.
    """
    processed = df.copy()
    
    # Binary encoding
    for col, mapping in BINARY_MAPPINGS.items():
        if col in processed.columns:
            processed[col] = processed[col].map(mapping).fillna(0).astype(int)
            
    # Ordinal encoding
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in processed.columns:
            processed[col] = processed[col].map(mapping).fillna(0).astype(int)
            
    return processed


def preprocess_data(raw_data_path: str = None, save_artifacts: bool = True):
    """
    Loads raw data, encodes features, scales numerical columns,
    splits into train/test, and persists preprocessed data and preprocessor artifacts.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if raw_data_path is None:
        raw_data_path = os.path.join(base_dir, 'data', 'raw', 'student_data.csv')
        
    print(f"Loading raw data from: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    
    # Ensure no missing values
    if df[FEATURE_COLUMNS].isnull().any().any():
        print("Imputing missing values with median/mode...")
        for col in NUMERICAL_FEATURES:
            df[col] = df[col].fillna(df[col].median())
        for col in list(BINARY_MAPPINGS.keys()) + list(ORDINAL_MAPPINGS.keys()):
            df[col] = df[col].fillna(df[col].mode()[0])
            
    # Encode categorical variables
    encoded_df = encode_features(df)
    
    X = encoded_df[FEATURE_COLUMNS].copy()
    y = encoded_df[TARGET_COLUMN].copy()
    
    # Stratified Train/Test split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Fit StandardScaler on training numerical columns
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[NUMERICAL_FEATURES] = scaler.fit_transform(X_train[NUMERICAL_FEATURES])
    X_test_scaled[NUMERICAL_FEATURES] = scaler.transform(X_test[NUMERICAL_FEATURES])
    
    if save_artifacts:
        processed_dir = os.path.join(base_dir, 'data', 'processed')
        models_dir = os.path.join(base_dir, 'models')
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        # Save datasets
        X_train_scaled.to_csv(os.path.join(processed_dir, 'X_train.csv'), index=False)
        X_test_scaled.to_csv(os.path.join(processed_dir, 'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False)
        
        # Save unscaled test set as well (helpful for human-interpretable tree visualization)
        X_test.to_csv(os.path.join(processed_dir, 'X_test_unscaled.csv'), index=False)
        
        # Save Preprocessor bundle
        preprocessor_bundle = {
            'scaler': scaler,
            'numerical_features': NUMERICAL_FEATURES,
            'feature_columns': FEATURE_COLUMNS,
            'binary_mappings': BINARY_MAPPINGS,
            'ordinal_mappings': ORDINAL_MAPPINGS
        }
        joblib.dump(preprocessor_bundle, os.path.join(models_dir, 'preprocessor.pkl'))
        print(f"Preprocessing completed successfully!")
        print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
        print(f"Preprocessor saved to: {os.path.join(models_dir, 'preprocessor.pkl')}")
        
    return X_train_scaled, X_test_scaled, y_train, y_test


def transform_single_student(input_data: dict, preprocessor_bundle: dict = None) -> np.ndarray:
    """
    Transforms a single student input dictionary into model-ready scaled features.
    Used by Streamlit dashboard for real-time predictions.
    """
    if preprocessor_bundle is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bundle_path = os.path.join(base_dir, 'models', 'preprocessor.pkl')
        preprocessor_bundle = joblib.load(bundle_path)
        
    df = pd.DataFrame([input_data])
    encoded = encode_features(df)
    
    # Scale numerical features
    scaler = preprocessor_bundle['scaler']
    num_cols = preprocessor_bundle['numerical_features']
    encoded[num_cols] = scaler.transform(encoded[num_cols])
    
    feature_cols = preprocessor_bundle['feature_columns']
    return encoded[feature_cols]


if __name__ == '__main__':
    preprocess_data()
