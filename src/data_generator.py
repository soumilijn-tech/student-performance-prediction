"""
Data Generator for Student Performance Prediction
Generates a realistic student academic dataset with genuine educational correlations.
"""

import os
import numpy as np
import pandas as pd

def generate_student_data(n_samples: int = 1200, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic student performance data with realistic relationships.
    
    Parameters:
        n_samples (int): Number of student records to generate.
        random_seed (int): Seed for reproducibility.
        
    Returns:
        pd.DataFrame: DataFrame containing student attributes and performance outcomes.
    """
    np.random.seed(random_seed)
    
    # 1. Study Hours per week: Mean 15, std 6, bounded between 2 and 35
    study_hours = np.clip(np.random.normal(15, 6, n_samples), 2, 35).round(1)
    
    # 2. Attendance Rate (%): Mean 82%, std 12%, bounded between 50% and 100%
    attendance_rate = np.clip(np.random.normal(82, 11, n_samples), 50, 100).round(1)
    
    # 3. Previous Exam Score (0-100): Mean 66, std 14, bounded between 35 and 98
    past_exam_score = np.clip(np.random.normal(66, 14, n_samples), 35, 98).round(1)
    
    # 4. Sleep Hours: Mean 7.0, std 1.1, bounded between 4.0 and 9.5
    sleep_hours = np.clip(np.random.normal(7.0, 1.1, n_samples), 4.0, 9.5).round(1)
    
    # 5. Tutoring Sessions per month (0 to 6)
    tutoring_sessions = np.random.choice([0, 1, 2, 3, 4, 5, 6], size=n_samples, p=[0.25, 0.25, 0.20, 0.15, 0.08, 0.05, 0.02])
    
    # 6. Extracurricular Activities: Yes / No
    extracurricular = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.55, 0.45])
    
    # 7. Parent Education Level: High School, Bachelor, Master, Doctorate
    parent_edu = np.random.choice(['High School', 'Bachelor', 'Master', 'Doctorate'], size=n_samples, p=[0.35, 0.40, 0.18, 0.07])
    
    # 8. Internet Access: Yes / No
    internet_access = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.88, 0.12])
    
    # 9. Stress Level: Low, Medium, High
    stress_level = np.random.choice(['Low', 'Medium', 'High'], size=n_samples, p=[0.25, 0.50, 0.25])
    
    # Base calculation for Final Performance Score (0-100)
    edu_weights = {'High School': 0.0, 'Bachelor': 2.5, 'Master': 4.5, 'Doctorate': 6.0}
    parent_bonus = np.array([edu_weights[e] for e in parent_edu])
    internet_bonus = np.where(internet_access == 'Yes', 2.0, -3.0)
    stress_penalty = np.where(stress_level == 'High', -4.5, np.where(stress_level == 'Medium', -1.0, 2.0))
    sleep_effect = np.where((sleep_hours >= 6.5) & (sleep_hours <= 8.5), 2.5, -2.0)
    extra_bonus = np.where(extracurricular == 'Yes', 1.5, 0.0)
    
    # Non-linear interaction: low attendance severely hurts even with high past scores
    attendance_factor = (attendance_rate / 100.0) ** 1.3
    study_factor = (study_hours / 35.0) * 20.0
    
    # Core score calculation
    raw_final_score = (
        0.38 * past_exam_score +
        35.0 * attendance_factor +
        study_factor +
        1.5 * tutoring_sessions +
        parent_bonus +
        internet_bonus +
        stress_penalty +
        sleep_effect +
        extra_bonus +
        np.random.normal(0, 4.0, n_samples)  # Stochastic noise
    )
    
    # Normalize score between 20 and 100
    final_score = np.clip(raw_final_score, 20.0, 100.0).round(1)
    
    # Binary Target: Passed (1 = Pass / Low Risk, 0 = At-Risk / Fail)
    # Passing threshold is 60.0 (grades >= 60 are passing)
    passed = (final_score >= 60.0).astype(int)
    
    # Student IDs
    student_ids = [f"STU_{1000 + i}" for i in range(n_samples)]
    
    df = pd.DataFrame({
        'Student_ID': student_ids,
        'Study_Hours_Per_Week': study_hours,
        'Attendance_Rate': attendance_rate,
        'Past_Exam_Score': past_exam_score,
        'Sleep_Hours': sleep_hours,
        'Tutoring_Sessions': tutoring_sessions,
        'Extracurricular_Activities': extracurricular,
        'Parent_Education_Level': parent_edu,
        'Internet_Access': internet_access,
        'Stress_Level': stress_level,
        'Final_Exam_Score': final_score,
        'Passed': passed
    })
    
    return df

def save_raw_dataset(output_dir: str = None):
    """Generate and save dataset to CSV."""
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, 'data', 'raw')
    
    os.makedirs(output_dir, exist_ok=True)
    df = generate_student_data(n_samples=1200, random_seed=42)
    output_path = os.path.join(output_dir, 'student_data.csv')
    df.to_csv(output_path, index=False)
    
    pass_count = (df['Passed'] == 1).sum()
    fail_count = (df['Passed'] == 0).sum()
    pass_pct = (pass_count / len(df)) * 100
    
    print(f"Dataset successfully created and saved to: {output_path}")
    print(f"Total Student Records: {len(df)}")
    print(f"Passed (Class 1): {pass_count} ({pass_pct:.1f}%)")
    print(f"At-Risk / Failed (Class 0): {fail_count} ({100 - pass_pct:.1f}%)")
    print("Sample Records:")
    print(df[['Student_ID', 'Study_Hours_Per_Week', 'Attendance_Rate', 'Past_Exam_Score', 'Final_Exam_Score', 'Passed']].head(3))
    return output_path

if __name__ == '__main__':
    save_raw_dataset()
