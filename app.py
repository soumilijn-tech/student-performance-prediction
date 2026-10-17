"""
Streamlit Web Dashboard for Student Performance Prediction
Provides interactive real-time predictions, model comparison, and feature insights.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ---------------------------------------------------------
# Helper Functions: Load Models, Metrics, and Preprocessor
# ---------------------------------------------------------
def ensure_artifacts_exist():
    """Ensures datasets and trained models exist, generating them on first run if missing."""
    required_files = [
        os.path.join(MODELS_DIR, 'preprocessor.pkl'),
        os.path.join(MODELS_DIR, 'random_forest.pkl'),
        os.path.join(MODELS_DIR, 'logistic_regression.pkl'),
        os.path.join(MODELS_DIR, 'decision_tree.pkl'),
        os.path.join(MODELS_DIR, 'model_metrics.json'),
        os.path.join(DATA_DIR, 'raw', 'student_data.csv')
    ]
    if not all(os.path.exists(p) for p in required_files):
        from src.data_generator import save_raw_dataset
        from src.preprocess import preprocess_data
        from src.train import train_models
        from src.evaluate import evaluate_models
        
        save_raw_dataset()
        preprocess_data()
        train_models()
        evaluate_models()

@st.cache_resource
def load_artifacts():
    """Load preprocessor, models, and evaluation metrics."""
    ensure_artifacts_exist()
    try:
        preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.pkl'))
        models = {
            'Random Forest (Recommended)': joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl')),
            'Logistic Regression': joblib.load(os.path.join(MODELS_DIR, 'logistic_regression.pkl')),
            'Decision Tree': joblib.load(os.path.join(MODELS_DIR, 'decision_tree.pkl'))
        }
    except Exception as e:
        return None, None, None
        
    metrics_path = os.path.join(MODELS_DIR, 'model_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = {}
        
    return preprocessor, models, metrics

@st.cache_data
def load_dataset_sample():
    """Load raw dataset sample for the explorer tab."""
    csv_path = os.path.join(DATA_DIR, 'raw', 'student_data.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def preprocess_single_input(input_dict, preprocessor):
    """Encodes and scales a single student profile dictionary."""
    df = pd.DataFrame([input_dict])
    
    # Binary mappings
    for col, mapping in preprocessor['binary_mappings'].items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)
            
    # Ordinal mappings
    for col, mapping in preprocessor['ordinal_mappings'].items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)
            
    # Scale numerical features
    num_cols = preprocessor['numerical_features']
    df[num_cols] = preprocessor['scaler'].transform(df[num_cols])
    
    feature_cols = preprocessor['feature_columns']
    return df[feature_cols]

# ---------------------------------------------------------
# UI Layout
# ---------------------------------------------------------
preprocessor, models, metrics = load_artifacts()

# App Header
st.title("🎓 Student Performance Prediction System")
st.markdown(
    """
    Predict whether a student is **on track to pass** or **at academic risk** using Machine Learning.
    Adjust the student's academic and lifestyle parameters in the sidebar to view instant predictions and recommendations.
    """
)

if preprocessor is None or models is None:
    st.error(
        "⚠️ Model artifacts not found! Please run the training pipeline first: \n"
        "1. `python src/data_generator.py` \n"
        "2. `python src/preprocess.py` \n"
        "3. `python src/train.py` \n"
        "4. `python src/evaluate.py`"
    )
    st.stop()

# ---------------------------------------------------------
# Sidebar Controls: Student Profile Inputs
# ---------------------------------------------------------
st.sidebar.header("📋 Student Profile")
st.sidebar.markdown("Configure student attributes below:")

# Academic parameters
st.sidebar.subheader("📚 Academic Factors")
past_score = st.sidebar.slider(
    "Previous Exam Score (0 - 100)",
    min_value=30.0,
    max_value=100.0,
    value=70.0,
    step=0.5,
    help="Student's score in the previous term examination."
)

attendance = st.sidebar.slider(
    "Attendance Rate (%)",
    min_value=50.0,
    max_value=100.0,
    value=85.0,
    step=1.0,
    help="Percentage of scheduled classes attended."
)

study_hours = st.sidebar.slider(
    "Weekly Study Hours",
    min_value=2.0,
    max_value=35.0,
    value=16.0,
    step=0.5,
    help="Number of hours dedicated to independent study per week."
)

tutoring = st.sidebar.slider(
    "Tutoring Sessions / Month",
    min_value=0,
    max_value=6,
    value=1,
    step=1,
    help="Number of extra tutoring or remedial sessions attended."
)

# Lifestyle & Background parameters
st.sidebar.subheader("🌱 Lifestyle & Background")
sleep_hours = st.sidebar.slider(
    "Average Daily Sleep (Hours)",
    min_value=4.0,
    max_value=10.0,
    value=7.0,
    step=0.5,
    help="Average sleep hours per night."
)

stress_level = st.sidebar.selectbox(
    "Self-Reported Stress Level",
    options=["Low", "Medium", "High"],
    index=1,
    help="Perceived level of academic or personal stress."
)

parent_edu = st.sidebar.selectbox(
    "Parent's Education Level",
    options=["High School", "Bachelor", "Master", "Doctorate"],
    index=1,
    help="Highest education level attained by primary guardian."
)

extracurricular = st.sidebar.selectbox(
    "Extracurricular Participation",
    options=["Yes", "No"],
    index=0,
    help="Engaged in sports, arts, music, or school clubs."
)

internet_access = st.sidebar.selectbox(
    "Home Internet Access",
    options=["Yes", "No"],
    index=0,
    help="Reliable internet connection available for schoolwork."
)

# Model Choice
st.sidebar.subheader("🤖 Prediction Engine")
selected_model_name = st.sidebar.selectbox(
    "Select Model",
    options=["Random Forest (Recommended)", "Logistic Regression", "Decision Tree", "All Models (Consensus)"]
)

# Construct student input dictionary
student_input = {
    'Study_Hours_Per_Week': study_hours,
    'Attendance_Rate': attendance,
    'Past_Exam_Score': past_score,
    'Sleep_Hours': sleep_hours,
    'Tutoring_Sessions': tutoring,
    'Extracurricular_Activities': extracurricular,
    'Parent_Education_Level': parent_edu,
    'Internet_Access': internet_access,
    'Stress_Level': stress_level
}

# ---------------------------------------------------------
# Tabs Setup
# ---------------------------------------------------------
tab_pred, tab_compare, tab_importance, tab_data = st.tabs([
    "🔮 Real-Time Prediction",
    "📊 Model Comparison",
    "🔍 Feature Importance",
    "📁 Dataset Explorer"
])

# ---------------------------------------------------------
# TAB 1: Real-Time Prediction & Actionable Feedback
# ---------------------------------------------------------
with tab_pred:
    st.subheader("Student Academic Risk Assessment")
    
    # Preprocess inputs
    X_input = preprocess_single_input(student_input, preprocessor)
    
    # Predictions
    preds = {}
    probs = {}
    
    for name, model in models.items():
        p = model.predict(X_input)[0]
        preds[name] = p
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X_input)[0][1]
        else:
            prob = float(p)
        probs[name] = prob

    # Determine main display model
    if selected_model_name == "All Models (Consensus)":
        avg_prob = np.mean(list(probs.values()))
        final_prediction = 1 if avg_prob >= 0.50 else 0
        final_prob = avg_prob
        model_label = "Consensus (Average of 3 Models)"
    else:
        final_prediction = preds[selected_model_name]
        final_prob = probs[selected_model_name]
        model_label = selected_model_name
        
    # Visual Result Card
    col1, col2, col3 = st.columns([1.5, 1.2, 1.3])
    
    with col1:
        st.markdown("#### Primary Outcome")
        if final_prediction == 1:
            st.success("### ✅ Status: ON TRACK (PASS)")
            st.write(f"The model anticipates the student will **pass** with strong probability.")
        else:
            st.error("### ⚠️ Status: AT-RISK (POTENTIAL FAIL)")
            st.write(f"The model identifies this student as **academically at-risk**.")
            
        st.caption(f"Evaluated via: **{model_label}**")

    with col2:
        st.markdown("#### Confidence Probability")
        pass_pct = final_prob * 100
        risk_pct = (1.0 - final_prob) * 100
        
        st.metric(label="Passing Probability", value=f"{pass_pct:.1f}%")
        st.progress(float(final_prob))
        st.caption(f"Risk Index: **{risk_pct:.1f}%**")

    with col3:
        st.markdown("#### Multi-Model Breakdown")
        for m_name, m_prob in probs.items():
            status_emoji = "🟢" if m_prob >= 0.50 else "🔴"
            clean_name = m_name.replace(" (Recommended)", "")
            st.write(f"{status_emoji} **{clean_name}**: `{m_prob*100:.1f}%`")
            
    st.divider()

    # Actionable AI Recommendations
    st.subheader("💡 Tailored Academic Recommendations")
    recommendations = []
    
    if attendance < 75.0:
        recommendations.append(
            "🚨 **Urgent Attendance Warning:** Current attendance is below 75%. Prioritizing regular class attendance is the single highest-impact action."
        )
    elif attendance < 85.0:
        recommendations.append(
            "📈 **Boost Class Attendance:** Increasing attendance by just 5-10% will notably stabilize academic retention."
        )

    if study_hours < 10.0:
        recommendations.append(
            "📖 **Increase Study Time:** Weekly self-study is under 10 hours. Target 14-16 hours per week with dedicated focus blocks."
        )

    if past_score < 60.0 and tutoring < 2:
        recommendations.append(
            "🧑‍🏫 **Tutoring Recommended:** Previous score was below 60%. Enrolling in 2-3 tutoring sessions per month will clarify foundational concepts."
        )

    if stress_level == "High":
        recommendations.append(
            "🧘 **Manage Academic Stress:** High stress is reported. Incorporate mindfulness, short breaks, and academic counseling."
        )

    if sleep_hours < 6.0:
        recommendations.append(
            "💤 **Sleep Hygiene:** Less than 6 hours of sleep significantly degrades memory consolidation and exam performance. Aim for 7-8 hours."
        )

    if not recommendations:
        recommendations.append(
            "🌟 **Strong Momentum:** The student demonstrates healthy academic habits, good attendance, and solid past preparation. Keep maintaining this consistency!"
        )

    for rec in recommendations:
        st.info(rec)

# ---------------------------------------------------------
# TAB 2: Model Comparison & Evaluation
# ---------------------------------------------------------
with tab_compare:
    st.subheader("Model Performance Comparison (Held-Out Test Set)")
    st.markdown(
        """
        The dataset was split into **80% Training** and **20% Testing** (unseen).
        Below are the standard classification metrics computed on the test set:
        """
    )
    
    if metrics:
        # Build comparison DataFrame
        comparison_records = []
        for m_key, m_val in metrics.items():
            comparison_records.append({
                "Model": m_key,
                "Accuracy": f"{m_val['Accuracy']*100:.2f}%",
                "Precision": f"{m_val['Precision']*100:.2f}%",
                "Recall": f"{m_val['Recall']*100:.2f}%",
                "F1-Score": f"{m_val['F1_Score']:.4f}",
                "ROC-AUC": f"{m_val['ROC_AUC']:.4f}"
            })
        
        comp_df = pd.DataFrame(comparison_records)
        st.dataframe(comp_df, width="stretch", hide_index=True)
        
        # Chart comparison
        chart_data = pd.DataFrame({
            'Model': list(metrics.keys()),
            'Accuracy': [m['Accuracy'] * 100 for m in metrics.values()],
            'F1-Score (%)': [m['F1_Score'] * 100 for m in metrics.values()],
            'ROC-AUC (%)': [m['ROC_AUC'] * 100 for m in metrics.values()]
        }).set_index('Model')
        
        st.bar_chart(chart_data)
        
        # Confusion Matrices Section
        st.subheader("Confusion Matrix Analysis")
        cols = st.columns(len(metrics))
        for idx, (m_key, m_val) in enumerate(metrics.items()):
            with cols[idx]:
                st.markdown(f"**{m_key}**")
                cm = m_val['Confusion_Matrix']
                cm_df = pd.DataFrame(
                    cm,
                    columns=["Pred: Fail", "Pred: Pass"],
                    index=["Actual: Fail", "Actual: Pass"]
                )
                st.dataframe(cm_df)
                tn, fp = cm[0]
                fn, tp = cm[1]
                st.caption(f"Correct: **{tn + tp}** | Errors: **{fp + fn}**")
                
        # Educational explanation of metrics
        with st.expander("📖 Guide: Understanding ML Evaluation Metrics (For Beginners)"):
            st.markdown(
                """
                - **Accuracy**: Overall percentage of correct predictions (both Pass and Fail).
                - **Precision**: When the model predicts 'Pass', how often is it actually right?
                - **Recall**: Out of all students who actually passed, how many did the model identify?
                - **F1-Score**: The harmonic mean of Precision and Recall. Great for balanced assessment.
                - **ROC-AUC**: Measures how well the model separates Pass students from At-Risk students across all possible classification thresholds (1.0 is perfect).
                """
            )
    else:
        st.warning("Metrics file not found. Run `python src/evaluate.py` to populate.")

# ---------------------------------------------------------
# TAB 3: Feature Importance
# ---------------------------------------------------------
with tab_importance:
    st.subheader("Feature Importance: What Drives Student Success?")
    st.markdown(
        """
        Machine Learning algorithms identify which student factors have the highest influence on academic outcomes.
        """
    )
    
    rf_metrics = metrics.get('Random Forest', {})
    if 'Feature_Importance' in rf_metrics:
        feat_dict = rf_metrics['Feature_Importance']
        feat_df = pd.DataFrame(
            list(feat_dict.items()),
            columns=['Feature', 'Importance']
        ).sort_values(by='Importance', ascending=False)
        
        col_chart, col_explain = st.columns([1.6, 1.0])
        with col_chart:
            st.bar_chart(data=feat_df.set_index('Feature'), horizontal=True)
            
        with col_explain:
            st.markdown("#### Key Takeaways")
            st.markdown(
                """
                1. **Past Exam Score & Attendance**: Strongly drive outcomes. A student with high attendance and good past mastery rarely fails.
                2. **Study Hours**: Essential for bridging knowledge gaps.
                3. **Sleep & Stress**: Act as compounding multipliers on learning efficiency.
                4. **Support Systems**: Tutoring and internet access provide key safety nets for students facing challenges.
                """
            )
    else:
        st.info("Feature importance data is generated during model evaluation.")

# ---------------------------------------------------------
# TAB 4: Dataset Explorer
# ---------------------------------------------------------
with tab_data:
    st.subheader("Raw Student Performance Dataset")
    raw_df = load_dataset_sample()
    
    if raw_df is not None:
        st.markdown(f"Total Student Records: **{len(raw_df):,}** | Features: **{raw_df.shape[1]}**")
        
        # Summary metrics
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Overall Pass Rate", f"{(raw_df['Passed'].mean()*100):.1f}%")
        col_b.metric("Avg Attendance", f"{raw_df['Attendance_Rate'].mean():.1f}%")
        col_c.metric("Avg Past Score", f"{raw_df['Past_Exam_Score'].mean():.1f}")
        col_d.metric("Avg Study Hours/Wk", f"{raw_df['Study_Hours_Per_Week'].mean():.1f} hrs")
        
        st.dataframe(raw_df.head(25), width="stretch")
        
        st.markdown("#### Summary Statistics")
        st.dataframe(raw_df.describe().round(2), width="stretch")
    else:
        st.warning("Raw dataset not found. Run `python src/data_generator.py` to generate it.")
