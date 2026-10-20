# 🎓 Student Performance Prediction

> **Smarter Insights for Brighter Futures**

A professional, SaaS-inspired machine learning web application to predict whether a student will pass or is at academic risk, using **Python**, **Pandas**, **NumPy**, **Scikit-learn**, and **Streamlit**.

---

## 📌 Project Overview

This project provides an enterprise-grade academic risk detection and advisory platform:
1. **Custom Branding**: Dedicated education + AI vector logo, deep navy sidebar, and gradient hero headers.
2. **Navigation Sections**:
   - **Home**: Executive overview, KPI metric counters, operational workflow.
   - **Student Prediction**: Comprehensive input form with validation, real-time risk assessment, passing probabilities, multi-model consensus, and tailored guidance.
   - **Model Performance**: Genuine held-out test benchmarks, interactive metric comparison charts, confusion matrix diagnostics, and feature importance rankings.
   - **Visual Insights**: Exploratory cohort visualizations (Attendance vs. Pass Rate, Study Hours vs. Scores, Stress & Guardian influence, and interactive data explorer).
   - **About**: Architecture, tech stack, and ethical AI in education.

---

## 📂 Project Structure

```text
Student-Prediction-Model/
│
├── .streamlit/
│   └── config.toml                   # Streamlit cloud theme and server configuration
│
├── assets/
│   └── logo.png                      # Custom AI education & prediction branding logo
│
├── data/
│   ├── raw/
│   │   └── student_data.csv          # 1,200 raw student records
│   └── processed/
│       ├── X_train.csv               # Scaled training features (80%)
│       ├── X_test.csv                # Scaled testing features (20%)
│       ├── y_train.csv               # Training labels (Pass=1, Fail=0)
│       └── y_test.csv                # Testing labels
│
├── models/
│   ├── logistic_regression.pkl       # Saved Logistic Regression model
│   ├── decision_tree.pkl             # Saved Decision Tree model
│   ├── random_forest.pkl             # Saved Random Forest model
│   ├── preprocessor.pkl              # Saved scaler and feature encoder bundle
│   └── model_metrics.json            # Model performance comparison metrics
│
├── src/
│   ├── __init__.py                   # Package initialization
│   ├── data_generator.py             # Generates realistic student dataset
│   ├── preprocess.py                 # Feature encoding, scaling, and splitting
│   ├── train.py                      # Trains and saves all 3 models
│   └── evaluate.py                   # Evaluates models and calculates metrics
│
├── app.py                            # Streamlit entry point (Web application)
├── requirements.txt                  # Python dependencies for local & cloud deployment
├── .gitignore                        # Git ignore rules for clean repository
└── README.md                         # Project documentation and guide
```

---

## 🚀 How to Run Locally

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. (Optional) Re-run the Machine Learning Pipeline
*Note: Trained models and dataset are already pre-generated and included! If you ever want to re-train from scratch:*
```bash
python src/data_generator.py
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

### 3. Launch the Streamlit Web Dashboard
```bash
streamlit run app.py
```
Your web browser will automatically open at `http://localhost:8501`.

---

## ☁️ How to Deploy on Streamlit Community Cloud (Step-by-Step)

Streamlit Community Cloud lets you host your app online for free in just a few clicks!

### Step 1: Upload Project to GitHub
1. Create a repository on GitHub named `student-performance-prediction` at [github.com/new](https://github.com/new) (keep it Public, uncheck all template boxes).
2. Push your project to GitHub:
   ```bash
   git remote add origin https://github.com/soumilijn-tech/student-performance-prediction.git
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **Create app** (or **New app**).
3. Fill in the deployment details:
   - **Repository**: `soumilijn-tech/student-performance-prediction`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy!**

### Step 3: View Your Live App
Streamlit Cloud will read `requirements.txt`, install dependencies, and provide you with a permanent public link (e.g., `https://student-prediction-model.streamlit.app`) that you can share with anyone!

---

## 🧠 Machine Learning Concepts Explained Simply

- **Features (Inputs)**: Information about each student:
  - `Attendance Rate`: How regularly the student attends lectures.
  - `Study Hours Per Week`: Dedicated independent study time.
  - `Past Exam Score`: Prior foundation and academic history.
  - `Sleep Hours` & `Stress Level`: Rest and mental well-being factors.
  - `Tutoring Sessions`: Extra academic assistance received.
- **Target (Output)**: `Passed` (1 = Pass / Low Risk, 0 = At-Risk / Fail).
- **Train vs. Test Split**: We train the models on 80% of students and evaluate them on 20% unseen students to verify real learning vs memorization.
- **Accuracy**: Percentage of correct predictions overall.
- **Recall**: Measures how well the model avoids missing students who need academic intervention.
- **F1-Score**: The harmonic mean of Precision and Recall.
