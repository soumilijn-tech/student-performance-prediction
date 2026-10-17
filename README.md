# 🎓 Student Performance Prediction Model

A machine learning web application built from scratch to predict whether a student will pass or is at academic risk, using **Python**, **Pandas**, **NumPy**, **Scikit-learn**, and **Streamlit**.

---

## 📌 Project Overview

This project simulates a complete end-to-end Machine Learning data science lifecycle:
1. **Data Generation**: Generates 1,200 realistic student academic and demographic records.
2. **Data Preprocessing**: Cleans missing data, encodes categorical features, and standardizes numerical values.
3. **Model Training**: Trains and compares three popular machine learning algorithms:
   - **Logistic Regression**: Linear probability baseline classifier.
   - **Decision Tree Classifier**: Interpretable rule-based classifier.
   - **Random Forest Classifier**: High-performance ensemble classifier.
4. **Model Evaluation**: Evaluates models on an unseen test set using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
5. **Interactive Web Dashboard**: A Streamlit web application allowing educators and students to test real-time predictions, explore what-if scenarios, and view actionable recommendations.

---

## 📂 Project Structure

```text
Student-Prediction-Model/
│
├── .streamlit/
│   └── config.toml                   # Streamlit cloud theme and server configuration
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
1. Create a free account at [github.com](https://github.com) if you do not have one.
2. In GitHub, click **New repository** (name it `Student-Prediction-Model`), set it to **Public**, and leave "Initialize with README" unchecked.
3. In your local terminal, navigate inside your project directory and run:
   ```bash
   git init
   git add .
   git commit -m "Deploy Student Performance Prediction Model"
   git branch -M main
   git remote add origin https://github.com/<YOUR-GITHUB-USERNAME>/Student-Prediction-Model.git
   git push -u origin main
   ```
   *(Replace `<YOUR-GITHUB-USERNAME>` with your actual GitHub username).*

### Step 2: Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **Create app** (or **New app**).
3. Fill in the deployment details:
   - **Repository**: `<YOUR-GITHUB-USERNAME>/Student-Prediction-Model`
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
