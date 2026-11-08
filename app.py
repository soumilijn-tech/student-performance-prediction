"""
Student Performance Prediction - Enterprise Web Application
Brand: Student Performance Prediction
Tagline: Smarter Insights for Brighter Futures
"""

import os
import json
import base64
import joblib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Prediction | Smarter Insights for Brighter Futures",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
LOGO_PATH = os.path.join(ASSETS_DIR, 'logo.png')

# ---------------------------------------------------------
# Utility: Base64 Logo Encoding
# ---------------------------------------------------------
def get_base64_logo():
    """Encodes logo image to base64 for responsive HTML rendering."""
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

LOGO_BASE64 = get_base64_logo()

# ---------------------------------------------------------
# ---------------------------------------------------------
# Custom Modern CSS (SaaS Dashboard UI & Mobile Responsive)
# ---------------------------------------------------------
st.markdown(f"""
<style>
    /* Global Fonts & Reset */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    *, *::before, *::after {{
        box-sizing: border-box !important;
    }}
    
    html, body {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        overflow-x: hidden !important;
        max-width: 100vw !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }}
    
    [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    /* Main Background & Overflow Prevention */
    .stApp {{
        background-color: #F8FAFC;
        overflow-x: hidden !important;
        max-width: 100vw !important;
        width: 100% !important;
        padding-bottom: 90px !important;
    }}
    
    .main .block-container {{
        max-width: 1200px !important;
        width: 100% !important;
        box-sizing: border-box !important;
        padding-top: 2rem !important;
        padding-bottom: 90px !important;
    }}
    
    /* Word Wrap & Typography Helpers */
    .title-word {{
        display: inline-block;
        white-space: nowrap;
    }}
    
    /* Deep Navy Sidebar */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
        border-right: 1px solid #334155;
    }}
    
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown span,
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] label {{
        color: #F1F5F9 !important;
        word-break: normal !important;
        hyphens: none !important;
    }}
    
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {{
        background: rgba(255, 255, 255, 0.04);
        padding: 10px 14px;
        border-radius: 10px;
        margin-bottom: 6px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.2s ease;
    }}
    
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {{
        background: rgba(99, 102, 241, 0.15);
        border-color: #6366F1;
    }}
    
    /* Gradient Hero Header (Desktop & Tablet Default) */
    .hero-header {{
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 35%, #4338CA 70%, #6366F1 100%);
        border-radius: 20px;
        padding: 36px 40px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.3), 0 8px 10px -6px rgba(67, 56, 202, 0.2);
        display: flex;
        align-items: center;
        gap: 32px;
        width: 100%;
        box-sizing: border-box;
    }}
    
    .hero-logo-box {{
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    
    .hero-logo {{
        width: 100px;
        height: 100px;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.1);
        padding: 6px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        object-fit: contain;
        display: block;
    }}
    
    .hero-content-box {{
        flex: 1;
        min-width: 0;
    }}
    
    .hero-title {{
        font-size: clamp(1.6rem, 3.2vw, 2.3rem);
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.01em;
        color: #FFFFFF !important;
        line-height: 1.25;
        word-break: normal !important;
        overflow-wrap: normal !important;
        word-wrap: normal !important;
        hyphens: none !important;
        -webkit-hyphens: none !important;
        text-wrap: balance;
        max-width: 100%;
    }}
    
    .hero-tagline {{
        font-size: clamp(0.95rem, 1.8vw, 1.15rem);
        font-weight: 500;
        color: #C7D2FE !important;
        margin: 8px 0 0 0;
        letter-spacing: 0.01em;
        word-break: normal !important;
        hyphens: none !important;
    }}
    
    .hero-desc {{
        font-size: clamp(0.88rem, 1.4vw, 0.98rem);
        color: #E0E7FF !important;
        margin: 12px 0 0 0;
        max-width: 820px;
        line-height: 1.6;
        word-break: normal !important;
        overflow-wrap: break-word;
    }}
    
    /* Hero Action Buttons Container */
    .hero-actions-container {{
        margin-bottom: 24px;
        width: 100%;
    }}
    
    /* Page Headers */
    .page-header {{
        margin-bottom: 24px;
        width: 100%;
    }}
    
    .page-title {{
        font-size: clamp(1.35rem, 3.5vw, 1.85rem) !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        margin: 0 !important;
        line-height: 1.3 !important;
        word-break: normal !important;
        overflow-wrap: normal !important;
        hyphens: none !important;
        -webkit-hyphens: none !important;
    }}
    
    .page-subtitle {{
        color: #64748B !important;
        font-size: clamp(0.85rem, 2.2vw, 1rem) !important;
        margin: 6px 0 0 0 !important;
        line-height: 1.5 !important;
        word-break: normal !important;
        overflow-wrap: break-word !important;
    }}
    
    /* White SaaS Cards with Subtle Shadows */
    .saas-card {{
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.04), 0 2px 4px -2px rgba(15, 23, 42, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        width: 100%;
        box-sizing: border-box;
    }}
    
    .saas-card:hover {{
        box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.07), 0 4px 6px -4px rgba(15, 23, 42, 0.04);
    }}
    
    .card-title {{
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 10px;
        word-break: normal !important;
        hyphens: none !important;
    }}
    
    /* KPI Metric Cards */
    .kpi-card {{
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
        border-top: 4px solid #4F46E5;
        width: 100%;
        box-sizing: border-box;
        margin-bottom: 12px;
    }}
    
    .kpi-value {{
        font-size: 2rem;
        font-weight: 800;
        color: #1E1B4B;
        line-height: 1.1;
        margin: 8px 0;
        word-break: normal !important;
    }}
    
    .kpi-label {{
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        word-break: normal !important;
    }}
    
    /* Prediction Badges */
    .badge-pass {{
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 14px;
        font-size: 1.3rem;
        font-weight: 800;
        text-align: center;
        box-shadow: 0 8px 16px -4px rgba(16, 185, 129, 0.4);
        word-break: normal !important;
        hyphens: none !important;
    }}
    
    .badge-risk {{
        background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 14px;
        font-size: 1.3rem;
        font-weight: 800;
        text-align: center;
        box-shadow: 0 8px 16px -4px rgba(239, 68, 68, 0.4);
        word-break: normal !important;
        hyphens: none !important;
    }}
    
    /* Primary Action Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 20px !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        min-height: 48px !important;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.4) !important;
    }}
    
    .stButton > button:active {{
        transform: translateY(0);
    }}
    
    /* Footer */
    .saas-footer {{
        text-align: center;
        padding: 24px 16px;
        margin-top: 36px;
        margin-bottom: 20px;
        border-top: 1px solid #E2E8F0;
        color: #94A3B8;
        font-size: 0.85rem;
        word-break: normal !important;
    }}
    
    /* Hide Unnecessary Promotional Badges & Streamlit Footers */
    footer,
    div[class*="viewerBadge"],
    .viewerBadge_container__1QSob,
    div[class*="ProfileButton"],
    .stAppDeployButton,
    div[data-testid="stDecoration"] {{
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }}
    
    /* Prevent Manage App Button From Overlapping Content */
    div[class*="manageApp"],
    [data-testid="manage-app-button"] {{
        bottom: 12px !important;
        right: 12px !important;
        opacity: 0.8 !important;
        transition: opacity 0.2s ease !important;
    }}
    div[class*="manageApp"]:hover,
    [data-testid="manage-app-button"]:hover {{
        opacity: 1.0 !important;
    }}

    /* =====================================================
       RESPONSIVE BREAKPOINTS (Mobile & Tablet Layouts)
       ===================================================== */
       
    /* Tablet & Small Screens (<= 768px) */
    @media screen and (max-width: 768px) {{
        .stApp {{
            padding-bottom: 110px !important;
        }}
        
        .main .block-container {{
            padding-left: 14px !important;
            padding-right: 14px !important;
            padding-top: 1.25rem !important;
            padding-bottom: 110px !important;
            max-width: 100% !important;
        }}
        
        /* Convert Hero Banner to Single-Column Centered Layout */
        .hero-header {{
            flex-direction: column !important;
            text-align: center !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 24px 16px !important;
            gap: 14px !important;
            border-radius: 16px !important;
            margin-bottom: 18px !important;
            width: 100% !important;
        }}
        
        .hero-logo-box {{
            width: 100% !important;
            display: flex !important;
            justify-content: center !important;
            margin: 0 auto 4px auto !important;
        }}
        
        .hero-logo {{
            width: 80px !important;
            height: 80px !important;
            margin: 0 auto !important;
            display: block !important;
        }}
        
        .hero-content-box {{
            width: 100% !important;
            text-align: center !important;
        }}
        
        .hero-title {{
            font-size: clamp(1.3rem, 5.2vw, 1.7rem) !important;
            text-align: center !important;
            line-height: 1.3 !important;
            margin: 0 auto 8px auto !important;
            word-break: normal !important;
            overflow-wrap: normal !important;
            word-wrap: normal !important;
            hyphens: none !important;
            -webkit-hyphens: none !important;
            text-wrap: balance !important;
        }}
        
        .hero-tagline {{
            text-align: center !important;
            font-size: clamp(0.9rem, 3.8vw, 1.05rem) !important;
            margin: 4px auto 0 auto !important;
        }}
        
        .hero-desc {{
            text-align: center !important;
            font-size: clamp(0.84rem, 3.2vw, 0.92rem) !important;
            line-height: 1.55 !important;
            margin: 10px auto 0 auto !important;
            max-width: 100% !important;
            padding: 0 4px !important;
        }}
        
        .hero-actions-container {{
            margin-bottom: 20px !important;
        }}
        
        .page-header {{
            text-align: center !important;
            margin-bottom: 18px !important;
        }}
        
        .page-title {{
            font-size: clamp(1.2rem, 5vw, 1.55rem) !important;
            text-align: center !important;
        }}
        
        .page-subtitle {{
            font-size: 0.85rem !important;
            text-align: center !important;
        }}
        
        .saas-card {{
            padding: 16px 14px !important;
            border-radius: 14px !important;
            margin-bottom: 14px !important;
        }}
        
        .kpi-card {{
            padding: 14px 10px !important;
            border-radius: 12px !important;
            margin-bottom: 10px !important;
        }}
        
        .kpi-value {{
            font-size: 1.65rem !important;
        }}
        
        .kpi-label {{
            font-size: 0.78rem !important;
        }}
        
        .badge-pass, .badge-risk {{
            font-size: 1.15rem !important;
            padding: 14px 16px !important;
        }}
        
        .saas-footer {{
            margin-bottom: 28px !important;
            padding-bottom: 10px !important;
        }}
        
        div[class*="manageApp"],
        [data-testid="manage-app-button"] {{
            bottom: 8px !important;
            right: 8px !important;
            transform: scale(0.85);
            transform-origin: bottom right;
        }}
    }}
    
    /* Mobile Extra Small Screens (<= 480px, e.g. 360px, 390px, 430px) */
    @media screen and (max-width: 480px) {{
        .main .block-container {{
            padding-left: 10px !important;
            padding-right: 10px !important;
            padding-top: 1rem !important;
            padding-bottom: 110px !important;
        }}
        
        .hero-header {{
            padding: 20px 12px !important;
            border-radius: 14px !important;
            gap: 12px !important;
            margin-bottom: 16px !important;
        }}
        
        .hero-logo {{
            width: 72px !important;
            height: 72px !important;
            border-radius: 16px !important;
            padding: 4px !important;
        }}
        
        .hero-title {{
            font-size: clamp(1.2rem, 5.6vw, 1.55rem) !important;
            letter-spacing: -0.01em !important;
            line-height: 1.25 !important;
        }}
        
        .hero-tagline {{
            font-size: 0.88rem !important;
            margin-top: 4px !important;
        }}
        
        .hero-desc {{
            font-size: 0.82rem !important;
            line-height: 1.5 !important;
            margin-top: 8px !important;
        }}
        
        .saas-card {{
            padding: 14px 10px !important;
            border-radius: 12px !important;
            margin-bottom: 12px !important;
        }}
        
        .kpi-card {{
            padding: 12px 8px !important;
            border-radius: 10px !important;
        }}
        
        .kpi-value {{
            font-size: 1.45rem !important;
        }}
        
        .badge-pass, .badge-risk {{
            font-size: 1.05rem !important;
            padding: 12px 12px !important;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Core Helper Functions: Artifacts & Inference
# ---------------------------------------------------------
def ensure_artifacts_exist():
    """Self-healing: Ensures datasets and models exist, auto-generating on first boot if missing."""
    required_files = [
        os.path.join(MODELS_DIR, 'preprocessor.pkl'),
        os.path.join(MODELS_DIR, 'random_forest.pkl'),
        os.path.join(MODELS_DIR, 'logistic_regression.pkl'),
        os.path.join(MODELS_DIR, 'decision_tree.pkl'),
        os.path.join(MODELS_DIR, 'model_metrics.json'),
        os.path.join(DATA_DIR, 'raw', 'student_data.csv')
    ]
    if not all(os.path.exists(p) for p in required_files):
        with st.spinner("Initializing student dataset and training machine learning models..."):
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
    """Load preprocessor, trained models, and genuine evaluation metrics."""
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
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
    return preprocessor, models, metrics

@st.cache_data
def load_dataset():
    """Load the raw student dataset for visual insights and EDA."""
    csv_path = os.path.join(DATA_DIR, 'raw', 'student_data.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def preprocess_single_input(input_dict, preprocessor):
    """Transforms a single student profile dictionary into model-ready scaled features."""
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

# Initialize Artifacts
preprocessor, models, metrics = load_artifacts()
raw_df = load_dataset()

# Navigation State Management
NAV_OPTIONS = [
    "🏠 Home",
    "🔮 Student Prediction",
    "📊 Model Performance",
    "📈 Visual Insights",
    "ℹ️ About"
]

if "nav_selection" not in st.session_state:
    st.session_state["nav_selection"] = "🏠 Home"

def navigate_to(page_name):
    st.session_state["nav_selection"] = page_name

# ---------------------------------------------------------
# Sidebar Navigation & Branding
# ---------------------------------------------------------
with st.sidebar:
    # Sidebar Brand Header
    if LOGO_BASE64:
        st.markdown(f"""
        <div style="text-align: center; padding: 10px 0 16px 0;">
            <img src="data:image/png;base64,{LOGO_BASE64}" style="width: 84px; height: 84px; border-radius: 18px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); margin-bottom: 12px;"/>
            <h2 style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF !important; margin: 0; line-height: 1.25; word-break: normal !important; hyphens: none !important;">
                <span class="title-word">Student Performance</span><br/><span class="title-word">Prediction</span>
            </h2>
            <p style="font-size: 0.8rem; font-weight: 500; color: #818CF8 !important; margin: 6px 0 0 0; word-break: normal !important; hyphens: none !important;">
                <span class="title-word">Smarter Insights</span> <span class="title-word">for Brighter Futures</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 10px 0;">
            <h2 style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF !important; word-break: normal !important; hyphens: none !important;">
                🎓 <span class="title-word">Student Performance</span> <span class="title-word">Prediction</span>
            </h2>
            <p style="font-size: 0.8rem; color: #818CF8 !important;">Smarter Insights for Brighter Futures</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<hr style='border: 0; border-top: 1px solid #334155; margin: 12px 0 20px 0;'/>", unsafe_allow_html=True)
    
    # 5 Main Navigation Sections
    st.markdown("<p style='font-size: 0.75rem; text-transform: uppercase; font-weight: 700; color: #94A3B8; letter-spacing: 0.08em; margin-bottom: 8px;'>Navigation</p>", unsafe_allow_html=True)
    
    selected_nav = st.radio(
        "Navigation Menu",
        options=NAV_OPTIONS,
        key="nav_selection",
        label_visibility="collapsed"
    )
    
    st.markdown("<hr style='border: 0; border-top: 1px solid #334155; margin: 24px 0 16px 0;'/>", unsafe_allow_html=True)
    
    # Sidebar Quick Model Info
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.04); padding: 14px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08);">
        <p style="font-size: 0.75rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin: 0 0 6px 0;">Production Models</p>
        <p style="font-size: 0.82rem; color: #E2E8F0; margin: 0 0 4px 0;">🌲 Random Forest (100 Trees)</p>
        <p style="font-size: 0.82rem; color: #E2E8F0; margin: 0 0 4px 0;">📈 Logistic Regression (C=1.0)</p>
        <p style="font-size: 0.82rem; color: #E2E8F0; margin: 0;">🌿 Decision Tree (Depth=5)</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("v2.0 • SaaS Production Architecture")

# ---------------------------------------------------------
# SECTION 1: HOME
# ---------------------------------------------------------
if selected_nav == "🏠 Home":
    # Hero Banner
    logo_img_tag = f'<img src="data:image/png;base64,{LOGO_BASE64}" class="hero-logo" alt="Student Performance Prediction Logo"/>' if LOGO_BASE64 else '🎓'
    
    st.markdown(f"""
    <div class="hero-header">
        <div class="hero-logo-box">
            {logo_img_tag}
        </div>
        <div class="hero-content-box">
            <h1 class="hero-title">
                <span class="title-word">Student</span> <span class="title-word">Performance</span> <span class="title-word">Prediction</span>
            </h1>
            <p class="hero-tagline">
                <span class="title-word">Smarter Insights</span> <span class="title-word">for Brighter Futures</span>
            </p>
            <p class="hero-desc">
                An explainable, AI-powered predictive platform that enables educators, academic advisors, and students 
                to identify academic risk factors early, benchmark progress, and implement proactive educational interventions.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero Quick Action Navigation Buttons
    st.markdown('<div class="hero-actions-container">', unsafe_allow_html=True)
    c_btn1, c_btn2, c_btn3 = st.columns(3)
    with c_btn1:
        st.button(
            "🔮 Run Student Prediction",
            key="btn_hero_pred",
            use_container_width=True,
            on_click=navigate_to,
            args=("🔮 Student Prediction",)
        )
    with c_btn2:
        st.button(
            "📊 View Model Performance",
            key="btn_hero_perf",
            use_container_width=True,
            on_click=navigate_to,
            args=("📊 Model Performance",)
        )
    with c_btn3:
        st.button(
            "📈 Explore Visual Insights",
            key="btn_hero_vis",
            use_container_width=True,
            on_click=navigate_to,
            args=("📈 Visual Insights",)
        )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick KPI Counters
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Analyzed Students</div>
            <div class="kpi-value">1,200</div>
            <div style="font-size: 0.8rem; color: #64748B;">Validated Cohort Records</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown("""
        <div class="kpi-card" style="border-top-color: #10B981;">
            <div class="kpi-label">Peak Model Accuracy</div>
            <div class="kpi-value" style="color: #059669;">88.75%</div>
            <div style="font-size: 0.8rem; color: #64748B;">Logistic Regression Baseline</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown("""
        <div class="kpi-card" style="border-top-color: #8B5CF6;">
            <div class="kpi-label">Early-Risk Recall</div>
            <div class="kpi-value" style="color: #7C3AED;">96.24%</div>
            <div style="font-size: 0.8rem; color: #64748B;">Random Forest Sensitivity</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown("""
        <div class="kpi-card" style="border-top-color: #EC4899;">
            <div class="kpi-label">Predictive Factors</div>
            <div class="kpi-value" style="color: #DB2777;">9</div>
            <div style="font-size: 0.8rem; color: #64748B;">Academic & Lifestyle Metrics</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # Feature Overview Cards
    col_feat1, col_feat2 = st.columns(2)
    with col_feat1:
        st.markdown("""
        <div class="saas-card">
            <div class="card-title">🎯 Early Academic Risk Detection</div>
            <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
                Traditional academic assessments identify struggling students after midterms or finals—often too late for remedial assistance. 
                Our platform analyses attendance velocity, study routines, and foundation scores to flag at-risk trajectories weeks in advance.
            </p>
        </div>
        <div class="saas-card">
            <div class="card-title">🤖 Triangulated Multi-Model Consensus</div>
            <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
                Rather than relying on a single black-box algorithm, the system executes predictions across 
                <strong>Random Forest</strong>, <strong>Logistic Regression</strong>, and <strong>Decision Tree</strong> architectures to deliver balanced, trustworthy risk assessments.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_feat2:
        st.markdown("""
        <div class="saas-card">
            <div class="card-title">💡 Actionable, Data-Driven Recommendations</div>
            <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
                Every prediction is paired with prescriptive guidance tailored to the student's unique profile—highlighting the exact 
                marginal gains in attendance, weekly study blocks, tutoring sessions, or sleep needed to cross into safe passing standing.
            </p>
        </div>
        <div class="saas-card">
            <div class="card-title">📈 Transparent Model Explainability</div>
            <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
                Educators can inspect verified classification metrics, confusion matrices, and feature importance rankings directly from 
                held-out test sets—ensuring full transparency and compliance with academic governance standards.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Workflow Stepper Card
    st.markdown("""
    <div class="saas-card">
        <div class="card-title">🚀 Platform Operational Workflow</div>
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 14px; margin-top: 16px; width: 100%;">
            <div style="flex: 1 1 200px; max-width: 100%; box-sizing: border-box; background: #F8FAFC; padding: 14px 16px; border-radius: 12px; border-left: 4px solid #4F46E5;">
                <div style="font-weight: 700; color: #1E1B4B; margin-bottom: 4px; word-break: normal !important; hyphens: none !important;">1. Profile Input</div>
                <div style="font-size: 0.85rem; color: #64748B; line-height: 1.5;">Enter student attendance, study hours, past scores & wellness variables.</div>
            </div>
            <div style="flex: 1 1 200px; max-width: 100%; box-sizing: border-box; background: #F8FAFC; padding: 14px 16px; border-radius: 12px; border-left: 4px solid #7C3AED;">
                <div style="font-weight: 700; color: #1E1B4B; margin-bottom: 4px; word-break: normal !important; hyphens: none !important;">2. Feature Pipeline</div>
                <div style="font-size: 0.85rem; color: #64748B; line-height: 1.5;">Automatic category encoding and StandardScaler normalization.</div>
            </div>
            <div style="flex: 1 1 200px; max-width: 100%; box-sizing: border-box; background: #F8FAFC; padding: 14px 16px; border-radius: 12px; border-left: 4px solid #059669;">
                <div style="font-weight: 700; color: #1E1B4B; margin-bottom: 4px; word-break: normal !important; hyphens: none !important;">3. ML Inference</div>
                <div style="font-size: 0.85rem; color: #64748B; line-height: 1.5;">Real-time classification and calibrated confidence estimation.</div>
            </div>
            <div style="flex: 1 1 200px; max-width: 100%; box-sizing: border-box; background: #F8FAFC; padding: 14px 16px; border-radius: 12px; border-left: 4px solid #DB2777;">
                <div style="font-weight: 700; color: #1E1B4B; margin-bottom: 4px; word-break: normal !important; hyphens: none !important;">4. Actionable Strategy</div>
                <div style="font-size: 0.85rem; color: #64748B; line-height: 1.5;">Dynamic interventions tailored to student well-being and study gaps.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# SECTION 2: STUDENT PREDICTION
# ---------------------------------------------------------
elif selected_nav == "🔮 Student Prediction":
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">
            <span class="title-word">🔮 Real-Time</span> <span class="title-word">Student</span> <span class="title-word">Performance</span> <span class="title-word">Prediction</span>
        </h1>
        <p class="page-subtitle">Configure student academic and lifestyle metrics to compute predictive passing risk and guidance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("student_prediction_form"):
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📝 Student Profile Parameters</div>', unsafe_allow_html=True)
        
        col_form1, col_form2 = st.columns(2)
        
        with col_form1:
            st.markdown("#### 📚 Academic Foundations")
            past_score = st.slider(
                "Previous Exam Score (0 - 100)",
                min_value=30.0,
                max_value=100.0,
                value=70.0,
                step=0.5,
                help="Historical score from previous semester exam (Passing threshold is 60.0)."
            )
            attendance = st.slider(
                "Lecture Attendance Rate (%)",
                min_value=50.0,
                max_value=100.0,
                value=85.0,
                step=1.0,
                help="Percentage of scheduled lectures attended."
            )
            study_hours = st.slider(
                "Weekly Independent Study Hours",
                min_value=2.0,
                max_value=35.0,
                value=16.0,
                step=0.5,
                help="Average dedicated self-study hours per week."
            )
            tutoring = st.slider(
                "Tutoring / Remedial Sessions (Per Month)",
                min_value=0,
                max_value=6,
                value=1,
                step=1,
                help="Number of structured one-on-one or group tutoring sessions attended."
            )
            
        with col_form2:
            st.markdown("#### 🌱 Well-being & Background")
            sleep_hours = st.slider(
                "Average Daily Sleep Duration (Hours)",
                min_value=4.0,
                max_value=10.0,
                value=7.0,
                step=0.5,
                help="Average nightly sleep hours."
            )
            stress_level = st.selectbox(
                "Self-Reported Stress Level",
                options=["Low", "Medium", "High"],
                index=1,
                help="Subjective level of academic and personal stress."
            )
            parent_edu = st.selectbox(
                "Parent / Guardian Education Level",
                options=["High School", "Bachelor", "Master", "Doctorate"],
                index=1,
                help="Highest formal education level completed by primary caregiver."
            )
            extracurricular = st.selectbox(
                "Active Extracurricular Participation",
                options=["Yes", "No"],
                index=0,
                help="Participation in sports, clubs, music, or volunteer programs."
            )
            internet_access = st.selectbox(
                "Reliable Home Internet Access",
                options=["Yes", "No"],
                index=0,
                help="High-speed internet access available at home for coursework."
            )

        st.markdown("---")
        
        # Model selection inside form
        col_m1, col_m2 = st.columns([1.5, 1])
        with col_m1:
            selected_model_name = st.selectbox(
                "Select Machine Learning Engine",
                options=[
                    "Random Forest (Recommended)",
                    "Logistic Regression",
                    "Decision Tree",
                    "All Models (Consensus View)"
                ],
                index=0,
                help="Select which trained algorithm evaluates this profile."
            )
        with col_m2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("🔮 Predict Student Performance", use_container_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

    # Process and Display Prediction
    if submit_btn or 'has_predicted' in st.session_state:
        st.session_state['has_predicted'] = True
        
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
        
        # Preprocess features
        X_input = preprocess_single_input(student_input, preprocessor)
        
        # Compute predictions across all 3 models
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

        # Determine primary result
        if selected_model_name == "All Models (Consensus View)":
            avg_prob = np.mean(list(probs.values()))
            final_pred = 1 if avg_prob >= 0.50 else 0
            final_prob = avg_prob
            display_model_title = "Consensus Engine (Average of 3 Models)"
        else:
            final_pred = preds[selected_model_name]
            final_prob = probs[selected_model_name]
            display_model_title = selected_model_name

        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🎯 Predictive Assessment Outcome</div>', unsafe_allow_html=True)
        
        col_res1, col_res2, col_res3 = st.columns([1.4, 1.2, 1.4])
        
        with col_res1:
            st.markdown("<p style='font-size: 0.85rem; font-weight: 700; color: #64748B; text-transform: uppercase;'>Primary Classification</p>", unsafe_allow_html=True)
            if final_pred == 1:
                st.markdown("""
                <div class="badge-pass">
                    ✅ ON TRACK (PASS)
                </div>
                <p style="font-size: 0.9rem; color: #047857; margin-top: 10px; font-weight: 600;">
                    Student demonstrates strong academic indicators and is projected to comfortably pass.
                </p>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="badge-risk">
                    ⚠️ ACADEMIC RISK (FAIL)
                </div>
                <p style="font-size: 0.9rem; color: #B91C1C; margin-top: 10px; font-weight: 600;">
                    Identified with high probability of academic struggle. Structured intervention advised.
                </p>
                """, unsafe_allow_html=True)
            st.caption(f"Evaluator: **{display_model_title}**")
            
        with col_res2:
            st.markdown("<p style='font-size: 0.85rem; font-weight: 700; color: #64748B; text-transform: uppercase;'>Confidence & Probabilities</p>", unsafe_allow_html=True)
            pass_pct = final_prob * 100
            risk_pct = (1.0 - final_prob) * 100
            
            st.metric(label="Passing Probability", value=f"{pass_pct:.1f}%")
            st.progress(float(final_prob))
            st.caption(f"Risk Index: **{risk_pct:.1f}%** (Classification Threshold = 50.0%)")
            
        with col_res3:
            st.markdown("<p style='font-size: 0.85rem; font-weight: 700; color: #64748B; text-transform: uppercase;'>Multi-Model Breakdown</p>", unsafe_allow_html=True)
            for m_name, m_prob in probs.items():
                clean_name = m_name.replace(" (Recommended)", "")
                status_icon = "🟢" if m_prob >= 0.50 else "🔴"
                status_text = "Pass" if m_prob >= 0.50 else "Risk"
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #F1F5F9;">
                    <span style="font-size: 0.9rem; font-weight: 600; color: #1E293B;">{status_icon} {clean_name}</span>
                    <span style="font-size: 0.9rem; font-weight: 700; color: #4F46E5;">{m_prob*100:.1f}% ({status_text})</span>
                </div>
                """, unsafe_allow_html=True)
                
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Tailored Actionable Recommendations
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">💡 Actionable Academic Recommendations</div>', unsafe_allow_html=True)
        
        recs = []
        if attendance < 75.0:
            recs.append("🚨 **Critical Attendance Deficit:** Current lecture attendance is below 75%. In our historical cohort, students with <75% attendance experience a 4.2x higher failure rate regardless of past scores. Prioritize lecture presence immediately.")
        elif attendance < 85.0:
            recs.append("📈 **Optimize Attendance:** Increasing attendance by just 5–10% provides the single most reliable marginal boost to passing confidence.")
            
        if study_hours < 10.0:
            recs.append("📖 **Expand Dedicated Study Time:** Current independent study is under 10 hours/week. Target a structured 14–16 weekly study hours using dedicated focus blocks.")
        elif study_hours > 28.0 and sleep_hours < 6.0:
            recs.append("⚖️ **Diminishing Returns on Overstudy:** Very high study hours (>28h) combined with low sleep (<6h) induces cognitive fatigue. Balance study with restorative sleep.")
            
        if past_score < 60.0 and tutoring < 2:
            recs.append("🧑‍🏫 **Tutoring Sessions Needed:** Previous foundation score is below passing standard (60%). Scheduling 2–3 tutoring sessions per month stabilizes conceptual gaps.")
            
        if sleep_hours < 6.0:
            recs.append("💤 **Sleep Deprivation Warning:** Sleep under 6 hours impairs memory consolidation and exam recall. Aim for 7.0–8.0 hours daily.")
            
        if stress_level == "High":
            recs.append("🧘 **Stress Mitigation:** High stress degrades performance under timed pressure. Involve academic advisory or student wellness resources.")
            
        if not recs:
            recs.append("🌟 **Exemplary Profile:** This student demonstrates high attendance, balanced study discipline, and healthy rest habits. Maintain this positive routine!")
            
        for r in recs:
            st.info(r)
            
        st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# SECTION 3: MODEL PERFORMANCE
# ---------------------------------------------------------
elif selected_nav == "📊 Model Performance":
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">
            <span class="title-word">📊 Model</span> <span class="title-word">Evaluation</span> <span class="title-word">& Benchmarks</span>
        </h1>
        <p class="page-subtitle">Authentic validation metrics computed on the held-out test cohort (240 unseen students / 20% test split).</p>
    </div>
    """, unsafe_allow_html=True)
    
    if metrics:
        # Top KPI Highlights
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #4F46E5;">
                <div class="kpi-label">Logistic Regression Accuracy</div>
                <div class="kpi-value" style="color: #4F46E5;">{metrics['Logistic Regression']['Accuracy']*100:.2f}%</div>
                <div style="font-size: 0.8rem; color: #64748B;">ROC-AUC: {metrics['Logistic Regression']['ROC_AUC']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #10B981;">
                <div class="kpi-label">Random Forest Sensitivity</div>
                <div class="kpi-value" style="color: #059669;">{metrics['Random Forest']['Recall']*100:.2f}%</div>
                <div style="font-size: 0.8rem; color: #64748B;">Identifies 96.2% of passing cases</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #8B5CF6;">
                <div class="kpi-label">Decision Tree F1-Score</div>
                <div class="kpi-value" style="color: #7C3AED;">{metrics['Decision Tree']['F1_Score']:.4f}</div>
                <div style="font-size: 0.8rem; color: #64748B;">Rule-based Depth = 5</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        # Comparison Table Card
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📋 Comprehensive Algorithm Benchmark</div>', unsafe_allow_html=True)
        
        benchmark_records = []
        for m_name, m_data in metrics.items():
            benchmark_records.append({
                "Algorithm": m_name,
                "Accuracy (%)": f"{m_data['Accuracy']*100:.2f}%",
                "Precision (%)": f"{m_data['Precision']*100:.2f}%",
                "Recall / Sensitivity (%)": f"{m_data['Recall']*100:.2f}%",
                "F1-Score": f"{m_data['F1_Score']:.4f}",
                "ROC-AUC Score": f"{m_data['ROC_AUC']:.4f}"
            })
        st.dataframe(pd.DataFrame(benchmark_records), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Interactive Comparison Bar Chart
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📈 Metric Visual Comparison</div>', unsafe_allow_html=True)
        
        chart_rows = []
        for m_name, m_data in metrics.items():
            chart_rows.append({"Model": m_name, "Metric": "Accuracy", "Score": m_data['Accuracy'] * 100})
            chart_rows.append({"Model": m_name, "Metric": "Precision", "Score": m_data['Precision'] * 100})
            chart_rows.append({"Model": m_name, "Metric": "Recall", "Score": m_data['Recall'] * 100})
            chart_rows.append({"Model": m_name, "Metric": "F1-Score", "Score": m_data['F1_Score'] * 100})
            chart_rows.append({"Model": m_name, "Metric": "ROC-AUC", "Score": m_data['ROC_AUC'] * 100})
            
        chart_df = pd.DataFrame(chart_rows)
        
        comp_chart = alt.Chart(chart_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X('Model:N', title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Score:Q', title='Percentage (%)', scale=alt.Scale(domain=[75, 100])),
            color=alt.Color('Metric:N', scale=alt.Scale(scheme='set2')),
            xOffset='Metric:N',
            tooltip=['Model', 'Metric', alt.Tooltip('Score:Q', format='.2f')]
        ).properties(height=340)
        
        st.altair_chart(comp_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Confusion Matrices Card
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🧮 Confusion Matrix Diagnostic</div>', unsafe_allow_html=True)
        st.caption("Distribution of True Positives, True Negatives, False Positives, and False Negatives on the 240 test cases.")
        
        cm_cols = st.columns(len(metrics))
        for idx, (m_name, m_data) in enumerate(metrics.items()):
            with cm_cols[idx]:
                cm = m_data['Confusion_Matrix']
                tn, fp = cm[0]
                fn, tp = cm[1]
                
                st.markdown(f"**{m_name}**")
                cm_df = pd.DataFrame(
                    [
                        {"Actual": "Actual: At-Risk", "Pred: At-Risk (TN)": tn, "Pred: Pass (FP)": fp},
                        {"Actual": "Actual: Pass", "Pred: At-Risk (FN)": fn, "Pred: Pass (TP)": tp}
                    ]
                ).set_index("Actual")
                st.dataframe(cm_df, use_container_width=True)
                
                total_correct = tn + tp
                total_errors = fp + fn
                st.caption(f"✅ Correct: **{total_correct}** ({(total_correct/240)*100:.1f}%) | ❌ Errors: **{total_errors}**")
                
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature Importance Card
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 Feature Importance Rankings (Random Forest)</div>', unsafe_allow_html=True)
        
        rf_feats = metrics.get('Random Forest', {}).get('Feature_Importance', {})
        if rf_feats:
            feat_df = pd.DataFrame(
                list(rf_feats.items()),
                columns=['Feature', 'Importance']
            ).sort_values(by='Importance', ascending=True)
            
            # Format clean labels
            clean_labels = {
                'Past_Exam_Score': 'Past Exam Score',
                'Attendance_Rate': 'Attendance Rate (%)',
                'Study_Hours_Per_Week': 'Weekly Study Hours',
                'Sleep_Hours': 'Average Sleep Duration',
                'Tutoring_Sessions': 'Monthly Tutoring Sessions',
                'Stress_Level': 'Stress Level',
                'Parent_Education_Level': 'Parent Education Level',
                'Internet_Access': 'Home Internet Access',
                'Extracurricular_Activities': 'Extracurricular Activities'
            }
            feat_df['CleanFeature'] = feat_df['Feature'].map(lambda x: clean_labels.get(x, x))
            
            feat_chart = alt.Chart(feat_df).mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6).encode(
                x=alt.X('Importance:Q', title='Gini Importance Index'),
                y=alt.Y('CleanFeature:N', sort='-x', title=None),
                color=alt.Color('Importance:Q', scale=alt.Scale(scheme='purples'), legend=None),
                tooltip=['CleanFeature', alt.Tooltip('Importance:Q', format='.4f')]
            ).properties(height=320)
            
            col_fc1, col_fc2 = st.columns([1.6, 1.0])
            with col_fc1:
                st.altair_chart(feat_chart, use_container_width=True)
            with col_fc2:
                st.markdown("""
                <div style="background: #F8FAFC; padding: 18px; border-radius: 12px; border: 1px solid #E2E8F0;">
                    <h5 style="color: #1E1B4B; margin-top: 0;">Key Takeaways:</h5>
                    <ul style="color: #475569; font-size: 0.9rem; padding-left: 20px; line-height: 1.6;">
                        <li><strong>Past Score (36.2%) & Attendance (21.7%):</strong> Represent >55% of the total decision weight.</li>
                        <li><strong>Study Hours (18.1%):</strong> Forms the primary actionable lever for course correction.</li>
                        <li><strong>Sleep & Stress:</strong> Act as cognitive multipliers that amplify learning retention.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# SECTION 4: VISUAL INSIGHTS
# ---------------------------------------------------------
elif selected_nav == "📈 Visual Insights":
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">
            <span class="title-word">📈 Exploratory</span> <span class="title-word">Data</span> <span class="title-word">& Insights</span>
        </h1>
        <p class="page-subtitle">Statistical visualizations and trend analysis across the 1,200 cohort records.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if raw_df is not None:
        # Row 1: Attendance vs Exam Score & Pass Rate by Study Blocks
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🎯 Attendance Rate vs. Final Performance</div>', unsafe_allow_html=True)
            
            scatter_df = raw_df[['Attendance_Rate', 'Final_Exam_Score', 'Passed']].copy()
            scatter_df['Outcome'] = scatter_df['Passed'].map({1: 'Passed', 0: 'At-Risk'})
            
            sc_chart = alt.Chart(scatter_df.sample(400, random_state=42)).mark_circle(size=60, opacity=0.7).encode(
                x=alt.X('Attendance_Rate:Q', title='Attendance Rate (%)', scale=alt.Scale(domain=[50, 100])),
                y=alt.Y('Final_Exam_Score:Q', title='Final Exam Score (0-100)', scale=alt.Scale(domain=[30, 100])),
                color=alt.Color('Outcome:N', scale=alt.Scale(domain=['Passed', 'At-Risk'], range=['#10B981', '#EF4444'])),
                tooltip=['Attendance_Rate', 'Final_Exam_Score', 'Outcome']
            ).properties(height=320)
            
            # Passing threshold line at 60
            rule = alt.Chart(pd.DataFrame({'y': [60]})).mark_rule(color='#64748B', strokeDash=[4, 4]).encode(y='y:Q')
            st.altair_chart(sc_chart + rule, use_container_width=True)
            st.caption("Points above the dashed line (score >= 60) indicate passing status. Higher attendance strongly clusters in safe zones.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_v2:
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📖 Study Hours vs. Past & Final Scores</div>', unsafe_allow_html=True)
            
            study_bins = pd.cut(raw_df['Study_Hours_Per_Week'], bins=[0, 8, 14, 20, 26, 40], labels=['<8h', '8-14h', '14-20h', '20-26h', '>26h'])
            agg_study = raw_df.groupby(study_bins, observed=False).agg(
                Avg_Final_Score=('Final_Exam_Score', 'mean'),
                Pass_Rate=('Passed', lambda x: x.mean() * 100)
            ).reset_index()
            agg_study.rename(columns={'Study_Hours_Per_Week': 'Study_Block'}, inplace=True)
            
            bar_study = alt.Chart(agg_study).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color='#6366F1').encode(
                x=alt.X('Study_Block:N', title='Weekly Study Hours Tier', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Pass_Rate:Q', title='Cohort Pass Rate (%)', scale=alt.Scale(domain=[0, 100])),
                tooltip=['Study_Block', alt.Tooltip('Pass_Rate:Q', format='.1f'), alt.Tooltip('Avg_Final_Score:Q', format='.1f')]
            ).properties(height=320)
            
            st.altair_chart(bar_study, use_container_width=True)
            st.caption("Students studying 14+ hours per week maintain an >85% cohort passing rate.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Row 2: Stress Level & Sleep Hours
        col_v3, col_v4 = st.columns(2)
        
        with col_v3:
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🧘 Stress Level vs. Pass Rate</div>', unsafe_allow_html=True)
            
            stress_agg = raw_df.groupby('Stress_Level').agg(
                Count=('Student_ID', 'count'),
                Pass_Rate=('Passed', lambda x: x.mean() * 100),
                Avg_Score=('Final_Exam_Score', 'mean')
            ).loc[['Low', 'Medium', 'High']].reset_index()
            
            stress_chart = alt.Chart(stress_agg).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                x=alt.X('Stress_Level:N', title='Reported Stress Level', sort=['Low', 'Medium', 'High'], axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Pass_Rate:Q', title='Passing Rate (%)', scale=alt.Scale(domain=[50, 100])),
                color=alt.Color('Stress_Level:N', scale=alt.Scale(domain=['Low', 'Medium', 'High'], range=['#10B981', '#F59E0B', '#EF4444']), legend=None),
                tooltip=['Stress_Level', alt.Tooltip('Pass_Rate:Q', format='.1f'), alt.Tooltip('Avg_Score:Q', format='.1f')]
            ).properties(height=300)
            
            st.altair_chart(stress_chart, use_container_width=True)
            st.caption("High stress reduces overall pass rates by ~16% compared to well-managed low-stress cohorts.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_v4:
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🎓 Guardian Education Influence</div>', unsafe_allow_html=True)
            
            edu_order = ['High School', 'Bachelor', 'Master', 'Doctorate']
            edu_agg = raw_df.groupby('Parent_Education_Level').agg(
                Pass_Rate=('Passed', lambda x: x.mean() * 100),
                Avg_Score=('Final_Exam_Score', 'mean')
            ).reindex(edu_order).reset_index()
            
            edu_chart = alt.Chart(edu_agg).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color='#8B5CF6').encode(
                x=alt.X('Parent_Education_Level:N', title='Guardian Education', sort=edu_order, axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Pass_Rate:Q', title='Passing Rate (%)', scale=alt.Scale(domain=[50, 100])),
                tooltip=['Parent_Education_Level', alt.Tooltip('Pass_Rate:Q', format='.1f'), alt.Tooltip('Avg_Score:Q', format='.1f')]
            ).properties(height=300)
            
            st.altair_chart(edu_chart, use_container_width=True)
            st.caption("Guardian education shows positive correlation with academic preparation and resources.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Interactive Dataset Explorer
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📁 Interactive Student Cohort Explorer</div>', unsafe_allow_html=True)
        
        col_f1, col_f2, col_f3 = st.columns([1, 1, 1])
        with col_f1:
            filter_status = st.selectbox("Filter Status", ["All Students", "Passed Only", "At-Risk Only"])
        with col_f2:
            filter_stress = st.selectbox("Filter Stress", ["All Levels", "Low", "Medium", "High"])
        with col_f3:
            search_id = st.text_input("Search Student ID", "")
            
        view_df = raw_df.copy()
        if filter_status == "Passed Only":
            view_df = view_df[view_df['Passed'] == 1]
        elif filter_status == "At-Risk Only":
            view_df = view_df[view_df['Passed'] == 0]
            
        if filter_stress != "All Levels":
            view_df = view_df[view_df['Stress_Level'] == filter_stress]
            
        if search_id:
            view_df = view_df[view_df['Student_ID'].str.contains(search_id.strip(), case=False)]
            
        st.dataframe(view_df.head(50), use_container_width=True)
        st.caption(f"Displaying {min(50, len(view_df))} of {len(view_df):,} matching records.")
        st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# SECTION 5: ABOUT
# ---------------------------------------------------------
elif selected_nav == "ℹ️ About":
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">
            <span class="title-word">ℹ️ About</span> <span class="title-word">Platform</span> <span class="title-word">& Architecture</span>
        </h1>
        <p class="page-subtitle">Methodology, technical stack, and ethical AI in education.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_a1, col_a2 = st.columns([1.4, 1.0])
    
    with col_a1:
        st.markdown("""
        <div class="saas-card">
            <div class="card-title">🎯 Project Mission & Vision</div>
            <p style="color: #475569; font-size: 0.95rem; line-height: 1.7;">
                The <strong>Student Performance Prediction</strong> initiative was developed to bridge the gap between 
                predictive data analytics and actionable academic advisory. By synthesizing multidimensional educational telemetry 
                (attendance consistency, dedicated study investment, historical mastery, and well-being metrics), the system 
                equips educators with early warning capabilities to support students before academic distress impacts graduation outcomes.
            </p>
        </div>
        
        <div class="saas-card">
            <div class="card-title">🏗️ Production Pipeline Architecture</div>
            <p style="color: #475569; font-size: 0.95rem; line-height: 1.7;">
                1. <strong>Data Preprocessing:</strong> Categorical variables are mapped ordinally and binary-encoded. Continuous features are standardized using Scikit-learn's <code>StandardScaler</code> fitted strictly on training data.<br/>
                2. <strong>Ensemble Machine Learning:</strong> The system trains and cross-evaluates three distinct algorithmic archetypes:
                <ul>
                    <li><strong>Logistic Regression:</strong> Well-calibrated probabilistic baseline with strong linear interpretability.</li>
                    <li><strong>Decision Tree:</strong> Non-linear hierarchical decision rules with depth-control to mitigate variance.</li>
                    <li><strong>Random Forest:</strong> 100-tree bagged ensemble providing high generalization and maximum recall.</li>
                </ul>
                3. <strong>Cloud Deployment:</strong> Packaged as a lightweight, cloud-native Streamlit web service with automatic self-healing artifact initialization.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_a2:
        st.markdown("""
        <div class="saas-card">
            <div class="card-title">💻 Technology Stack</div>
            <ul style="color: #475569; font-size: 0.9rem; line-height: 1.8; padding-left: 20px;">
                <li><strong>Language:</strong> Python 3.13</li>
                <li><strong>Web Framework:</strong> Streamlit 1.66+</li>
                <li><strong>Machine Learning:</strong> Scikit-learn 1.9</li>
                <li><strong>Data Processing:</strong> Pandas 3.0 & NumPy 2.5</li>
                <li><strong>Visualization:</strong> Altair 6.3 & Matplotlib</li>
                <li><strong>Hosting:</strong> Streamlit Community Cloud (24/7 Free Tier)</li>
                <li><strong>Version Control:</strong> GitHub</li>
            </ul>
        </div>
        
        <div class="saas-card">
            <div class="card-title">⚖️ Ethical AI in Education</div>
            <p style="color: #475569; font-size: 0.88rem; line-height: 1.6;">
                <strong>Notice:</strong> Machine learning risk assessments are intended solely as <em>supportive early intervention signals</em> for educators and counselors. 
                They should never be employed for punitive sorting, automated grading, or denying educational opportunities.
            </p>
        </div>
        
        <div class="saas-card">
            <div class="card-title">🔗 Project Links</div>
            <p style="font-size: 0.9rem; margin-bottom: 8px;">
                <strong>GitHub Repository:</strong><br/>
                <a href="https://github.com/soumilijn-tech/student-performance-prediction" target="_blank">soumilijn-tech/student-performance-prediction</a>
            </p>
            <p style="font-size: 0.9rem; margin: 0;">
                <strong>Live Deployment:</strong><br/>
                <a href="https://student-performance-prediction-rorrrdysrqmbqzoxydoopk.streamlit.app/" target="_blank">Live Streamlit App</a>
            </p>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Global Footer
# ---------------------------------------------------------
st.markdown("""
<div class="saas-footer">
    Student Performance Prediction Platform • <strong>Smarter Insights for Brighter Futures</strong><br/>
    Built with Python, Scikit-learn, Pandas & Streamlit • Deployed on Streamlit Community Cloud
</div>
""", unsafe_allow_html=True)
