import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
from streamlit_lottie import st_lottie

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI Student Performance System",
    page_icon="🎓",
    layout="wide"
)

# ---------------- CSS ---------------- #

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("assets/style.css")

# ---------------- LOTTIE ---------------- #

def load_lottie(url):
    return requests.get(url).json()

lottie_ai = load_lottie("https://assets5.lottiefiles.com/packages/lf20_kyu7xb1v.json")

st_lottie(lottie_ai, height=250)

# ---------------- HEADER ---------------- #

st.markdown(
"""
<h1 style='
text-align:center;
font-size:50px;
color:black;
font-weight:bold;
margin-bottom:10px;
'>
🎓 AI Student Performance Analytics
</h1>
""",
unsafe_allow_html=True
)

# ---------------- TYPING DESCRIPTION ---------------- #

description = """
This dashboard helps us understand how students are performing based on their academic data.  
It analyzes factors like reading, writing, and other details to find patterns.  
Using machine learning, it can predict student scores and identify their performance level.  
It also helps in spotting students who may need extra support.
"""

placeholder = st.empty()
typed_text = ""

for char in description:
    typed_text += char
    placeholder.markdown(f"<p style='font-size:18px'>{typed_text}</p>", unsafe_allow_html=True)
    time.sleep(0.01)

st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)

# ---------------- PROJECT DESCRIPTION ---------------- #

col1, col2 = st.columns([2,1])

with col1:
    st.markdown("""
    ## 🚀 Project Overview
    
    This dashboard helps analyze student data and predict outcomes.

    ✔ EDA Analysis  
    ✔ ML Predictions  
    ✔ Risk Detection  
    ✔ Model Comparison  
    """)

with col2:
    st.info("""
    📌 **Project Info**

    Internship Project  
    Machine Learning Dashboard  

    Built With:
    - Python
    - Streamlit
    - XGBoost
    """)

# ---------------- DATA ---------------- #

@st.cache_data
def load_data():
    df = pd.read_csv("data/students.csv")
    df.rename(columns={"race/ethnicity": "race_ethnicity"}, inplace=True)
    return df

df = load_data()

# ---------------- DASHBOARD SUMMARY ---------------- #

st.markdown("## 📊 Dashboard Summary")

col1, col2, col3, col4 = st.columns(4)

total_students = df.shape[0]

avg_score = round(
    (df["math_score"].mean() +
     df["reading_score"].mean() +
     df["writing_score"].mean()) / 3,
    2
)

# Smooth small animation
with col1:
    placeholder1 = st.empty()
    for i in range(0, total_students, max(1, int(total_students/5))):
        placeholder1.metric("Total Students", i)
        time.sleep(0.05)
    placeholder1.metric("Total Students", total_students)

with col2:
    placeholder2 = st.empty()
    for i in np.linspace(0, avg_score, 10):
        placeholder2.metric("Average Score", round(i,2))
        time.sleep(0.05)
    placeholder2.metric("Average Score", avg_score)

with col3:
    st.metric("ML Models Used", "4")

with col4:
    st.metric("Best Model Accuracy", "91%")

st.markdown("---")

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("🎓 AI Dashboard")

st.sidebar.info("""
Dataset Overview  
EDA Visualizations  
Model Explanation  
Prediction System  
Model Evaluation  
Sample Inputs  
AI Chatbot
""")

# ---------------- FINAL SYSTEM READY ---------------- #

with st.spinner("🚀 Finalizing system..."):
    time.sleep(1)

st.success("System Ready !")