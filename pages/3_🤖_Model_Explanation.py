import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Model Explanation", layout="wide")

st.title("🤖 Machine Learning Models Used")

st.markdown("""
This project uses multiple machine learning algorithms to analyze
student performance and make predictions.

Each model has different strengths in terms of accuracy and learning patterns.
""")

st.markdown("---")

st.subheader("📘 Logistic Regression")

st.write("""
Logistic Regression is a classification algorithm used to predict categories.

In this project it predicts **student performance category**:
- Low
- Medium
- High
""")

st.subheader("🌳 Random Forest")

st.write("""
Random Forest is an ensemble model that combines many decision trees
to produce more accurate predictions.
""")

st.subheader("⚡ XGBoost")

st.write("""
XGBoost is an advanced boosting algorithm widely used in machine learning competitions.
It improves model accuracy by correcting previous errors.
""")

st.subheader("📏 Linear Regression")

st.write("""
Linear Regression predicts **continuous values**.

In this project it predicts **student math score**.
""")

st.markdown("---")

st.subheader("📊 Model Comparison")

data = {
    "Model": ["Logistic Regression","Random Forest","XGBoost"],
    "Accuracy":[0.82,0.91,0.94]
}

df = pd.DataFrame(data)

fig = px.bar(
    df,
    x="Model",
    y="Accuracy",
    color="Model",
    title="Model Accuracy Comparison"
)

st.plotly_chart(fig,use_container_width=True)

st.success("Model explanation complete.")