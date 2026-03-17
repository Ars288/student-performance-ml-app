import streamlit as st
import pandas as pd
from utils.preprocessing import load_dataset

st.set_page_config(page_title="Dataset Overview", layout="wide")

# -------- Background Image --------
# -------- Background Image --------
st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            rgba(255,255,255,0.88),
            rgba(255,255,255,0.88)
        ),
        url("https://tse3.mm.bing.net/th/id/OIP.Zmv7JT2fJRqX6gXFqyts_wHaDt?pid=Api&P=0&h=180");
        background-size: cover;
        background-attachment: fixed;
    }

    /* FORCE TEXT COLOR TO BLACK */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: black !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Dataset Overview")

st.markdown(
"""
This section introduces the dataset used for the machine learning model.

The dataset contains **student academic performance information** such as:

- Gender
- Parental Education
- Lunch Type
- Test Preparation
- Reading Score
- Writing Score
- Math Score

We use this data to analyze patterns and build predictive models.
"""
)

df = load_dataset()

st.subheader("📂 Dataset Shape")

col1, col2 = st.columns(2)

with col1:
    st.metric("Number of Rows", df.shape[0])

with col2:
    st.metric("Number of Columns", df.shape[1])

st.markdown("---")

st.subheader("🔎 Dataset Preview")

st.dataframe(df.head(10), use_container_width=True)

st.markdown("---")

st.subheader("📋 Column Descriptions")

column_info = {
    "gender": "Student Gender",
    "race_ethnicity": "Ethnicity Group",
    "parental_level_of_education": "Parent Education Level",
    "lunch": "Lunch Type (standard/free)",
    "test_preparation_course": "Test Preparation Course",
    "math_score": "Math Exam Score",
    "reading_score": "Reading Score",
    "writing_score": "Writing Score"
}

info_df = pd.DataFrame(
    column_info.items(),
    columns=["Column Name", "Description"]
)

st.table(info_df)

st.markdown("---")

st.subheader("📊 Data Types")

st.write(df.dtypes)

st.markdown("---")

st.subheader("❗ Missing Values")

missing = df.isnull().sum()

st.write(missing)

st.success("Dataset successfully loaded and verified.")