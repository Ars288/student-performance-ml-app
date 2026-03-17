
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediction System", layout="wide")

# -----------------------------
# Animation & Styling
# -----------------------------
st.markdown("""
<style>

@keyframes fadeInUp{
from{
opacity:0;
transform:translateY(40px);
}
to{
opacity:1;
transform:translateY(0px);
}
}

.result-card{
animation: fadeInUp 0.8s ease-in-out;
padding:20px;
border-radius:14px;
background:rgba(255,255,255,0.7);
backdrop-filter:blur(10px);
box-shadow:0 8px 20px rgba(0,0,0,0.15);
margin-top:15px;
}

.stButton>button{
background:linear-gradient(90deg,#000000,#000000);
color:white;
font-weight:bold;
border-radius:12px;
height:50px;
width:100%;
transition:0.3s;
border:none;
}

.stButton>button:hover{
transform:scale(1.05);
box-shadow:0 6px 20px rgba(0,0,0,0.2);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.title("Student Performance Prediction System")

st.markdown("""
Enter student information below to predict performance.
""")

# -----------------------------
# Load Model
# -----------------------------
models = joblib.load("models/trained_models.pkl")
model = models["linear_regression"]

st.markdown("---")

# -----------------------------
# Input Fields
# -----------------------------
col1,col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["female","male"]
    )

    lunch = st.selectbox(
        "Lunch Type",
        ["standard","free/reduced"]
    )

    test_course = st.selectbox(
        "Test Preparation",
        ["none","completed"]
    )

with col2:

    reading_score = st.slider(
        "Reading Score",
        0,100,50
    )

    writing_score = st.slider(
        "Writing Score",
        0,100,50
    )

st.markdown("---")

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Performance"):

    gender_val = 0 if gender=="female" else 1
    lunch_val = 0 if lunch=="free/reduced" else 1
    prep_val = 0 if test_course=="none" else 1

    input_data = pd.DataFrame({
        "gender":[gender_val],
        "race_ethnicity":[0],
        "parental_level_of_education":[0],
        "lunch":[lunch_val],
        "test_preparation_course":[prep_val],
        "reading_score":[reading_score],
        "writing_score":[writing_score],
        "average_score":[(reading_score+writing_score)/2]
    })

    with st.spinner("🤖 AI is analyzing student performance..."):
        prediction = model.predict(input_data)[0]

# -----------------------------
# Animated Result Card
# -----------------------------
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        st.subheader("Prediction Result")

        st.metric(
            "Predicted Math Score",
            round(prediction,2)
        )

        st.progress(min(int(prediction),100))

        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Performance Message
# -----------------------------
        if prediction < 40:
            st.error("⚠ Student is at Risk")

        elif prediction < 75:
            st.warning("Student Performance: Average")

        else:
            st.success("🌟 High Performing Student")

