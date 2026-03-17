
import streamlit as st
from utils.preprocessing import load_dataset
from utils.visualization import score_distribution
from utils.visualization import gender_performance
from utils.visualization import lunch_vs_score
from utils.visualization import parental_education
from utils.visualization import correlation_heatmap

st.set_page_config(page_title="EDA Visualizations", layout="wide")

# -----------------------------
# Animated Dashboard Styling
# -----------------------------
st.markdown("""
<style>

.main-title{
font-size:95px;
font-weight:800;
text-align:center;
color:#2E86C1;
margin-bottom:25px;
}

@keyframes slideUp{
from{
opacity:0;
transform:translateY(60px);
}
to{
opacity:1;
transform:translateY(0px);
}
}

.graph-container{
animation: slideUp 0.9s ease-in-out;
padding:15px;
border-radius:15px;
background-color:#ffffff10;
box-shadow:0 4px 12px rgba(0,0,0,0.1);
margin-bottom:25px;
}

.kpi-card{
background: linear-gradient(135deg,#6dd5ed,#2193b0);
padding:20px;
border-radius:15px;
text-align:center;
color:white;
font-weight:bold;
box-shadow:0 6px 15px rgba(0,0,0,0.2);
transition: transform 0.3s;
}

.kpi-card:hover{
transform: scale(1.05);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown(
'<p class="main-title">📈 Student Performance Analysis Dashboard</p>',
unsafe_allow_html=True
)

st.markdown(
"""
Exploratory Data Analysis (EDA) helps us understand patterns and relationships
within the dataset before applying machine learning models.

Below are various visualizations that reveal insights into student performance.
"""
)

# -----------------------------
# Load Dataset
# -----------------------------
df = load_dataset()

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.title("⚙️ Dashboard Filters")
st.sidebar.markdown("Filter dataset to analyze student performance.")

gender_filter = st.sidebar.multiselect(
    "Select Gender",
    options=df["gender"].unique(),
    default=df["gender"].unique()
)

lunch_filter = st.sidebar.multiselect(
    "Select Lunch Type",
    options=df["lunch"].unique(),
    default=df["lunch"].unique()
)

df = df[
    (df["gender"].isin(gender_filter)) &
    (df["lunch"].isin(lunch_filter))
]

# -----------------------------
# KPI Cards
# -----------------------------
avg_math = round(df["math_score"].mean(),2)
avg_read = round(df["reading_score"].mean(),2)
avg_write = round(df["writing_score"].mean(),2)
total_students = df.shape[0]

col1,col2,col3,col4 = st.columns(4)

with col1:
    st.markdown(
        f'<div class="kpi-card">👨‍🎓 Students<br><h2>{total_students}</h2></div>',
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f'<div class="kpi-card">📘 Avg Math<br><h2>{avg_math}</h2></div>',
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f'<div class="kpi-card">📗 Avg Reading<br><h2>{avg_read}</h2></div>',
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f'<div class="kpi-card">📕 Avg Writing<br><h2>{avg_write}</h2></div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# -----------------------------
# Score Distribution
# -----------------------------
st.subheader("📊 Score Distribution")

fig1 = score_distribution(df)

st.markdown('<div class="graph-container">', unsafe_allow_html=True)
st.plotly_chart(
    fig1,
    use_container_width=True,
    config={"displayModeBar": False}
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# Gender vs Performance
# -----------------------------
st.subheader("👩 Gender vs Performance")

fig2 = gender_performance(df)

st.markdown('<div class="graph-container">', unsafe_allow_html=True)
st.plotly_chart(
    fig2,
    use_container_width=True,
    config={"displayModeBar": False}
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# Lunch Type vs Score
# -----------------------------
st.subheader("🍽 Lunch Type vs Score")

fig3 = lunch_vs_score(df)

st.markdown('<div class="graph-container">', unsafe_allow_html=True)
st.plotly_chart(
    fig3,
    use_container_width=True,
    config={"displayModeBar": False}
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# Parental Education vs Marks
# -----------------------------
st.subheader("🎓 Parental Education vs Marks")

fig4 = parental_education(df)

st.markdown('<div class="graph-container">', unsafe_allow_html=True)
st.plotly_chart(
    fig4,
    use_container_width=True,
    config={"displayModeBar": False}
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# Correlation Heatmap
# -----------------------------
st.subheader("🔥 Correlation Heatmap")

heatmap = correlation_heatmap(df)

st.markdown('<div class="graph-container">', unsafe_allow_html=True)
st.pyplot(heatmap)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# Dataset Viewer
# -----------------------------
with st.expander("🔎 View Dataset"):
    st.dataframe(df)

# -----------------------------
# Success Message
# -----------------------------
st.success("EDA completed successfully.")

