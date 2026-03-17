import streamlit as st
import pandas as pd

st.set_page_config(page_title="Sample Inputs", layout="wide")

st.title("🧪 Example Inputs for Users")

st.markdown("""
These example inputs help users understand what kind of values
should be entered in the prediction system.
""")

data = {
"Example":["Example 1","Example 2","Example 3"],
"Gender":["Female","Male","Female"],
"Reading Score":[90,60,30],
"Writing Score":[92,58,35],
"Expected Result":["High Performer","Average","At Risk"]
}

df = pd.DataFrame(data)

st.table(df)

st.markdown("---")

st.subheader("📊 Performance Rule Table")

rules = {
"Average Score":["< 40","40 - 75","> 90"],
"Category":["At Risk","Normal","Distinction"]
}

rule_df = pd.DataFrame(rules)

st.table(rule_df)

st.success("Sample inputs provided successfully.")