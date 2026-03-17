import streamlit as st
import requests
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

st.title("🤖 AI Student Data Assistant")

st.write("Ask questions about the project or dataset.")

# Load dataset
df = pd.read_csv("data/students.csv")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask your question")

if prompt:

    st.chat_message("user").write(prompt)

    lower_prompt = prompt.lower()

    # Dataset based responses
    if "average math" in lower_prompt:
        answer = f"The average math score is {round(df['math_score'].mean(),2)}."

    elif "average reading" in lower_prompt:
        answer = f"The average reading score is {round(df['reading_score'].mean(),2)}."

    elif "average writing" in lower_prompt:
        answer = f"The average writing score is {round(df['writing_score'].mean(),2)}."

    elif "how many students" in lower_prompt:
        answer = f"There are {df.shape[0]} students in the dataset."

    elif "gender scored higher" in lower_prompt:
        avg = df.groupby("gender")[["math_score","reading_score","writing_score"]].mean()
        answer = f"Average scores by gender:\n{avg}"

    else:
        # AI response
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "meta-llama/llama-3-8b-instruct",
                "messages": [{"role": "user", "content": prompt}],
            },
        )

        answer = response.json()["choices"][0]["message"]["content"]

    st.chat_message("assistant").write(answer)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": answer})