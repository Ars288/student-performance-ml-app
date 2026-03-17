import streamlit as st
import requests
import os
import pandas as pd
import pyttsx3
from dotenv import load_dotenv
from streamlit_mic_recorder import speech_to_text

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

st.title("🤖 AI Student Assistant")
st.write("Ask anything using text or microphone.")

# Load dataset
df = pd.read_csv("data/students.csv")

# Text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Voice input
voice_prompt = speech_to_text(
    language='en',
    start_prompt="🎤 Speak",
    stop_prompt="Stop",
    just_once=True,
    use_container_width=True,
    key="voice"
)

# Text input
text_prompt = st.chat_input("Ask anything...")

prompt = text_prompt if text_prompt else voice_prompt

if prompt:

    st.chat_message("user").write(prompt)

    lower_prompt = prompt.lower()

    # Dataset answers
    if "average math" in lower_prompt:
        answer = f"The average math score is {round(df['math_score'].mean(),2)}."

    elif "how many students" in lower_prompt:
        answer = f"There are {df.shape[0]} students in the dataset."

    else:
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

    # Speak the response aloud
    speak(answer)

    st.session_state.messages.append({"role":"user","content":prompt})
    st.session_state.messages.append({"role":"assistant","content":answer})