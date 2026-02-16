import streamlit as st
import requests
import base64
import uuid
from streamlit_mic_recorder import mic_recorder

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Tutor", page_icon="🎓", layout="centered")

# -----------------------------
# Session Setup
# -----------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_processing_voice" not in st.session_state:
    st.session_state.is_processing_voice = False

# -----------------------------
# Title
# -----------------------------
st.title("🎓 AI Tutor")

# -----------------------------
# Display Chat History
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # 🔥 Play audio if exists
        if message.get("audio"):
            st.audio(message["audio"], format="audio/mp3")

# -----------------------------
# TEXT INPUT
# -----------------------------
if prompt := st.chat_input("Ask your question..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            response = requests.post(
                f"{API_BASE}/ask-text",
                params={
                    "question": prompt,
                    "session_id": st.session_state.session_id,
                    "output_lang": "en"
                }
            )

            if response.status_code == 200:
                data = response.json()
                reply = data.get("response_text", "No response")

                audio_bytes = None
                if data.get("audio_base64"):
                    audio_bytes = base64.b64decode(data["audio_base64"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply,
                    "audio": audio_bytes
                })

                st.rerun()

            else:
                st.error("Backend error")

# -----------------------------
# VOICE INPUT
# -----------------------------
st.divider()
st.markdown("### 🎙 Voice Input")

# Disable mic while processing
if not st.session_state.is_processing_voice:

    audio = mic_recorder(
        start_prompt="🎤 Speak",
        stop_prompt="🛑 Stop",
        key="voice_recorder"
    )

    if audio:
        st.session_state.is_processing_voice = True
        st.session_state.recorded_audio = audio["bytes"]
        st.rerun()

# -----------------------------
# PROCESS VOICE
# -----------------------------
if st.session_state.is_processing_voice:

    with st.spinner("Processing voice..."):

        response = requests.post(
            f"{API_BASE}/ask-voice",
            params={
                "session_id": st.session_state.session_id,
                "output_lang": "en"
            },
            files={"file": ("audio.wav", st.session_state.recorded_audio, "audio/wav")}
        )

        st.session_state.is_processing_voice = False

        if response.status_code == 200:
            data = response.json()

            if data.get("error"):
                st.warning("⚠️ Please speak clearly and try again.")
            else:
                user_text = data.get("transcribed_text", "").strip()
                assistant_text = data.get("response_text", "")

                if user_text:

                    # Save user message
                    st.session_state.messages.append({
                        "role": "user",
                        "content": user_text
                    })

                    # Decode audio
                    audio_bytes = None
                    if data.get("audio_base64"):
                        audio_bytes = base64.b64decode(data["audio_base64"])

                    # Save assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_text,
                        "audio": audio_bytes
                    })

        else:
            st.error("Voice processing failed")

        st.rerun()
