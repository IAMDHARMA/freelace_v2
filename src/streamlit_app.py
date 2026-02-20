# streamlit_app.py

import streamlit as st
import requests
import base64
import uuid
from streamlit_mic_recorder import mic_recorder

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Tutor", page_icon="🎓", layout="centered")

# -------------------------------------------------------
# Session Setup
# -------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_processing_voice" not in st.session_state:
    st.session_state.is_processing_voice = False

if "recorded_audio" not in st.session_state:
    st.session_state.recorded_audio = None

# -------------------------------------------------------
# Title
# -------------------------------------------------------
st.title("🎓 AI Tutor")
st.caption("Ask questions in any language — get English lessons back!")

# -------------------------------------------------------
# Display Chat History
# -------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Play audio response if available
        if message.get("audio"):
            st.audio(message["audio"], format="audio/mp3")

# -------------------------------------------------------
# TEXT INPUT
# -------------------------------------------------------
if prompt := st.chat_input("Ask your question..."):

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # ✅ FIXED: Do NOT include output_lang=None in params — omit it entirely
                # Passing None sends the literal string "None" in query params
                params = {
                    "question": prompt,
                    "session_id": st.session_state.session_id,
                }

                response = requests.post(
                    f"{API_BASE}/ask-text",
                    params=params,
                    timeout=60,
                )

                if response.status_code == 200:
                    data = response.json()

                    if data.get("error"):
                        st.warning(f"⚠️ {data['error']}")
                    else:
                        reply = data.get("response_text", "No response received.")

                        audio_bytes = None
                        if data.get("audio_base64"):
                            audio_bytes = base64.b64decode(data["audio_base64"])

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": reply,
                            "audio": audio_bytes,
                        })
                else:
                    st.error(f"Backend error: {response.status_code}")

            except requests.exceptions.Timeout:
                st.error("⏱ Request timed out. Please try again.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Cannot connect to the backend. Make sure the FastAPI server is running.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    st.rerun()

# -------------------------------------------------------
# VOICE INPUT
# -------------------------------------------------------
st.divider()
st.markdown("### 🎙 Voice Input")

# Only show mic while not processing to prevent double submissions
if not st.session_state.is_processing_voice:
    audio = mic_recorder(
        start_prompt="🎤 Speak",
        stop_prompt="🛑 Stop",
        key="voice_recorder",
    )

    if audio and audio.get("bytes"):
        st.session_state.is_processing_voice = True
        st.session_state.recorded_audio = audio["bytes"]
        st.rerun()

# -------------------------------------------------------
# PROCESS VOICE (runs after rerun, while flag is True)
# -------------------------------------------------------
if st.session_state.is_processing_voice and st.session_state.recorded_audio:

    with st.spinner("Processing voice..."):
        # ✅ FIXED: Wrap entire block in try/finally so flag always resets
        # Previously, if an exception was raised, is_processing_voice stayed True forever
        try:
            # ✅ FIXED: Do NOT pass output_lang=None — omit from params entirely
            params = {"session_id": st.session_state.session_id}

            response = requests.post(
                f"{API_BASE}/ask-voice",
                params=params,
                files={"file": ("audio.wav", st.session_state.recorded_audio, "audio/wav")},
                timeout=60,
            )

            if response.status_code == 200:
                data = response.json()

                if data.get("error"):
                    st.warning("⚠️ Please speak clearly and try again.")
                else:
                    user_text = data.get("transcribed_text", "").strip()
                    assistant_text = data.get("response_text", "")

                    if user_text:
                        st.session_state.messages.append({
                            "role": "user",
                            "content": user_text,
                        })

                        audio_bytes = None
                        if data.get("audio_base64"):
                            audio_bytes = base64.b64decode(data["audio_base64"])

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": assistant_text,
                            "audio": audio_bytes,
                        })
            else:
                st.error(f"Voice processing failed: {response.status_code}")

        except requests.exceptions.Timeout:
            st.error("⏱ Voice request timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Cannot connect to the backend. Make sure the FastAPI server is running.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

        finally:
            # ✅ FIXED: Always reset state, even if an exception occurred
            st.session_state.is_processing_voice = False
            st.session_state.recorded_audio = None

    st.rerun()
