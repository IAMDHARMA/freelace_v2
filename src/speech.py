# src/speech.py

import os
import base64
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

VOICE_MAP = {
    "en": "JBFqnCBsd6RMkjVDRZzb",
    "ta": "JBFqnCBsd6RMkjVDRZzb",
    "hi": "JBFqnCBsd6RMkjVDRZzb"
}


# ===============================
# SPEECH TO TEXT (ElevenLabs STT)
# ===============================
def speech_to_text(audio_path: str):

    try:
        with open(audio_path, "rb") as audio:
            transcript = client.speech_to_text.convert(
                file=audio,
                model_id="scribe_v1",
                diarize=False   # 🔥 reduces overlapping detection sensitivity
            )

        if not transcript or not transcript.text:
            return None

        text = transcript.text.strip()
        text_lower = text.lower()

        # 🔥 Invalid / Noise outputs
        invalid_outputs = [
            "(traffic noise)",
            "(background noise)",
            "(overlapping dialogue)",
            "(music)",
            ""
        ]

        # Ignore known noise labels
        if text_lower in invalid_outputs:
            return None

        # Ignore very short speech (like "uh", "hmm")
        if len(text.split()) < 2:
            return None

        return text

    except Exception as e:
        print("STT ERROR:", e)
        return None


# ===============================
# TEXT TO SPEECH (ElevenLabs TTS)
# ===============================
def text_to_speech(text: str, language: str = "en"):

    try:
        if not text or len(text.strip()) == 0:
            return None

        voice_id = VOICE_MAP.get(language, VOICE_MAP["en"])

        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2"
        )

        audio_bytes = b"".join(audio_stream)

        if not audio_bytes:
            return None

        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return audio_base64

    except Exception as e:
        print("TTS ERROR:", e)
        return None
