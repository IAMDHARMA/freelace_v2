# src/speech.py

import os
import base64
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

assert os.getenv("ELEVENLABS_API_KEY"), "ELEVENLABS_API_KEY is not set in .env"

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# ✅ NOTE: All currently point to the same voice ID.
# Replace values with actual multilingual ElevenLabs voice IDs when available.
# Using a single multilingual voice (eleven_multilingual_v2) is intentional —
# it handles Tamil, Hindi, and English without separate voices.
VOICE_MAP = {
    "en": "JBFqnCBsd6RMkjVDRZzb",
    "ta": "JBFqnCBsd6RMkjVDRZzb",
    "hi": "JBFqnCBsd6RMkjVDRZzb",
}
DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"

# Languages where splitting by space is used to avoid TTS timeout on long text
CHUNK_LANGUAGES = {"ta", "hi"}
MAX_CHUNK_LEN = 180


def split_text_safe(text: str, max_len: int = MAX_CHUNK_LEN) -> list[str]:
    """
    Split long text into chunks of at most max_len characters.
    Splits on word boundaries to avoid cutting mid-word.
    Used for Tamil and Hindi where ElevenLabs may time out on long inputs.
    """
    parts = []
    current = ""

    for word in text.split():
        if len(current) + len(word) + 1 > max_len:
            if current.strip():
                parts.append(current.strip())
            current = word + " "
        else:
            current += word + " "

    if current.strip():
        parts.append(current.strip())

    return parts if parts else [text]


# ============================================================
# SPEECH TO TEXT — ElevenLabs STT (Scribe)
# ============================================================
def speech_to_text(audio_path: str) -> str | None:
    """
    Transcribe an audio file to text using ElevenLabs Scribe STT.
    Returns None if no valid speech is detected.
    """
    # Noise-only or invalid STT outputs to ignore
    INVALID_OUTPUTS = {
        "(traffic noise)",
        "(background noise)",
        "(overlapping dialogue)",
        "(music)",
        "",
    }

    try:
        with open(audio_path, "rb") as audio:
            transcript = client.speech_to_text.convert(
                file=audio,
                model_id="scribe_v1",
                diarize=False,
            )

        if not transcript or not transcript.text:
            return None

        text = transcript.text.strip()

        if text.lower() in INVALID_OUTPUTS:
            return None

        # Reject single-word transcriptions — likely noise
        if len(text.split()) < 2:
            return None

        return text

    except Exception as e:
        print(f"[STT ERROR] {e}")
        return None


# ============================================================
# TEXT TO SPEECH — ElevenLabs TTS
# ============================================================
def text_to_speech(text: str, language: str = "en") -> str | None:
    """
    Convert text to speech using ElevenLabs.
    Returns base64-encoded MP3 audio, or None on failure.

    For Tamil and Hindi, splits text into smaller chunks first
    to avoid ElevenLabs timeouts on long inputs.
    """
    if not text or not text.strip():
        return None

    voice_id = VOICE_MAP.get(language, DEFAULT_VOICE)

    # Split only for languages that need it
    chunks = split_text_safe(text) if language in CHUNK_LANGUAGES else [text]

    try:
        final_audio = b""

        for chunk in chunks:
            if not chunk.strip():
                continue

            stream = client.text_to_speech.convert(
                text=chunk,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",
            )
            final_audio += b"".join(stream)

        if not final_audio:
            return None

        return base64.b64encode(final_audio).decode("utf-8")

    except Exception as e:
        print(f"[TTS ERROR] {e}")
        return None