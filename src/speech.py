# speech.py

import os
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))


# AC-2: STT
def speech_to_text(audio_path):
    with open(audio_path, "rb") as audio:
        transcript = client.speech_to_text.convert(
            file=audio,
            model_id="scribe_v1"
        )
    return transcript.text


# AC-3: TTS
def text_to_speech(text, language="en"):

    voice_id = "JBFqnCBsd6RMkjVDRZzb"

    audio = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2"
    )

    output_file = "response.mp3"

    with open(output_file, "wb") as f:
        for chunk in audio:
            f.write(chunk)

    return output_file
