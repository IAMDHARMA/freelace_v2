# src/app.py

import os
import uuid
import shutil
import tempfile
import traceback

from fastapi import FastAPI, UploadFile, File, HTTPException
from .rag import get_qa_chain
from .speech import speech_to_text, text_to_speech
from .language import detect_language, translate_text

app = FastAPI()

# Initialize the RAG chain once at startup
qa_chain = get_qa_chain()


def sanitize_lang(value: str | None) -> str | None:
    """
    ✅ FIXED: Streamlit sends None as the string "None" in query params.
    This helper converts the string "None" back to Python None.
    """
    if value is None or value.strip().lower() == "none":
        return None
    return value.strip()


def invoke_chain(question: str, language: str, session_id: str) -> str:
    """
    Invoke the RAG chain and extract the text response.
    ✅ FIXED: Returns result.content directly since ChatGroq always returns AIMessage.
    No longer uses fragile getattr fallback.
    """
    result = qa_chain.invoke(
        {"question": question, "language": language},
        config={"configurable": {"session_id": session_id}},
    )
    return result.content


# ===============================
# TEXT QUERY
# ===============================
@app.post("/ask-text")
async def ask_text(
    question: str,
    session_id: str = None,
    output_lang: str = None,
    input_lang: str = None,
):
    # ✅ FIXED: Sanitize "None" strings from Streamlit query params
    output_lang = sanitize_lang(output_lang)
    input_lang = sanitize_lang(input_lang)

    if not session_id:
        session_id = str(uuid.uuid4())

    if not question or not question.strip():
        return {"error": "Question is empty"}

    try:
        detected_lang = detect_language(question)
        input_language = input_lang or detected_lang

        response = invoke_chain(question, input_language, session_id)

        final_output_lang = output_lang or input_language

        # Translate only when output language differs from input
        if final_output_lang != input_language:
            response = translate_text(response, final_output_lang)

        audio_base64 = None
        try:
            audio_base64 = text_to_speech(response, final_output_lang)
        except Exception as e:
            print(f"[TTS ERROR] {e}")

        return {
            "session_id": session_id,
            "input_language": input_language,
            "output_language": final_output_lang,
            "response_text": response,
            "audio_base64": audio_base64,
        }

    except Exception:
        print("\n===== TEXT ENDPOINT ERROR =====")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Text processing failed")


# ===============================
# VOICE QUERY
# ===============================
@app.post("/ask-voice")
async def ask_voice(
    file: UploadFile = File(...),
    session_id: str = None,
    output_lang: str = None,
    input_lang: str = None,
):
    # ✅ FIXED: Sanitize "None" strings from Streamlit query params
    output_lang = sanitize_lang(output_lang)
    input_lang = sanitize_lang(input_lang)

    if not session_id:
        session_id = str(uuid.uuid4())

    if not file:
        return {"error": "No audio file received"}

    # ✅ FIXED: Use tempfile.NamedTemporaryFile instead of a relative-path UUID file
    # This avoids polluting the working directory and is cleaned up safely in finally block
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = tmp.name
    tmp.close()

    try:
        # Save uploaded audio to temp file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Speech → Text
        transcribed_text = speech_to_text(temp_path)

        if not transcribed_text:
            return {
                "error": "No speech detected",
                "transcribed_text": "",
                "response_text": "",
                "audio_base64": None,
            }

        detected_lang = detect_language(transcribed_text)
        input_language = input_lang or detected_lang

        # RAG → Response
        response = invoke_chain(transcribed_text, input_language, session_id)

        final_output_lang = output_lang or input_language

        # Translate if needed
        if final_output_lang != input_language:
            response = translate_text(response, final_output_lang)

        audio_base64 = None
        try:
            audio_base64 = text_to_speech(response, final_output_lang)
        except Exception as e:
            print(f"[TTS ERROR] {e}")

        return {
            "session_id": session_id,
            "transcribed_text": transcribed_text,
            "input_language": input_language,
            "output_language": final_output_lang,
            "response_text": response,
            "audio_base64": audio_base64,
        }

    except Exception:
        print("\n===== VOICE ENDPOINT ERROR =====")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Voice processing failed")

    finally:
        # Always clean up the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=False)