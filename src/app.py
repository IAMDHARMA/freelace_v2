from fastapi import FastAPI, UploadFile, File, HTTPException
from .rag import get_qa_chain
from .speech import speech_to_text, text_to_speech
from .language import detect_language, translate_text
import shutil
import os
import uuid
import traceback

app = FastAPI()
qa_chain = get_qa_chain()


# ===============================
# TEXT QUERY
# ===============================
@app.post("/ask-text")
async def ask_text(
    question: str,
    session_id: str = None,
    output_lang: str = None,
    input_lang: str = None
):

    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        if not question:
            return {"error": "Question is empty"}

        detected_lang = detect_language(question)
        input_language = input_lang or detected_lang

        result = qa_chain.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}}
        )

        response = result.content if result else "No response generated."

        final_output_lang = output_lang or input_language

        if final_output_lang != input_language:
            response = translate_text(response, final_output_lang)

        try:
            audio_base64 = text_to_speech(response, final_output_lang)
        except Exception as e:
            print("TTS ERROR:", e)
            audio_base64 = None

        return {
            "session_id": session_id,
            "input_language": input_language,
            "output_language": final_output_lang,
            "response_text": response,
            "audio_base64": audio_base64
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
    input_lang: str = None
):

    if not session_id:
        session_id = str(uuid.uuid4())

    temp_file = f"{uuid.uuid4()}.wav"

    try:
        # Validate file
        if not file:
            return {"error": "No audio file received"}

        # Save audio
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Convert speech to text
        try:
            transcribed_text = speech_to_text(temp_file)
        except Exception as e:
            print("STT ERROR:", e)
            return {"error": "Speech recognition failed"}

        if not transcribed_text:
            return {"error": "No speech detected",
                     "transcribed_text": "",
                     "response_text": "",
                     "audio_base64": None
                     }

        detected_lang = detect_language(transcribed_text)
        input_language = input_lang or detected_lang

        # RAG call
        result = qa_chain.invoke(
            {"question": transcribed_text},
            config={"configurable": {"session_id": session_id}}
        )

        response = result.content if result else "No response generated."

        final_output_lang = output_lang or input_language

        if final_output_lang != input_language:
            response = translate_text(response, final_output_lang)

        # Safe TTS
        try:
            audio_base64 = text_to_speech(response, final_output_lang)
        except Exception as e:
            print("TTS ERROR:", e)
            audio_base64 = None

        return {
            "session_id": session_id,
            "transcribed_text": transcribed_text,
            "input_language": input_language,
            "output_language": final_output_lang,
            "response_text": response,
            "audio_base64": audio_base64
        }

    except Exception:
        print("\n===== VOICE ENDPOINT ERROR =====")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Voice processing failed")

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000)
