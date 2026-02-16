# app.py
from fastapi import FastAPI, UploadFile, File
from src.rag import get_qa_chain
from src.speech import speech_to_text, text_to_speech
from src.language import detect_language, translate_text
import shutil
import os

app = FastAPI()
qa_chain = get_qa_chain()


# TEXT QUERY
@app.post("/ask-text")
async def ask_text(question: str, output_lang: str = None):

    input_lang = detect_language(question)

    # ✅ LangChain 1.x way
    result = qa_chain.invoke(question)
    response = result.content

    if output_lang and output_lang != input_lang:
        response = translate_text(response, output_lang)

    audio_file = text_to_speech(response)

    return {
        "input_language": input_lang,
        "output_language": output_lang or input_lang,
        "response_text": response,
        "audio_file": audio_file
    }


# VOICE QUERY
@app.post("/ask-voice")
async def ask_voice(file: UploadFile = File(...), output_lang: str = None):

    temp_file = "input.wav"

    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # STT
    transcribed_text = speech_to_text(temp_file)

    input_lang = detect_language(transcribed_text)

    # ✅ LangChain 1.x way
    result = qa_chain.invoke(transcribed_text)
    response = result.content

    if output_lang and output_lang != input_lang:
        response = translate_text(response, output_lang)

    audio_file = text_to_speech(response)

    os.remove(temp_file)

    return {
        "transcribed_text": transcribed_text,
        "input_language": input_lang,
        "output_language": output_lang or input_lang,
        "response_text": response,
        "audio_file": audio_file
    }
