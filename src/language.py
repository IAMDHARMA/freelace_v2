# language.py

from langdetect import detect
from langchain_google_genai import ChatGoogleGenerativeAI

def detect_language(text):
    return detect(text)


def translate_text(text, target_lang):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0
    )

    prompt = f"Translate the following text to {target_lang}:\n{text}"

    response = llm.invoke(prompt)

    return response.content
