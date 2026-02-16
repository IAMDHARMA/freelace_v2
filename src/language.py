# src/language.py

from langdetect import detect, DetectorFactory
from .llm import llm  # ✅ Import from shared LLM file

DetectorFactory.seed = 0  # consistent detection


def detect_language(text: str) -> str:
    """
    Detect input language.
    Short text defaults to English to avoid misclassification.
    """
    if len(text.strip().split()) < 3:
        return "en"

    try:
        return detect(text)
    except Exception:
        return "en"


def translate_text(text: str, target_lang: str) -> str:
    """
    Translate text to target language using shared LLM.
    """

    prompt = f"""
You are a professional translator.

Translate the following text into {target_lang}.
Return ONLY the translated text.
Do not add explanations.

Text:
{text}
"""

    response = llm.invoke(prompt)
    return response.content.strip()
