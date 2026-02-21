# src/language.py

from langdetect import detect, DetectorFactory
from .llm import llm

DetectorFactory.seed = 0  # Consistent detection

# Allowed languages in your tutor system
SUPPORTED_LANGS = {"en", "ta", "hi"}

LANG_NAMES = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
}


def get_language_name(lang_code: str) -> str:
    """Convert a BCP-47 code to a full language name."""
    return LANG_NAMES.get(lang_code, lang_code)


def is_tamil(text: str) -> bool:
    """Check if Tamil characters are present."""
    return any("\u0B80" <= ch <= "\u0BFF" for ch in text)


def detect_language(text: str) -> str:
    """
    Only detect English / Tamil / Hindi.
    Block Spanish, French, German, etc.
    """

    text_l = text.lower().strip()

    # ----- Tamil Unicode detection -----
    if any("\u0B80" <= ch <= "\u0BFF" for ch in text):
        return "ta"

    # ----- Tamil keyword detection -----
    tamil_words = ["enna", "epdi", "solunga", "ungal", "enaku", "podu"]
    if any(w in text_l for w in tamil_words):
        return "ta"

    # ----- Hindi keyword detection -----
    hindi_words = ["kya", "kaise", "mera", "tum", "hain", "kartaa"]
    if any(w in text_l for w in hindi_words):
        return "hi"

    # ----- Block langdetect for short English text -----
    if len(text_l.split()) < 5:
        return "en"  # prevents Spanish detection

    # ----- Safe langdetect for longer sentences -----
    try:
        lang = detect(text_l)
        if lang in ["en", "ta", "hi"]:
            return lang
        return "en"   # BLOCK es, fr, de
    except:
        return "en"


def translate_text(text: str, target_lang: str) -> str:
    """Translate text to target language using the LLM."""
    lang_name = get_language_name(target_lang)

    prompt = f"""
You are a professional translator.

Translate the following text into {lang_name}.
Return ONLY the translated text — no notes, no explanations.

Text:
{text}
"""

    response = llm.invoke(prompt)
    return response.content.strip()