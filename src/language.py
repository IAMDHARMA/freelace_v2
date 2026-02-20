# src/language.py

from langdetect import detect, DetectorFactory
from .llm import llm

DetectorFactory.seed = 0  # Consistent detection across runs

# ✅ Map BCP-47 codes to full language names for LLM prompts
LANG_NAMES = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "pt": "Portuguese",
    "ru": "Russian",
}


def get_language_name(lang_code: str) -> str:
    """Convert a BCP-47 language code to a full language name for prompts."""
    return LANG_NAMES.get(lang_code, lang_code)


def detect_language(text: str) -> str:
    """
    Detect input language code (e.g. 'en', 'ta', 'hi').
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
    Translate text to target language using the shared LLM.
    Accepts BCP-47 language codes (e.g. 'ta', 'hi', 'en').
    """
    # ✅ FIXED: Use full language name in prompt, not raw code like "ta"
    lang_name = get_language_name(target_lang)

    prompt = f"""You are a professional translator.

Translate the following text into {lang_name}.
Return ONLY the translated text.
Do not add explanations, notes, or labels.

Text:
{text}
"""

    response = llm.invoke(prompt)
    return response.content.strip()