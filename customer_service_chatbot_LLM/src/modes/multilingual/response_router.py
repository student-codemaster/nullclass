"""Response router and cultural adapter for multilingual replies.

Provides helpers to wrap or adjust responses to be culturally appropriate for
Hindi, Kannada, and Tamil users. Uses `translator` to convert between English
and target language and adds polite prefixes/suffixes.
"""
import logging
from typing import Dict

from .translator import translate_from_english, detect_language

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


_GREETING_MAP = {
    "hi": "नमस्ते",
    "ta": "வணக்கம்",
    "kn": "ನಮಸ್ಕಾರ",
    "en": "Hello",
}

_POLITE_PREFIX = {
    "hi": "कृपया ध्यान दें:",
    "ta": "குறிப்பாக கவனிக்கவும்:",
    "kn": "ದಯವಿಟ್ಟು ಗಮನಿಸಿ:",
    "en": "Please note:",
}


def get_greeting_for_lang(lang: str) -> str:
    return _GREETING_MAP.get(lang, _GREETING_MAP["en"])


def adapt_response_for_culture(response_en: str, target_lang: str, user_input: str = None) -> Dict[str, str]:
    """Adapt an English response to the cultural/linguistic style of `target_lang`.

    Returns dict with keys: `language`, `translated_response`, `prefixed_response`.
    """
    # Translate to target language
    translated = translate_from_english(response_en, target_lang)

    # Add polite prefix appropriate for the language
    prefix = _POLITE_PREFIX.get(target_lang, _POLITE_PREFIX["en"])

    # Some languages prefer the prefix before the sentence; we construct sensibly
    prefixed = f"{prefix} {translated}"

    return {"language": target_lang, "translated_response": translated, "prefixed_response": prefixed}


if __name__ == "__main__":
    en = "We recommend restarting the application and checking your internet connection."
    for lang in ("hi", "ta", "kn"):
        out = adapt_response_for_culture(en, lang)
        print(lang, out)
