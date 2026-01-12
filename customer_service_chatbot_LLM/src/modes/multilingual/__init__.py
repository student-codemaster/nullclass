"""Multilingual utilities: translator and response router for supported languages.

Supports automatic language detection for Hindi (`hi`), Kannada (`kn`), Tamil (`ta`) and English (`en`).
Provides translation functions (to/from English) and a cultural response adapter.
"""

from .translator import detect_language, detect_and_translate, translate_from_english, translate_to_english
from .response_router import adapt_response_for_culture, get_greeting_for_lang

__all__ = [
    "detect_language",
    "detect_and_translate",
    "translate_from_english",
    "translate_to_english",
    "adapt_response_for_culture",
    "get_greeting_for_lang",
]
