"""Translator utilities with multiple backends.

Behavior:
- Detect language using `langdetect` if available, otherwise use script-based heuristics.
- Translate using Hugging Face Marian models where available (cached pipelines).
- Fallback to `googletrans` if transformers models are not available.

The code is defensive: missing optional dependencies won't crash the whole app,
but will return the original text (with warnings) if translation can't be performed.
"""
import re
import os
import logging
from typing import Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Map language pairs to reasonable HF models when available
_HF_MODEL_MAP = {
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("ta", "en"): "Helsinki-NLP/opus-mt-ta-en",
    ("en", "ta"): "Helsinki-NLP/opus-mt-en-ta",
    ("kn", "en"): "Helsinki-NLP/opus-mt-kn-en",
    ("en", "kn"): "Helsinki-NLP/opus-mt-en-kn",
}

_PIPELINE_CACHE = {}


def _detect_language_langdetect(text: str) -> str:
    try:
        from langdetect import detect
        lang = detect(text)
        return lang
    except Exception as e:
        logger.info("langdetect not available or failed: %s", e)
        raise


def _detect_language_script(text: str) -> str:
    # Very lightweight script detection for Devanagari (Hindi), Tamil, Kannada
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    if re.search(r"[\u0B80-\u0BFF]", text):
        return "ta"
    if re.search(r"[\u0C80-\u0CFF]", text):
        return "kn"
    # fallback to english
    return "en"


def detect_language(text: str) -> str:
    """Detect language code for given text. Returns two-letter ISO code (hi, kn, ta, en, etc.)."""
    if not text or not text.strip():
        return "en"
    # Prefer langdetect if available
    try:
        lang = _detect_language_langdetect(text)
        return lang
    except Exception:
        return _detect_language_script(text)


def _get_pipeline(src: str, tgt: str):
    key = (src, tgt)
    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key]

    model_name = _HF_MODEL_MAP.get(key)
    if model_name:
        try:
            from transformers import pipeline
            pipe = pipeline("translation", model=model_name)
            _PIPELINE_CACHE[key] = pipe
            return pipe
        except Exception as e:
            logger.info("Failed to load HF pipeline %s: %s", model_name, e)

    # No HF pipeline available for this pair
    return None


def _googletrans_translate(text: str, src: str, tgt: str) -> str:
    try:
        from googletrans import Translator
        t = Translator()
        # googletrans auto-detects; specify src/tgt when possible
        res = t.translate(text, src=src if src != "en" else "auto", dest=tgt)
        return res.text
    except Exception as e:
        logger.info("googletrans fallback failed: %s", e)
        raise


def translate_to_english(text: str, src_lang: str) -> str:
    """Translate `text` from `src_lang` into English. Returns English text (or original on failure)."""
    if src_lang == "en":
        return text

    # Try HF model
    pipe = _get_pipeline(src_lang, "en")
    if pipe is not None:
        try:
            out = pipe(text)
            if isinstance(out, list) and out:
                return out[0].get("translation_text") or list(out[0].values())[0]
            return str(out)
        except Exception as e:
            logger.info("HF translation failed: %s", e)

    # Fallback to googletrans
    try:
        return _googletrans_translate(text, src_lang, "en")
    except Exception:
        logger.warning("Translation to English unavailable; returning original text.")
        return text


def translate_from_english(text: str, tgt_lang: str) -> str:
    """Translate `text` from English into `tgt_lang`. Returns translated text or original on failure."""
    if tgt_lang == "en":
        return text

    pipe = _get_pipeline("en", tgt_lang)
    if pipe is not None:
        try:
            out = pipe(text)
            if isinstance(out, list) and out:
                return out[0].get("translation_text") or list(out[0].values())[0]
            return str(out)
        except Exception as e:
            logger.info("HF translation (en->%s) failed: %s", tgt_lang, e)

    try:
        return _googletrans_translate(text, "en", tgt_lang)
    except Exception:
        logger.warning("Translation from English unavailable; returning original English text.")
        return text


def detect_and_translate(text: str) -> Tuple[str, str]:
    """Detect user language and translate the text into English.

    Returns `(detected_lang, english_text)`.
    """
    lang = detect_language(text)
    en = translate_to_english(text, lang)
    return lang, en


if __name__ == "__main__":
    samples = [
        "हेलो, क्या आप मेरी मदद कर सकते हैं?",
        "ನमಸ್ಕಾರ, ನನ್ನ ಸಹಾಯ ಮಾಡಬಹುದು?",
        "வணக்கம், உதவி தேவை",
    ]
    for s in samples:
        l, e = detect_and_translate(s)
        print(l, "->", e)
