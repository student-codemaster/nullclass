def detect_and_translate(text: str, target_lang: str = "en"):
    """
    Minimal fallback: naive detection and passthrough translation.
    Returns (detected_language, translated_text).
    """
    if not text:
        return ("und", "")
    try:
        text.encode("ascii")
        detected = "en"
    except UnicodeEncodeError:
        detected = "other"
    return (detected, text)


def adapt_response_for_culture(response: str, user_locale: str = "en"):
    """
    Minimal adaptation placeholder — returns response unchanged for now.
    """
    if not response:
        return response
    return response