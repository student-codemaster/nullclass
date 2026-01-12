class SentimentClassifier:
    """Minimal fallback sentiment classifier to avoid import errors."""
    def __init__(self):
        pass

    def predict(self, text: str) -> dict:
        t = (text or "").lower()
        if any(w in t for w in ("good", "great", "awesome", "happy", "excellent")):
            label = "positive"
            score = 0.9
        elif any(w in t for w in ("bad", "terrible", "sad", "angry", "awful")):
            label = "negative"
            score = 0.9
        else:
            label = "neutral"
            score = 0.6
        return {"label": label, "score": score}