"""Sentiment detection utilities with multiple fallbacks.

Primary method: Hugging Face `pipeline('sentiment-analysis')` if available.
Fallback: `vaderSentiment` analyzer. Final fallback: simple keyword heuristic.

Provides `classify_sentiment(text)` returning a dict: `{label: str, score: float}`
where `label` is one of `positive`, `negative`, `neutral`.
"""
from typing import Dict
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _hf_sentiment(text: str) -> Dict:
    try:
        from transformers import pipeline
    except Exception as e:
        raise RuntimeError("transformers not available") from e

    try:
        pipe = pipeline("sentiment-analysis")
        out = pipe(text[:1000])  # limit length for safety
        if isinstance(out, list) and out:
            res = out[0]
            label = res.get("label", "")
            score = float(res.get("score", 0.0))
            # Normalize common labels
            label_l = label.lower()
            if "neg" in label_l:
                return {"label": "negative", "score": score}
            if "pos" in label_l:
                return {"label": "positive", "score": score}
            # Some models use LABEL_0/1 — heuristically map by score
            return {"label": "positive" if score >= 0.5 else "negative", "score": score}
    except Exception as e:
        logger.info("HF sentiment pipeline failed: %s", e)
        raise


def _vader_sentiment(text: str) -> Dict:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except Exception as e:
        raise RuntimeError("vaderSentiment not available") from e

    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores.get("compound", 0.0)
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {"label": label, "score": compound}


_POS_WORDS = set(["good", "great", "thanks", "thank you", "awesome", "love", "happy", "excellent"])
_NEG_WORDS = set(["bad", "terrible", "hate", "unhappy", "angry", "frustrated", "disappointed", "problem"])


def _heuristic_sentiment(text: str) -> Dict:
    t = text.lower()
    pos = sum(t.count(w) for w in _POS_WORDS)
    neg = sum(t.count(w) for w in _NEG_WORDS)
    score = (pos - neg) / max(1, pos + neg)
    if pos == neg:
        label = "neutral"
    elif pos > neg:
        label = "positive"
    else:
        label = "negative"
    return {"label": label, "score": float(score)}


def classify_sentiment(text: str) -> Dict[str, float]:
    """Classify sentiment using available backends.

    Returns {'label': 'positive'|'negative'|'neutral', 'score': float}.
    Tries HF pipeline first, then Vader, then heuristic.
    """
    if not text or not text.strip():
        return {"label": "neutral", "score": 0.0}

    # Try Hugging Face pipeline
    try:
        return _hf_sentiment(text)
    except Exception:
        logger.info("HF pipeline not available or failed, trying VADER.")

    try:
        return _vader_sentiment(text)
    except Exception:
        logger.info("VADER not available, using heuristic sentiment.")

    return _heuristic_sentiment(text)


if __name__ == "__main__":
    sample = "I am very disappointed and frustrated with the product."
    print(classify_sentiment(sample))
