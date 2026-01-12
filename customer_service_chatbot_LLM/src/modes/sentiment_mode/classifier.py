"""Sentiment classifier and response tone adjuster.

Exposes `SentimentClassifier` which can `classify(text)` and
`adjust_response(response, sentiment, use_llm=False)` to rewrite assistant
responses with an appropriate tone. If `use_llm=True`, it will call
`langchain_helper.llm` to rewrite the text (requires API key and may cost).
"""
from typing import Dict, Optional
import logging

import langchain_helper
from .sentiment_model import classify_sentiment

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SentimentClassifier:
    def __init__(self, use_llm_for_rewrite: bool = False):
        self.use_llm_for_rewrite = use_llm_for_rewrite

    def classify(self, text: str) -> Dict[str, float]:
        return classify_sentiment(text)

    def adjust_response(self, response: str, sentiment: Dict[str, float], user_text: Optional[str] = None) -> str:
        """Adjust the assistant response according to detected sentiment.

        If `use_llm_for_rewrite` is True, the function will call the configured
        LLM from `langchain_helper` to rewrite the response in a requested tone.
        Otherwise it applies lightweight template-based modifications.
        """
        label = sentiment.get("label", "neutral")
        score = sentiment.get("score", 0.0)

        if self.use_llm_for_rewrite:
            # Create a small instruction to the LLM to rewrite the response
            tone_instruction = {
                "negative": "Rewrite the response to be empathetic and reassuring while providing the same information.",
                "positive": "Rewrite the response to be enthusiastic and affirming while preserving the information.",
                "neutral": "Rewrite the response to be clear and professional while preserving the information.",
            }.get(label, "Rewrite the response to be clear and professional.")

            prompt = (
                f"User message: {user_text or ''}\n\n"
                f"Assistant draft: {response}\n\n"
                f"Instruction: {tone_instruction}\n\nRewritten response:")

            try:
                out = langchain_helper.llm(prompt)
                # Attempt to extract text
                if isinstance(out, dict):
                    return out.get("result") or out.get("text") or str(out)
                return str(out)
            except Exception as e:
                logger.info("LLM rewrite failed: %s, falling back to template.", e)

        # Template-based adjustments (cheap, deterministic)
        if label == "negative":
            prefix = "I'm sorry you're experiencing this. I understand this can be frustrating. "
            # If very negative, add escalation suggestion
            if score is not None and isinstance(score, (int, float)) and score < -0.6:
                prefix += "If you'd like, I can escalate this to our support team or provide next steps. "
            return prefix + response

        if label == "positive":
            prefix = "That's great to hear! "
            return prefix + response

        # neutral
        return response


if __name__ == "__main__":
    sc = SentimentClassifier()
    sample = "I'm really unhappy with the service, it broke on me"
    s = sc.classify(sample)
    print(s)
    print(sc.adjust_response("Here's what you can try: restart the app.", s, user_text=sample))
