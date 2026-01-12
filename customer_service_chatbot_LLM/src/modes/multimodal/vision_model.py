"""Vision utilities for multimodal chatbot.

This module provides lightweight OCR, optional image captioning, and a
placeholder for image generation. The implementations are intentionally
non-blocking: heavy dependencies are imported only when needed and clear
error messages guide the user on how to enable features.
"""
import os
import logging
from typing import Optional

try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        class Document:
            def __init__(self, page_content, metadata=None):
                self.page_content = page_content
                self.metadata = metadata or {}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def extract_text_from_image(image_path: str) -> str:
    """Extract text from an image using pytesseract.

    Raises informative RuntimeError if dependencies are missing.
    """
    try:
        from PIL import Image
        import pytesseract
    except Exception as e:
        raise RuntimeError(
            "Missing OCR dependencies. Install with: `pip install pillow pytesseract` "
            "and install the Tesseract binary (https://github.com/tesseract-ocr/tesseract)."
        ) from e

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text.strip()


def caption_image(image_path: str) -> Optional[str]:
    """Generate an image caption using a transformers pipeline if available.

    Returns None if captioning dependencies are not installed.
    """
    try:
        from transformers import pipeline
    except Exception:
        logger.info("Transformers not available for image captioning; skipping caption.")
        return None

    try:
        captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    except Exception as e:
        logger.info("Failed to create caption pipeline: %s", e)
        return None

    try:
        res = captioner(image_path)
        # pipeline returns a list of dicts with 'generated_text'
        if isinstance(res, list) and res:
            return res[0].get("generated_text", None)
        return None
    except Exception as e:
        logger.info("Image captioning failed: %s", e)
        return None


def generate_image(prompt: str, output_path: Optional[str] = None, provider: str = "gemini") -> str:
    """Placeholder for image generation.

    If a provider API key (e.g. `GEMINI_API_KEY`) is available this function
    can be extended to call that provider's image generation endpoint. By
    default this function raises a clear error instructing how to enable it.
    """
    # Check for provider keys
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key is None:
        raise RuntimeError(
            "No image-generation provider configured. Set `GEMINI_API_KEY` or implement a local generator."
        )

    # Here you would add provider-specific code. For now we raise NotImplementedError
    raise NotImplementedError(
        "Image generation is not implemented in this repo. Add calls to your provider's API (e.g. Gemini) "
        "in `vision_model.generate_image` and return a path or URL to the generated image."
    )


def image_to_document(image_path: str, caption: Optional[str] = None, ocr_text: Optional[str] = None) -> Document:
    """Convert image data into a LangChain Document with metadata.

    Always includes `source` metadata pointing to the image path and optional
    `caption` and `ocr` fields.
    """
    parts = []
    if caption:
        parts.append(f"Caption: {caption}")
    if ocr_text:
        parts.append(f"OCR: {ocr_text}")
    page_content = "\n\n".join(parts) if parts else f"Image: {os.path.basename(image_path)}"
    metadata = {"source": image_path, "type": "image"}
    if caption:
        metadata["caption"] = caption
    if ocr_text:
        metadata["ocr"] = ocr_text

    return Document(page_content=page_content, metadata=metadata)


if __name__ == "__main__":
    print("vision_model helpers: extract_text_from_image, caption_image, generate_image")
