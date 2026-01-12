"""Handler to integrate image and text inputs with the QA chain and index.

This module provides `handle_multimodal_query` which accepts an optional image
path and question text, optionally adds image-derived content to the FAISS
index, and queries the existing QA chain from `langchain_helper`.
"""
import os
import logging
from typing import Optional, Dict, Any

try:
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    HAS_LANGCHAIN_MULTI = True
except Exception:
    FAISS = None
    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    HAS_LANGCHAIN_MULTI = False

import langchain_helper
from .vision_model import (
    extract_text_from_image,
    caption_image,
    generate_image,
    image_to_document,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _add_documents_to_index(docs: list[Document], index_dir: Optional[str] = None):
    index_dir = index_dir or langchain_helper.vectordb_file_path
    if os.path.exists(index_dir):
        vectordb = FAISS.load_local(index_dir, langchain_helper.instructor_embeddings)
        vectordb.add_documents(docs)
        vectordb.save_local(index_dir)
    else:
        vectordb = FAISS.from_documents(documents=docs, embedding=langchain_helper.instructor_embeddings)
        vectordb.save_local(index_dir)


def handle_multimodal_query(
    question: Optional[str] = None,
    image_path: Optional[str] = None,
    add_to_index: bool = False,
    generate_image_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Process a multimodal user input and return a QA response.

    Returns a dict with keys: `response` (the chain output dict), and optional
    `generated_image` (path/URL) and `image_insights` (caption/ocr).
    """
    image_insights = {}
    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            ocr = extract_text_from_image(image_path)
        except Exception as e:
            ocr = None
            logger.info("OCR skipped/failed: %s", e)

        try:
            caption = caption_image(image_path)
        except Exception as e:
            caption = None
            logger.info("Captioning skipped/failed: %s", e)

        image_insights = {"caption": caption, "ocr": ocr}

        if add_to_index:
            doc = image_to_document(image_path, caption=caption, ocr_text=ocr)
            try:
                _add_documents_to_index([doc])
            except Exception as e:
                logger.exception("Failed to add image document to index: %s", e)

    # Build query text combining image insights and question
    pieces = []
    if image_insights.get("caption"):
        pieces.append(f"Image caption: {image_insights['caption']}")
    if image_insights.get("ocr"):
        pieces.append(f"Image OCR text: {image_insights['ocr']}")
    if question:
        pieces.append(f"Question: {question}")

    query_text = "\n\n".join(pieces).strip() or ""

    # Query the QA chain
    chain = langchain_helper.get_qa_chain()
    response = chain(query_text)

    result: Dict[str, Any] = {"response": response, "image_insights": image_insights}

    if generate_image_prompt:
        try:
            img_ref = generate_image(generate_image_prompt)
            result["generated_image"] = img_ref
        except NotImplementedError as e:
            logger.info("Image generation not implemented: %s", e)
            result["generated_image_error"] = str(e)
        except Exception as e:
            logger.exception("Image generation failed: %s", e)
            result["generated_image_error"] = str(e)

    return result


if __name__ == "__main__":
    # Simple smoke test (no heavy dependencies required for OCR/caption unless used)
    sample_image = None
    res = handle_multimodal_query(question="What is this?", image_path=sample_image)
    print(res)
