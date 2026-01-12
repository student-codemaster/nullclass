"""Medical retrieval utilities: build index from MedQuAD-like data, query chain, and entity extraction."""
import os
import json
import logging
from typing import List, Dict, Optional

try:
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
    HAS_LANGCHAIN_MED = True
except Exception:
    # provide lightweight fallbacks so the module can be imported without langchain
    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    FAISS = None
    HAS_LANGCHAIN_MED = False

import langchain_helper
from .prompts import MEDICAL_PROMPT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _load_json_or_jsonl(path: str) -> List[Document]:
    """Load a data file (json/jsonl/xml/csv) and return a list of Documents.

    Supports the MedQuAD XML layout (QAPair/Question/Answer) as well as
    JSON/JSONL lists and simple CSVs with `answer`/`response` columns.
    """
    import csv
    docs: List[Document] = []
    lower = path.lower()

    if lower.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                text = item.get("answer") or item.get("question") or json.dumps(item)
                source = item.get("source") or item.get("url") or path
                docs.append(Document(page_content=text, metadata={"source": source}))
    elif lower.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    text = item.get("answer") or item.get("question") or json.dumps(item)
                    source = item.get("source") or item.get("url") or path
                    docs.append(Document(page_content=text, metadata={"source": source}))
            else:
                docs.append(Document(page_content=json.dumps(data), metadata={"source": path}))
    elif lower.endswith(".xml"):
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(path)
            root = tree.getroot()
            # MedQuAD uses QAPair / Question / Answer structure
            for qa in root.findall('.//QAPair'):
                q_el = qa.find('.//Question')
                a_el = qa.find('.//Answer')
                qtxt = q_el.text.strip() if q_el is not None and q_el.text else ''
                atxt = ''.join(a_el.itertext()).strip() if a_el is not None else ''
                text = atxt or qtxt
                if text:
                    docs.append(Document(page_content=text, metadata={"source": path}))
            # Fallback: also try searching for Answer elements
            if not docs:
                for a_el in root.findall('.//Answer'):
                    atxt = ''.join(a_el.itertext()).strip()
                    if atxt:
                        docs.append(Document(page_content=atxt, metadata={"source": path}))
        except Exception:
            # On parse failure, include raw file content
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    txt = f.read()
                    docs.append(Document(page_content=txt, metadata={"source": path}))
            except Exception:
                pass
    elif lower.endswith('.csv'):
        try:
            with open(path, 'r', encoding='utf-8', errors='replace', newline='') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    text = (r.get('answer') or r.get('response') or r.get('text') or r.get('Answer') or '').strip()
                    source = r.get('source') or path
                    if text:
                        docs.append(Document(page_content=text, metadata={"source": source}))
        except Exception:
            pass
    else:
        # Unknown extension: try to read raw
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                docs.append(Document(page_content=f.read(), metadata={"source": path}))
        except Exception:
            pass

    return docs


def build_medical_index(dataset_path: str, index_dir: str = "faiss_index_medical") -> None:
    """Build or re-build a FAISS index for medical data from a MedQuAD JSON/JSONL file.

    dataset_path: path to a JSON/JSONL file or a directory containing such files.
    index_dir: folder to store FAISS index.
    """
    docs: List[Document] = []
    if os.path.isdir(dataset_path):
        for root, _, files in os.walk(dataset_path):
            for fn in files:
                p = os.path.join(root, fn)
                if fn.lower().endswith((".json", ".jsonl", ".xml", ".csv")):
                    docs.extend(_load_json_or_jsonl(p))
    else:
        docs.extend(_load_json_or_jsonl(dataset_path))

    if not docs:
        logger.warning("No medical documents found in %s", dataset_path)
        return

    logger.info("Building medical index at %s with %d docs", index_dir, len(docs))
    if FAISS is None:
        logger.warning("FAISS/langchain not available; skipping FAISS index build. Consider installing langchain_community/faiss.")
        return
    vectordb = FAISS.from_documents(documents=docs, embedding=langchain_helper.instructor_embeddings)
    vectordb.save_local(index_dir)
    logger.info("Medical index saved.")


def get_medical_chain(index_dir: str = "faiss_index_medical"):
    """Load the medical index and return a RetrievalQA chain configured with medical prompt.

    If LangChain/FAISS are unavailable, this function falls back to the generic
    `langchain_helper.get_qa_chain()` which provides a CSV-based retriever.
    """
    # If FAISS/langchain not available, reuse generic helper fallback
    if not HAS_LANGCHAIN_MED or FAISS is None:
        return langchain_helper.get_qa_chain()

    if not os.path.exists(index_dir):
        raise FileNotFoundError(f"Medical index not found at {index_dir}. Build it first.")

    vectordb = FAISS.load_local(index_dir, langchain_helper.instructor_embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # Lazy import to avoid hard dependency at module import time
    try:
        from langchain_community.chains import RetrievalQA
    except Exception:
        # fallback to generic chain
        return langchain_helper.get_qa_chain()

    chain = RetrievalQA.from_chain_type(
        llm=langchain_helper.llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": MEDICAL_PROMPT},
    )

    return chain


def extract_medical_entities(text: str) -> Dict[str, List[str]]:
    """Basic medical entity extraction with spaCy fallback to keyword matching.

    Returns dict with keys 'symptoms', 'diseases', 'treatments'.
    """
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        ents = [ent.text for ent in doc.ents]
        # Very rough mapping by keywords in entity text
        symptoms = [e for e in ents if e.lower() in _COMMON_SYMPTOMS]
        diseases = [e for e in ents if e.lower() in _COMMON_DISEASES]
        treatments = [e for e in ents if e.lower() in _COMMON_TREATMENTS]
        return {"symptoms": symptoms, "diseases": diseases, "treatments": treatments}
    except Exception:
        # Fallback keyword matching
        text_l = text.lower()
        symptoms = [s for s in _COMMON_SYMPTOMS if s in text_l]
        diseases = [d for d in _COMMON_DISEASES if d in text_l]
        treatments = [t for t in _COMMON_TREATMENTS if t in text_l]
        return {"symptoms": symptoms, "diseases": diseases, "treatments": treatments}


# Small keyword lists for fallback entity extraction (extendable)
_COMMON_SYMPTOMS = set([
    "fever",
    "cough",
    "headache",
    "nausea",
    "fatigue",
    "shortness of breath",
])

_COMMON_DISEASES = set([
    "diabetes",
    "hypertension",
    "asthma",
    "covid-19",
    "covid",
    "flu",
])

_COMMON_TREATMENTS = set([
    "ibuprofen",
    "acetaminophen",
    "insulin",
    "antibiotics",
    "rest",
])


if __name__ == "__main__":
    print("Medical utilities: build_medical_index, get_medical_chain, extract_medical_entities")
