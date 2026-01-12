import os
import sys
import logging

# ensure project `src` is on sys.path so local modules like `langchain_helper` can be imported
project_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_src not in sys.path:
    sys.path.insert(0, project_src)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import yaml
except Exception:
    yaml = None
from typing import List

try:
    import langchain_helper
except Exception as exc:
    logger.exception("Failed to import local module 'langchain_helper': %s", exc)
    raise SystemExit("Missing or broken local module 'langchain_helper'. Ensure c:\\Users\\ASHOKA MS\\Desktop\\chatbot\\customer_service_chatbot_LLM\\src is on PYTHONPATH and langchain_helper.py exists and is importable.")

try:
    from langchain_core.documents import Document
except Exception:
    logger.warning("LangChain Document class not available; using dict fallback.")
    Document = dict


def _load_csv(path: str) -> List:
    try:
        from langchain_community.document_loaders import CSVLoader
        from langchain_core.documents import Document
    except Exception:
        logger.warning("LangChain CSV loader not available; cannot load CSV: %s", path)
        return []

    loader = CSVLoader(file_path=path, source_column="prompt")
    return loader.load()


def _load_text_file(path: str) -> List:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    try:
        from langchain_core.documents import Document
    except Exception:
        logger.warning("LangChain Document class not available; returning plain dict for %s", path)
        return [type("Doc", (), {"page_content": content, "metadata": {"source": path}})()]

    return [Document(page_content=content, metadata={"source": path})]


def _collect_documents(sources: List[str]) -> List:
    docs: List = []
    for src in sources:
        src_path = os.path.normpath(src)
        if os.path.isdir(src_path):
            for root, _, files in os.walk(src_path):
                for fn in files:
                    p = os.path.join(root, fn)
                    if fn.lower().endswith(".csv"):
                        docs.extend(_load_csv(p))
                    elif fn.lower().endswith(".txt"):
                        docs.extend(_load_text_file(p))
        else:
            if src_path.lower().endswith(".csv"):
                docs.extend(_load_csv(src_path))
            elif src_path.lower().endswith(".txt"):
                docs.extend(_load_text_file(src_path))
            else:
                logger.info("Skipping unsupported source file: %s", src_path)
    return docs


def update_vector_db(config_path: str = None):
    """Update (or create) a FAISS vector database from configured sources.

    The function attempts to load an existing FAISS index and add new documents.
    If loading fails, it will create a new index from the provided sources.
    """
    base_dir = os.path.dirname(__file__)
    if config_path is None:
        config_path = os.path.join(base_dir, "config.yaml")

    logger.info("Loading updater config: %s", config_path)
    if yaml is None:
        logger.warning("PyYAML not installed; cannot read config.yaml. Provide sources directly or install pyyaml.")
        return
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.error("Config file not found: %s", config_path)
        return

    sources = cfg.get("sources", [])
    index_dir = cfg.get("index_dir", langchain_helper.vectordb_file_path)

    # Normalize relative paths: config paths are relative to `src` when running app
    # If config uses relative path starting with .., make relative to this file's parent (src/modes/dynamic_updater)
    norm_sources = []
    for s in sources:
        if os.path.isabs(s):
            norm_sources.append(s)
        else:
            # try relative to src (two levels up from this file)
            candidate = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", s))
            if os.path.exists(candidate):
                norm_sources.append(candidate)
            else:
                # fall back to as-is
                norm_sources.append(s)

    docs = _collect_documents(norm_sources)
    if not docs:
        logger.info("No documents found to index.")
        return

    try:
        from langchain_community.vectorstores import FAISS
    except Exception:
        logger.warning("LangChain FAISS not available; skipping index update.")
        return

    try:
        if os.path.exists(index_dir):
            logger.info("Loading existing FAISS index from %s", index_dir)
            vectordb = FAISS.load_local(index_dir, langchain_helper.instructor_embeddings)
            logger.info("Adding %d documents to existing index", len(docs))
            vectordb.add_documents(docs)
            vectordb.save_local(index_dir)
        else:
            logger.info("Creating new FAISS index at %s with %d documents", index_dir, len(docs))
            vectordb = FAISS.from_documents(documents=docs, embedding=langchain_helper.instructor_embeddings)
            vectordb.save_local(index_dir)
        logger.info("Index update complete.")
    except Exception as e:
        logger.exception("Failed to update FAISS index: %s", e)


if __name__ == "__main__":
    update_vector_db()
