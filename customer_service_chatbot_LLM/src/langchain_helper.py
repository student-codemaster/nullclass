import os
import logging
from typing import Optional, Any
import csv
import difflib
from functools import lru_cache

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Try to import LangChain/FAISS dependencies; if unavailable, provide safe fallbacks so
# the application can run in a degraded mode without heavy 3rd-party installs.
_has_langchain = True
try:
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.document_loaders import CSVLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
except Exception:
    _has_langchain = False


# load env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# where FAISS index will be stored
vectordb_file_path = os.path.join(os.path.dirname(__file__), "faiss_index")

# Default placeholders
llm = None
embeddings = None

if _has_langchain:
    # Allow overriding the model via env var `GOOGLE_MODEL` or try a small fallback list.
    model_candidates = []
    if os.environ.get("GOOGLE_MODEL"):
        model_candidates.append(os.environ.get("GOOGLE_MODEL"))
    # Prefer smaller/stable models first to avoid unsupported model errors.
    # Users can override with the `GOOGLE_MODEL` env var (recommended).
    # Note: some models (e.g., newer "pro" models) may not be available
    # for your API version; keep them at the end of the list.
    model_candidates.extend([
        "gemini-1.0-mini",
        "gemini-1.0",
        "gemini-1.5",
        "gemini-1.5-pro",
    ])
    llm = None
    for mdl in model_candidates:
        if not mdl:
            continue
        try:
            tmp = ChatGoogleGenerativeAI(
                model=mdl,
                google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
                temperature=0.1,
            )
            llm = tmp
            logger.info("Selected Google LLM model: %s", mdl)
            break
        except Exception as e:
            logger.debug("Model %s unavailable: %s", mdl, e)

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception:
        embeddings = None


def create_vector_db():
    """Create a FAISS vector DB from the CSV dataset when LangChain is available.

    If LangChain is not installed, a placeholder directory will be created instead.
    """
    if not _has_langchain:
        os.makedirs(vectordb_file_path, exist_ok=True)
        return {"status": "noop", "message": "langchain not available; placeholder created"}

    # dataset lives under dataset/dataset.csv
    loader = CSVLoader(file_path=os.path.join(os.path.dirname(__file__), "..", "dataset", "dataset.csv"), source_column="prompt")
    documents = loader.load()

    vectordb = FAISS.from_documents(documents=documents, embedding=embeddings)
    vectordb.save_local(vectordb_file_path)
    return {"status": "ok", "message": "vector DB created"}


def get_qa_chain():
    """Return a QA chain. Prefer LangChain+FAISS; otherwise return a simple CSV-based retriever."""
    # If langchain isn't available, return the lightweight CSV retriever chain.
    if not _has_langchain:
        def _csv_chain(query: str):
            return _simple_retriever(query)

        return _csv_chain
    # ensure index exists; if missing create a stub index or return a safe fallback
    idx_path = os.path.join(vectordb_file_path, "index.faiss")
    if not os.path.exists(idx_path):
        # try to create a minimal stub index (requires faiss); if that fails, fall back to CSV retriever
        try:
            ok = ensure_index_stub()
            if not ok:
                raise FileNotFoundError(idx_path)
        except Exception:
            def _csv_chain(query: str):
                return _simple_retriever(query)

            return _csv_chain

    try:
        vectordb = FAISS.load_local(
            vectordb_file_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception:
        def _csv_chain(query: str):
            return _simple_retriever(query)

        return _csv_chain

    # If no LLM was successfully created, fall back to CSV retriever only.
    if llm is None:
        def _csv_chain(query: str):
            return _simple_retriever(query)

        return _csv_chain

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate.from_template(
        """
Context:
{context}

Question:
{question}
"""
    )

    chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm)

    # Wrap Runnable/Chain into a callable that returns a dict with keys
    
    def _call_chain(query: str):
        try:
            # RunnableSequence (newer langchain) supports `invoke` with inputs
            if hasattr(chain, "invoke"):
                try:
                    res = chain.invoke({"question": query, "context": retriever})
                except Exception:
                    # some runnables accept raw string
                    res = chain.invoke(query)
                # Try to normalize result
                if isinstance(res, dict):
                    return res
                # Some runnables return objects with `content` or `text`
                text = getattr(res, "content", None) or getattr(res, "text", None) or str(res)
                return {"result": text, "source_documents": []}

            # LangChain Chain objects often expose `run`
            if hasattr(chain, "run"):
                out = chain.run(query)
                if isinstance(out, dict):
                    return out
                return {"result": out, "source_documents": []}

            # Fallback: if callable, call it
            if callable(chain):
                out = chain(query)
                if isinstance(out, dict):
                    return out
                return {"result": out, "source_documents": []}

            return {"result": "Unsupported chain type", "source_documents": []}
        except Exception as e:
            # Detect common model/API NOT_FOUND errors and fall back to CSV retriever
            msg = str(e)
            logger.warning("Chain execution raised exception: %s", msg)
            if "NOT_FOUND" in msg or "not found" in msg.lower() or "models/" in msg:
                    # Provide a clearer hint to the operator about fixing model selection.
                    logger.info(
                        "Model appears unavailable for this API/version. "
                        "Set the environment variable GOOGLE_MODEL to a supported model, "
                        "or choose a smaller/stable model like 'gemini-1.0-mini'. "
                        "Falling back to simple CSV retriever for this query."
                    )
                    try:
                        return _simple_retriever(query)
                    except Exception:
                        return {"result": "Model unavailable and CSV fallback failed.", "source_documents": []}
            # generic fallback to answer_query or return error
            try:
                fb = answer_query(query)
                return {"result": fb.get("answer", str(e)), "source_documents": fb.get("sources", [])}
            except Exception:
                return {"result": f"Chain execution error: {e}", "source_documents": []}

    return _call_chain


# Minimal fallback implementations so other modules can import `langchain_helper`
def create_knowledge_base(*args, **kwargs):
    os.makedirs(vectordb_file_path, exist_ok=True)
    return {"status": "noop", "message": "knowledge base placeholder created"}


def load_index(path: str = "faiss_index/index.faiss") -> bool:
    return os.path.exists(path)


def answer_query(query: str, top_k: int = 3) -> dict:
    if not query:
        return {"answer": "", "sources": []}
    return {"answer": f"Echo reply: {query}", "sources": ["stub"]}


# Lightweight CSV-based retriever fallback (no external deps).
@lru_cache(maxsize=1)
def _load_simple_dataset() -> list:
    path = os.path.join(os.path.dirname(__file__), "..", "dataset", "dataset.csv")
    path = os.path.normpath(path)
    rows = []
    try:
        # Open with newline='' so the csv module correctly handles quoted newlines
        with open(path, "r", encoding="utf-8", errors='replace', newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if not r:
                    continue
                prompt = (r.get("prompt") or r.get("question") or "").strip()
                response = (r.get("response") or r.get("answer") or "").strip()
                if prompt or response:
                    rows.append({"prompt": prompt, "response": response, "source": path})
    except Exception:
        return []
    return rows


def _simple_retriever(query: str, top_k: int = 3) -> dict:
    """Return a simple retrieval result using difflib on the `prompt` column.

    Returns a dict with keys `result` and `source_documents` to mimic the
    expected structure used elsewhere in the codebase.
    """
    data = _load_simple_dataset()
    if not data:
        return {"result": "No dataset available for simple retrieval.", "source_documents": []}

    prompts = [d["prompt"] for d in data]
    matches = difflib.get_close_matches(query, prompts, n=top_k, cutoff=0.1)
    docs = []
    texts = []
    for m in matches:
        for d in data:
            if d["prompt"] == m:
                texts.append(d["response"])
                docs.append(type("Doc", (), {"page_content": d["response"], "metadata": {"source": d.get("source")}})())
                break

    if texts:
        # simple concatenation of top responses
        return {"result": "\n\n".join(texts), "source_documents": docs}
    else:
        # fallback: return the single best fuzzy match by ratio
        best = None
        best_ratio = 0.0
        for d in data:
            ratio = difflib.SequenceMatcher(None, query, d["prompt"]).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best = d
        if best and best_ratio > 0.1:
            doc = type("Doc", (), {"page_content": best["response"], "metadata": {"source": best.get("source")}})()
            return {"result": best["response"], "source_documents": [doc]}
        return {"result": "No good match found in dataset.", "source_documents": []}


    def _append_to_dataset(rows: list):
        """Append rows (list of dicts with keys 'prompt' and 'response') to dataset/dataset.csv."""
        ds_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
        os.makedirs(ds_dir, exist_ok=True)
        path = os.path.join(ds_dir, "dataset.csv")
        write_header = not os.path.exists(path)
        try:
            with open(path, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["prompt", "response"])
                for r in rows:
                    prompt = (r.get("prompt") or "").replace("\n", " ").strip()
                    response = (r.get("response") or "").strip()
                    writer.writerow([prompt, response])
            # clear loader cache so new rows are seen
            try:
                _load_simple_dataset.cache_clear()
            except Exception:
                pass
            return True
        except Exception:
            return False


    def ingest_uploaded_files(uploaded_files) -> dict:
        """Ingest uploaded files (streamlit UploadedFile objects or file-like tuples).

        Accepts an iterable of objects with `.name` and `.getvalue()` (streamlit) or
        (filename, bytes) tuples. Supports .csv, .json, .xml files. Returns dict with
        counts and failures.
        """
        results = {"ingested": 0, "failed": 0, "files": []}
        for uf in uploaded_files:
            try:
                if hasattr(uf, "getvalue"):
                    name = uf.name
                    data = uf.getvalue()
                elif isinstance(uf, (list, tuple)) and len(uf) >= 2:
                    name = uf[0]
                    data = uf[1]
                else:
                    continue

                name_lower = name.lower()
                rows = []
                if name_lower.endswith(".csv"):
                    # write file to dataset directory and let loader pick it up
                    # also append rows to main dataset
                    text = data.decode("utf-8", errors="replace")
                    reader = csv.DictReader(text.splitlines())
                    for r in reader:
                        prompt = r.get("prompt") or r.get("question") or ""
                        response = r.get("response") or r.get("answer") or ""
                        if prompt or response:
                            rows.append({"prompt": prompt, "response": response})
                elif name_lower.endswith(".json"):
                    import json
                    text = data.decode("utf-8", errors="replace")
                    obj = json.loads(text)
                    if isinstance(obj, list):
                        for it in obj:
                            if isinstance(it, dict):
                                p = it.get("prompt") or it.get("question") or ""
                                r = it.get("response") or it.get("answer") or ""
                                if p or r:
                                    rows.append({"prompt": p, "response": r})
                    elif isinstance(obj, dict):
                        # single mapping
                        p = obj.get("prompt") or obj.get("question") or ""
                        r = obj.get("response") or obj.get("answer") or ""
                        if p or r:
                            rows.append({"prompt": p, "response": r})
                elif name_lower.endswith(".xml"):
                    try:
                        import xml.etree.ElementTree as ET
                        text = data.decode("utf-8", errors="replace")
                        root = ET.fromstring(text)
                        # support MedQuAD structure: QAPairs/QAPair/Question and Answer
                        for qa in root.findall('.//QAPair'):
                            q_el = qa.find('.//Question')
                            a_el = qa.find('.//Answer')
                            qtxt = q_el.text.strip() if q_el is not None and q_el.text else ''
                            atxt = ''.join(a_el.itertext()).strip() if a_el is not None else ''
                            if qtxt or atxt:
                                rows.append({"prompt": qtxt, "response": atxt})
                    except Exception:
                        # fallback: try to find Focus and Answer pairs
                        try:
                            import xml.etree.ElementTree as ET
                            text = data.decode("utf-8", errors="replace")
                            root = ET.fromstring(text)
                            focus = root.find('.//Focus')
                            if focus is not None and focus.text:
                                # search for all Answer elements
                                for a_el in root.findall('.//Answer'):
                                    atxt = ''.join(a_el.itertext()).strip()
                                    if atxt:
                                        rows.append({"prompt": focus.text.strip(), "response": atxt})
                        except Exception:
                            pass
                else:
                    # unknown extension - save raw file in dataset dir
                    rows = []

                ok = False
                if rows:
                    ok = _append_to_dataset(rows)

                results["files"].append({"name": name, "rows": len(rows), "ok": bool(ok)})
                if ok:
                    results["ingested"] += len(rows)
                else:
                    results["failed"] += 1
            except Exception:
                results["failed"] += 1
        return results


def ensure_index_stub(d: int = 384):
    try:
        import faiss
        os.makedirs(vectordb_file_path, exist_ok=True)
        idx_path = os.path.join(vectordb_file_path, "index.faiss")
        if not os.path.exists(idx_path):
            index = faiss.IndexFlatL2(d)
            faiss.write_index(index, idx_path)
        return os.path.exists(idx_path)
    except Exception:
        return False


# Try to create an optional instructor_embeddings object; fall back to None.
try:
    if _has_langchain:
        from langchain.embeddings import OpenAIEmbeddings
        instructor_embeddings = OpenAIEmbeddings()
    else:
        instructor_embeddings = None
except Exception:
    instructor_embeddings = None
