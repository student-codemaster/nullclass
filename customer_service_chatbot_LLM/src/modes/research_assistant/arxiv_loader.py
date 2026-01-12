
import os
import logging
from typing import List

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
except Exception:
    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    FAISS = None

import langchain_helper

from typing import Optional
import zipfile
import io

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_arxiv_csv(path: str, subset_category: str = None) -> List[Document]:
    """Load arXiv CSV and return a list of Documents filtered by category (optional).

    The CSV is expected to contain columns `title` and `abstract`. If `categories`
    column is present and `subset_category` is provided, filter rows that contain
    the category token.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"ArXiv CSV not found: {path}")

    if pd is None:
        raise RuntimeError("pandas is required to load arXiv CSVs. Install pandas to use this feature.")

    df = pd.read_csv(path)
    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    title_col = cols.get("title")
    abstract_col = cols.get("abstract")
    cat_col = cols.get("categories") or cols.get("category")

    if title_col is None or abstract_col is None:
        raise ValueError("CSV must contain `title` and `abstract` columns")

    if subset_category and cat_col:
        df = df[df[cat_col].astype(str).str.contains(subset_category, case=False, na=False)]

    docs: List[Document] = []
    for _, row in df.iterrows():
        title = str(row[title_col])
        abstract = str(row[abstract_col])
        content = f"Title: {title}\n\nAbstract: {abstract}"
        metadata = {}
        # copy some useful metadata if present
        for key in ("id", "authors", "pdf_url", "url", "categories"):
            if key in row and not pd.isna(row[key]):
                metadata[key] = row[key]
        docs.append(Document(page_content=content, metadata=metadata))

    logger.info("Loaded %d documents from %s", len(docs), path)
    return docs


def build_arxiv_index(csv_path: str, index_dir: str = "faiss_index_arxiv", subset_category: str = None):
    docs = load_arxiv_csv(csv_path, subset_category=subset_category)
    if not docs:
        logger.warning("No documents to index for arXiv dataset at %s", csv_path)
        return

    logger.info("Building arXiv FAISS index at %s with %d docs", index_dir, len(docs))
    if FAISS is None:
        logger.warning("FAISS/langchain not available; skipping arXiv index build.")
        return
    vectordb = FAISS.from_documents(documents=docs, embedding=langchain_helper.instructor_embeddings)
    vectordb.save_local(index_dir)
    logger.info("ArXiv index saved to %s", index_dir)


def download_arxiv_from_kaggle(dataset: str = "Cornell-University/arxiv", dest_dir: str = "data/kaggle_arxiv") -> Optional[str]:
    """Download the specified Kaggle dataset and return a path to the main CSV file.

    Attempts to use the `kaggle` package (Kaggle API). If not available, tries
    `kagglehub` if the environment contains it. The function unzips dataset files
    into `dest_dir` and returns the first CSV file found, or None on failure.
    """
    os.makedirs(dest_dir, exist_ok=True)

    # First, try Kaggle API
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
        api = KaggleApi()
        api.authenticate()
        logger.info("Downloading Kaggle dataset %s to %s", dataset, dest_dir)
        api.dataset_download_files(dataset, path=dest_dir, unzip=True, quiet=True)
    except Exception:
        # try kagglehub (user provided snippet)
        try:
            import importlib
            kagglehub = importlib.import_module("kagglehub")
            logger.info("Using kagglehub to download %s", dataset)
            path = kagglehub.dataset_download(dataset)
            # if kagglehub returns a zip bytes-like object, try to extract
            if isinstance(path, (bytes, bytearray)):
                with zipfile.ZipFile(io.BytesIO(path)) as z:
                    z.extractall(dest_dir)
            elif isinstance(path, str) and zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as z:
                    z.extractall(dest_dir)
        except Exception as exc:
            logger.warning("Kaggle download failed (%s) — ensure kaggle API or kagglehub is installed and configured.", exc)
            return None

    # find a CSV in dest_dir
    for root, _, files in os.walk(dest_dir):
        for fn in files:
            if fn.lower().endswith(".csv"):
                return os.path.join(root, fn)
    return None


if __name__ == "__main__":
    print("ArXiv loader module. Use build_arxiv_index(csv_path, index_dir, subset_category)")
    # Example: attempt to download from Kaggle and build index
    kaggle_csv = download_arxiv_from_kaggle()
    if kaggle_csv:
        build_arxiv_index(kaggle_csv)
    else:
        logger.info("No Kaggle CSV found or download failed. Provide csv_path to build_arxiv_index manually.")
