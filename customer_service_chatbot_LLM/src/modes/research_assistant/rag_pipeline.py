"""RAG pipeline for research assistant: retrieval, summarization and explanation using an open-source LLM."""
import os
import logging
from typing import Optional, List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
logging.basicConfig(level=logging.INFO)


def _get_local_llm(model_name: str = "google/flan-t5-small"):
    """Attempt to create a simple HuggingFace pipeline-backed LLM for explanations.

    This function requires `transformers` and optionally `torch` installed. If the
    dependencies are missing, it raises an informative RuntimeError.
    """
    # Configure transformers/torch logging to reduce verbose output that can appear
    # during model loading (these settings are safe to set repeatedly).
    try:
        import logging
        import os
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "0")
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("torch").setLevel(logging.ERROR)
    except Exception:
        pass

    try:
        from transformers import pipeline
        # use text2text-generation (flan) or text-generation models
        # avoid pre-downloading large files unless explicitly requested by the user
        pipe = pipeline("text2text-generation", model=model_name)
    except Exception as e:
        raise RuntimeError(
            "Unable to create local LLM pipeline. Install `transformers` and a compatible model, "
            "for example: `pip install transformers torch` and ensure the model is available." 
        ) from e

    try:
        from langchain import HuggingFacePipeline
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        raise RuntimeError("langchain HuggingFacePipeline integration failed: %s" % e)


EXPLAIN_PROMPT_TEMPLATE = (
    "You are an expert research assistant. Read the following CONTEXT (excerpts from research papers) and "
    "answer the QUESTION with a clear, step-by-step explanation suitable for a graduate student. "
    "Provide references to the source documents when relevant.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer:"
)


def get_research_chain(index_dir: str = "faiss_index_arxiv", llm_name: Optional[str] = None):
    """Load an arXiv FAISS index and return a RetrievalQA chain using a local open-source LLM.

    If `llm_name` is None, the pipeline will try to use `google/flan-t5-small`.
    """
    if not os.path.exists(index_dir):
        raise FileNotFoundError(f"ArXiv index not found at {index_dir}. Build it first.")

    # Lazy-import langchain components so this module can be imported even when
    # langchain/transformers aren't installed.
    try:
        from langchain_community.vectorstores import FAISS
        from langchain.chains import RetrievalQA
        from langchain_core.prompts import PromptTemplate
    except Exception as e:
        raise RuntimeError("langchain is required to build the research chain") from e

    embeddings = HuggingFaceEmbeddings()
    vectordb = FAISS.load_local(index_dir, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # Choose LLM: prefer a local HF model if available, otherwise fall back to a default LLM
    local_llm = None
    if llm_name:
        try:
            local_llm = _get_local_llm(llm_name)
        except Exception as e:
            logging.info("Local LLM creation failed: %s", e)

    if not local_llm:
        local_llm = _get_local_llm()
    llm = local_llm

    prompt = PromptTemplate.from_template(EXPLAIN_PROMPT_TEMPLATE)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return chain


def summarize_documents(docs: List[Any], llm_name: Optional[str] = None) -> str:
    """Produce a short summary of the provided documents using the chosen LLM.

    The `docs` parameter is a list of LangChain Documents or plain strings.
    """
    # Build a simple context concatenation
    parts = []
    for d in docs:
        if hasattr(d, "page_content"):
            parts.append(d.page_content)
        else:
            parts.append(str(d))
    context = "\n\n".join(parts[:6])  # limit context size

    explain_llm = None
    if llm_name:
        try:
            explain_llm = _get_local_llm(llm_name)
        except Exception as e:
            logger.info("Local LLM creation failed: %s", e)

    if not explain_llm:
        explain_llm = _get_local_llm()
    llm = explain_llm

    prompt_text = EXPLAIN_PROMPT_TEMPLATE.format(context=context, question="Give a concise summary of the above papers (4-6 sentences).")
    # If the llm supports calling as a callable
    try:
        out = llm(prompt_text)
        if isinstance(out, dict):
            return out.get("result") or out.get("text") or str(out)
        return str(out)
    except Exception as e:
        logger.exception("Failed to summarize with LLM: %s", e)
        raise


if __name__ == "__main__":
    print("RAG pipeline helpers: get_research_chain, summarize_documents")
