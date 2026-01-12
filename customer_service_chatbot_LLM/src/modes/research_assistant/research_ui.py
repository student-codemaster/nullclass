"""Streamlit UI for research assistant: paper search, summaries, and follow-up chat."""
import streamlit as st
import os
import logging
from typing import List

from .arxiv_loader import build_arxiv_index
from .rag_pipeline import get_research_chain, summarize_documents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


st.set_page_config(page_title="Research Assistant")

st.title("Research Assistant — ArXiv Explorer")

st.markdown("Upload an ArXiv CSV or provide its path, build an index for a chosen category (e.g., cs). Search papers, get summaries and ask follow-up questions.")

csv_path = st.text_input("ArXiv CSV path", value="../dataset/arxiv.csv")
category = st.text_input("Subset category (e.g., cs)", value="cs")
index_dir = st.text_input("Index directory", value="faiss_index_arxiv")

if st.button("Build index"):
    if not os.path.exists(csv_path):
        st.error("CSV path not found. Provide a valid path or upload the file.")
    else:
        with st.spinner("Building arXiv index..."):
            try:
                build_arxiv_index(csv_path, index_dir=index_dir, subset_category=category)
                st.success("ArXiv index built")
            except Exception as e:
                st.error(f"Failed to build index: {e}")

st.write("---")

query = st.text_input("Search query (paper topic or keywords)")
llm_choice = st.text_input("Local LLM model name (optional)", value="google/flan-t5-small")

if st.button("Search") and query:
    try:
        chain = get_research_chain(index_dir=index_dir, llm_name=llm_choice)
    except Exception as e:
        st.error(f"Failed to load research chain: {e}")
    else:
        with st.spinner("Retrieving relevant papers..."):
            res = chain(query)
            answer = res.get("result") if isinstance(res, dict) else str(res)
            st.subheader("Assistant answer")
            st.write(answer)

            docs = res.get("source_documents") or []
            if docs:
                st.subheader("Retrieved papers")
                for i, d in enumerate(docs):
                    title_preview = d.page_content.split('\n')[0][:200]
                    st.markdown(f"**{i+1}.** {title_preview}")
                    if d.metadata:
                        md_lines = []
                        for k, v in d.metadata.items():
                            md_lines.append(f"- **{k}**: {v}")
                        st.write('\n'.join(md_lines))

            # Store retrieved docs in session for follow-ups
            st.session_state.setdefault("last_docs", docs)
            st.session_state.setdefault("chat_history", [])

st.write("---")

st.subheader("Follow-up question")
followup = st.text_input("Ask a follow-up about the retrieved papers")
if st.button("Ask follow-up") and followup:
    docs = st.session_state.get("last_docs")
    if not docs:
        st.error("No previous retrieval found. Run a search first.")
    else:
        # Use the same chain, but include previous chat history in the prompt
        try:
            chain = get_research_chain(index_dir=index_dir, llm_name=llm_choice)
            # Build a context from last retrieved docs
            res = chain(f"{followup}")
            ans = res.get("result") if isinstance(res, dict) else str(res)
            st.write(ans)
            st.session_state["chat_history"].append({"q": followup, "a": ans})
        except Exception as e:
            st.error(f"Follow-up failed: {e}")

st.write("---")
st.subheader("Summarize retrieved papers")
if st.button("Summarize last retrieved"):
    docs = st.session_state.get("last_docs") or []
    if not docs:
        st.error("No retrieved papers to summarize")
    else:
        with st.spinner("Summarizing..."):
            try:
                summary = summarize_documents(docs, llm_name=llm_choice)
                st.write(summary)
            except Exception as e:
                st.error(f"Summarization failed: {e}")
