
import os
import sys
import logging
from datetime import datetime

# ensure src directory is on sys.path so `modes.*` imports work when running via streamlit
project_src = os.path.dirname(__file__)
if project_src not in sys.path:
    sys.path.insert(0, project_src)

from modes.dynamic_updater.schedule import start_background_scheduler
import langchain_helper
from modes.sentiment_mode import SentimentClassifier
from modes.multilingual import detect_and_translate, adapt_response_for_culture
from interaction_store import log_interaction

# Reduce noisy output from torch/transformers/huggingface when optional models are used.
# These environment variables and logger level changes suppress C++/internal diagnostics
# that otherwise appear in the output window during optional model initialization.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "0")

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
try:
    import streamlit as st
    _HAS_STREAMLIT = True
except Exception:
    st = None
    _HAS_STREAMLIT = False

if not _HAS_STREAMLIT:
    def cli_main():
        print("Running in CLI fallback mode (streamlit not installed).")
        username = input("Your name (optional): ").strip() or "cli_user"
        while True:
            print("\nMenu:\n1) Build/Update KB\n2) Ask a question\n3) Exit")
            choice = input("Choose an option: ").strip()
            if choice == "1":
                print("Building knowledge base...")
                try:
                    langchain_helper.create_vector_db()
                    print("Done")
                except Exception as e:
                    print("Error building KB:", e)
            elif choice == "2":
                q = input("Enter your question: ").strip()
                if not q:
                    print("Please enter a non-empty question.")
                    continue
                try:
                    chain = langchain_helper.get_qa_chain()
                    resp = chain(q)
                    answer = resp.get("result") or resp.get("text") or str(resp)
                    sources = resp.get("source_documents", [])
                except Exception:
                    fallback = langchain_helper.answer_query(q)
                    answer = fallback.get("answer")
                    sources = fallback.get("sources", [])
                print("\nAnswer:\n", answer)
                if sources:
                    print("Sources:")
                    for s in sources:
                        print(" -", getattr(s, "metadata", s))
                try:
                    src_docs = []
                    for s in sources:
                        try:
                            src_docs.append({"source": s.metadata.get("source") if hasattr(s, "metadata") else str(s)})
                        except Exception:
                            src_docs.append({"source": str(s)})
                    log_interaction(username, q, answer, source=src_docs)
                except Exception:
                    pass
            elif choice == "3":
                print("Goodbye")
                break
            else:
                print("Invalid choice")

    if __name__ == "__main__":
        cli_main()
        sys.exit(0)

# STREAMLIT UI MODE

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "language" not in st.session_state:
    st.session_state.language = "en"
if "show_sentiment" not in st.session_state:
    st.session_state.show_sentiment = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "scheduler_started" not in st.session_state:
    st.session_state.scheduler_started = False
if "last_ingest_time" not in st.session_state:
    st.session_state.last_ingest_time = None

with st.sidebar:
    st.markdown("##  Settings")
    st.markdown("### Advanced Options")

    language_map = {"English": "en", "Hindi": "hi", "Kannada": "kn", "Tamil": "ta"}
    language_display = st.selectbox("Language", list(language_map.keys()), index=0)
    st.session_state.language = language_map[language_display]

    st.session_state.show_sentiment = st.checkbox("Show Sentiment Analysis", value=False)

    enable_scheduler = st.checkbox("Enable Auto-Update", value=False)
    if enable_scheduler:
        try:
            start_background_scheduler()
            st.success(" Scheduler started")
            st.session_state.scheduler_started = True
        except Exception as e:
            st.warning(f"Scheduler: {e}")

    st.markdown("---")
    uploaded = st.file_uploader("Upload dataset files (CSV, JSON, XML)", accept_multiple_files=True)
    if uploaded:
        if st.button(" Ingest uploaded files"):
            with st.spinner("Ingesting files..."):
                try:
                    res = langchain_helper.ingest_uploaded_files(uploaded)
                    st.session_state.last_ingest_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.success(f"Ingested rows: {res.get('ingested')} (failed files: {res.get('failed')})")
                except Exception as e:
                    st.error(f"Ingest error: {e}")

    if st.button(" Build/Update KB"):
        with st.spinner("Building knowledge base..."):
            try:
                langchain_helper.create_vector_db()
                st.success(" Knowledge base created successfully!")
            except Exception as e:
                st.error(f"Error: {e}")

    st.metric("Language", st.session_state.get("language", "en").upper())
    st.metric("Messages", len(st.session_state.chat_history))
    # show scheduler status and last ingest time
    if st.session_state.scheduler_started:
        st.success("Scheduler: running")
    if st.session_state.last_ingest_time:
        st.caption(f"Last ingest: {st.session_state.last_ingest_time}")


# Main chat UI
st.title(" CUSTOMER SERVICE CHATBOT")
st.markdown("Ask a question below. enter email  below")

col1, col2, col3 = st.columns([1, 8, 1])
with col2:
    # Inline username input so the chat dynamically collects the user's name during conversation
    st.session_state.username = st.text_input("enter your email (optional)", value=st.session_state.username)
    query = st.text_area("Your Question:", placeholder="Ask anything (supports multilingual input)", height=120)

col1, col2, col3 = st.columns(3)
with col1:
    submit = st.button(" Search")
with col2:
    clear = st.button(" Clear History")
with col3:
    show_history = st.button(" History")

if clear:
    st.session_state.chat_history = []
    st.experimental_rerun()

if submit and query:
    # Multilingual preprocessing
    if st.session_state.get("language", "en") != "en":
        try:
            detected_lang, query_en = detect_and_translate(query)
        except Exception as e:
            st.warning(f"Translation issue: {e}, using original text")
            query_en = query
    else:
        query_en = query

    # Get response
    with st.spinner("Processing..."):
        try:
            chain = langchain_helper.get_qa_chain()
            response = chain(query_en)
            answer = response.get("result", "No answer found")
            sources = response.get("source_documents", [])
        except Exception as e:
            answer = f"Error: {str(e)}"
            sources = []

    # Sentiment analysis
    sentiment = None
    if st.session_state.get("show_sentiment", False):
        try:
            sc = SentimentClassifier(use_llm_for_rewrite=False)
            sentiment = sc.classify(query)
            adjusted = sc.adjust_response(answer, sentiment, user_text=query)
            answer = adjusted
        except Exception as e:
            st.warning(f"Sentiment analysis: {e}")

    # Multilingual postprocessing
    if st.session_state.get("language", "en") != "en":
        try:
            answer = adapt_response_for_culture(answer, st.session_state.get("language", "en"), user_input=query)
            if isinstance(answer, dict):
                answer = answer.get("prefixed_response", answer.get("translated_response", str(answer)))
        except Exception as e:
            st.warning(f"Response adaptation: {e}")

    # Store in history (ensure key exists to avoid KeyError on some Streamlit reruns)
    if "chat_history" not in st.session_state or st.session_state.get("chat_history") is None:
        st.session_state["chat_history"] = []
    st.session_state["chat_history"].append({
        "timestamp": datetime.now(),
        "question": query,
        "answer": answer,
        "sentiment": sentiment,
        "username": st.session_state.get("username", "anonymous") or "anonymous",
    })
    # Log to central interaction store
    try:
        src_docs = []
        if sources:
            for doc in sources:
                src_docs.append({"source": doc.metadata.get("source", "Unknown") if hasattr(doc, "metadata") else getattr(doc, "source", "Unknown")})
        log_interaction(st.session_state.username or "anonymous", query, answer, source=src_docs)
    except Exception:
        pass

    # Display results
    st.markdown("###  Answer")
    st.write(answer)

    if sources:
        with st.expander(" Sources"):
            for i, doc in enumerate(sources, 1):
                src = doc.metadata.get("source", "Unknown") if hasattr(doc, "metadata") else "Unknown"
                st.write(f"**{i}**: {src}")

    if st.session_state.get("show_sentiment", False) and sentiment:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sentiment", sentiment.get("label", "N/A").upper())
        with col2:
            st.metric("Score", f"{sentiment.get('score', 0):.2f}")
        with col3:
            st.metric("Language", st.session_state.get("language", "en").upper())

if show_history and st.session_state.chat_history:
    st.markdown("---")
    st.markdown("###  Chat History")
    for i, msg in enumerate(st.session_state.chat_history, 1):
        with st.expander(f"**{i}**. {msg['question'][:50]}... ({msg['timestamp'].strftime('%H:%M:%S')})"):
            st.write(f"**Q**: {msg['question']}")
            st.write(f"**A**: {msg['answer']}")
            if msg.get('sentiment'):
                st.write(f"**Sentiment**: {msg['sentiment'].get('label')}")
        with st.spinner("Processing..."):
                try:
                    chain = langchain_helper.get_qa_chain()
                    response = chain(query_en)
                    answer = response.get("result", "No answer found")
                    sources = response.get("source_documents", [])
                except Exception as e:
                    answer = f"Error: {str(e)}"
                    sources = []
        
        # Sentiment analysis
        sentiment = None
        if st.session_state.get("show_sentiment", False):
            try:
                sc = SentimentClassifier(use_llm_for_rewrite=False)
                sentiment = sc.classify(query)
                adjusted = sc.adjust_response(answer, sentiment, user_text=query)
                answer = adjusted
            except Exception as e:
                st.warning(f"Sentiment analysis: {e}")

        # Multilingual postprocessing
        if st.session_state.get("language", "en") != "en":
            try:
                answer = adapt_response_for_culture(answer, st.session_state.get("language", "en"), user_input=query)
                if isinstance(answer, dict):
                    answer = answer.get("prefixed_response", answer.get("translated_response", str(answer)))
            except Exception as e:
                st.warning(f"Response adaptation: {e}")
        
        # Store in history (ensure key exists)
        if "chat_history" not in st.session_state or st.session_state.get("chat_history") is None:
            st.session_state["chat_history"] = []
        st.session_state["chat_history"].append({
            "timestamp": datetime.now(),
            "question": query,
            "answer": answer,
            "sentiment": sentiment
        })
        
        # Display results
        st.markdown("###  Answer")
        st.write(answer)
        
        if sources:
            with st.expander(" Sources"):
                for i, doc in enumerate(sources, 1):
                    src = doc.metadata.get("source", "Unknown") if hasattr(doc, "metadata") else "Unknown"
                    st.write(f"**{i}**: {src}")
        
        if st.session_state.get("show_sentiment", False) and sentiment:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sentiment", sentiment.get("label", "N/A").upper())
            with col2:
                st.metric("Score", f"{sentiment.get('score', 0):.2f}")
            with col3:
                st.metric("Language", st.session_state.get("language", "en").upper())
    
    if show_history and st.session_state.chat_history:
        st.markdown("---")
        st.markdown("###  Chat History")
        for i, msg in enumerate(st.session_state.chat_history, 1):
            with st.expander(f"**{i}**. {msg['question'][:50]}... ({msg['timestamp'].strftime('%H:%M:%S')})"):
                st.write(f"**Q**: {msg['question']}")
                st.write(f"**A**: {msg['answer']}")
                if msg.get('sentiment'):
                    st.write(f"**Sentiment**: {msg['sentiment'].get('label')}")

# FOOTER

st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        " This chatbot Integrated 6 features"
    )

with col2:
    st.markdown(
        "See side bar to see the Features"
    )

with col3:
    st.markdown(
        "This chatbot educational purpose only. This is not official"
    )
