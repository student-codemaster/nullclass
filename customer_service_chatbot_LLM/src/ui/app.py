# advanced chatbot 
import streamlit as st
import os
import sys
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import langchain_helper
from modes.sentiment_mode import SentimentClassifier
from modes.sentiment_mode.dashboard import render_sentiment_dashboard
from modes.multilingual import detect_and_translate, adapt_response_for_culture
from modes.dynamic_updater.schedule import start_background_scheduler



# PAGE CONFIG AND STATE INITIALIZATION

st.set_page_config(
    page_title="Advanced Chatbot System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "mode" not in st.session_state:
    st.session_state.mode = "General Q&A"
if "language" not in st.session_state:
    st.session_state.language = "en"
if "show_sentiment" not in st.session_state:
    st.session_state.show_sentiment = False
if "scheduler_started" not in st.session_state:
    st.session_state.scheduler_started = False


# HELPING FUNCTION
def get_general_qa_response(question: str):
    try:
        chain = langchain_helper.get_qa_chain()
        result = chain(question)
        return result.get("result", "No answer found"), result.get("source_documents", [])
    except Exception as e:
        return f"Error: {str(e)}", []


def get_medical_qa_response(question: str):
    try:
        from modes.medical_qa.medical_retrieval import get_medical_chain, extract_medical_entities
        chain = get_medical_chain(index_dir="faiss_index_medical")
        result = chain(question)
        answer = result.get("result", "No answer found")
        sources = result.get("source_documents", [])
        entities = extract_medical_entities(question)
        return answer, sources, entities
    except Exception as e:
        return f"Error (Medical index not built): {str(e)}", [], {}


def get_research_response(question: str):
    try:
        from modes.research_assistant.rag_pipeline import get_research_chain
        chain = get_research_chain(index_dir="faiss_index_arxiv", llm_name=None)
        result = chain(question)
        answer = result.get("result", "No answer found")
        sources = result.get("source_documents", [])
        return answer, sources
    except Exception as e:
        return f"Error (Research index not built): {str(e)}", []


def process_multimodal_input(question: str, image_path: str = None):
    try:
        from modes.multimodal.handler import handle_multimodal_query
        result = handle_multimodal_query(
            question=question,
            image_path=image_path,
            add_to_index=False
        )
        return result
    except Exception as e:
        return {"response": {"result": f"Error: {str(e)}"}, "image_insights": {}}


def detect_and_adjust_sentiment(user_text: str, response: str):
    try:
        sc = SentimentClassifier(use_llm_for_rewrite=False)
        sentiment = sc.classify(user_text)
        adjusted = sc.adjust_response(response, sentiment, user_text=user_text)
        return sentiment, adjusted
    except Exception as e:
        return {"label": "neutral", "score": 0.0}, response


def handle_multilingual_flow(user_input: str, response_en: str, target_lang: str):
    if target_lang == "en":
        return response_en
    try:
        adapted = adapt_response_for_culture(response_en, target_lang, user_input=user_input)
        return adapted.get("prefixed_response", response_en)
    except Exception as e:
        return response_en


# SIDEBAR 

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Mode selection
    mode = st.selectbox(
        "Select Mode",
        ["General Q&A", "Medical Q&A", "Research Assistant", "Multimodal", "Sentiment Dashboard", "Combined Mode"],
        index=0
    )
    st.session_state.mode = mode
    
    # Language selection
    lang_map = {"English": "en", "Hindi": "hi", "Kannada": "kn", "Tamil": "ta"}
    language_display = st.selectbox(
        "Select Language",
        list(lang_map.keys()),
        index=0
    )
    st.session_state.language = lang_map[language_display]
    
    # Features toggles
    st.markdown("### 🎯 Feature Toggles")
    st.session_state.show_sentiment = st.checkbox("Show Sentiment Analysis", value=False)
    enable_multimodal = st.checkbox("Enable Multimodal Input", value=False)
    enable_scheduler = st.checkbox("Enable Auto-Update Scheduler", value=False)
    
    if enable_scheduler and not st.session_state.scheduler_started:
        try:
            start_background_scheduler()
            st.session_state.scheduler_started = True
            st.success("✅ Scheduler started in background")
        except Exception as e:
            st.warning(f"Scheduler failed: {e}")
    
    # Index management
    st.markdown("### 📚 Index Management")
    if st.button("🔄 Build General Index"):
        with st.spinner("Building general knowledge base..."):
            try:
                langchain_helper.create_vector_db()
                st.success("✅ General index built")
            except Exception as e:
                st.error(f"Error: {e}")
    
    if st.button("🏥 Build Medical Index"):
        with st.spinner("Building medical index..."):
            medical_csv = st.text_input("Medical dataset path (JSON/JSONL)")
            if medical_csv:
                try:
                    from modes.medical_qa.medical_retrieval import build_medical_index
                    build_medical_index(medical_csv)
                    st.success("✅ Medical index built")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    if st.button("📖 Build Research Index"):
        with st.spinner("Building research index..."):
            arxiv_csv = st.text_input("ArXiv dataset path (CSV)")
            if arxiv_csv:
                try:
                    from modes.research_assistant.arxiv_loader import build_arxiv_index
                    build_arxiv_index(arxiv_csv)
                    st.success("✅ Research index built")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    st.markdown("---")
    st.markdown("### 📊 Chat Stats")
    st.metric("Messages Sent", len(st.session_state.chat_history))
    st.metric("Current Language", st.session_state.language.upper())
    st.metric("Active Mode", st.session_state.mode)


# MAIN CONTENT AREA
st.markdown("# 🤖 Advanced Chatbot System")
st.markdown(f"*Mode: **{st.session_state.mode}** | Language: **{st.session_state.language.upper()}** | Sentiment: {'**ON**' if st.session_state.show_sentiment else '**OFF**'}*")



# MODE-SPECIFIC UI

if st.session_state.mode == "General Q&A":
    st.markdown("## 💬 General Question & Answer")
    st.info("Ask any question from the knowledge base. Supports multilingual input and sentiment detection.")
    
    query = st.text_area("Your Question:", placeholder="Ask anything...")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        submit = st.button("🔍 Get Answer", key="gen_submit")
    with col2:
        clear = st.button("🗑️ Clear", key="gen_clear")
    
    if clear:
        st.session_state.chat_history = []
        st.rerun()
    
    if submit and query:
        # Multilingual preprocessing
        if st.session_state.language != "en":
            detected_lang, query_en = detect_and_translate(query)
        else:
            query_en = query
            detected_lang = "en"
        
        with st.spinner("Processing..."):
            response, sources = get_general_qa_response(query_en)
            
            # Sentiment analysis
            sentiment = None
            if st.session_state.show_sentiment:
                sentiment, response = detect_and_adjust_sentiment(query, response)
            
            # Multilingual postprocessing
            if st.session_state.language != "en":
                response = handle_multilingual_flow(query, response, st.session_state.language)
        
        # Store in history
        st.session_state.chat_history.append({
            "timestamp": datetime.now(),
            "mode": "General Q&A",
            "question": query,
            "answer": response,
            "sentiment": sentiment
        })
        
        # Display response
        st.markdown("### 📝 Answer")
        st.write(response)
        
        if sources:
            with st.expander("📖 Sources"):
                for i, doc in enumerate(sources, 1):
                    st.write(f"**Source {i}**: {doc.metadata.get('source', 'Unknown')}")
        
        if st.session_state.show_sentiment and sentiment:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sentiment", sentiment.get("label", "N/A").upper())
            with col2:
                st.metric("Score", f"{sentiment.get('score', 0):.2f}")


elif st.session_state.mode == "Medical Q&A":
    st.markdown("## 🏥 Medical Q&A System")
    st.warning("⚠️ For educational purposes only. Always consult qualified healthcare professionals.")
    
    query = st.text_area("Medical Question:", placeholder="Ask a medical question...")
    col1, col2 = st.columns([3, 1])
    with col1:
        submit = st.button("🔍 Get Medical Answer", key="med_submit")
    with col2:
        clear = st.button("🗑️ Clear", key="med_clear")
    
    if clear:
        st.session_state.chat_history = []
        st.rerun()
    
    if submit and query:
        if st.session_state.language != "en":
            detected_lang, query_en = detect_and_translate(query)
        else:
            query_en = query
        
        with st.spinner("Retrieving medical information..."):
            response, sources, entities = get_medical_qa_response(query_en)
        
        if st.session_state.language != "en":
            response = handle_multilingual_flow(query, response, st.session_state.language)
        
        st.session_state.chat_history.append({
            "timestamp": datetime.now(),
            "mode": "Medical Q&A",
            "question": query,
            "answer": response,
            "entities": entities
        })
        
        st.markdown("### 📋 Medical Answer")
        st.write(response)
        
        if entities and any(entities.values()):
            st.markdown("### 🔬 Extracted Entities")
            col1, col2, col3 = st.columns(3)
            with col1:
                if entities.get("symptoms"):
                    st.write(f"**Symptoms**: {', '.join(entities['symptoms'])}")
            with col2:
                if entities.get("diseases"):
                    st.write(f"**Diseases**: {', '.join(entities['diseases'])}")
            with col3:
                if entities.get("treatments"):
                    st.write(f"**Treatments**: {', '.join(entities['treatments'])}")
        
        if sources:
            with st.expander("📖 Medical Sources"):
                for i, doc in enumerate(sources, 1):
                    st.write(f"**Source {i}**: {doc.metadata.get('source', 'Unknown')}")


elif st.session_state.mode == "Research Assistant":
    st.markdown("## 📚 Research Assistant (ArXiv Papers)")
    st.info("Explore research papers, get summaries, and ask follow-up questions.")
    
    query = st.text_area("Research Query:", placeholder="Search for papers or ask a research question...")
    col1, col2 = st.columns([3, 1])
    with col1:
        submit = st.button("🔍 Search Papers", key="res_submit")
    with col2:
        clear = st.button("🗑️ Clear", key="res_clear")
    
    if clear:
        st.session_state.chat_history = []
        st.rerun()
    
    if submit and query:
        if st.session_state.language != "en":
            detected_lang, query_en = detect_and_translate(query)
        else:
            query_en = query
        
        with st.spinner("Searching research papers..."):
            response, sources = get_research_response(query_en)
        
        if st.session_state.language != "en":
            response = handle_multilingual_flow(query, response, st.session_state.language)
        
        st.session_state.chat_history.append({
            "timestamp": datetime.now(),
            "mode": "Research",
            "question": query,
            "answer": response,
            "paper_count": len(sources)
        })
        
        st.markdown("### 📄 Research Answer")
        st.write(response)
        
        if sources:
            st.markdown("### 📑 Retrieved Papers")
            for i, doc in enumerate(sources, 1):
                with st.expander(f"Paper {i}"):
                    preview = doc.page_content[:500] if hasattr(doc, "page_content") else str(doc)[:500]
                    st.write(preview)
                    if hasattr(doc, "metadata") and doc.metadata:
                        st.json(doc.metadata)


elif st.session_state.mode == "Multimodal":
    st.markdown("## 🖼️ Multimodal Input (Image + Text)")
    st.info("Upload an image and ask a question. The system will analyze the image and answer.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        image_path = None
        if uploaded_file:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(uploaded_file.getbuffer())
                image_path = tmp.name
            st.image(image_path, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        query = st.text_area("Question about the image:", placeholder="What is in this image?")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        submit = st.button("🔍 Analyze", key="mm_submit")
    with col2:
        clear = st.button("🗑️ Clear", key="mm_clear")
    
    if clear:
        st.session_state.chat_history = []
        st.rerun()
    
    if submit and (query or image_path):
        with st.spinner("Analyzing multimodal input..."):
            result = process_multimodal_input(query, image_path)
            response = result.get("response", {}).get("result", "No answer")
            insights = result.get("image_insights", {})
        
        if st.session_state.language != "en":
            response = handle_multilingual_flow(query or "", response, st.session_state.language)
        
        st.markdown("### 📝 Analysis")
        st.write(response)
        
        if insights:
            st.markdown("### 🔎 Image Insights")
            if insights.get("caption"):
                st.write(f"**Caption**: {insights['caption']}")
            if insights.get("ocr"):
                st.write(f"**OCR Text**: {insights['ocr']}")


elif st.session_state.mode == "Sentiment Dashboard":
    try:
        render_sentiment_dashboard()
    except Exception as e:
        st.error(f"Failed to render dashboard: {e}")

elif st.session_state.mode == "Combined Mode":
    st.markdown("## 🌟 Combined Mode (All Features)")
    st.info("Use multiple features together in one interface.")
    
    # Tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["💬 General", "🏥 Medical", "📚 Research", "🖼️ Multimodal"])
    
    with tab1:
        query = st.text_area("General Question:", key="combined_gen")
        if st.button("Search", key="cb_gen"):
            if query:
                if st.session_state.language != "en":
                    _, query_en = detect_and_translate(query)
                else:
                    query_en = query
                response, sources = get_general_qa_response(query_en)
                if st.session_state.language != "en":
                    response = handle_multilingual_flow(query, response, st.session_state.language)
                st.write(response)
    
    with tab2:
        query = st.text_area("Medical Question:", key="combined_med")
        if st.button("Search", key="cb_med"):
            if query:
                if st.session_state.language != "en":
                    _, query_en = detect_and_translate(query)
                else:
                    query_en = query
                response, sources, entities = get_medical_qa_response(query_en)
                if st.session_state.language != "en":
                    response = handle_multilingual_flow(query, response, st.session_state.language)
                st.write(response)
                if entities:
                    st.json(entities)
    
    with tab3:
        query = st.text_area("Research Query:", key="combined_res")
        if st.button("Search", key="cb_res"):
            if query:
                if st.session_state.language != "en":
                    _, query_en = detect_and_translate(query)
                else:
                    query_en = query
                response, sources = get_research_response(query_en)
                if st.session_state.language != "en":
                    response = handle_multilingual_flow(query, response, st.session_state.language)
                st.write(response)
    
    with tab4:
        uploaded_file = st.file_uploader("Upload Image:", key="combined_mm", type=["jpg", "jpeg", "png"])
        query = st.text_area("Question:", key="combined_mm_q")
        if st.button("Analyze", key="cb_mm"):
            if query or uploaded_file:
                image_path = None
                if uploaded_file:
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        image_path = tmp.name
                result = process_multimodal_input(query, image_path)
                response = result.get("response", {}).get("result")
                st.write(response)


# CHAT HISTORY FOOTER
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 📜 Chat History")
    with st.expander(f"View {len(st.session_state.chat_history)} messages"):
        for i, msg in enumerate(st.session_state.chat_history, 1):
            st.write(f"**{i}. [{msg['mode']}]** {msg['question'][:50]}...")
            st.caption(f"Time: {msg['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
