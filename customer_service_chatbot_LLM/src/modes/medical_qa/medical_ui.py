import streamlit as st
import os
import logging

from medical_retrieval import build_medical_index, get_medical_chain, extract_medical_entities
from interaction_store import log_interaction, add_correction, get_recent

# optional append helper from langchain_helper
try:
    import langchain_helper
    _append_to_dataset = getattr(langchain_helper, "_append_to_dataset", None)
except Exception:
    _append_to_dataset = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


st.set_page_config(page_title="Medical QA Chatbot")

st.title("Medical Q&A Chatbot")

st.markdown("This is a specialist medical QA mode. Upload a MedQuAD JSON/JSONL dataset or provide a path, then build the medical knowledge index. The assistant will cite sources and avoid hallucination.")

username = st.text_input("Your name (optional)", value="")

default_path = os.path.join(os.path.dirname(__file__), "data", "MedQuAD")
dataset_path = st.text_input("Path to MedQuAD file/directory (json/jsonl/xml/csv)", value=default_path)
upload = st.file_uploader("Or upload a MedQuAD file (json/jsonl/xml/csv)", type=["json", "jsonl", "xml", "csv"]) 

if upload is not None:
    # Save uploaded file to a temp location under src/ so relative paths resolve
    save_name = getattr(upload, "name", "uploaded_medquad")
    save_path = os.path.join(os.getcwd(), save_name)
    with open(save_path, "wb") as f:
        f.write(upload.getbuffer())
    dataset_path = save_path
    st.success(f"Uploaded dataset saved to {save_path}")

index_dir = st.text_input("Index directory", value="faiss_index_medical")

if st.button("Build medical knowledge index"):
    if not os.path.exists(dataset_path):
        st.error("Dataset path not found. Please upload or provide a valid path.")
    else:
        with st.spinner("Building index — this may take a while..."):
            try:
                build_medical_index(dataset_path, index_dir=index_dir)
                st.success("Medical index built and saved to %s" % index_dir)
            except Exception as e:
                st.error(f"Failed to build index: {e}")

st.write("---")

question = st.text_input("Ask a medical question:")
ner_box = st.checkbox("Show extracted medical entities (symptoms/diseases/treatments)")

if st.button("Get Answer") and question:
    try:
        chain = get_medical_chain(index_dir=index_dir)
    except Exception as e:
        st.error(f"Medical index not found or failed to load: {e}")
    else:
        with st.spinner("Running retrieval and answer generation..."):
            result = chain(question)
            answer = result.get("result") or result.get("answer") or str(result)
            st.subheader("Answer")
            st.write(answer)

            src_docs = []
            if result.get("source_documents"):
                st.subheader("Sources")
                for doc in result["source_documents"]:
                    src = doc.metadata.get("source") if hasattr(doc, "metadata") else getattr(doc, "source", "unknown")
                    st.write(f"- {src}")
                    src_docs.append({"source": src})

            if ner_box:
                entities = extract_medical_entities(question)
                st.subheader("Extracted medical entities")
                st.write(entities)

            # log interaction
            try:
                iid = log_interaction(username or "anonymous", question, answer, source=src_docs)
                st.info(f"Interaction logged (id={iid})")
            except Exception as e:
                st.warning(f"Failed to log interaction: {e}")

            # correction UI
            st.write("---")
            st.write("If the answer is incorrect or incomplete, submit a corrected answer below to save it to the local dataset.")
            correction = st.text_area("Corrected answer (optional)")
            if st.button("Submit correction") and correction.strip():
                try:
                    ok = add_correction(iid, correction.strip())
                    if ok:
                        st.success("Correction saved to interaction log.")
                        if _append_to_dataset is not None:
                            try:
                                _append_to_dataset([{"prompt": question, "response": correction.strip()}])
                                st.success("Correction appended to local dataset.")
                            except Exception as e:
                                st.warning(f"Failed to append correction to dataset: {e}")
                    else:
                        st.warning("Could not save correction to interaction log.")
                except Exception as e:
                    st.error(f"Failed to save correction: {e}")
