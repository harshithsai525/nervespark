import streamlit as st
import os
from utils import setup_rag_pipeline, process_query

# --- Page Configuration ---
st.set_page_config(
    page_title="Multi-Document Legal Research Assistant",
    page_icon="⚖️",
    layout="wide"
)

# --- Groq API Key Management ---
# For demonstration, we use the provided key directly.
# In a production environment, use st.secrets or environment variables for security.
GROQ_API_KEY = "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"

# --- UI and Application Logic ---
st.title("⚖️ Multi-Document Legal Research Assistant")
st.markdown("""
This tool helps you analyze multiple legal documents to get contextual answers with citations. [cite: 3]
Upload your legal documents (e.g., contracts, case law) and ask a question.
""")

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload your legal PDFs here.",
        type="pdf",
        accept_multiple_files=True,
        help="You can upload multiple contracts, statutes, or case law documents."
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} document(s) uploaded successfully!")
        # Save uploaded files temporarily to be processed by PyPDFLoader
        for file in uploaded_files:
            with open(file.name, "wb") as f:
                f.write(file.getbuffer())
    else:
        st.warning("Please upload at least one document.")

# --- Main Content Area ---
st.header("2. Ask a Legal Question")

# Initialize the RAG pipeline in session state if documents are uploaded
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if uploaded_files and st.button("Process Documents and Build Index"):
    with st.spinner("Processing documents, creating embeddings, and building the index... Please wait."):
        st.session_state.rag_chain = setup_rag_pipeline(uploaded_files, GROQ_API_KEY)
        st.success("✅ Index built successfully! You can now ask questions.")
        
# Disable query input if the pipeline is not ready
query_disabled = st.session_state.rag_chain is None

query = st.text_input(
    "Enter your legal query:",
    placeholder="e.g., What are the termination clauses in the agreements?",
    disabled=query_disabled
)

if query and not query_disabled:
    with st.spinner("Searching documents and generating an answer..."):
        answer = process_query(st.session_state.rag_chain, query)
        st.markdown("### Answer")
        st.markdown(answer)

# Clean up temporary files after the session
if uploaded_files:
    for file in uploaded_files:
        if os.path.exists(file.name):
            os.remove(file.name)