import os
import streamlit as st
from retriever import add_to_db, retrieve
from utils import load_pdf, chunk_text

st.set_page_config(page_title="Legal RAG Assistant", layout="wide")

# UI: File Upload
st.title("📚 Legal Research Assistant")
uploaded_files = st.file_uploader("Upload legal documents (PDFs)", type=["pdf"], accept_multiple_files=True)

# Process Uploaded PDFs
if uploaded_files:
    os.makedirs("temp", exist_ok=True)  # ✅ Ensure temp/ directory exists

    all_chunks = []
    all_metadata = []
    
    for uploaded_file in uploaded_files:
        file_path = f"temp/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        text = load_pdf(file_path)
        chunks = chunk_text(text)
        
        metadata = [{"source": uploaded_file.name}] * len(chunks)
        all_chunks.extend(chunks)
        all_metadata.extend(metadata)

    st.success("✅ Documents processed and chunked.")

    
    # Save to FAISS
    if st.button("➕ Add to Vector Store"):
        add_to_db(all_chunks, all_metadata)
        st.success("✅ Vector store updated.")

# Query Interface
query = st.text_input("🔎 Ask a legal question")

if query:
    st.info(f"Searching for: **{query}**")
    docs = retrieve(query)
    for i, doc in enumerate(docs):
        st.markdown(f"**Result {i+1}:**")
        st.write(doc.page_content)
        st.caption(f"📄 Source: {doc.metadata.get('source', 'Unknown')}")
