# Legal RAG Assistant

A multi-document legal research assistant leveraging Retrieval-Augmented Generation (RAG). Upload legal PDFs such as contracts and case law, and ask contextual questions to get well-cited, context-aware answers.

---

## Key Features

- Upload multiple PDF documents at once.
- Supports various legal documents including contracts, case law, and statutes.
- Combines semantic search using FAISS with large language model (LLM) generated answers.
- Uses HuggingFace embeddings (`all-MiniLM-L6-v2`) for efficient semantic vectorization.
- Built with Streamlit for a fast, interactive web user interface.
- Processes PDFs and chunks text intelligently for better search and retrieval.

---

## Tech Stack

- **Frontend:** Streamlit
- **LLM Backend:** DistilGPT-2 (via Hugging Face)
- **Embeddings:** `all-MiniLM-L6-v2` (Hugging Face)
- **Vector Store:** FAISS
- **PDF Processing:** PyPDF2
- **Dependencies:** transformers, torch, chromadb, sentence-transformers, langchain, requests, and others.

---

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/klahari09/legal-rag-assistant.git
cd legal-rag-assistant
2. Create and Activate a Python Virtual Environment (Recommended)
On macOS/Linux:
bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate
On Windows (CMD):
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate
3. Install Dependencies
Make sure you have Python 3.7+ installed.
Install required Python packages from the requirements.txt file:

bash
Copy
Edit
pip install -r requirements.txt
4. Google Drive Vectorstore Setup
Since the FAISS vectorstore file (faiss_store.pkl) can be large and is not included in the repository, it will be downloaded automatically on app start from Google Drive.

No manual action required here if you use the provided code.

5. Run the Application
Run the Streamlit app with:

bash
Copy
Edit
streamlit run app.py
The app will open in your default browser or provide a URL to access it locally.

Usage
Upload one or more legal PDF documents (contracts, case law, statutes) via the file uploader.

The app will process and chunk the PDFs.

Click "Add to Vector Store" to update the FAISS vectorstore with the document chunks.

Type your legal question in the query box.

Get contextual, cited answers retrieved from your uploaded documents.

File Structure Overview
graphql
Copy
Edit
legal-rag-assistant/
├── app.py                 # Main Streamlit app
├── retriever.py           # FAISS vectorstore build & retrieval logic
├── utils.py               # Utility functions for PDF loading, chunking, downloading vectorstore
├── requirements.txt       # Python dependencies
├── faiss_store.pkl        # (Not included; downloaded automatically)
├── temp/                  # Temporary folder for uploaded PDFs
└── README.md
Important Notes
The vectorstore file (faiss_store.pkl) is large and managed separately. It is downloaded automatically from Google Drive when the app starts, if not present locally.

Make sure you have an internet connection the first time you run the app for the vectorstore download.

Uploaded PDFs are temporarily saved in the temp/ folder, which is created automatically.

This project currently supports English language legal documents.

Troubleshooting
If the app fails to download the vectorstore, check your internet connection and Google Drive file sharing permissions.

If you encounter package errors, ensure you are using Python 3.7+ and have installed all dependencies.

Contributing
Contributions and improvements are welcome! Please fork the repo, create a feature branch, and submit a pull request.


