# Legal RAG Assistant

A multi-document legal research assistant powered by Retrieval-Augmented Generation (RAG). This app allows users to upload legal documents such as **contracts, case law, and statutes** in PDF format and ask contextual legal questions. The system retrieves relevant content chunks and generates well-cited, context-aware responses.

## Features

- **Multi-document upload support**
- Handles legal documents: contracts, case law, statutes
- RAG pipeline: combines semantic search with LLM-generated answers
- Legal citation-aware answers
- Uses FAISS for efficient document chunk similarity search
- Embedding support via OpenAI or HuggingFace
- Built with Streamlit for a fast and interactive web interface

## 🛠 Tech Stack

- **Frontend**: Streamlit
- **LLM Backend**: OpenAI GPT-4 (or your preferred model)
- **Embeddings**: OpenAIEmbeddings / HuggingFaceEmbeddings (SentenceTransformers)
- **Vector Store**: FAISS
- **PDF Processing**: PyPDF2
- **Storage**: Pickle (for vector store)
- 
How to Run Locally
1. Clone the Repository
git clone https://github.com/klahari09/legal-rag-assistant.git
cd legal-rag-assistant

2. Install Dependencies
pip install -r requirements.txt

3.Add API Keys
OPENAI_API_KEY = "your-openai-api-key"

4. Run the App
streamlit run app.py

🌐 Live App
🔗 https://klahari09-legal-rag-assistant-app-llbxaj.streamlit.app/

