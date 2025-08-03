from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import pickle
import os

# Use a free HuggingFace model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def add_to_db(text_chunks, metadata, db_path="faiss_store.pkl"):
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embedding_model, metadatas=metadata)

    with open(db_path, "wb") as f:
        pickle.dump(vectorstore, f)
    print(f"[INFO] Vectorstore saved to {db_path}")

def retrieve(query, db_path="faiss_store.pkl", k=5):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"[ERROR] Vectorstore file '{db_path}' not found. Please run add_to_db() first.")
    
    with open(db_path, "rb") as f:
        vectorstore = pickle.load(f)

    docs = vectorstore.similarity_search(query, k=k)
    return docs
