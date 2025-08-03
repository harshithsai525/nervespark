import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# === Set your GROQ API Key ===
os.environ["GROQ_API_KEY"] = "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"

# === Embedding model ===
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# === Add to DB ===
def add_to_db(text_chunks, metadata=None, db_path="faiss_store.pkl"):
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embedding_model, metadatas=metadata)
    with open(db_path, "wb") as f:
        pickle.dump(vectorstore, f)
    print(f"[INFO] Vectorstore saved to {db_path}")

# === Retrieve similar documents ===
def retrieve(query, db_path="faiss_store.pkl", k=5):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"[ERROR] Vectorstore file '{db_path}' not found. Please run add_to_db() first.")
    with open(db_path, "rb") as f:
        vectorstore = pickle.load(f)
    docs = vectorstore.similarity_search(query, k=k)
    return docs

# === RAG pipeline using GROQ ===
def answer_with_groq(query, docs):
    # Combine retrieved docs into a context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Prompt Template
    prompt = PromptTemplate.from_template(
        """Answer the question using the following context:

{context}

Question: {question}

Answer:"""
    )

    # Chain
    llm = ChatGroq(model="llama3-8b-8192")  # You can try "llama3-70b-8192" if needed
    chain = prompt | llm | StrOutputParser()

    # Generate Answer
    answer = chain.invoke({"context": context, "question": query})
    return answer

# === Example Usage ===
if __name__ == "__main__":
    # Only run add_to_db() once to create FAISS index
    if not os.path.exists("faiss_store.pkl"):
        text_chunks = [
            "Clause 1: All users must agree to the terms.",
            "Clause 2: This contract is governed by Indian Law.",
            "Clause 3: Data privacy is a top priority."
        ]
        metadata = [{"source": "contract1.txt"}, {"source": "contract2.txt"}, {"source": "contract3.txt"}]
        add_to_db(text_chunks, metadata)

    # Run a query
    query = "What does the contract say about privacy?"
    docs = retrieve(query)
    answer = answer_with_groq(query, docs)

    print("\n[QUESTION]", query)
    print("[ANSWER]", answer)
