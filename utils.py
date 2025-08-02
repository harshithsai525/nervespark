import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

def setup_rag_pipeline(uploaded_files, groq_api_key):
    """
    Sets up the RAG pipeline by processing uploaded PDF files.

    This function performs the following steps:
    1.  Loads documents from the uploaded PDF files.
    2.  Splits the documents into smaller, manageable chunks. [cite: 26]
    3.  Generates embeddings for these chunks using a HuggingFace model. [cite: 24]
    4.  Stores the chunks and their embeddings in a Chroma vector database. [cite: 25]
    5.  Initializes a ChatGroq model for generation.
    6.  Creates a retrieval chain that fetches relevant documents and generates a response. [cite: 27]

    Args:
        uploaded_files: A list of uploaded file objects from Streamlit.
        groq_api_key (str): The API key for the Groq language model.

    Returns:
        A LangChain retrieval chain object ready to process queries.
    """
    if not uploaded_files:
        return None

    # Load documents
    docs = []
    for file in uploaded_files:
        loader = PyPDFLoader(file.name)
        docs.extend(loader.load())

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create vector store
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # Initialize Groq LLM
    llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-70b-8192")

    # Create Prompt Template
    # This prompt guides the LLM to provide contextual answers with citations and handle conflicts. [cite: 4, 7, 8]
    system_prompt = (
        "You are an expert legal research assistant. Analyze the provided context "
        "from legal documents to answer the user's query. Your task is to:\n"
        "1. Provide a direct and comprehensive answer to the query based *only* on the retrieved context.\n"
        "2. For each piece of information in your answer, you MUST cite the specific source document and relevant section. Use the format.\n"
        "3. If the context contains conflicting information across different documents, identify and clearly state the conflict in your answer.\n"
        "4. If the context does not contain enough information to answer the query, state that clearly.\n"
        "5. Do not make up information or use external knowledge.\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create the retrieval chain
    Youtube_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever()
    rag_chain = create_retrieval_chain(retriever, Youtube_chain)

    return rag_chain

def process_query(chain, query):
    """
    Processes a user's query using the RAG chain and returns the answer.

    Args:
        chain: The LangChain retrieval chain object.
        query (str): The user's legal query.

    Returns:
        str: The generated answer from the language model.
    """
    if not chain:
        return "The document processing pipeline has not been set up. Please upload documents first."
    
    result = chain.invoke({"input": query})
    return result["answer"]