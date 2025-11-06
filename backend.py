import os
import pickle
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Client

# --- Environment-aware secret loading ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    SYSTEM_PROMPT = st.secrets["SYSTEM_PROMPT"]
except (FileNotFoundError, KeyError):
    import config

    GROQ_API_KEY = config.GROQ_API_KEY
    SYSTEM_PROMPT = config.SYSTEM_PROMPT

FAISS_INDEX_PATH = "faiss_index"


def load_or_build_index():
    """Load FAISS index if exists, else build from documents."""
    if os.path.exists(FAISS_INDEX_PATH):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    # Build index
    docs_path = os.path.join(os.path.dirname(__file__), "documents")
    all_pages = []
    for filename in os.listdir(docs_path):
        file_path = os.path.join(docs_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            continue
        pages = loader.load()
        all_pages.extend(pages)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    texts = text_splitter.split_documents(all_pages)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(texts, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    return vector_store


# Initialize Groq client
client = Client(api_key=GROQ_API_KEY)


def query_groq(messages):
    """
    Send messages to Groq LLM. Messages = list of {"role": "user"/"assistant"/"system", "content": str}.
    Returns: assistant response string
    """
    formatted_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted_messages.append({"role": role, "content": content})

    response = client.chat(
        model="llama-3.1-8b-instant",
        messages=formatted_messages
    )
    return response["choices"][0]["message"]["content"]


def get_qa_chain():
    """
    Returns a function that can be called as:
    response = qa_chain(user_query, chat_history)
    """
    vector_store = load_or_build_index()

    def qa_chain(user_query, chat_history):
        # Keep last 5 turns
        recent_history = chat_history[-5:]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add recent conversation
        for msg in recent_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Retrieve relevant docs from FAISS
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.get_relevant_documents(user_query)
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        messages.append({"role": "assistant", "content": f"Context:\n{context_text}"})

        # Add user question
        messages.append({"role": "user", "content": user_query})

        # Query Groq
        answer = query_groq(messages)
        return answer

    return qa_chain
