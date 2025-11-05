# backend.py

# backend.py

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

GROQ_API_KEY = None
SYSTEM_PROMPT = None
try:
    # Running on Streamlit Cloud
    GROQ_API_KEY = str(st.secrets["GROQ_API_KEY"])
    SYSTEM_PROMPT = str(st.secrets["SYSTEM_PROMPT"])
except (FileNotFoundError, KeyError):
    # Running locally, load from config.py
    import config
    GROQ_API_KEY = config.GROQ_API_KEY
    SYSTEM_PROMPT = config.SYSTEM_PROMPT


FAISS_INDEX_PATH = "faiss_index"


def load_and_build_index():
    if os.path.exists(FAISS_INDEX_PATH):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        docs_path = os.path.join(os.path.dirname(__file__), 'documents')
        all_pages = []
        for filename in os.listdir(docs_path):
            file_path = os.path.join(docs_path, filename)
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                all_pages.extend(pages)
            elif filename.endswith('.txt'):
                loader = TextLoader(file_path)
                pages = loader.load()
                all_pages.extend(pages)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        texts = text_splitter.split_documents(all_pages)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(texts, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        return vector_store


def get_qa_chain():
    vector_store = load_and_build_index()
    retriever = vector_store.as_retriever()
    llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)

    # 1. This is the prompt for the "history-aware" part of the chain.
    # It turns the new question + old history into a single, standalone question.
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2. This is the main answering prompt, which now has access to the history.
    qa_system_prompt = SYSTEM_PROMPT + """

    CONTEXT:
    {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )

    # This chain takes the question and the retrieved documents and generates the answer.
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 3. This is the final chain that ties it all together.
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain