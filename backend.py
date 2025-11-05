# backend.py

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

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

# This prompt combines the persona and the instructions into one template
PROMPT_TEMPLATE = SYSTEM_PROMPT + """

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


def load_and_build_index():
    if os.path.exists(FAISS_INDEX_PATH):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        docs_path = os.path.join(os.path.dirname(__file__), 'documents')
        all_pages = []
        for filename in os.listdir(docs_path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(docs_path, filename)
                loader = PyPDFLoader(file_path)
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
    llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'  # Specify the key for the AI's response
    )

    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vector_store.as_retriever(),
        memory=memory,  # Plug in the memory
        combine_docs_chain_kwargs={"prompt": prompt},  # Pass our custom prompt
        output_key='answer'  # Ensure the output key is consistent
    )
    return qa_chain