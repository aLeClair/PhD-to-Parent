# backend.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_stuff_documents_chain, create_retrieval_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import config

FAISS_INDEX_PATH = "faiss_index"

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
    retriever = vector_store.as_retriever()
    llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=config.GROQ_API_KEY)
    system_prompt = config.SYSTEM_PROMPT
    human_prompt = config.HUMAN_PROMPT_TEMPLATE
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)
    return qa_chain