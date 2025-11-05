# backend.py

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# --- Environment-Aware Secret Loading ---
GROQ_API_KEY = None
SYSTEM_PROMPT = None
try:
    GROQ_API_KEY = str(st.secrets["GROQ_API_KEY"])
    SYSTEM_PROMPT = str(st.secrets["SYSTEM_PROMPT"])
except (FileNotFoundError, KeyError):
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
            file_path = os.path.join(docs_path, filename)
            if filename.endswith(('.pdf', '.txt')):
                loader = TextLoader(file_path, encoding='utf-8') if filename.endswith('.txt') else PyPDFLoader(
                    file_path)
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

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

    condense_question_prompt_text = "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:"
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_prompt_text)

    qa_prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        output_key='answer'
    )
    return qa_chain