# backend.py

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate

# --- Environment-Aware Secret Loading ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    SYSTEM_PROMPT = st.secrets["SYSTEM_PROMPT"]
except (FileNotFoundError, KeyError):
    import config
    GROQ_API_KEY = config.GROQ_API_KEY
    SYSTEM_PROMPT = config.SYSTEM_PROMPT

FAISS_INDEX_PATH = "vector_store"

# -------------------------
# Load or Build FAISS Index
# -------------------------
def load_and_build_index():
    if os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
        print("🔍 Loading existing FAISS index...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("🛠️ Building new FAISS index...")
        docs_path = os.path.join(os.path.dirname(__file__), 'documents')
        all_pages = []

        for filename in os.listdir(docs_path):
            file_path = os.path.join(docs_path, filename)
            if filename.endswith(('.pdf', '.txt')):
                loader = TextLoader(file_path, encoding='utf-8') if filename.endswith('.txt') else PyPDFLoader(file_path)
                pages = loader.load()
                all_pages.extend(pages)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        texts = text_splitter.split_documents(all_pages)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(texts, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)

        print("✅ FAISS index built and saved!")
        return vector_store


# -------------------------
# Rephrase Chain
# -------------------------
def build_rephrase_chain(llm):
    rephrase_prompt = ChatPromptTemplate.from_template(
        SYSTEM_PROMPT +
        "\n\nChat History:\n{chat_history}\n\nUser's latest message:\n{question}\n\n"
        "Rephrased, clear version of their question:"
    )
    return LLMChain(llm=llm, prompt=rephrase_prompt, output_key="rephrased_question")


# -------------------------
# Main QA Chain
# -------------------------
def get_qa_chain():
    # 1️⃣ Load or build FAISS index
    index = load_and_build_index()

    # 2️⃣ Create retriever
    retriever = index.as_retriever(search_kwargs={"k": 6})

    # 3️⃣ Set up memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5
    )

    # 4️⃣ Initialize LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        groq_api_key=GROQ_API_KEY
    )

    # 5️⃣ Chains
    rephrase_chain = build_rephrase_chain(llm)

    qa_prompt = ChatPromptTemplate.from_template(
        SYSTEM_PROMPT + "\n\n"
        "Context:\n{context}\n\n"
        "Chat history:\n{chat_history}\n\n"
        "Question:\n{question}"
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=False,
    )

    # 6️⃣ Run full logic (rephrase → answer)
    def run_full_chain(user_input, chat_history):
        rephrased = rephrase_chain.invoke({
            "question": user_input,
            "chat_history": chat_history
        })["rephrased_question"]

        result = qa_chain.invoke({
            "question": rephrased,
            "chat_history": chat_history
        })
        return result

    return run_full_chain
