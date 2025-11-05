# app.py

import streamlit as st
from backend import get_qa_chain  # Import our main function from the backend

# --- The App's Title ---
st.title("PhD to Parent 🎓")

# --- Initialize Session State for Conversation History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Load the RAG chain and show a status message ---
try:
    qa_chain = get_qa_chain()
    st.toast("AI brain is ready!", icon="🧠")
except Exception as e:
    st.error(f"An error occurred while loading the AI brain: {e}")
    st.stop()

# --- Display Past Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- The Chat Input Box (at the bottom) ---
if prompt := st.chat_input("Ask a question about the research"):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the AI's response
    with st.chat_message("assistant"):
        with st.spinner("The AI is thinking..."):
            response = qa_chain.invoke(prompt)
            st.markdown(response["result"])

    # Add AI response to session state
    st.session_state.messages.append({"role": "assistant", "content": response["result"]})