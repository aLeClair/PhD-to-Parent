# app.py

import streamlit as st
from backend import get_qa_chain

st.title("PhD to Parent")
# --- Initialize Session State and Add the Greeting Message ---
if "messages" not in st.session_state:
    # This is the one-time greeting that sets the context.
    welcome_message = (
        "Hello! I'm an AI assistant who has read all of Andrew's research. "
        "My purpose is to help you understand his work. Think of me as a friendly translator for his complex ideas. "
        "Please feel free to ask me anything, like 'What does he do?' or 'What is an ontology?'!"
    )
    st.session_state.messages = [{"role": "assistant", "content": welcome_message}]

# Load the RAG chain
try:
    qa_chain = get_qa_chain()
    # We remove the toast here for a cleaner interface
except Exception as e:
    st.error(f"An error occurred while loading the AI brain: {e}")
    st.stop()

# Display the entire chat history from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# The Chat Input Box
if prompt := st.chat_input("Ask a question about the research"):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get the AI's response
    with st.spinner("The AI is thinking..."):
        response = qa_chain.invoke({
            "question": prompt,
            "chat_history": st.session_state.messages
        })
        # Add the AI's response to the session state
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

    st.rerun()