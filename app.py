# app.py

import streamlit as st
from backend import get_qa_chain

st.title("PhD to Parent 🎓")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load the RAG chain
try:
    qa_chain = get_qa_chain()
    st.toast("AI brain is ready!", icon="🧠")
except Exception as e:
    st.error(f"An error occurred while loading the AI brain: {e}")
    st.stop()

# Display past messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# The Chat Input Box
if prompt := st.chat_input("Ask a question about the research"):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the AI's response, now passing in the history
    with st.chat_message("assistant"):
        with st.spinner("The AI is thinking..."):
            # This is the new way to call the chain, with history
            response = qa_chain.invoke({
                "question": prompt,
                "chat_history": st.session_state.messages
            })
            st.markdown(response["answer"])

    # Add the AI's response to the session state
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})