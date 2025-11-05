# app.py

import streamlit as st
from backend import get_qa_chain

st.title("PhD to Parent 🎓")

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    qa_chain = get_qa_chain()
    st.toast("AI brain is ready!", icon="🧠")
except Exception as e:
    st.error(f"An error occurred while loading the AI brain: {e}")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the research"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("The AI is thinking..."):
            # This is the new way to call the chain
            response = qa_chain.invoke({"input": prompt})
            # The answer is now in a key called "answer"
            st.markdown(response["answer"])

    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})