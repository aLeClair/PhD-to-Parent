import streamlit as st
from backend import get_qa_chain

st.title("PhD to Parent")

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    qa_chain = get_qa_chain()
except Exception as e:
    st.error(f"Error loading AI brain: {e}")
    st.stop()

# Display greeting only if the conversation is new
if len(st.session_state.messages) == 0:
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I'm an AI assistant who has read all of Andrew's research. "
                   "Ask me anything and I'll explain it clearly!"
    })

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
if prompt := st.chat_input("Ask a question about the research"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get response from QA chain
            response = qa_chain(prompt, st.session_state.messages)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
