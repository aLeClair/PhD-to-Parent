import streamlit as st
from backend import get_qa_chain
from langchain_core.messages import HumanMessage, AIMessage

st.title("PhD to Parent")

# --- Initialize chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(
            content=(
                "Hello! I'm an AI assistant who has read all of Andrew's research. "
                "My purpose is to help you understand his work. Think of me as a friendly translator "
                "for his complex ideas. Please feel free to ask me anything!"
            )
        )
    ]

# --- Load QA chain once ---
if "qa_chain" not in st.session_state:
    try:
        st.session_state.qa_chain = get_qa_chain()
    except Exception as e:
        st.error(f"An error occurred while loading the AI brain: {e}")
        st.stop()

qa_chain = st.session_state.qa_chain

# --- Display previous chat ---
for message in st.session_state.chat_history:
    role = "assistant" if isinstance(message, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(message.content)

# --- Handle new input ---
if prompt := st.chat_input("Ask a question about the research"):
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("The AI is thinking..."):
            response = qa_chain(prompt, st.session_state.chat_history)
            st.markdown(response["answer"])

    st.session_state.chat_history.append(AIMessage(content=response["answer"]))
