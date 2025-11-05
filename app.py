import streamlit as st
from backend import get_qa_chain
from langchain_core.messages import HumanMessage, AIMessage

st.title("PhD to Parent")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(
            content="Hello! I'm an AI assistant who has read all of Andrew's research. My purpose is to help you understand his work. Think of me as a friendly translator for his complex ideas. Please feel free to ask me anything!")
    ]

try:
    qa_chain = get_qa_chain()
except Exception as e:
    st.error(f"An error occurred while loading the AI brain: {e}")
    st.stop()

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

if prompt := st.chat_input("Ask a question about the research"):
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("The AI is thinking..."):
            response = qa_chain.invoke({
                "input": prompt,
                "chat_history": st.session_state.chat_history
            })
            st.markdown(response["answer"])

    st.session_state.chat_history.append(AIMessage(content=response["answer"]))