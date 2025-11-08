# app.py

import streamlit as st
from backend import get_qa_chain
from langchain_core.messages import HumanMessage, AIMessage

st.title("Andrew's Research Sherpa 🏔️")

# --- Custom CSS for Chat Bubbles ---
css = """
<style>
    .st-emotion-cache-1c7y2kd {
        padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    [data-testid="chat-message-container"]:has(div[data-testid="stChatMessageContent-assistant"]) {
        background-color: #f0f2f6;
    }
    [data-testid="chat-message-container"]:has(div[data-testid="stChatMessageContent-user"]) {
        background-color: #dcf8c6;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# --- Main App Logic ---

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    # Load the chain only once and store it in session state
    try:
        st.session_state.chain = get_qa_chain()
    except Exception as e:
        st.error(f"An error occurred while loading the AI brain: {e}")
        st.stop()

# --- Display Onboarding OR Chat History ---

# If the conversation is new, display the special onboarding section.
if len(st.session_state.chat_history) == 0:
    st.chat_message("assistant").markdown(
        "Hey there! I'm Andrew's Research Sherpa—your guide through the 'mountain' of his academic work. I've read all his papers so you don't have to. What are you curious about first?"
    )

    st.markdown(
        "From my research mountaineering, I find these to be great starting points, but if you're curious about something else, feel free to ask in the chat box below!")

    col1, col2, col3 = st.columns(3)
    if col1.button("What does Andrew do, in simple terms?"):
        st.session_state.chat_history.append(HumanMessage(content="What does Andrew do, in simple terms?"))
        st.rerun()
    if col2.button("What is 'Knowledge Representation?'"):
        st.session_state.chat_history.append(HumanMessage(content="What is 'Knowledge Representation?'"))
        st.rerun()
    if col3.button("Show me a real-world example of his work."):
        st.session_state.chat_history.append(HumanMessage(content="Show me a real-world example of his work."))
        st.rerun()

# If the conversation has started, display the history and handle new turns.
else:
    # Display all past messages
    for message in st.session_state.chat_history:
        with st.chat_message("assistant" if isinstance(message, AIMessage) else "user"):
            st.markdown(message.content)

    # Check if the last message was from the user (and needs a response)
    if isinstance(st.session_state.chat_history[-1], HumanMessage):
        with st.chat_message("assistant"):
            with st.spinner("The Sherpa is thinking..."):
                response = st.session_state.chain.invoke({
                    "input": st.session_state.chat_history[-1].content,
                    "chat_history": st.session_state.chat_history[:-1]  # Pass all history *except* the latest question
                })
                st.session_state.chat_history.append(AIMessage(content=response["answer"]))
                st.rerun()  # Rerun to display the new AI message

# --- The regular chat input box ---
if prompt := st.chat_input("Ask a follow-up, or your own question..."):
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.rerun()