import streamlit as st
from backend import get_qa_chain
from langchain_core.messages import HumanMessage, AIMessage

st.title("Andrew's Research Sherpa 🏔️")

def handle_chat_turn(prompt):
    """
    Adds the user's prompt to the history, gets the AI's response,
    and adds that to the history as well.
    """
    # Add user message to history
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    # Get the AI's response
    with st.spinner("The Sherpa is thinking..."):
        response = qa_chain.invoke({
            "input": prompt,
            "chat_history": st.session_state.chat_history
        })
        # Add AI response to history
        st.session_state.chat_history.append(AIMessage(content=response["answer"]))

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

try:
    qa_chain = get_qa_chain()
except Exception as e:
    st.error(f"An error occurred while loading the AI brain: {e}")
    st.stop()

# Display the greeting and suggested questions ONLY if the conversation is new
if len(st.session_state.chat_history) == 0:
    # Add the initial greeting to the history
    st.session_state.chat_history.append(
        AIMessage(
            content="Hey there! I'm Andrew's Research Sherpa—your guide through the 'mountain' of his academic work. I've read all his papers so you don't have to. What are you curious about first?")
    )

    # Display the full chat history (which is just the greeting at this point)
    for message in st.session_state.chat_history:
        with st.chat_message("assistant"):
            st.markdown(message.content)

    st.markdown(
        "From my research mountaineering, I find these to be great starting points, but if you're curious about something else, feel free to ask in the chat box below!")

    # Create columns for the suggested question buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("What does Andrew do, in simple terms?"):
            handle_chat_turn("What does Andrew do, in simple terms?")
            st.rerun()  # Rerun to display the new messages
    with col2:
        if st.button("What is 'Knowledge Representation?'"):
            handle_chat_turn("What is 'Knowledge Representation?'")
            st.rerun()
    with col3:
        if st.button("Show me a real-world example of his work."):
            handle_chat_turn("Show me a real-world example of his work.")
            st.rerun()

else:  # If the conversation has already started, just display the full history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

# --- The regular chat input box ---
if prompt := st.chat_input("Ask a follow-up question..."):
    handle_chat_turn(prompt)
    st.rerun()  # Rerun to display the new messages